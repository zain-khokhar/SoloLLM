import json
import logging

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from core.config import settings
from core.inference import ollama_client
from core.token_budget import TokenTracker, estimate_token_count
from core.continuation import detect_truncation, build_continuation_messages, stitch_content
from core.kv_cache import kv_cache_manager
from core.distillation import (
    distillation_pipeline, conversation_memory, confidence_scorer,
    self_verifier, prompt_engine,
)
from storage.database import (
    create_conversation, add_message, get_messages,
    get_conversation, get_message, update_message,
    save_conversation_summary, get_conversation_summary,
    save_distillation_metric,
)
from storage.schemas import ChatRequest, ContinueRequest

logger = logging.getLogger(__name__)
router = APIRouter()


async def _stream_chat(messages: list[dict], model: str, max_tokens: int, temperature: float, conversation_id: str, message_id: str, distillation_meta: dict | None = None):
    """Generator that streams SSE events for a chat response."""
    tracker = TokenTracker(max_tokens=max_tokens)
    full_content = ""
    done_reason = "stop"
    eval_count = 0

    # Emit distillation metadata event at the start if present
    if distillation_meta:
        yield {
            "event": "distillation",
            "data": json.dumps(distillation_meta),
        }

    try:
        async for chunk in ollama_client.chat_stream(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            if chunk["type"] == "token":
                token = chunk["content"]
                tracker.add_token(token)
                full_content += token
                yield {
                    "event": "token",
                    "data": json.dumps({"content": token}),
                }

            elif chunk["type"] == "done":
                eval_count = chunk.get("eval_count", 0)
                done_reason = chunk.get("done_reason", "stop")
                tracker.set_final_count(eval_count)

        # Check for truncation
        truncation = detect_truncation(full_content, done_reason, eval_count, max_tokens)

        # Save message to database
        token_count = eval_count if eval_count > 0 else estimate_token_count(full_content)
        await update_message(message_id, full_content, token_count)

        # Track KV-cache state
        kv_cache_manager.save_cache_state(conversation_id, model, token_count)

        if truncation["is_truncated"] and settings.truncation_detection:
            yield {
                "event": "truncated",
                "data": json.dumps({
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "tokens_used": token_count,
                    "reason": truncation["reason"],
                    "confidence": truncation["confidence"],
                    "last_content": truncation["last_content"],
                }),
            }
        else:
            yield {
                "event": "done",
                "data": json.dumps({
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "tokens_used": token_count,
                    "truncated": False,
                }),
            }

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)}),
        }


@router.post("/chat")
async def chat(request: ChatRequest):
    """Start or continue a conversation with streaming SSE response."""
    model = request.model or settings.default_model
    max_tokens = request.max_tokens or settings.max_tokens
    temperature = request.temperature if request.temperature is not None else settings.temperature

    # Check Ollama availability
    if not await ollama_client.is_available():
        raise HTTPException(status_code=503, detail="Ollama is not running. Please start Ollama first.")

    # Create or get conversation
    if request.conversation_id:
        conversation = await get_conversation(request.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation_id = request.conversation_id
    else:
        # New conversation — use first 50 chars of message as title
        title = request.message[:50].strip()
        if len(request.message) > 50:
            title += "..."
        conversation = await create_conversation(title=title, model=model, system_prompt=request.system_prompt or "")
        conversation_id = conversation["id"]

    # Save user message
    await add_message(conversation_id, "user", request.message, estimate_token_count(request.message))

    # Build message list for Ollama
    db_messages = await get_messages(conversation_id)
    ollama_messages = []

    # Determine base system prompt
    system_prompt = request.system_prompt or (conversation.get("system_prompt") if isinstance(conversation, dict) else "")

    distillation_meta = None

    # Phase 3: Run distillation pipeline if enabled and RAG documents exist
    if settings.distillation_enabled:
        try:
            from rag.pipeline import rag_pipeline
            workspace_id = request.workspace_id if hasattr(request, 'workspace_id') else "default"

            # Get conversation history for memory compression
            conv_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in db_messages
                if msg["role"] in ("user", "assistant")
            ]

            # Run distilled query through the pipeline
            distill_result = await rag_pipeline.distilled_query(
                query=request.message,
                workspace_id=workspace_id,
                top_k=5,
                conversation_messages=conv_history[:-1] if len(conv_history) > 1 else None,
                base_system_prompt=system_prompt or "",
            )

            # Use the distillation-enhanced system prompt if we got context
            if distill_result.get("processed_context"):
                system_prompt = distill_result["system_prompt"]

                # Build metadata for frontend
                distillation_meta = {
                    "confidence": distill_result["confidence"],
                    "query_type": distill_result["query_type"],
                    "sub_queries": distill_result["sub_queries"],
                    "compression_ratio": distill_result["compression_ratio"],
                    "original_tokens": distill_result["original_tokens"],
                    "compressed_tokens": distill_result["compressed_tokens"],
                    "hops_used": distill_result["hops_used"],
                    "chunks_before_dedup": distill_result["chunks_before_dedup"],
                    "chunks_after_dedup": distill_result["chunks_after_dedup"],
                    "citations": [
                        {
                            "index": c.index,
                            "document_title": c.document_title,
                            "section_title": c.section_title,
                            "page_number": c.page_number,
                        }
                        for c in distill_result["cited_context"].citations
                    ] if distill_result.get("cited_context") else [],
                }

                # Save distillation metric
                await save_distillation_metric({
                    "conversation_id": conversation_id,
                    "query": request.message[:500],
                    "compression_ratio": distill_result["compression_ratio"],
                    "confidence_score": distill_result["confidence"].get("overall", 0),
                    "confidence_level": distill_result["confidence"].get("level", ""),
                    "retrieval_quality": distill_result["confidence"].get("retrieval_quality", 0),
                    "coverage": distill_result["confidence"].get("coverage", 0),
                    "source_diversity": distill_result["confidence"].get("source_diversity", 0),
                    "query_type": distill_result["query_type"],
                    "sub_queries": json.dumps(distill_result["sub_queries"]),
                    "hops_used": distill_result["hops_used"],
                    "verified": False,
                })

                logger.info(
                    f"Distillation: confidence={distill_result['confidence']['level']}, "
                    f"compression={distill_result['compression_ratio']:.2f}, "
                    f"type={distill_result['query_type']}, "
                    f"hops={distill_result['hops_used']}"
                )
            else:
                # No RAG context — still apply conversation memory if enabled
                if settings.conversation_memory_compression and len(conv_history) > settings.max_recent_messages:
                    memory_ctx = conversation_memory.build_memory_context(conv_history[:-1])
                    if memory_ctx and system_prompt:
                        system_prompt += f"\n\n--- CONVERSATION CONTEXT ---\n{memory_ctx}"
                    elif memory_ctx:
                        system_prompt = f"You are a helpful assistant.\n\n--- CONVERSATION CONTEXT ---\n{memory_ctx}"

        except Exception as e:
            logger.warning(f"Distillation pipeline error (falling back to standard): {e}")

    # Build Ollama messages
    if system_prompt:
        ollama_messages.append({"role": "system", "content": system_prompt})

    # Add conversation history
    for msg in db_messages:
        if msg["role"] in ("user", "assistant"):
            ollama_messages.append({"role": msg["role"], "content": msg["content"]})

    # Create placeholder assistant message
    assistant_msg = await add_message(conversation_id, "assistant", "", 0)
    message_id = assistant_msg["id"]

    return EventSourceResponse(
        _stream_chat(ollama_messages, model, max_tokens, temperature, conversation_id, message_id, distillation_meta),
        media_type="text/event-stream",
    )


@router.post("/chat/continue")
async def continue_chat(request: ContinueRequest):
    """Continue a truncated response from where it stopped."""
    # Get conversation and message
    conversation = await get_conversation(request.conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    original_message = await get_message(request.message_id)
    if not original_message:
        raise HTTPException(status_code=404, detail="Message not found")

    if original_message["role"] != "assistant":
        raise HTTPException(status_code=400, detail="Can only continue assistant messages")

    if not await ollama_client.is_available():
        raise HTTPException(status_code=503, detail="Ollama is not running.")

    model = conversation["model"]
    max_tokens = settings.max_tokens
    original_content = original_message["content"]

    # Get all messages up to (but not including) the truncated message
    all_messages = await get_messages(request.conversation_id)
    prior_messages = []
    for msg in all_messages:
        if msg["id"] == request.message_id:
            break
        prior_messages.append({"role": msg["role"], "content": msg["content"]})

    # Build continuation messages
    system_prompt = conversation.get("system_prompt", "")
    if system_prompt:
        context_messages = [{"role": "system", "content": system_prompt}] + prior_messages
    else:
        context_messages = prior_messages

    continuation_messages = build_continuation_messages(
        context_messages, original_content
    )

    async def _stream_continuation():
        tracker = TokenTracker(max_tokens=max_tokens)
        continuation_content = ""
        done_reason = "stop"
        eval_count = 0

        try:
            async for chunk in ollama_client.chat_stream(
                messages=continuation_messages,
                model=model,
                max_tokens=max_tokens,
            ):
                if chunk["type"] == "token":
                    token = chunk["content"]
                    tracker.add_token(token)
                    continuation_content += token
                    yield {
                        "event": "token",
                        "data": json.dumps({"content": token}),
                    }

                elif chunk["type"] == "done":
                    eval_count = chunk.get("eval_count", 0)
                    done_reason = chunk.get("done_reason", "stop")
                    tracker.set_final_count(eval_count)

            # Stitch the content together
            full_content = stitch_content(original_content, continuation_content)
            total_tokens = (original_message.get("token_count", 0) or 0) + eval_count

            # Update the original message with stitched content
            await update_message(request.message_id, full_content, total_tokens)

            # Track continuation in KV-cache
            kv_cache_manager.record_continuation(request.conversation_id, eval_count)

            # Check if this continuation is also truncated
            truncation = detect_truncation(continuation_content, done_reason, eval_count, max_tokens)

            if truncation["is_truncated"] and settings.truncation_detection:
                yield {
                    "event": "truncated",
                    "data": json.dumps({
                        "message_id": request.message_id,
                        "conversation_id": request.conversation_id,
                        "tokens_used": total_tokens,
                        "reason": truncation["reason"],
                        "confidence": truncation["confidence"],
                        "last_content": truncation["last_content"],
                    }),
                }
            else:
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "message_id": request.message_id,
                        "conversation_id": request.conversation_id,
                        "tokens_used": total_tokens,
                        "truncated": False,
                    }),
                }

        except Exception as e:
            logger.error(f"Continuation streaming error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(
        _stream_continuation(),
        media_type="text/event-stream",
    )
