import json
import logging

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from core.config import settings
from core.inference import ollama_client
from core.token_budget import TokenTracker, estimate_token_count
from core.continuation import detect_truncation, build_continuation_messages, stitch_content
from core.kv_cache import kv_cache_manager
from core.context_manager import thread_context_builder
from core.distillation import (
    distillation_pipeline, conversation_memory, confidence_scorer,
    self_verifier, prompt_engine,
)
from storage.database import (
    create_conversation, add_message, get_messages,
    get_conversation, get_message, update_message,
    save_conversation_summary, get_conversation_summary,
    save_distillation_metric,
    get_default_thread, get_thread, get_thread_settings,
    get_thread_documents,
)
from storage.schemas import ChatRequest, ContinueRequest

logger = logging.getLogger(__name__)
router = APIRouter()


async def _stream_chat(messages: list[dict], model: str, max_tokens: int, temperature: float, conversation_id: str, message_id: str, thread_id: str | None = None, distillation_meta: dict | None = None):
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
                    "thread_id": thread_id,
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
                    "thread_id": thread_id,
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
    """Start or continue a conversation with streaming SSE response.
    
    Context Isolation Architecture:
    - Every conversation has threads
    - Each thread has its own isolated context (messages, documents, settings)
    - No context from other threads or conversations leaks into the current thread
    - thread_id is required for context-aware chat; if omitted, the default thread is used
    """
    model = request.model or settings.default_model
    temperature = request.temperature if request.temperature is not None else settings.temperature

    # Check Ollama availability
    if not await ollama_client.is_available():
        raise HTTPException(status_code=503, detail="Ollama is not running. Please start Ollama first.")

    # Validate the model is actually installed
    try:
        installed = await ollama_client.list_models()
        installed_names = [m["name"] for m in installed]
        if model not in installed_names:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' is not installed. Available: {', '.join(installed_names) or 'none'}. Please pull the model first or choose an installed one.",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # If we can't check, proceed anyway

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

    # Resolve thread — CRITICAL for context isolation
    thread_id = request.thread_id
    if not thread_id:
        # Use default thread for this conversation
        default_thread = await get_default_thread(conversation_id)
        if default_thread:
            thread_id = default_thread["id"]
        else:
            # Fallback: if conversation was created above, use the auto-created thread
            thread_id = conversation.get("default_thread_id")

    # Load thread settings for this specific thread
    thread_settings_data = None
    thread = None
    if thread_id:
        thread = await get_thread(thread_id)
        thread_settings_data = await get_thread_settings(thread_id)

    # Resolve max_tokens and temperature from thread settings if available
    max_tokens = request.max_tokens or (thread_settings_data or {}).get("max_tokens") or settings.max_tokens
    if thread_settings_data and request.temperature is None:
        temperature = thread_settings_data.get("temperature", temperature)

    # Save user message — scoped to this thread
    await add_message(
        conversation_id, "user", request.message,
        estimate_token_count(request.message),
        thread_id=thread_id,
    )

    # Get ONLY messages for this thread (context isolation!)
    db_messages = await get_messages(conversation_id, thread_id=thread_id)

    # Determine system prompt — thread-specific takes priority
    system_prompt = request.system_prompt or ""
    if thread and thread.get("system_prompt"):
        system_prompt = thread["system_prompt"]
    elif not system_prompt and conversation.get("system_prompt"):
        system_prompt = conversation["system_prompt"]

    distillation_meta = None

    # Determine which documents are attached to this thread (for scoped RAG)
    thread_doc_ids = None
    if thread_id:
        thread_docs = await get_thread_documents(thread_id)
        if thread_docs:
            thread_doc_ids = [d["document_id"] for d in thread_docs]

    # Phase 3: Run distillation pipeline if enabled and RAG is enabled for this thread
    # CRITICAL: Only run RAG if this thread has attached documents.
    # If no documents are attached, skip entirely — the thread should start clean.
    rag_enabled_for_thread = (thread_settings_data or {}).get("rag_enabled", True)
    if settings.distillation_enabled and rag_enabled_for_thread and thread_doc_ids:
        try:
            from rag.pipeline import rag_pipeline
            workspace_id = request.workspace_id if hasattr(request, 'workspace_id') else "default"

            # Get ONLY this thread's conversation history (isolated)
            conv_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in db_messages
                if msg["role"] in ("user", "assistant")
            ]

            rag_top_k = (thread_settings_data or {}).get("rag_top_k", 5)

            # Run distilled query — scoped to ONLY this thread's documents
            distill_result = await rag_pipeline.distilled_query(
                query=request.message,
                workspace_id=workspace_id,
                top_k=rag_top_k,
                conversation_messages=conv_history[:-1] if len(conv_history) > 1 else None,
                base_system_prompt=system_prompt or "",
                document_ids=thread_doc_ids,
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
                    f"Distillation [thread={thread_id}]: confidence={distill_result['confidence']['level']}, "
                    f"compression={distill_result['compression_ratio']:.2f}, "
                    f"type={distill_result['query_type']}, "
                    f"hops={distill_result['hops_used']}"
                )

        except Exception as e:
            logger.warning(f"Distillation pipeline error (falling back to standard): {e}")

    # Build context using thread-aware context builder (ISOLATED)
    ollama_messages = thread_context_builder.build_isolated_context(
        thread_messages=db_messages,
        system_prompt=system_prompt,
        thread_settings=thread_settings_data,
        current_query=request.message,
    )

    # Create placeholder assistant message — scoped to this thread
    assistant_msg = await add_message(conversation_id, "assistant", "", 0, thread_id=thread_id)
    message_id = assistant_msg["id"]

    return EventSourceResponse(
        _stream_chat(ollama_messages, model, max_tokens, temperature, conversation_id, message_id, thread_id, distillation_meta),
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

    # Get thread_id from the original message for context isolation
    thread_id = original_message.get("thread_id")

    # Get only messages from the same thread (context isolation!)
    all_messages = await get_messages(request.conversation_id, thread_id=thread_id)
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
