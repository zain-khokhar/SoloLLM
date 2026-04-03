import json
import re
import logging
import urllib.parse

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from core.config import settings
from core.inference import ollama_client
from core.token_budget import TokenTracker, estimate_token_count
from core.kv_cache import kv_cache_manager
from core.context_manager import thread_context_builder
from core.intent_classifier import classify as classify_intent
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
from storage.schemas import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Web Search helpers ──────────────────────────────────────

class WebSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model: str | None = None
    conversation_id: str | None = None
    thread_id: str | None = None
    num_results: int = 5


def _search_duckduckgo(query: str, num_results: int = 5) -> list[dict]:
    """Search DuckDuckGo HTML and return structured results."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    with httpx.Client(timeout=15, follow_redirects=True) as client:
        response = client.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    results = []
    links = re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        response.text, re.DOTALL,
    )
    for href, title, snippet in links[:num_results]:
        clean_title = re.sub(r'<[^>]+>', '', title).strip()
        clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
        if clean_title:
            results.append({"title": clean_title, "snippet": clean_snippet, "url": href})
    # Fallback: titles only
    if not results:
        titles = re.findall(r'<a[^>]+class="result__a"[^>]*>(.*?)</a>', response.text)
        for t in titles[:num_results]:
            clean = re.sub(r'<[^>]+>', '', t).strip()
            if clean:
                results.append({"title": clean, "snippet": "", "url": ""})
    return results


async def _scrape_url_text(url: str, max_chars: int = 2000) -> str:
    """Fetch a URL and extract clean text."""
    try:
        from bs4 import BeautifulSoup
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)
        return text[:max_chars] if len(text) > max_chars else text
    except Exception:
        return ""


@router.post("/chat/web-search")
async def web_search_chat(request: WebSearchRequest):
    """Search the web, scrape top results, and stream an LLM-summarized answer."""
    model = request.model or settings.default_model

    # Validate Ollama
    if not await ollama_client.is_available():
        raise HTTPException(status_code=503, detail="Ollama is not running.")

    # Validate model exists; fall back to first available
    try:
        installed = await ollama_client.list_models()
        installed_names = [m["name"] for m in installed]
        if model not in installed_names and installed_names:
            model = installed_names[0]
    except Exception:
        pass

    async def event_generator():
        # Step 1: Search
        yield {"event": "status", "data": json.dumps({"phase": "searching", "message": f"Searching the web for: {request.query}"})}

        try:
            search_results = _search_duckduckgo(request.query, request.num_results)
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"content": f"Web search failed: {e}"})}
            return

        if not search_results:
            yield {"event": "error", "data": json.dumps({"content": "No search results found. Try a different query."})}
            return

        yield {"event": "status", "data": json.dumps({"phase": "scraping", "message": f"Reading {len(search_results)} results..."})}

        # Step 2: Scrape top 3 result pages for more content
        scraped_texts = []
        for result in search_results[:3]:
            if result.get("url"):
                content = await _scrape_url_text(result["url"], max_chars=1500)
                if content:
                    scraped_texts.append({"title": result["title"], "content": content})

        # Build context from search results
        context_parts = []
        for i, r in enumerate(search_results, 1):
            entry = f"[{i}] {r['title']}"
            if r.get("snippet"):
                entry += f"\n{r['snippet']}"
            context_parts.append(entry)

        search_context = "\n\n".join(context_parts)

        # Add scraped content for richer context
        if scraped_texts:
            scraped_section = "\n\n---\n\n".join(
                f"Content from \"{s['title']}\":\n{s['content']}" for s in scraped_texts
            )
            # Keep total context under ~4000 chars for small models
            if len(search_context) + len(scraped_section) > 4000:
                scraped_section = scraped_section[:4000 - len(search_context)]
            search_context += "\n\n--- Detailed Content ---\n\n" + scraped_section

        yield {"event": "status", "data": json.dumps({"phase": "answering", "message": "Generating answer..."})}
        yield {"event": "sources", "data": json.dumps({"results": search_results})}

        # Step 3: Ask the LLM to summarize with a small-model-friendly prompt
        system_prompt = (
            "You are a helpful assistant. The user asked a question and web search results are provided below. "
            "Use ONLY the provided search results to answer. Be concise and direct. "
            "If the results don't contain the answer, say so.\n\n"
            f"=== WEB SEARCH RESULTS ===\n{search_context}\n=== END RESULTS ==="
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.query},
        ]

        # Resolve conversation/thread for saving if provided
        conversation_id = request.conversation_id
        thread_id = request.thread_id

        if conversation_id:
            if not thread_id:
                default_thread = await get_default_thread(conversation_id)
                if default_thread:
                    thread_id = default_thread["id"]
        else:
            title = f"Web: {request.query[:45]}"
            if len(request.query) > 45:
                title += "..."
            conversation = await create_conversation(title=title, model=model, system_prompt="")
            conversation_id = conversation["id"]
            thread_id = conversation.get("default_thread_id")
            if not thread_id:
                default_thread = await get_default_thread(conversation_id)
                if default_thread:
                    thread_id = default_thread["id"]

        # Save user message
        if thread_id:
            await add_message(
                conversation_id, "user", f"[Web Search] {request.query}",
                estimate_token_count(request.query), thread_id=thread_id,
            )
            assistant_msg = await add_message(conversation_id, "assistant", "", 0, thread_id=thread_id)
            message_id = assistant_msg["id"]
        else:
            message_id = None

        # Stream LLM response
        full_content = ""
        try:
            async for chunk in ollama_client.chat_stream(
                messages=messages, model=model,
                max_tokens=settings.max_tokens, temperature=0.3,
            ):
                if chunk["type"] == "token":
                    token = chunk["content"]
                    full_content += token
                    yield {"event": "token", "data": json.dumps({"content": token})}
                elif chunk["type"] == "done":
                    pass

            # Save assistant message
            if message_id:
                await update_message(message_id, full_content, estimate_token_count(full_content))

            yield {"event": "done", "data": json.dumps({
                "message_id": message_id,
                "conversation_id": conversation_id,
                "thread_id": thread_id,
            })}

        except Exception as e:
            yield {"event": "error", "data": json.dumps({"content": f"LLM error: {e}"})}

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


async def _stream_chat(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    conversation_id: str,
    message_id: str,
    thread_id: str | None = None,
    distillation_meta: dict | None = None,
    document_ids_used: list[str] | None = None,
):
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

        # Save message to database
        token_count = eval_count if eval_count > 0 else estimate_token_count(full_content)
        await update_message(
            message_id,
            full_content,
            token_count,
            documents_used=document_ids_used or [],
        )

        # Track KV-cache state
        kv_cache_manager.save_cache_state(conversation_id, model, token_count)

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

    if not thread_id:
        raise HTTPException(status_code=400, detail="No thread found for this conversation. Cannot proceed without context isolation.")

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

    # Smart Context: classify intent to determine optimal history depth
    intent = classify_intent(request.message, len(db_messages))
    logger.info(f"Intent classified: {intent['intent']} (confidence={intent['confidence']:.2f}, depth={intent['history_depth']})")

    # Apply intent-based message filtering before context building
    if intent["history_depth"] == 0:
        # Standalone — only include the current user message
        context_messages = [msg for msg in db_messages if msg.get("role") == "user"][-1:]
    elif intent["history_depth"] == -1:
        # Full history — include all messages
        context_messages = db_messages
    else:
        # Recent context — take the last N messages
        depth = intent["history_depth"]
        context_messages = db_messages[-depth:] if len(db_messages) > depth else db_messages

    # Determine system prompt — thread-specific takes priority
    system_prompt = request.system_prompt or ""
    if thread and thread.get("system_prompt"):
        system_prompt = thread["system_prompt"]
    elif not system_prompt and conversation.get("system_prompt"):
        system_prompt = conversation["system_prompt"]

    distillation_meta = None
    document_ids_used: list[str] = []

    # Determine which documents are attached to this thread (for scoped RAG)
    thread_doc_ids = None
    if thread_id:
        thread_docs = await get_thread_documents(thread_id)
        if thread_docs:
            thread_doc_ids = [d["document_id"] for d in thread_docs]

    documents_only_mode = bool(getattr(request, "documents_only", False))
    if documents_only_mode and not thread_doc_ids:
        raise HTTPException(
            status_code=400,
            detail="Document-only mode requires at least one document attached to this thread.",
        )

    if documents_only_mode:
        # Strict mode: ignore prior assistant/user history and answer from thread documents only.
        context_messages = [msg for msg in db_messages if msg.get("role") == "user"][-1:]

    # Phase 3: Run distillation pipeline if enabled and RAG is enabled for this thread
    # CRITICAL: Only run RAG if this thread has attached documents.
    # If no documents are attached, skip entirely — the thread should start clean.
    rag_enabled_for_thread = (thread_settings_data or {}).get("rag_enabled", True)
    # Smart Context: only run RAG when intent says so AND conditions are met
    if documents_only_mode:
        should_run_rag = bool(thread_doc_ids)
    else:
        should_run_rag = intent["needs_rag"] or (settings.distillation_enabled and rag_enabled_for_thread and thread_doc_ids)

    rag_context_applied = False
    if should_run_rag and thread_doc_ids:
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
                if documents_only_mode:
                    strict_mode_prefix = (
                        "DOCUMENT-ONLY MODE: Answer using ONLY the retrieved sources from the documents "
                        "attached to this thread. Do not use web knowledge, prior assumptions, or "
                        "outside information. If the answer is not in the sources, say exactly that."
                    )
                    system_prompt = f"{strict_mode_prefix}\n\n{system_prompt}".strip()
                rag_context_applied = True

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
                            "document_id": c.document_id,
                        }
                        for c in distill_result["cited_context"].citations
                    ] if distill_result.get("cited_context") else [],
                }

                if distill_result.get("cited_context"):
                    document_ids_used = sorted(
                        {
                            c.document_id
                            for c in distill_result["cited_context"].citations
                            if getattr(c, "document_id", None)
                        }
                    )

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

    # Fallback for strict document-only mode if distillation produced no usable context.
    if documents_only_mode and thread_doc_ids and not rag_context_applied:
        try:
            from rag.pipeline import rag_pipeline
            from rag.citations import citation_tracker

            workspace_id = request.workspace_id if hasattr(request, 'workspace_id') else "default"
            rag_top_k = (thread_settings_data or {}).get("rag_top_k", 5)
            cited = await rag_pipeline.query(
                query=request.message,
                workspace_id=workspace_id,
                top_k=rag_top_k,
                document_ids=thread_doc_ids,
                rerank=True,
            )
            retrieved_context = citation_tracker.format_system_context(cited)
            strict_mode_prompt = (
                "DOCUMENT-ONLY MODE: Answer using ONLY the retrieved sources from documents attached "
                "to this thread. If sources are insufficient, say: 'I could not find this in the "
                "documents attached to this thread.'"
            )
            if retrieved_context:
                system_prompt = f"{strict_mode_prompt}\n\n{retrieved_context}\n\n{system_prompt}".strip()
            else:
                system_prompt = (
                    f"{strict_mode_prompt}\n\n"
                    "No relevant passages were retrieved from the attached thread documents for this question."
                )
        except Exception as e:
            logger.warning(f"Document-only fallback retrieval failed: {e}")

    # Build context using thread-aware context builder (ISOLATED)
    # Use intent-filtered messages instead of all db_messages
    if not document_ids_used and documents_only_mode and thread_doc_ids:
        document_ids_used = sorted(set(thread_doc_ids))

    ollama_messages = thread_context_builder.build_isolated_context(
        thread_messages=context_messages,
        system_prompt=system_prompt,
        thread_settings=thread_settings_data,
        current_query=request.message,
        thread_id=thread_id,
    )

    # Create placeholder assistant message — scoped to this thread
    assistant_msg = await add_message(conversation_id, "assistant", "", 0, thread_id=thread_id)
    message_id = assistant_msg["id"]

    return EventSourceResponse(
        _stream_chat(
            ollama_messages,
            model,
            max_tokens,
            temperature,
            conversation_id,
            message_id,
            thread_id,
            distillation_meta,
            document_ids_used=document_ids_used,
        ),
        media_type="text/event-stream",
    )


