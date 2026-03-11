"""
Context Distillation API Endpoints for SoloLLM — Phase 3.

Provides endpoints for:
- Distillation settings management
- Distilled RAG queries
- Chain-of-density summarization
- Self-verification
- Confidence scoring
- Distillation metrics dashboard
"""

import json
import logging

from fastapi import APIRouter, HTTPException

from core.config import settings
from core.inference import ollama_client
from core.distillation import (
    distillation_pipeline,
    context_compressor,
    dedup_engine,
    prompt_engine,
    query_decomposer,
    confidence_scorer,
    conversation_memory,
)
from storage.database import (
    get_messages,
    get_conversation,
    get_distillation_metrics,
    save_distillation_metric,
)
from storage.schemas import (
    DistillationSettings,
    ChainOfDensityRequest,
    SelfVerifyRequest,
    DistilledQueryRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/distillation", tags=["Distillation"])


# ── Settings ────────────────────────────────────────────────

@router.get("/settings")
async def get_distillation_settings():
    """Get current distillation pipeline settings."""
    return {
        "settings": {
            "distillation_enabled": settings.distillation_enabled,
            "context_compression": settings.context_compression,
            "compression_target_ratio": settings.compression_target_ratio,
            "deduplication_enabled": settings.deduplication_enabled,
            "dedup_similarity_threshold": settings.dedup_similarity_threshold,
            "adaptive_prompts": settings.adaptive_prompts,
            "query_decomposition": settings.query_decomposition,
            "multi_hop_retrieval": settings.multi_hop_retrieval,
            "multi_hop_max_hops": settings.multi_hop_max_hops,
            "self_verification": settings.self_verification,
            "chain_of_density": settings.chain_of_density,
            "chain_of_density_iterations": settings.chain_of_density_iterations,
            "confidence_scoring": settings.confidence_scoring,
            "conversation_memory_compression": settings.conversation_memory_compression,
            "max_recent_messages": settings.max_recent_messages,
            "max_memory_tokens": settings.max_memory_tokens,
        }
    }


@router.put("/settings")
async def update_distillation_settings(update: DistillationSettings):
    """Update distillation pipeline settings (runtime only)."""
    updates = update.model_dump(exclude_none=True)
    for key, value in updates.items():
        if hasattr(settings, key):
            object.__setattr__(settings, key, value)
            logger.info(f"Distillation setting updated: {key}={value}")

    return await get_distillation_settings()


# ── Distilled Query ─────────────────────────────────────────

@router.post("/query")
async def distilled_query(request: DistilledQueryRequest):
    """
    Run a distillation-enhanced RAG query.

    Returns the full pipeline result including confidence scores,
    compression stats, and the distilled system prompt.
    """
    try:
        from rag.pipeline import rag_pipeline

        # Get conversation history if provided
        conversation_messages = None
        if request.conversation_id:
            conv = await get_conversation(request.conversation_id)
            if conv:
                msgs = await get_messages(request.conversation_id)
                conversation_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in msgs
                    if m["role"] in ("user", "assistant")
                ]

        result = await rag_pipeline.distilled_query(
            query=request.query,
            workspace_id=request.workspace_id,
            top_k=request.top_k,
            conversation_messages=conversation_messages,
        )

        # Save metric
        await save_distillation_metric({
            "conversation_id": request.conversation_id or "",
            "query": request.query[:500],
            "compression_ratio": result["compression_ratio"],
            "confidence_score": result["confidence"].get("overall", 0),
            "confidence_level": result["confidence"].get("level", ""),
            "retrieval_quality": result["confidence"].get("retrieval_quality", 0),
            "coverage": result["confidence"].get("coverage", 0),
            "source_diversity": result["confidence"].get("source_diversity", 0),
            "query_type": result["query_type"],
            "sub_queries": json.dumps(result["sub_queries"]),
            "hops_used": result["hops_used"],
            "verified": False,
        })

        return {
            "confidence": result["confidence"],
            "query_type": result["query_type"],
            "sub_queries": result["sub_queries"],
            "compression": {
                "ratio": result["compression_ratio"],
                "original_tokens": result["original_tokens"],
                "compressed_tokens": result["compressed_tokens"],
            },
            "retrieval": {
                "hops_used": result["hops_used"],
                "chunks_before_dedup": result["chunks_before_dedup"],
                "chunks_after_dedup": result["chunks_after_dedup"],
            },
            "citations": [
                {
                    "index": c.index,
                    "document_title": c.document_title,
                    "section_title": c.section_title,
                    "page_number": c.page_number,
                    "excerpt": c.excerpt,
                }
                for c in result["cited_context"].citations
            ] if result.get("cited_context") else [],
            "context_preview": result["processed_context"][:500] + "..."
            if len(result.get("processed_context", "")) > 500
            else result.get("processed_context", ""),
        }

    except Exception as e:
        logger.error(f"Distilled query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Query Analysis ──────────────────────────────────────────

@router.post("/analyze-query")
async def analyze_query(query: str):
    """
    Analyze a query without running retrieval.

    Returns query classification, decomposition, and template selection.
    """
    query_type = prompt_engine.classify_query(query)
    sub_queries = query_decomposer.decompose(query)

    return {
        "query": query,
        "query_type": query_type,
        "sub_queries": sub_queries,
        "template_used": query_type,
        "decomposed": len(sub_queries) > 1,
    }


# ── Chain-of-Density ────────────────────────────────────────

@router.post("/chain-of-density")
async def run_chain_of_density(request: ChainOfDensityRequest):
    """
    Run chain-of-density summarization on provided content.

    Uses the LLM to iteratively produce denser summaries.
    """
    if not await ollama_client.is_available():
        raise HTTPException(status_code=503, detail="Ollama is not running.")

    model = request.model or settings.default_model

    async def _llm_fn(messages):
        result = await ollama_client.chat(messages=messages, model=model, temperature=0.3)
        return result["content"]

    try:
        summary = await distillation_pipeline.run_chain_of_density(
            content=request.content,
            llm_fn=_llm_fn,
            iterations=min(request.iterations, 5),
        )
        return {"summary": summary, "iterations": request.iterations}
    except Exception as e:
        logger.error(f"Chain-of-density failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Self-Verification ──────────────────────────────────────

@router.post("/verify")
async def verify_response(request: SelfVerifyRequest):
    """
    Verify a response against source context.

    Uses the LLM to check factual accuracy.
    """
    if not await ollama_client.is_available():
        raise HTTPException(status_code=503, detail="Ollama is not running.")

    model = request.model or settings.default_model

    async def _llm_fn(messages):
        result = await ollama_client.chat(messages=messages, model=model, temperature=0.1)
        return result["content"]

    try:
        result = await distillation_pipeline.run_self_verification(
            response=request.response,
            context=request.context,
            query=request.query,
            llm_fn=_llm_fn,
        )
        return result
    except Exception as e:
        logger.error(f"Self-verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Compression Preview ────────────────────────────────────

@router.post("/compress")
async def compress_text(text: str, query: str = ""):
    """Preview context compression on provided text."""
    result = context_compressor.compress(text, query)
    return {
        "original_tokens": result.original_tokens,
        "compressed_tokens": result.compressed_tokens,
        "compression_ratio": result.compression_ratio,
        "compressed_text": result.compressed_text[:2000],
    }


# ── Metrics ─────────────────────────────────────────────────

@router.get("/metrics")
async def get_metrics(conversation_id: str | None = None, limit: int = 50):
    """Get distillation metrics, optionally filtered by conversation."""
    metrics = await get_distillation_metrics(conversation_id, limit)

    # Compute aggregate stats
    if metrics:
        avg_confidence = sum(m.get("confidence_score", 0) for m in metrics) / len(metrics)
        avg_compression = sum(m.get("compression_ratio", 0) for m in metrics) / len(metrics)
        total_verified = sum(1 for m in metrics if m.get("verified"))
        query_types = {}
        for m in metrics:
            qt = m.get("query_type", "unknown")
            query_types[qt] = query_types.get(qt, 0) + 1
    else:
        avg_confidence = 0
        avg_compression = 0
        total_verified = 0
        query_types = {}

    return {
        "metrics": metrics,
        "summary": {
            "total_queries": len(metrics),
            "avg_confidence": round(avg_confidence, 3),
            "avg_compression_ratio": round(avg_compression, 3),
            "total_verified": total_verified,
            "query_type_distribution": query_types,
        },
    }
