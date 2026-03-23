"""
Hybrid Retriever for SoloLLM.

Combines vector search (semantic) + BM25 keyword search (lexical)
using Reciprocal Rank Fusion (RRF) to produce a single ranked list
of results that captures both semantic meaning and exact matches.

Precision upgrade: adds precision_fusion mode with calibrated score
fusion, hard threshold gates, per-document caps, MMR diversity
selection, and retrieval diagnostics for observability.
"""

import logging
import math
from dataclasses import dataclass, field

from rag.vectorstore import vector_store, SearchResult
from rag.keyword_index import keyword_index, KeywordResult
from rag.embeddings import embedding_engine
from rag.query_understanding import query_analyzer

logger = logging.getLogger(__name__)


@dataclass
class RetrievalDiagnostics:
    """Observability info for a precision-mode retrieval."""
    vector_candidates: int = 0
    vector_after_threshold: int = 0
    keyword_candidates: int = 0
    keyword_after_coverage: int = 0
    fused_candidates: int = 0
    after_doc_cap: int = 0
    after_mmr: int = 0
    final_count: int = 0
    precision_mode: str = "legacy_rrf"
    query_type: str = "default"
    ambiguity_score: float = 0.0


@dataclass
class RetrievalResult:
    """A unified retrieval result from hybrid search."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    document_title: str = ""
    section_title: str = ""
    page_number: int | None = None
    chunk_index: int = 0
    source: str = "hybrid"  # "vector", "keyword", or "hybrid"
    metadata: dict = field(default_factory=dict)
    # Precision metadata
    term_coverage: float = 0.0
    phrase_match_score: float = 0.0
    fused_score: float = 0.0
    score_normalized: float = 0.0


class HybridRetriever:
    """
    Hybrid retriever combining vector similarity and BM25 keyword search.

    Supports two modes:
    1. legacy_rrf: Reciprocal Rank Fusion (original behavior)
    2. precision_fusion: Calibrated score fusion with hard gates, MMR, and per-doc caps
    """

    def __init__(
        self,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        workspace_id: str = "default",
        top_k: int = 10,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        vector_top_k: int = 20,
        keyword_top_k: int = 20,
        # Precision mode parameters
        precision_mode: str = "legacy_rrf",
        vector_min_score: float = 0.28,
        lexical_required_coverage: float = 0.5,
        per_document_cap: int = 0,
        use_mmr: bool = False,
        mmr_lambda: float = 0.65,
        candidate_pool_size: int = 80,
    ) -> tuple[list[RetrievalResult], RetrievalDiagnostics | None]:
        """
        Perform hybrid retrieval: vector + keyword search with fusion.

        Args:
            precision_mode: 'legacy_rrf' or 'precision_fusion'
            vector_min_score: Minimum normalized vector similarity (precision mode)
            lexical_required_coverage: Minimum term coverage for keyword results
            per_document_cap: Max chunks per document_id (0 = unlimited)
            use_mmr: Apply MMR diversity selection
            mmr_lambda: MMR relevance-diversity tradeoff (higher = more relevance)
            candidate_pool_size: How many candidates to fetch from each source

        Returns:
            (results, diagnostics) — diagnostics is None in legacy mode
        """
        is_precision = precision_mode == "precision_fusion"

        # Query understanding (for precision mode)
        analysis = None
        if is_precision:
            analysis = query_analyzer.analyze(query)

        # Get query embedding
        query_embedding = embedding_engine.embed_query(query)

        # Determine fetch sizes
        vec_fetch = candidate_pool_size if is_precision else vector_top_k
        kw_fetch = candidate_pool_size if is_precision else keyword_top_k

        # Vector search
        vector_results = await vector_store.search(
            query_embedding=query_embedding,
            workspace_id=workspace_id,
            top_k=vec_fetch,
            document_id=document_id,
            document_ids=document_ids,
            score_threshold=vector_min_score if is_precision else None,
            fetch_k=candidate_pool_size if is_precision else None,
        )

        # Keyword search
        keyword_results = await keyword_index.search(
            query=query,
            workspace_id=workspace_id,
            top_k=kw_fetch,
            document_id=document_id,
            document_ids=document_ids,
            precision_mode=is_precision,
            required_terms=analysis.required_terms if analysis else None,
            keyphrases=analysis.keyphrases if analysis else None,
        )

        if is_precision:
            return self._precision_fusion(
                query, vector_results, keyword_results, top_k,
                vector_min_score, lexical_required_coverage,
                per_document_cap, use_mmr, mmr_lambda, analysis,
            )
        else:
            # Legacy RRF path
            fused = self._rrf_fusion(vector_results, keyword_results)
            logger.info(
                f"Hybrid retrieval: query='{query[:50]}...', "
                f"vector={len(vector_results)}, keyword={len(keyword_results)}, "
                f"fused={len(fused)}"
            )
            return fused[:top_k], None

    def _precision_fusion(
        self,
        query: str,
        vector_results: list[SearchResult],
        keyword_results: list[KeywordResult],
        top_k: int,
        vector_min_score: float,
        lexical_required_coverage: float,
        per_document_cap: int,
        use_mmr: bool,
        mmr_lambda: float,
        analysis,
    ) -> tuple[list[RetrievalResult], RetrievalDiagnostics]:
        """Precision-gated calibrated score fusion pipeline."""

        diag = RetrievalDiagnostics(
            precision_mode="precision_fusion",
            query_type=analysis.query_type if analysis else "default",
            ambiguity_score=analysis.ambiguity_score if analysis else 0.0,
        )

        # ── Stage 1: Collect and gate vector candidates ──────
        diag.vector_candidates = len(vector_results)
        # Already threshold-filtered by vectorstore, but double-check
        gated_vec = [
            vr for vr in vector_results
            if vr.score_normalized >= vector_min_score
        ]
        diag.vector_after_threshold = len(gated_vec)

        # ── Stage 2: Gate keyword candidates by coverage ─────
        diag.keyword_candidates = len(keyword_results)
        gated_kw = [
            kr for kr in keyword_results
            if kr.term_coverage >= lexical_required_coverage or kr.phrase_match > 0
        ]
        diag.keyword_after_coverage = len(gated_kw)

        # ── Stage 3: Calibrated score fusion ─────────────────
        # Weights: wv=0.45, wl=0.35, wp=0.15, ws=0.05
        wv, wl, wp, ws = 0.45, 0.35, 0.15, 0.05

        candidates: dict[str, dict] = {}

        for vr in gated_vec:
            candidates[vr.chunk_id] = {
                "chunk_id": vr.chunk_id,
                "document_id": vr.document_id,
                "content": vr.content,
                "document_title": vr.document_title,
                "section_title": vr.section_title,
                "page_number": vr.page_number,
                "chunk_index": vr.chunk_index,
                "metadata": vr.metadata,
                "vec_norm": vr.score_normalized,
                "lex_norm": 0.0,
                "phrase_bonus": 0.0,
                "section_signal": 0.0,
                "vector_score": vr.score,
                "keyword_score": 0.0,
                "term_coverage": 0.0,
                "phrase_match_score": 0.0,
                "source": "vector",
            }

        for kr in gated_kw:
            # Section signal: does the section title overlap with query?
            section_signal = 0.0
            if kr.section_title:
                q_words = set(query.lower().split())
                sec_words = set(kr.section_title.lower().split())
                section_signal = len(q_words & sec_words) / max(len(q_words), 1)

            if kr.chunk_id in candidates:
                c = candidates[kr.chunk_id]
                c["lex_norm"] = kr.bm25_normalized
                c["phrase_bonus"] = kr.phrase_match
                c["section_signal"] = max(c["section_signal"], section_signal)
                c["keyword_score"] = kr.score
                c["term_coverage"] = kr.term_coverage
                c["phrase_match_score"] = kr.phrase_match
                c["source"] = "hybrid"
            else:
                candidates[kr.chunk_id] = {
                    "chunk_id": kr.chunk_id,
                    "document_id": kr.document_id,
                    "content": kr.content,
                    "document_title": kr.document_title,
                    "section_title": kr.section_title,
                    "page_number": kr.page_number,
                    "chunk_index": kr.chunk_index,
                    "metadata": {},
                    "vec_norm": 0.0,
                    "lex_norm": kr.bm25_normalized,
                    "phrase_bonus": kr.phrase_match,
                    "section_signal": section_signal,
                    "vector_score": 0.0,
                    "keyword_score": kr.score,
                    "term_coverage": kr.term_coverage,
                    "phrase_match_score": kr.phrase_match,
                    "source": "keyword",
                }

        # Compute fused score
        for c in candidates.values():
            c["fused_score"] = (
                wv * c["vec_norm"]
                + wl * c["lex_norm"]
                + wp * c["phrase_bonus"]
                + ws * c["section_signal"]
            )

        # Sort by fused score
        ranked = sorted(candidates.values(), key=lambda x: x["fused_score"], reverse=True)
        diag.fused_candidates = len(ranked)

        # ── Stage 4: Per-document cap ────────────────────────
        if per_document_cap > 0:
            doc_counts: dict[str, int] = {}
            capped = []
            for c in ranked:
                doc_id = c["document_id"]
                count = doc_counts.get(doc_id, 0)
                if count < per_document_cap:
                    capped.append(c)
                    doc_counts[doc_id] = count + 1
            ranked = capped
        diag.after_doc_cap = len(ranked)

        # ── Stage 5: MMR diversity selection ─────────────────
        if use_mmr and len(ranked) > top_k:
            ranked = self._mmr_select(ranked, top_k, mmr_lambda)
        diag.after_mmr = len(ranked)

        # Final truncation
        ranked = ranked[:top_k]
        diag.final_count = len(ranked)

        results = [
            RetrievalResult(
                chunk_id=c["chunk_id"],
                document_id=c["document_id"],
                content=c["content"],
                score=c["fused_score"],
                vector_score=c["vector_score"],
                keyword_score=c["keyword_score"],
                document_title=c["document_title"],
                section_title=c["section_title"],
                page_number=c["page_number"],
                chunk_index=c["chunk_index"],
                source=c["source"],
                metadata=c["metadata"],
                term_coverage=c["term_coverage"],
                phrase_match_score=c["phrase_match_score"],
                fused_score=c["fused_score"],
                score_normalized=c["vec_norm"],
            )
            for c in ranked
        ]

        logger.info(
            "Precision retrieval: query='%s...', vec=%d→%d, kw=%d→%d, "
            "fused=%d, capped=%d, mmr=%d, final=%d",
            query[:50],
            diag.vector_candidates, diag.vector_after_threshold,
            diag.keyword_candidates, diag.keyword_after_coverage,
            diag.fused_candidates, diag.after_doc_cap,
            diag.after_mmr, diag.final_count,
        )

        return results, diag

    def _mmr_select(
        self,
        candidates: list[dict],
        k: int,
        lambda_param: float = 0.65,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance selection.

        MMR(d) = lambda * Rel(d,q) - (1-lambda) * max_{s in S} Sim(d,s)

        Uses word-overlap as a fast similarity proxy.
        """
        if not candidates or k <= 0:
            return []

        selected: list[dict] = []
        remaining = list(candidates)

        # Pre-compute word sets for fast similarity
        word_sets = {}
        for c in remaining:
            words = set(c["content"].lower().split())
            word_sets[c["chunk_id"]] = words

        while len(selected) < k and remaining:
            best_mmr = -float("inf")
            best_idx = 0

            for i, cand in enumerate(remaining):
                rel = cand["fused_score"]

                # Max similarity to already-selected
                max_sim = 0.0
                cand_words = word_sets.get(cand["chunk_id"], set())
                for sel in selected:
                    sel_words = word_sets.get(sel["chunk_id"], set())
                    union_len = len(cand_words | sel_words)
                    if union_len > 0:
                        sim = len(cand_words & sel_words) / union_len
                        max_sim = max(max_sim, sim)

                mmr = lambda_param * rel - (1 - lambda_param) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _rrf_fusion(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[KeywordResult],
    ) -> list[RetrievalResult]:
        """
        Reciprocal Rank Fusion (legacy mode).

        Merges two ranked lists into one, giving credit to items
        that appear in both lists.
        """
        scores: dict[str, dict] = {}

        # Process vector results
        for rank, vr in enumerate(vector_results):
            rrf_score = self.vector_weight / (self.rrf_k + rank + 1)
            if vr.chunk_id not in scores:
                scores[vr.chunk_id] = {
                    "chunk_id": vr.chunk_id,
                    "document_id": vr.document_id,
                    "content": vr.content,
                    "document_title": vr.document_title,
                    "section_title": vr.section_title,
                    "page_number": vr.page_number,
                    "chunk_index": vr.chunk_index,
                    "metadata": vr.metadata,
                    "rrf_score": 0.0,
                    "vector_score": vr.score,
                    "keyword_score": 0.0,
                    "source": "vector",
                }
            scores[vr.chunk_id]["rrf_score"] += rrf_score
            scores[vr.chunk_id]["vector_score"] = vr.score

        # Process keyword results
        for rank, kr in enumerate(keyword_results):
            rrf_score = self.keyword_weight / (self.rrf_k + rank + 1)
            if kr.chunk_id not in scores:
                scores[kr.chunk_id] = {
                    "chunk_id": kr.chunk_id,
                    "document_id": kr.document_id,
                    "content": kr.content,
                    "document_title": kr.document_title,
                    "section_title": kr.section_title,
                    "page_number": kr.page_number,
                    "chunk_index": kr.chunk_index,
                    "metadata": {},
                    "rrf_score": 0.0,
                    "vector_score": 0.0,
                    "keyword_score": 0.0,
                    "source": "keyword",
                }
            scores[kr.chunk_id]["rrf_score"] += rrf_score
            scores[kr.chunk_id]["keyword_score"] = kr.score

            # Mark as hybrid if found in both
            if scores[kr.chunk_id]["vector_score"] > 0:
                scores[kr.chunk_id]["source"] = "hybrid"

        # Sort by RRF score
        ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)

        return [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                content=r["content"],
                score=r["rrf_score"],
                vector_score=r["vector_score"],
                keyword_score=r["keyword_score"],
                document_title=r["document_title"],
                section_title=r["section_title"],
                page_number=r["page_number"],
                chunk_index=r["chunk_index"],
                source=r["source"],
                metadata=r["metadata"],
            )
            for r in ranked
        ]


# Singleton
hybrid_retriever = HybridRetriever()
