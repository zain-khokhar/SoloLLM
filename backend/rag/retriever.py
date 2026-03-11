"""
Hybrid Retriever for SoloLLM.

Combines vector search (semantic) + BM25 keyword search (lexical)
using Reciprocal Rank Fusion (RRF) to produce a single ranked list
of results that captures both semantic meaning and exact matches.
"""

import logging
from dataclasses import dataclass, field

from rag.vectorstore import vector_store, SearchResult
from rag.keyword_index import keyword_index, KeywordResult
from rag.embeddings import embedding_engine

logger = logging.getLogger(__name__)


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


class HybridRetriever:
    """
    Hybrid retriever combining vector similarity and BM25 keyword search.

    Uses Reciprocal Rank Fusion (RRF) to merge rankings:
    RRF_score = 1 / (k + rank_vector) + 1 / (k + rank_keyword)

    The 'k' parameter (default 60) controls how much weight is given
    to lower-ranked results.
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
        vector_top_k: int = 20,
        keyword_top_k: int = 20,
    ) -> list[RetrievalResult]:
        """
        Perform hybrid retrieval: vector + keyword search with RRF fusion.

        Args:
            query: The search query
            workspace_id: Workspace to search in
            top_k: Number of final results to return
            document_id: Optional filter to a specific document
            vector_top_k: Number of vector results to fetch before fusion
            keyword_top_k: Number of keyword results to fetch before fusion
        """
        # Get query embedding
        query_embedding = embedding_engine.embed_query(query)

        # Run both searches in parallel conceptually (sequential for simplicity)
        vector_results = await vector_store.search(
            query_embedding=query_embedding,
            workspace_id=workspace_id,
            top_k=vector_top_k,
            document_id=document_id,
        )

        keyword_results = await keyword_index.search(
            query=query,
            workspace_id=workspace_id,
            top_k=keyword_top_k,
            document_id=document_id,
        )

        # Fuse results using RRF
        fused = self._rrf_fusion(vector_results, keyword_results)

        logger.info(
            f"Hybrid retrieval: query='{query[:50]}...', "
            f"vector={len(vector_results)}, keyword={len(keyword_results)}, "
            f"fused={len(fused)}"
        )

        return fused[:top_k]

    def _rrf_fusion(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[KeywordResult],
    ) -> list[RetrievalResult]:
        """
        Reciprocal Rank Fusion.

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
