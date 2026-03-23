"""
Cross-Encoder Reranker for SoloLLM.

Re-scores retrieved results using a cross-encoder model
for more accurate relevance scoring. Falls back to a
heuristic reranker when the model is unavailable.
"""

import logging
import re
from dataclasses import dataclass

from core.config import settings

logger = logging.getLogger(__name__)

_cross_encoder = None
_cross_encoder_failed = False


def _load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazily load the cross-encoder model."""
    global _cross_encoder, _cross_encoder_failed

    if not settings.reranker_enabled and settings.rag_precision_mode != "precision_fusion":
        return None

    if _cross_encoder is not None:
        return _cross_encoder
    if _cross_encoder_failed:
        return None

    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder: {model_name}")

        cross_encoder_kwargs = {}
        if settings.reranker_local_files_only:
            cross_encoder_kwargs = {
                "automodel_args": {"local_files_only": True},
                "tokenizer_args": {"local_files_only": True},
            }

        _cross_encoder = CrossEncoder(model_name, **cross_encoder_kwargs)
        logger.info(f"Cross-encoder loaded: {model_name}")
        return _cross_encoder
    except ImportError:
        _cross_encoder_failed = True
        logger.warning("sentence-transformers not installed. Using heuristic reranker.")
        return None
    except Exception as e:
        _cross_encoder_failed = True
        logger.error(f"Failed to load cross-encoder: {e}")
        return None


class Reranker:
    """
    Re-ranks retrieval results for better relevance.

    Uses a cross-encoder model when available (much more accurate
    than bi-encoder similarity). Falls back to heuristic scoring
    based on term overlap, position, and section relevance.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = settings.reranker_model_name or model_name

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> dict:
        """
        Re-rank results by relevance to the query.

        Returns dict with:
        - results: top_k most relevant results with updated scores
        - confidence: average score of top results (0-1 range)
        """
        if not results:
            return {"results": [], "confidence": 0.0}

        # Try cross-encoder first
        model = _load_cross_encoder(self.model_name)

        if model is not None:
            reranked = self._cross_encoder_rerank(model, query, results, top_k)
        else:
            reranked = self._heuristic_rerank(query, results, top_k)

        # Apply score threshold if provided
        if score_threshold is not None:
            reranked = [r for r in reranked if r.score >= score_threshold]

        # Compute confidence as average score of returned results
        confidence = 0.0
        if reranked:
            confidence = sum(r.score for r in reranked) / len(reranked)
            confidence = min(max(confidence, 0.0), 1.0)

        return {"results": reranked, "confidence": round(confidence, 4)}

    def _cross_encoder_rerank(self, model, query: str, results: list, top_k: int) -> list:
        """Re-rank using cross-encoder model."""
        try:
            pairs = [(query, r.content) for r in results]
            scores = model.predict(pairs)

            for result, score in zip(results, scores):
                result.score = float(score)

            results.sort(key=lambda r: r.score, reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return self._heuristic_rerank(query, results, top_k)

    def _heuristic_rerank(self, query: str, results: list, top_k: int) -> list:
        """
        Heuristic reranking based on multiple signals:
        1. Term overlap (exact matches)
        2. Query term coverage (what fraction of query terms appear)
        3. Section title relevance
        4. Content length preference (not too short, not too long)
        """
        query_terms = set(self._tokenize(query.lower()))

        for result in results:
            content_lower = result.content.lower()
            content_terms = set(self._tokenize(content_lower))

            # Feature 1: Term overlap ratio
            overlap = query_terms & content_terms
            coverage = len(overlap) / max(len(query_terms), 1)

            # Feature 2: Exact phrase match bonus
            phrase_bonus = 0.2 if query.lower() in content_lower else 0

            # Feature 3: Section title relevance
            section_title = getattr(result, 'section_title', '') or ''
            title_terms = set(self._tokenize(section_title.lower()))
            title_overlap = len(query_terms & title_terms) / max(len(query_terms), 1)

            # Feature 4: Content length quality score
            content_len = len(result.content)
            if content_len < 50:
                length_score = 0.3
            elif content_len < 200:
                length_score = 0.7
            elif content_len < 1000:
                length_score = 1.0
            else:
                length_score = 0.8

            # Combined heuristic score
            heuristic_score = (
                coverage * 0.4
                + phrase_bonus * 0.2
                + title_overlap * 0.15
                + length_score * 0.1
                + result.score * 0.15  # Original retrieval score
            )

            result.score = heuristic_score

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for heuristic scoring."""
        return [w for w in re.findall(r'\w+', text) if len(w) > 1]

    def warmup(self):
        """Pre-load the cross-encoder model to avoid first-request latency."""
        model = _load_cross_encoder(self.model_name)
        if model is not None:
            try:
                # Run a dummy prediction to warm up
                model.predict([("warmup query", "warmup content")])
                logger.info("Reranker model warmed up successfully")
            except Exception as e:
                logger.warning(f"Reranker warmup failed: {e}")
        else:
            logger.info("Reranker warmup skipped (model not available)")


# Singleton
reranker = Reranker()
