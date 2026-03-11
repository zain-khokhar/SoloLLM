"""
Cross-Encoder Reranker for SoloLLM.

Re-scores retrieved results using a cross-encoder model
for more accurate relevance scoring. Falls back to a
heuristic reranker when the model is unavailable.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_cross_encoder = None


def _load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazily load the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder: {model_name}")
        _cross_encoder = CrossEncoder(model_name)
        logger.info(f"Cross-encoder loaded: {model_name}")
        return _cross_encoder
    except ImportError:
        logger.warning("sentence-transformers not installed. Using heuristic reranker.")
        return None
    except Exception as e:
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
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int = 5,
    ) -> list:
        """
        Re-rank results by relevance to the query.

        Returns the top_k most relevant results with updated scores.
        """
        if not results:
            return []

        # Try cross-encoder first
        model = _load_cross_encoder(self.model_name)

        if model is not None:
            return self._cross_encoder_rerank(model, query, results, top_k)
        else:
            return self._heuristic_rerank(query, results, top_k)

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


# Singleton
reranker = Reranker()
