"""
Local Embedding Engine for SoloLLM.

Provides document and query embedding using sentence-transformers.
Falls back to a TF-IDF based approach if sentence-transformers is not installed.
"""

import logging
import numpy as np

from core.config import settings

logger = logging.getLogger(__name__)

# Try to load sentence-transformers
_model = None
_model_name = None


def _load_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazily load the embedding model."""
    global _model, _model_name

    if _model is not None and _model_name == model_name:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        logger.info(f"Embedding model loaded: {model_name} (dim={_model.get_sentence_embedding_dimension()})")
        return _model
    except ImportError:
        logger.warning("sentence-transformers not installed. Using fallback TF-IDF embeddings.")
        return None
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return None


class EmbeddingEngine:
    """
    Embedding engine that generates vector representations of text.

    Uses sentence-transformers when available, falls back to
    TF-IDF + hashing for basic vector similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self._fallback_mode = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Returns a list of float vectors.
        """
        if not texts:
            return []

        model = _load_model(self.model_name)

        if model is not None:
            try:
                embeddings = model.encode(
                    texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Embedding failed, falling back to TF-IDF: {e}")

        # Fallback: TF-IDF-like hash embeddings
        self._fallback_mode = True
        return [self._hash_embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query."""
        results = self.embed([query])
        return results[0] if results else [0.0] * self.dimension

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of documents."""
        return self.embed(documents)

    def _hash_embed(self, text: str) -> list[float]:
        """
        Fallback embedding using character-level hashing.
        Produces a deterministic vector that captures some text similarity.
        """
        import hashlib

        vec = [0.0] * self.dimension
        words = text.lower().split()

        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for d in range(self.dimension):
                idx = (h + d * 7919) % self.dimension
                vec[idx] += 1.0 / (1 + i * 0.01)  # position-weighted

        # Normalize
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @property
    def is_fallback(self) -> bool:
        return self._fallback_mode

    def get_info(self) -> dict:
        """Get info about the current embedding engine."""
        model = _load_model(self.model_name)
        if model is not None:
            return {
                "model": self.model_name,
                "dimension": model.get_sentence_embedding_dimension(),
                "fallback": False,
            }
        return {
            "model": "hash-fallback",
            "dimension": self.dimension,
            "fallback": True,
        }


# Singleton instance
embedding_engine = EmbeddingEngine(
    model_name=settings.embedding_model_name,
    dimension=settings.embedding_dimension,
)
