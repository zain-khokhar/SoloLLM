"""
KV-Cache Manager for SoloLLM.

Tracks KV-cache state per conversation for auto-continuation.
With Ollama as the inference backend, Ollama manages the actual KV cache
internally. This module provides a tracking/metadata layer so the
application can reason about cache availability and make decisions
about continuation strategy.
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Metadata for a single conversation's KV-cache state."""
    conversation_id: str
    model: str
    token_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    continuation_count: int = 0
    is_warm: bool = True

    def touch(self):
        self.last_accessed = time.time()

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_accessed


class KVCacheManager:
    """
    Manages KV-cache metadata for conversations.

    This is a lightweight tracking layer. The actual KV cache lives
    inside Ollama's runtime. This manager tracks which conversations
    have been recently active (and therefore likely still have warm
    caches in Ollama) so the continuation system can decide whether
    to attempt a fast resume or fall back to full context rebuild.
    """

    def __init__(self, max_entries: int = 20, ttl_seconds: float = 300.0):
        self._cache: dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds

    def save_cache_state(
        self,
        conversation_id: str,
        model: str,
        token_count: int,
    ) -> CacheEntry:
        """Record that a conversation has an active KV-cache."""
        self._evict_stale()

        entry = CacheEntry(
            conversation_id=conversation_id,
            model=model,
            token_count=token_count,
        )
        self._cache[conversation_id] = entry
        logger.debug(
            f"KV-cache state saved: conv={conversation_id}, "
            f"model={model}, tokens={token_count}"
        )
        return entry

    def get_cache_state(self, conversation_id: str) -> CacheEntry | None:
        """
        Get the cache state for a conversation.
        Returns None if no cache exists or it has expired.
        """
        entry = self._cache.get(conversation_id)
        if entry is None:
            return None

        if entry.age_seconds > self._ttl_seconds:
            del self._cache[conversation_id]
            logger.debug(f"KV-cache expired: conv={conversation_id}")
            return None

        entry.touch()
        return entry

    def is_cache_warm(self, conversation_id: str) -> bool:
        """Check if a conversation likely has a warm KV-cache."""
        entry = self.get_cache_state(conversation_id)
        return entry is not None and entry.is_warm

    def record_continuation(self, conversation_id: str, additional_tokens: int):
        """Record that a continuation was performed using this cache."""
        entry = self._cache.get(conversation_id)
        if entry:
            entry.token_count += additional_tokens
            entry.continuation_count += 1
            entry.touch()
            logger.debug(
                f"Continuation recorded: conv={conversation_id}, "
                f"total_tokens={entry.token_count}, "
                f"continuations={entry.continuation_count}"
            )

    def clear_cache(self, conversation_id: str):
        """Clear the cache state for a conversation."""
        if conversation_id in self._cache:
            del self._cache[conversation_id]
            logger.debug(f"KV-cache cleared: conv={conversation_id}")

    def clear_all(self):
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("All KV-cache entries cleared")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        self._evict_stale()
        return {
            "active_entries": len(self._cache),
            "max_entries": self._max_entries,
            "total_tokens_cached": sum(e.token_count for e in self._cache.values()),
            "entries": [
                {
                    "conversation_id": e.conversation_id,
                    "model": e.model,
                    "token_count": e.token_count,
                    "age_seconds": round(e.age_seconds, 1),
                    "continuation_count": e.continuation_count,
                    "is_warm": e.is_warm,
                }
                for e in self._cache.values()
            ],
        }

    def _evict_stale(self):
        """Remove expired entries and enforce max size."""
        # Remove expired
        expired = [
            cid for cid, entry in self._cache.items()
            if entry.age_seconds > self._ttl_seconds
        ]
        for cid in expired:
            del self._cache[cid]

        # Enforce max size — remove oldest first
        if len(self._cache) > self._max_entries:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_accessed,
            )
            to_remove = len(self._cache) - self._max_entries
            for cid, _ in sorted_entries[:to_remove]:
                del self._cache[cid]


# Singleton instance
kv_cache_manager = KVCacheManager()
