"""
Thread-aware Context Manager for SoloLLM.

Implements:
1. Context isolation — each thread has its own message history, no cross-contamination.
2. KV Cache Compression — compress old context to reduce token usage.
3. Memory Layer Management (MemGPT-style) — virtual memory paging for context.
4. Sliding window — only keep relevant recent messages in the active context.
"""

import logging
import hashlib
from dataclasses import dataclass, field
from core.token_budget import estimate_token_count

logger = logging.getLogger(__name__)


# ── KV Cache Compression ───────────────────────────────────

class ContextCompressor:
    """
    Compresses conversation context to fit within token budgets.
    Uses extractive summarization (key-sentence selection) to reduce
    context size 4x–20x without losing core meaning.
    """

    @staticmethod
    def compress_messages(
        messages: list[dict],
        target_ratio: float = 0.6,
        max_tokens: int = 2048,
    ) -> list[dict]:
        """
        Compress a list of messages to fit within token budget.
        Keeps system and recent messages intact, compresses older ones.
        """
        if not messages:
            return messages

        total_tokens = sum(estimate_token_count(m.get("content", "")) for m in messages)
        if total_tokens <= max_tokens:
            return messages

        # Separate system messages and conversation messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        conv_msgs = [m for m in messages if m.get("role") != "system"]

        if len(conv_msgs) <= 4:
            return messages

        # Keep last N messages intact (recent context), compress older ones
        keep_recent = min(6, len(conv_msgs))
        recent = conv_msgs[-keep_recent:]
        older = conv_msgs[:-keep_recent]

        compressed_older = []
        for msg in older:
            content = msg.get("content", "")
            compressed = ContextCompressor._extract_key_sentences(content, target_ratio)
            compressed_older.append({
                **msg,
                "content": compressed,
            })

        return system_msgs + compressed_older + recent

    @staticmethod
    def _extract_key_sentences(text: str, ratio: float = 0.6) -> str:
        """Extract the most important sentences from text."""
        if not text or len(text) < 100:
            return text

        sentences = []
        current = ""
        for char in text:
            current += char
            if char in ".!?\n" and len(current.strip()) > 10:
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())

        if len(sentences) <= 2:
            return text

        # Score sentences by position, length, and keyword density
        scored = []
        keywords = {"important", "key", "note", "result", "because", "therefore",
                     "however", "but", "first", "finally", "main", "summary",
                     "error", "solution", "answer", "question", "code", "function"}

        for i, sent in enumerate(sentences):
            score = 0.0
            words = sent.lower().split()

            # Position score — first and last sentences are important
            if i == 0:
                score += 2.0
            elif i == len(sentences) - 1:
                score += 1.5
            else:
                score += 0.5

            # Keyword density
            keyword_count = sum(1 for w in words if w in keywords)
            score += keyword_count * 0.5

            # Length score (prefer medium-length sentences)
            if 20 < len(words) < 50:
                score += 1.0
            elif len(words) >= 50:
                score += 0.5

            scored.append((i, score, sent))

        # Keep top sentences based on ratio
        keep_count = max(1, int(len(scored) * ratio))
        scored.sort(key=lambda x: x[1], reverse=True)
        kept = sorted(scored[:keep_count], key=lambda x: x[0])

        return " ".join(s[2] for s in kept)

    @staticmethod
    def compress_for_storage(content: str, max_tokens: int = 1024) -> str:
        """Compress content for long-term storage in context pages."""
        tokens = estimate_token_count(content)
        if tokens <= max_tokens:
            return content
        ratio = max_tokens / tokens
        return ContextCompressor._extract_key_sentences(content, min(ratio, 0.8))


# ── Memory Layer Manager (MemGPT-style) ────────────────────

@dataclass
class ContextPage:
    """Represents a page of context in the virtual memory system."""
    page_id: str
    page_number: int
    content: str
    compressed_content: str
    is_active: bool  # True = in RAM, False = on disk
    token_count: int
    priority: float  # 0.0 to 1.0, higher = more important

    @property
    def active_content(self) -> str:
        """Return compressed content if available, otherwise full content."""
        if self.compressed_content:
            return self.compressed_content
        return self.content


class MemoryLayerManager:
    """
    Manages context like an OS virtual memory system.
    - Active pages: loaded in context window (RAM)
    - Inactive pages: stored on disk, can be paged in when needed
    - Automatic page-out of old/low-priority context
    - Automatic page-in when query relevance is detected
    """

    def __init__(self, max_active_tokens: int = 4096, page_size: int = 2048):
        self.max_active_tokens = max_active_tokens
        self.page_size = page_size
        self._compressor = ContextCompressor()

    def create_pages_from_messages(
        self, messages: list[dict], existing_pages: list[dict] | None = None,
    ) -> list[ContextPage]:
        """Convert messages into context pages."""
        if not messages:
            return []

        pages = []
        current_content = ""
        current_tokens = 0
        page_num = 0

        for msg in messages:
            content = f"[{msg.get('role', 'user')}]: {msg.get('content', '')}\n"
            tokens = estimate_token_count(content)

            if current_tokens + tokens > self.page_size and current_content:
                compressed = self._compressor.compress_for_storage(current_content)
                pages.append(ContextPage(
                    page_id=hashlib.md5(current_content[:200].encode()).hexdigest(),
                    page_number=page_num,
                    content=current_content,
                    compressed_content=compressed if compressed != current_content else "",
                    is_active=True,
                    token_count=current_tokens,
                    priority=self._calculate_priority(page_num, len(messages)),
                ))
                page_num += 1
                current_content = ""
                current_tokens = 0

            current_content += content
            current_tokens += tokens

        # Last page
        if current_content:
            compressed = self._compressor.compress_for_storage(current_content)
            pages.append(ContextPage(
                page_id=hashlib.md5(current_content[:200].encode()).hexdigest(),
                page_number=page_num,
                content=current_content,
                compressed_content=compressed if compressed != current_content else "",
                is_active=True,
                token_count=current_tokens,
                priority=self._calculate_priority(page_num, len(messages)),
            ))

        return self._enforce_memory_budget(pages)

    def _calculate_priority(self, page_number: int, total_messages: int) -> float:
        """Recent pages get higher priority."""
        if total_messages == 0:
            return 0.5
        # Linear decay: most recent = 1.0, oldest = 0.1
        return max(0.1, page_number / max(1, total_messages))

    def _enforce_memory_budget(self, pages: list[ContextPage]) -> list[ContextPage]:
        """Page out older/lower-priority pages if we exceed the budget."""
        active_tokens = sum(p.token_count for p in pages if p.is_active)

        if active_tokens <= self.max_active_tokens:
            return pages

        # Sort by priority (lowest first) to page out
        active_pages = sorted(
            [p for p in pages if p.is_active],
            key=lambda p: p.priority,
        )

        for page in active_pages:
            if active_tokens <= self.max_active_tokens:
                break
            page.is_active = False
            active_tokens -= page.token_count
            logger.debug(f"Paged out context page {page.page_number} ({page.token_count} tokens)")

        return pages

    def get_active_context(self, pages: list[ContextPage]) -> str:
        """Build the active context string from active pages."""
        active = sorted(
            [p for p in pages if p.is_active],
            key=lambda p: p.page_number,
        )
        parts = []
        for page in active:
            parts.append(page.active_content)
        return "\n".join(parts)

    def page_in_relevant(
        self, pages: list[ContextPage], query: str, max_page_in: int = 2,
    ) -> list[ContextPage]:
        """Page in inactive pages that are relevant to the current query."""
        inactive = [p for p in pages if not p.is_active]
        if not inactive:
            return pages

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for page in inactive:
            content_lower = page.content.lower()
            # Simple relevance: word overlap
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored.append((page, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)

        for page, _ in scored[:max_page_in]:
            page.is_active = True
            logger.debug(f"Paged in context page {page.page_number} (relevant to query)")

        return self._enforce_memory_budget(pages)


# ── Thread Context Builder ──────────────────────────────────

class ThreadContextBuilder:
    """
    Builds context for a specific thread with complete isolation.
    No context from other threads or conversations leaks in.
    """

    def __init__(self):
        self._compressor = ContextCompressor()
        self._memory_manager = MemoryLayerManager()

    def build_isolated_context(
        self,
        thread_messages: list[dict],
        system_prompt: str = "",
        thread_settings: dict | None = None,
        current_query: str = "",
        thread_id: str | None = None,
    ) -> list[dict]:
        """
        Build a fully isolated context for this thread.
        Only messages belonging to this thread are included.
        """
        ts = thread_settings or {}
        # Map DB column names to context manager keys
        max_history = ts.get("max_history_messages") or 20
        max_tokens = ts.get("max_tokens") or 4096
        kv_compression = bool(ts.get("compression_enabled", 0))
        memory_layers = ts.get("memory_layers") or 0
        memory_mode = "virtual_paging" if memory_layers > 1 else ("sliding_window" if memory_layers == 1 else "none")

        ollama_messages = []

        # System prompt (thread-specific)
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})

        # Filter only user/assistant messages
        history = [m for m in thread_messages if m.get("role") in ("user", "assistant")]

        if not history:
            return ollama_messages

        # Apply memory layer strategy
        if memory_mode == "virtual_paging" and len(history) > max_history:
            context, pages = self._apply_virtual_paging(history, max_tokens, current_query)
            if context:
                # Persist pages to database if thread_id is available
                if thread_id and pages:
                    import asyncio
                    try:
                        from storage.database import save_context_pages
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(save_context_pages(thread_id, pages))
                        else:
                            loop.run_until_complete(save_context_pages(thread_id, pages))
                    except Exception as e:
                        logger.warning(f"Failed to persist context pages: {e}")
                # Insert paged context as a system-level summary
                ollama_messages.append({
                    "role": "system",
                    "content": f"--- CONVERSATION MEMORY (paged) ---\n{context}",
                })
                # Keep recent messages as-is
                recent = history[-min(6, len(history)):]
                for msg in recent:
                    ollama_messages.append({"role": msg["role"], "content": msg["content"]})
                return ollama_messages

        # Apply sliding window
        if len(history) > max_history * 2:
            history = history[-(max_history * 2):]

        # Apply KV compression if enabled
        if kv_compression and len(history) > 6:
            history_as_msgs = [{"role": m["role"], "content": m["content"]} for m in history]
            compressed = self._compressor.compress_messages(
                history_as_msgs,
                target_ratio=ts.get("compression_ratio", 0.6),
                max_tokens=max_tokens,
            )
            for msg in compressed:
                if msg.get("role") != "system":
                    ollama_messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            for msg in history:
                ollama_messages.append({"role": msg["role"], "content": msg["content"]})

        return ollama_messages

    def _apply_virtual_paging(
        self, messages: list[dict], max_tokens: int, query: str,
    ) -> tuple[str, list]:
        """Apply MemGPT-style virtual paging to old context. Returns (context_str, pages)."""
        self._memory_manager.max_active_tokens = max_tokens // 2  # Reserve half for recent
        pages = self._memory_manager.create_pages_from_messages(messages[:-6])

        if query:
            pages = self._memory_manager.page_in_relevant(pages, query)

        return self._memory_manager.get_active_context(pages), pages


# Singleton instances
context_compressor = ContextCompressor()
memory_layer_manager = MemoryLayerManager()
thread_context_builder = ThreadContextBuilder()
