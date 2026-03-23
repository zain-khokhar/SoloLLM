"""
Context Distillation Engine for SoloLLM — Phase 3.

Compresses and distills retrieved context so small models
receive pure signal instead of noisy, redundant chunks.

Includes:
- Context compression (remove fluff, keep signal)
- Deduplication (remove near-duplicate chunks)
- Adaptive prompt templates
- Query decomposition
- Chain-of-density prompting
- Self-verification loop
- Confidence scoring
- Multi-hop retrieval
- Conversation memory with compression
- Distillation pipeline orchestrator
"""

import re
import json
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Context Compression ────────────────────────────────────

@dataclass
class CompressedContext:
    """Result of context compression."""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    removed_items: list[str] = field(default_factory=list)


class ContextCompressor:
    """
    Compresses retrieved context by removing noise, redundancy,
    and formatting artifacts that waste precious tokens.
    """

    def __init__(self, target_ratio: float = 0.6):
        self.target_ratio = target_ratio  # Keep ~60% of original

    def compress(self, context: str, query: str = "") -> CompressedContext:
        """
        Compress context by applying multiple strategies:
        1. Remove excessive whitespace and formatting
        2. Remove boilerplate/legal text
        3. Remove sentences with low relevance to query
        4. Consolidate repeated information
        """
        original = context
        original_tokens = len(context.split())

        # Step 1: Clean formatting
        text = self._clean_formatting(context)

        # Step 2: Remove boilerplate
        text = self._remove_boilerplate(text)

        # Step 3: Query-aware filtering
        if query:
            text = self._query_filter(text, query)

        # Step 4: Sentence deduplication
        text = self._dedup_sentences(text)

        compressed_tokens = len(text.split())
        ratio = compressed_tokens / max(original_tokens, 1)

        return CompressedContext(
            original_text=original,
            compressed_text=text.strip(),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=round(ratio, 3),
        )

    def _clean_formatting(self, text: str) -> str:
        """Remove excessive whitespace, blank lines, and formatting noise."""
        # Collapse multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Collapse multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        # Remove page numbers / headers-footers patterns
        text = re.sub(r'\n\s*Page \d+ of \d+\s*\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*-\s*\d+\s*-\s*\n', '\n', text)
        # Remove URLs that are just noise
        text = re.sub(r'https?://\S{80,}', '[URL]', text)
        return text

    def _remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate patterns."""
        boilerplate_patterns = [
            r'(?i)all rights reserved\.?',
            r'(?i)copyright ©?\s*\d{4}.*?\.',
            r'(?i)table of contents',
            r'(?i)this (page|document) (is|was) (intentionally )?(left )?blank',
            r'(?i)for internal use only',
            r'(?i)confidential and proprietary',
            r'(?i)please see the (appendix|glossary)',
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text)
        return text

    def _query_filter(self, text: str, query: str) -> str:
        """Keep sentences that are relevant to the query using keyphrase matching."""
        from rag.query_understanding import query_analyzer

        analysis = query_analyzer.analyze(query)
        keyphrases = analysis.keyphrases
        required_terms = set(analysis.required_terms)

        if not required_terms and not keyphrases:
            return text

        sentences = re.split(r'(?<=[.!?])\s+', text)
        kept = []

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Keep if any keyphrase matches
            if any(kp in sentence_lower for kp in keyphrases):
                kept.append(sentence)
                continue

            # Keep if >= 2 required terms are present (not just one generic token)
            sentence_terms = set(re.findall(r'\w+', sentence_lower))
            matched_required = required_terms & sentence_terms
            if len(matched_required) >= 2:
                kept.append(sentence)
                continue

            # Keep very short connector sentences
            if len(sentence.split()) < 5:
                kept.append(sentence)

        return ' '.join(kept)

    def _dedup_sentences(self, text: str) -> str:
        """Remove near-duplicate sentences using hash comparison."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        seen_hashes = set()
        unique = []

        for sentence in sentences:
            # Normalize for comparison
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            if len(normalized) < 10:
                unique.append(sentence)
                continue

            h = hashlib.md5(normalized.encode()).hexdigest()[:12]
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(sentence)

        return ' '.join(unique)


# ── Chunk Deduplication ─────────────────────────────────────

class DeduplicationEngine:
    """
    Removes near-duplicate chunks from retrieval results.
    Uses SimHash-like fingerprinting for fast comparison.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold

    def deduplicate(self, chunks: list[dict]) -> list[dict]:
        """Remove near-duplicate chunks, keeping the highest-scored one."""
        if len(chunks) <= 1:
            return chunks

        unique = []
        seen_fingerprints: list[set] = []

        for chunk in chunks:
            content = chunk.get("content", "")
            fp = self._fingerprint(content)

            is_dup = False
            for seen_fp in seen_fingerprints:
                similarity = self._jaccard(fp, seen_fp)
                if similarity >= self.threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique.append(chunk)
                seen_fingerprints.append(fp)

        logger.info(f"Dedup: {len(chunks)} → {len(unique)} chunks")
        return unique

    def _fingerprint(self, text: str) -> set:
        """Create a shingle-based fingerprint."""
        words = re.findall(r'\w+', text.lower())
        # 3-word shingles
        shingles = set()
        for i in range(len(words) - 2):
            shingle = f"{words[i]}_{words[i+1]}_{words[i+2]}"
            shingles.add(hashlib.md5(shingle.encode()).hexdigest()[:8])
        return shingles

    def _jaccard(self, a: set, b: set) -> float:
        """Jaccard similarity between two sets."""
        if not a and not b:
            return 1.0
        union = len(a | b)
        return len(a & b) / union if union else 0.0


# ── Adaptive Prompt Templates ──────────────────────────────

class PromptTemplateEngine:
    """
    Selects and fills prompt templates based on query type.

    Templates are optimized for different query patterns:
    - Factual questions → direct answer format
    - Analytical questions → structured analysis format
    - Creative tasks → open-ended format
    - Code questions → code-first format
    - Comparison questions → table/comparative format
    """

    TEMPLATES = {
        "factual": (
            "Answer the following question directly and concisely based on the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Provide a clear, accurate answer. Cite sources using [Source N] notation."
        ),
        "analytical": (
            "Analyze the following question using the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Provide a structured analysis with:\n"
            "1. Key findings\n2. Supporting evidence (cite sources)\n3. Conclusion"
        ),
        "code": (
            "Answer the following code-related question using the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Provide code examples where applicable. Explain the approach, "
            "then show the implementation. Cite sources with [Source N]."
        ),
        "comparison": (
            "Compare and contrast based on the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Structure your response as a clear comparison. "
            "Use a table if appropriate. Cite sources with [Source N]."
        ),
        "creative": (
            "Using the context below as reference, address the following.\n\n"
            "Context:\n{context}\n\n"
            "Task: {query}\n\n"
            "Be thorough and creative while staying grounded in the sources. "
            "Cite relevant sources with [Source N]."
        ),
        "default": (
            "Using the following context, answer the question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Answer accurately based on the context. Cite sources using [Source N] notation."
        ),
    }

    def classify_query(self, query: str) -> str:
        """Classify query type for template selection."""
        q = query.lower().strip()

        # Code-related
        code_signals = ['code', 'function', 'class', 'implement', 'bug', 'error',
                        'syntax', 'api', 'method', 'variable', 'debug', 'script']
        if any(s in q for s in code_signals):
            return "code"

        # Comparison
        comparison_signals = ['compare', 'difference', 'versus', 'vs', 'better',
                              'pros and cons', 'advantages', 'contrast']
        if any(s in q for s in comparison_signals):
            return "comparison"

        # Factual
        factual_signals = ['what is', 'who is', 'when', 'where', 'how many',
                           'define', 'what does', 'name the', 'list']
        if any(q.startswith(s) or s in q for s in factual_signals):
            return "factual"

        # Analytical
        analytical_signals = ['why', 'how does', 'explain', 'analyze', 'evaluate',
                              'assess', 'impact', 'effect', 'cause']
        if any(s in q for s in analytical_signals):
            return "analytical"

        # Creative
        creative_signals = ['write', 'create', 'generate', 'draft', 'compose',
                            'suggest', 'design', 'brainstorm', 'imagine']
        if any(s in q for s in creative_signals):
            return "creative"

        return "default"

    def render(self, query: str, context: str, template_type: str | None = None) -> str:
        """Render a prompt template with query and context."""
        if not template_type:
            template_type = self.classify_query(query)

        template = self.TEMPLATES.get(template_type, self.TEMPLATES["default"])
        return template.format(context=context, query=query)


# ── Query Decomposition ────────────────────────────────────

class QueryDecomposer:
    """
    Decomposes complex queries into simpler sub-queries
    for more targeted retrieval.
    """

    def decompose(self, query: str) -> list[str]:
        """
        Break a complex query into simpler components.

        Uses heuristic patterns — no LLM call needed.
        """
        sub_queries = []

        # Pattern 1: "and" conjunctions
        if " and " in query.lower():
            parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
            if len(parts) > 1 and all(len(p.split()) > 2 for p in parts):
                sub_queries.extend([p.strip() for p in parts])
                return sub_queries

        # Pattern 2: Multiple questions (separated by ?)
        if query.count('?') > 1:
            parts = [p.strip() + '?' for p in query.split('?') if p.strip()]
            if len(parts) > 1:
                return parts

        # Pattern 3: "also" / "additionally"
        split_words = [' also ', ' additionally ', ' furthermore ', ' moreover ']
        for word in split_words:
            if word in query.lower():
                parts = re.split(re.escape(word), query, flags=re.IGNORECASE)
                sub_queries.extend([p.strip() for p in parts if p.strip()])
                return sub_queries

        # No decomposition needed
        return [query]


# ── Chain-of-Density Prompting ─────────────────────────────

class ChainOfDensity:
    """
    Implements Chain-of-Density prompting for progressive
    summarization. Each iteration produces a denser summary
    without losing critical information.
    """

    def build_density_prompt(
        self,
        content: str,
        iteration: int = 1,
        max_iterations: int = 3,
        previous_summary: str = "",
    ) -> str:
        """Build a chain-of-density prompt for iterative summarization."""
        if iteration == 1:
            return (
                f"Summarize the following content in approximately 3-4 sentences. "
                f"Focus on the most important facts and findings.\n\n"
                f"Content:\n{content}\n\n"
                f"Summary:"
            )
        else:
            return (
                f"The following is a summary of a document. Make it denser by:\n"
                f"1. Identifying 1-2 missing important entities or facts\n"
                f"2. Adding them to the summary\n"
                f"3. Keeping the summary the same length or shorter\n"
                f"4. Removing filler words and vague phrases\n\n"
                f"Previous summary:\n{previous_summary}\n\n"
                f"Original content:\n{content[:2000]}\n\n"
                f"Denser summary (iteration {iteration}/{max_iterations}):"
            )


# ── Self-Verification ──────────────────────────────────────

class SelfVerificationLoop:
    """
    Builds verification prompts to check LLM responses
    against retrieved context for factual accuracy.
    """

    def build_verification_prompt(
        self,
        response: str,
        context: str,
        query: str,
    ) -> str:
        """Build a self-verification prompt."""
        return (
            f"Verify the following response against the provided context.\n\n"
            f"Original question: {query}\n\n"
            f"Response to verify:\n{response}\n\n"
            f"Context (source of truth):\n{context}\n\n"
            f"Check for:\n"
            f"1. Factual accuracy — does the response match the context?\n"
            f"2. Unsupported claims — is anything stated that isn't in the context?\n"
            f"3. Completeness — is any critical information from the context missing?\n\n"
            f"If corrections are needed, provide the corrected response. "
            f"If the response is accurate, respond with: VERIFIED."
        )


# ── Confidence Scoring ─────────────────────────────────────

class ConfidenceScorer:
    """
    Assigns confidence scores to RAG responses based on
    retrieval quality and context coverage.
    """

    def score(
        self,
        query: str,
        retrieval_results: list,
        context_text: str,
    ) -> dict:
        """
        Score confidence of a RAG response.

        Returns:
        - overall: 0.0-1.0 aggregate confidence
        - retrieval_quality: How good were the retrieved chunks
        - coverage: How well does the context cover the query
        - source_diversity: Variety of sources used
        - phrase_coverage: How well keyphrases are covered
        - dispersion_penalty: Penalty for scattered/noisy results
        """
        if not retrieval_results:
            return {
                "overall": 0.1,
                "retrieval_quality": 0.0,
                "coverage": 0.0,
                "source_diversity": 0.0,
                "phrase_coverage": 0.0,
                "dispersion_penalty": 0.0,
                "level": "low",
            }

        # Retrieval quality — average similarity score
        scores = [getattr(r, 'score', 0) for r in retrieval_results]
        avg_score = sum(scores) / max(len(scores), 1)
        retrieval_quality = min(avg_score * 1.5, 1.0)  # Scale up

        # Coverage — what fraction of query terms appear in context
        query_terms = set(re.findall(r'\w+', query.lower()))
        context_terms = set(re.findall(r'\w+', context_text.lower()))
        coverage = len(query_terms & context_terms) / max(len(query_terms), 1)

        # Source diversity — unique documents
        doc_ids = set(getattr(r, 'document_id', '') for r in retrieval_results)
        diversity = min(len(doc_ids) / 3, 1.0)  # Ideal: 3+ distinct sources

        # Phrase coverage — how many keyphrases appear in context
        phrase_cov = 0.0
        try:
            from rag.query_understanding import query_analyzer
            analysis = query_analyzer.analyze(query)
            if analysis.keyphrases:
                context_lower = context_text.lower()
                matched = sum(1 for kp in analysis.keyphrases if kp in context_lower)
                phrase_cov = matched / len(analysis.keyphrases)
        except Exception:
            pass

        # Dispersion penalty — penalize when scores vary widely (noisy retrieval)
        dispersion_penalty = 0.0
        if len(scores) > 1:
            mean_s = avg_score
            variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
            std_dev = variance ** 0.5
            dispersion_penalty = min(std_dev * 0.5, 0.2)  # Cap at 0.2

        # Overall
        overall = (
            retrieval_quality * 0.30
            + coverage * 0.25
            + diversity * 0.15
            + phrase_cov * 0.20
            - dispersion_penalty
        )
        overall = max(0.0, min(1.0, overall + 0.1))  # Floor boost

        # Level
        if overall >= 0.7:
            level = "high"
        elif overall >= 0.4:
            level = "medium"
        else:
            level = "low"

        return {
            "overall": round(overall, 3),
            "retrieval_quality": round(retrieval_quality, 3),
            "coverage": round(coverage, 3),
            "source_diversity": round(diversity, 3),
            "phrase_coverage": round(phrase_cov, 3),
            "dispersion_penalty": round(dispersion_penalty, 3),
            "level": level,
        }


# ── Multi-Hop Retrieval ────────────────────────────────────

class MultiHopRetriever:
    """
    Performs multi-hop retrieval by iteratively refining
    the query based on initial retrieval results.

    Hop 1: Original query → initial results
    Hop 2: Extract entities from results → refined query → more results
    Hop 3: (optional) Combine insights → final query → targeted results
    """

    def extract_follow_up_queries(
        self,
        original_query: str,
        initial_results: list,
        max_hops: int = 2,
    ) -> list[str]:
        """
        Generate follow-up queries based on initial retrieval results.

        Extracts key entities and terms from results to form
        more targeted queries for subsequent retrieval hops.
        """
        if not initial_results or max_hops <= 1:
            return []

        # Extract unique entities from initial results
        all_text = " ".join(
            getattr(r, 'content', '')[:500] for r in initial_results[:3]
        )

        # Find capitalized multi-word entities (simple NER)
        entities = set(re.findall(
            r'\b([A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+)\b', all_text
        ))

        # Find quoted terms
        quoted = set(re.findall(r'"([^"]+)"', all_text))
        entities.update(quoted)

        # Build follow-up queries by combining original intent with new entities
        follow_ups = []
        query_lower = original_query.lower()

        for entity in list(entities)[:3]:
            if entity.lower() not in query_lower:
                follow_ups.append(f"{original_query} {entity}")

        return follow_ups[:max_hops - 1]


# ── Conversation Memory ────────────────────────────────────

class ConversationMemory:
    """
    Manages conversation memory with compression.

    Keeps recent messages in full and compresses older messages
    into summaries to maintain context within token limits.
    """

    def __init__(
        self,
        max_recent_messages: int = 10,
        max_total_tokens: int = 4000,
    ):
        self.max_recent = max_recent_messages
        self.max_tokens = max_total_tokens

    def build_memory_context(
        self,
        messages: list[dict],
    ) -> str:
        """
        Build a memory context string from conversation messages.

        Returns recent messages in full + compressed older messages.
        """
        if not messages:
            return ""

        # Split into recent and old
        recent = messages[-self.max_recent:]
        old = messages[:-self.max_recent] if len(messages) > self.max_recent else []

        parts = []

        # Compress older messages
        if old:
            summary = self._compress_messages(old)
            parts.append(f"[Earlier conversation summary: {summary}]")

        # Add recent messages in full
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")

        result = "\n\n".join(parts)

        # Truncate if still too long
        words = result.split()
        if len(words) > self.max_tokens:
            result = " ".join(words[-self.max_tokens:])

        return result

    def _compress_messages(self, messages: list[dict]) -> str:
        """
        Compress older messages into a brief summary.
        Uses extractive summarization (key sentences).
        """
        topics = set()
        key_info = []

        for msg in messages:
            content = msg.get("content", "")
            # Extract questions asked
            questions = re.findall(r'[^.!?]*\?', content)
            for q in questions[:2]:
                topics.add(q.strip())

            # Extract first sentence of each message as key info
            sentences = re.split(r'(?<=[.!?])\s+', content)
            if sentences:
                key_info.append(sentences[0][:100])

        summary_parts = []
        if topics:
            summary_parts.append(f"Topics discussed: {'; '.join(list(topics)[:5])}")
        if key_info:
            summary_parts.append(f"Key points: {'; '.join(key_info[:5])}")

        return " | ".join(summary_parts) if summary_parts else "General conversation"


# ── Singletons ──────────────────────────────────────────────

context_compressor = ContextCompressor()
dedup_engine = DeduplicationEngine()
prompt_engine = PromptTemplateEngine()
query_decomposer = QueryDecomposer()
chain_of_density = ChainOfDensity()
self_verifier = SelfVerificationLoop()
confidence_scorer = ConfidenceScorer()
multi_hop = MultiHopRetriever()
conversation_memory = ConversationMemory()


# ── Distillation Pipeline Orchestrator ──────────────────────

@dataclass
class DistillationResult:
    """Full result of the distillation pipeline."""
    # The final system prompt to send to the LLM
    system_prompt: str
    # The processed context (compressed, deduped)
    processed_context: str
    # Original context before processing
    original_context: str
    # Query classification
    query_type: str
    # Sub-queries from decomposition
    sub_queries: list[str] = field(default_factory=list)
    # Confidence scores
    confidence: dict = field(default_factory=dict)
    # Compression stats
    compression_ratio: float = 1.0
    original_tokens: int = 0
    compressed_tokens: int = 0
    # Number of retrieval hops performed
    hops_used: int = 1
    # Chunks before/after deduplication
    chunks_before_dedup: int = 0
    chunks_after_dedup: int = 0
    # Retrieval results for verification
    retrieval_results: list = field(default_factory=list)


class DistillationPipeline:
    """
    Orchestrates the full Phase 3 distillation pipeline.

    Flow:
    1. Decompose query into sub-queries (if complex)
    2. Retrieve chunks for each sub-query
    3. Deduplicate retrieved chunks
    4. Multi-hop retrieval (refine with follow-up queries)
    5. Compress context (remove noise)
    6. Score confidence
    7. Select adaptive prompt template
    8. Build final system prompt
    """

    def __init__(self):
        self.compressor = context_compressor
        self.dedup = dedup_engine
        self.prompts = prompt_engine
        self.decomposer = query_decomposer
        self.density = chain_of_density
        self.verifier = self_verifier
        self.scorer = confidence_scorer
        self.multi_hop = multi_hop
        self.memory = conversation_memory

    async def process(
        self,
        query: str,
        retriever_fn: Any,
        workspace_id: str = "default",
        top_k: int = 5,
        conversation_messages: list[dict] | None = None,
        base_system_prompt: str = "",
        config: dict | None = None,
    ) -> DistillationResult:
        """
        Run the full distillation pipeline.

        Args:
            query: The user's query
            retriever_fn: Async callable(query, workspace_id, top_k) -> list[RetrievalResult]
            workspace_id: Document workspace
            top_k: Number of chunks to retrieve per query
            conversation_messages: Previous messages for memory compression
            base_system_prompt: Base system prompt to augment
            config: Override settings dict
        """
        from core.config import settings

        cfg = config or {}
        do_compression = cfg.get("context_compression", settings.context_compression)
        do_dedup = cfg.get("deduplication_enabled", settings.deduplication_enabled)
        do_decompose = cfg.get("query_decomposition", settings.query_decomposition)
        do_multi_hop = cfg.get("multi_hop_retrieval", settings.multi_hop_retrieval)
        do_adaptive = cfg.get("adaptive_prompts", settings.adaptive_prompts)
        do_confidence = cfg.get("confidence_scoring", settings.confidence_scoring)
        do_memory = cfg.get("conversation_memory_compression", settings.conversation_memory_compression)
        max_hops = cfg.get("multi_hop_max_hops", settings.multi_hop_max_hops)

        # Step 1: Query decomposition
        sub_queries = [query]
        if do_decompose:
            sub_queries = self.decomposer.decompose(query)
            if len(sub_queries) > 1:
                logger.info(f"Decomposed query into {len(sub_queries)} sub-queries: {sub_queries}")

        # Step 2: Retrieve chunks for all sub-queries
        all_results = []
        for sq in sub_queries:
            results = await retriever_fn(sq, workspace_id, top_k)
            all_results.extend(results)

        if not all_results:
            # No documents found — return minimal result
            prompt = base_system_prompt or "You are a helpful assistant."
            if do_memory and conversation_messages:
                memory_ctx = self.memory.build_memory_context(conversation_messages)
                if memory_ctx:
                    prompt += f"\n\n--- CONVERSATION CONTEXT ---\n{memory_ctx}"
            return DistillationResult(
                system_prompt=prompt,
                processed_context="",
                original_context="",
                query_type=self.prompts.classify_query(query) if do_adaptive else "default",
                sub_queries=sub_queries,
                confidence={"overall": 0.1, "level": "low", "retrieval_quality": 0, "coverage": 0, "source_diversity": 0},
            )

        chunks_before_dedup = len(all_results)

        # Step 3: Deduplicate chunks
        if do_dedup and len(all_results) > 1:
            chunk_dicts = [
                {"content": getattr(r, "content", ""), "score": getattr(r, "score", 0), "_obj": r}
                for r in all_results
            ]
            unique_dicts = self.dedup.deduplicate(chunk_dicts)
            all_results = [d["_obj"] for d in unique_dicts]

        chunks_after_dedup = len(all_results)

        # Step 4: Multi-hop retrieval
        hops_used = 1
        if do_multi_hop and max_hops > 1:
            follow_ups = self.multi_hop.extract_follow_up_queries(
                query, all_results, max_hops=max_hops
            )
            for fq in follow_ups:
                hop_results = await retriever_fn(fq, workspace_id, top_k // 2 or 2)
                if hop_results:
                    all_results.extend(hop_results)
                    hops_used += 1
                    logger.info(f"Multi-hop {hops_used}: '{fq[:60]}' → {len(hop_results)} results")

            # Deduplicate again after multi-hop
            if do_dedup and hops_used > 1:
                chunk_dicts = [
                    {"content": getattr(r, "content", ""), "score": getattr(r, "score", 0), "_obj": r}
                    for r in all_results
                ]
                unique_dicts = self.dedup.deduplicate(chunk_dicts)
                all_results = [d["_obj"] for d in unique_dicts]

        # Build raw context from results
        raw_context = "\n\n".join(
            getattr(r, "content", "")[:1500] for r in all_results[:top_k * 2]
        )
        original_context = raw_context

        # Step 5: Compress context
        compression_ratio = 1.0
        original_tokens = len(raw_context.split())
        compressed_tokens = original_tokens

        if do_compression and raw_context:
            compressed = self.compressor.compress(raw_context, query)
            raw_context = compressed.compressed_text
            compression_ratio = compressed.compression_ratio
            original_tokens = compressed.original_tokens
            compressed_tokens = compressed.compressed_tokens
            logger.info(
                f"Compression: {original_tokens} → {compressed_tokens} tokens "
                f"(ratio={compression_ratio:.2f})"
            )

        # Step 6: Score confidence
        confidence = {"overall": 0.5, "level": "medium", "retrieval_quality": 0.5, "coverage": 0.5, "source_diversity": 0.5}
        if do_confidence:
            confidence = self.scorer.score(query, all_results, raw_context)
            logger.info(f"Confidence: {confidence['level']} ({confidence['overall']:.2f})")

        # Step 7: Build adaptive prompt
        query_type = "default"
        if do_adaptive:
            query_type = self.prompts.classify_query(query)
            context_prompt = self.prompts.render(query, raw_context, query_type)
        else:
            context_prompt = (
                f"Using the following context, answer the question.\n\n"
                f"Context:\n{raw_context}\n\n"
                f"Question: {query}\n\n"
                f"Answer accurately based on the context."
            )

        # Step 8: Build final system prompt
        system_parts = []

        if base_system_prompt:
            system_parts.append(base_system_prompt)

        # Add conversation memory
        if do_memory and conversation_messages:
            memory_ctx = self.memory.build_memory_context(conversation_messages)
            if memory_ctx:
                system_parts.append(f"--- CONVERSATION CONTEXT ---\n{memory_ctx}")

        # Add confidence hint
        if do_confidence and confidence["level"] == "low":
            system_parts.append(
                "Note: The retrieved context has LOW relevance to this query. "
                "If the context doesn't fully answer the question, supplement "
                "with your general knowledge and clearly indicate which parts "
                "are not from the provided sources."
            )

        # Add the RAG context prompt
        system_parts.append(context_prompt)

        final_prompt = "\n\n".join(system_parts)

        return DistillationResult(
            system_prompt=final_prompt,
            processed_context=raw_context,
            original_context=original_context,
            query_type=query_type,
            sub_queries=sub_queries,
            confidence=confidence,
            compression_ratio=compression_ratio,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            hops_used=hops_used,
            chunks_before_dedup=chunks_before_dedup,
            chunks_after_dedup=chunks_after_dedup,
            retrieval_results=all_results,
        )

    async def run_chain_of_density(
        self,
        content: str,
        llm_fn: Any,
        iterations: int = 2,
    ) -> str:
        """
        Run chain-of-density summarization.

        Args:
            content: The content to summarize
            llm_fn: Async callable(messages) -> str (LLM response)
            iterations: Number of density iterations
        """
        summary = ""
        for i in range(1, iterations + 1):
            prompt = self.density.build_density_prompt(
                content=content,
                iteration=i,
                max_iterations=iterations,
                previous_summary=summary,
            )
            summary = await llm_fn([{"role": "user", "content": prompt}])
            logger.info(f"Chain-of-density iteration {i}/{iterations}: {len(summary)} chars")

        return summary

    async def run_self_verification(
        self,
        response: str,
        context: str,
        query: str,
        llm_fn: Any,
    ) -> dict:
        """
        Verify a response against context using self-verification.

        Args:
            response: The LLM response to verify
            context: The source context
            query: Original query
            llm_fn: Async callable(messages) -> str (LLM response)

        Returns:
            {"verified": bool, "corrected_response": str | None, "feedback": str}
        """
        prompt = self.verifier.build_verification_prompt(response, context, query)
        verification = await llm_fn([{"role": "user", "content": prompt}])

        if "VERIFIED" in verification.upper():
            return {"verified": True, "corrected_response": None, "feedback": "Response verified against sources."}
        else:
            return {"verified": False, "corrected_response": verification, "feedback": "Response was corrected based on source verification."}


# Pipeline singleton
distillation_pipeline = DistillationPipeline()
