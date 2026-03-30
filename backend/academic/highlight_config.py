"""
Highlight Configuration & System Prompt for VU Academic Highlighted Handouts.

Centralizes all configurable settings, the AI system prompt, and the
retry/repair prompt for the highlighted handout generation pipeline.
"""

from dataclasses import dataclass, field


@dataclass
class HighlightConfig:
    """Centralized configuration for the highlight pipeline."""

    # ── Batching ──────────────────────────────────────────────
    batch_pages: int = 5
    max_batch_chars: int = 7000
    char_safety_margin: float = 0.85       # use 85% of max to prevent overflow
    max_review_evidence_chars: int = 2500

    # ── AI Output ─────────────────────────────────────────────
    max_spans_per_batch: int = 20
    min_span_length: int = 15
    max_span_length: int = 500
    ai_temperature: float = 0.1
    ai_max_tokens: int = 2000

    # ── Retry / Error ─────────────────────────────────────────
    max_retries_per_batch: int = 1
    batch_timeout_seconds: int = 120
    continue_on_error: bool = True
    max_failed_batch_pct: float = 0.50     # stop if >50% batches fail

    # ── Highlight density ─────────────────────────────────────
    target_highlights_per_page: int = 5
    max_highlights_per_page: int = 15

    # ── Matching ──────────────────────────────────────────────
    fuzzy_match_fallback: bool = True
    merge_adjacent_highlights: bool = True
    merge_threshold_pt: float = 3.0

    # ── Logging ───────────────────────────────────────────────
    debug_mode: bool = False
    save_ai_logs: bool = True

    @property
    def effective_max_batch_chars(self) -> int:
        """Char budget per batch after applying safety margin."""
        return int(self.max_batch_chars * self.char_safety_margin)


# ── System prompt ─────────────────────────────────────────────

HIGHLIGHT_SYSTEM_PROMPT = """\
You are an Expert Highlighted Handout Generator for Virtual University (VU) Pakistan students.

YOUR ROLE:
You analyze course handout pages alongside student exam review evidence to identify and extract the most exam-relevant text spans for yellow highlighting.

TARGET AUDIENCE:
VU students preparing for midterm and final exams. Your highlights must help them focus on what actually appears in exams.

HIGHLIGHTING RULES:
1. ONLY return text that appears EXACTLY (verbatim) in the CURRENT_BATCH_TEXT.
2. Each highlight MUST be a COMPLETE logical unit:
   - Full definitions (term + explanation)
   - Complete formulas with context
   - Entire rules, principles, or key statements
   - Important paragraphs (2-5 sentences) that explain a core concept
3. NEVER highlight single words, incomplete sentences, or partial definitions.
4. Minimum highlight length: 30 characters.
5. Preferred highlight length: 50-250 characters.
6. DO NOT paraphrase or summarize — copy text exactly as it appears.

PRIORITY ORDER (what to highlight first):
1. Content that MATCHES topics/concepts mentioned in REVIEW_EVIDENCE (exam-proven).
2. Definitions of key terms and concepts.
3. Formulas, equations, and their explanations.
4. Rules, principles, and important theorems.
5. Frequently tested concepts (based on review patterns).
6. Important examples that explain core concepts.

OUTPUT FORMAT:
Return ONLY a valid JSON array. No explanations, no thinking, no markdown headers.
[
  {"text": "exact copied text span from batch", "reason": "definition|formula|rule|concept|exam_point", "confidence": 0.0}
]

CRITICAL:
- Return 8-20 spans per batch.
- If you cannot find enough relevant content return fewer spans rather than highlighting irrelevant text.
- Do NOT wrap JSON in markdown code fences.
- Do NOT include any text before or after the JSON array.\
"""

# ── Batch user prompt template ────────────────────────────────

HIGHLIGHT_BATCH_PROMPT_TEMPLATE = """\
Identify exam-important text spans in the pages below.
Use the REVIEW_EVIDENCE to prioritise content students reported as appearing in past exams.

Rules reminder:
- Return ONLY exact text copied from CURRENT_BATCH_TEXT
- No paraphrase, no summaries, no single words
- Each span must be a complete logical unit (definition, formula, rule, concept)
- Minimum span length: 30 characters
- Return max {max_spans} spans
- Output strict JSON array: [{{"text": "...", "reason": "...", "confidence": 0.0}}]

REVIEW_EVIDENCE:
{review_evidence}

CURRENT_BATCH_TEXT:
{batch_text}\
"""

# ── Repair prompt (used on retry after invalid response) ──────

HIGHLIGHT_REPAIR_PROMPT = """\
Your previous response was not valid JSON or did not contain any valid highlight spans.

Return ONLY a JSON array of objects, nothing else.
Each object: {{"text": "exact text from the page content above", "reason": "...", "confidence": 0.0}}

Rules:
- Copy text EXACTLY from the batch pages — do not paraphrase.
- Minimum text length per span: 30 characters.
- Return 5-15 spans.
- No markdown, no explanations, no thinking tags.\
"""
