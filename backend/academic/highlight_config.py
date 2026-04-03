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
    max_spans_per_batch: int = 25
    min_span_length: int = 40              # reject tiny useless spans
    max_span_length: int = 800
    ai_temperature: float = 0.1
    ai_max_tokens: int = 4000

    # ── Model requirements ──────────────────────────────────
    min_model_params_b: float = 3.0        # block models below 3B — they hallucinate

    # ── Retry / Error ─────────────────────────────────────────
    max_retries_per_batch: int = 1
    batch_timeout_seconds: int = 120
    continue_on_error: bool = True
    max_failed_batch_pct: float = 0.50     # stop if >50% batches fail

    # ── Highlight density ─────────────────────────────────────
    target_highlights_per_page: int = 8
    max_highlights_per_page: int = 20

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
You extract exam-important text from VU Pakistan course handouts for yellow highlighting.

RULES:
1. Read CURRENT_BATCH_TEXT. Find important sections that match REVIEW_EVIDENCE topics.
2. Copy each section EXACTLY character-for-character from CURRENT_BATCH_TEXT. Never paraphrase.
3. Each "text" value MUST be 50-600 characters long. Never use single words or short phrases.
4. Return 10-25 spans per batch.
5. Output ONLY a JSON array. Nothing else. No explanation. No summary. No markdown.

WHAT TO COPY:
- Complete definitions with explanations (the term AND its meaning)
- Full paragraphs about key concepts
- Complete bullet-point groups
- Formulas, rules, theorems with context

NEVER DO:
- Never write your own text. Only copy from CURRENT_BATCH_TEXT.
- Never output single words like "ENIAC" or "Transistor" — always include the full sentence/paragraph.
- Never output markdown, HTML, or XML tags.
- Never summarize or explain. Just output the JSON array.

OUTPUT — ONLY THIS, NOTHING ELSE:
[{"text": "exact text copied from the batch, minimum 50 characters long", "reason": "definition", "confidence": 0.9}]\
"""

# ── Batch user prompt template ────────────────────────────────

HIGHLIGHT_BATCH_PROMPT_TEMPLATE = """\
Copy exam-important text sections from CURRENT_BATCH_TEXT below.
Each "text" must be copied EXACTLY from the batch (50-600 chars). Return 10-{max_spans} spans.
Output ONLY the JSON array — no explanation, no summary, no markdown.

[{{"text": "exact copied text from batch", "reason": "definition|formula|rule|concept|exam_point", "confidence": 0.0-1.0}}]

REVIEW_EVIDENCE:
{review_evidence}

CURRENT_BATCH_TEXT:
{batch_text}\
"""

# ── Repair prompt (used on retry after invalid response) ──────

HIGHLIGHT_REPAIR_PROMPT = """\
Your previous response was NOT valid. You MUST return ONLY a JSON array.

DO NOT output any thinking, explanation, or commentary. ONLY output the JSON array.

Step 1: Find important text sections in the batch pages above.
Step 2: Copy each section EXACTLY (verbatim, no changes).
Step 3: Return this JSON format:

[{{"text": "exact text copied from the pages above", "reason": "concept", "confidence": 0.8}}]

Rules:
- Each "text" value must be copied EXACTLY from the page content above.
- Minimum 50 characters per span.
- Return 5-20 spans.
- Output ONLY the JSON array. Nothing else. No ```json```. No explanations.\
"""
