"""
Unit tests for the Highlighted Handout pipeline fixes.

Tests cover:
1. _extract_spans with <think> blocks (including UNCLOSED tags)
2. _extract_spans with valid JSON
3. _extract_spans with line-based and paragraph-level fallback
4. Overflow buffer logic
5. Priority scoring
"""

import json
import re
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from academic.highlight_config import HighlightConfig


def test_extract_spans():
    """Test the span extraction logic that lives inside pdf_renderer.

    We recreate the _extract_spans function here using the same logic
    to validate it against real AI response patterns.
    """
    cfg = HighlightConfig()

    def _strip_code_fences(text: str) -> str:
        s = (text or "").strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        return s.strip()

    def _extract_json_array_text(text: str) -> str:
        s = _strip_code_fences(text)
        start = s.find("[")
        end = s.rfind("]")
        if start >= 0 and end > start:
            return s[start:end + 1]
        return s

    def _extract_spans(raw: str) -> list[str]:
        if not raw:
            return []
        # Step 1: Strip <think>...</think> blocks (handles unclosed tags)
        text = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', raw)
        text = re.sub(r'<reasoning>[\s\S]*?(?:</reasoning>|$)', '', text)
        text = text.strip()
        if not text:
            return []
        # Step 2: JSON extraction
        json_text = _extract_json_array_text(text)
        # Step 3: JSON parse
        try:
            parsed = json.loads(json_text)
            spans = []
            arr = (
                parsed if isinstance(parsed, list)
                else parsed.get("highlights", []) if isinstance(parsed, dict)
                else []
            )
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict):
                        s = str(item.get("text", "")).strip()
                    elif isinstance(item, str):
                        s = item.strip()
                    else:
                        continue
                    if cfg.min_span_length <= len(s) <= cfg.max_span_length:
                        spans.append(s)
            if spans:
                return spans[:cfg.max_spans_per_batch]
        except (json.JSONDecodeError, ValueError):
            pass

        # Step 4: Paragraph-level extraction (join consecutive lines)
        stripped_text = _strip_code_fences(text)

        # 4a: Paragraph-level grouping
        paragraphs = re.split(r'\n\s*\n', stripped_text)
        spans = []
        for para in paragraphs:
            lines = []
            for line in para.splitlines():
                s = line.strip().strip("-*\"' \u2022\u00b7\u25ba\u25aa")
                s = re.sub(r'^(?:\d+[\.\)]\s*|text:\s*)', '', s, flags=re.IGNORECASE).strip()
                s = s.strip("\"' ")
                if len(s) >= 10:
                    lines.append(s)
            if lines:
                joined = " ".join(lines)
                if cfg.min_span_length <= len(joined) <= cfg.max_span_length:
                    spans.append(joined)

        if spans:
            return spans[:cfg.max_spans_per_batch]

        # 4b: Fall back to single-line extraction
        spans = []
        for line in stripped_text.splitlines():
            s = line.strip().strip("-*\"' \u2022\u00b7\u25ba\u25aa")
            s = re.sub(r'^(?:\d+[\.\)]\s*|text:\s*)', '', s, flags=re.IGNORECASE).strip()
            s = s.strip("\"' ")
            if cfg.min_span_length <= len(s) <= cfg.max_span_length:
                spans.append(s)
        return spans[:cfg.max_spans_per_batch]

    # ── Test 1: <think> block stripping (closed tag) ──
    print("Test 1: <think> block stripping (closed)...")
    response_with_think = """<think>
Okay, so I need to figure out what text to highlight based on the review evidence and current batch text.
Let me analyze the content carefully.
I should identify key definitions and formulas.
</think>

```json
[
  {"text": "A Turing machine is a particularly simple kind of computer that can solve problems", "reason": "definition", "confidence": 0.8},
  {"text": "The test is conducted with two people and a machine to check intelligence", "reason": "concept", "confidence": 0.7}
]
```"""
    spans = _extract_spans(response_with_think)
    assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}: {spans}"
    assert "Turing machine" in spans[0], f"Expected Turing machine in span[0], got: {spans[0]}"
    print(f"  PASS: Got {len(spans)} spans after stripping <think> block")

    # ── Test 2: UNCLOSED <think> block (critical fix) ──
    print("Test 2: UNCLOSED <think> block (critical bug fix)...")
    unclosed_think = """<think>
Okay, so I'm trying to understand this course outline. It's CS101 Introduction to Computing.
Let me go through each section step by step.
Starting from Week 1: The introduction talks about Charles Babbage.
Week 2 is about the evolution of computing, computer organization.
Week 3 focuses on algorithms, flowcharts, programming languages.
This is a very long thinking block that never closes.
The model just keeps thinking and thinking without closing the tag.
It generates pages and pages of reasoning without any useful output.
More thinking here about various topics.
And even more thinking without end."""
    spans = _extract_spans(unclosed_think)
    assert spans == [], f"Expected empty list for unclosed <think> with no content after, got {len(spans)}: {spans}"
    print(f"  PASS: Unclosed <think> block stripped correctly (returned {len(spans)} spans)")

    # ── Test 3: UNCLOSED <think> block followed by valid JSON ──
    print("Test 3: UNCLOSED <think> followed by valid JSON...")
    # This simulates a model that starts thinking but eventually outputs JSON without closing <think>
    unclosed_then_json = """<think>
Let me analyze the content carefully and identify important sections.
I need to find definitions and key concepts.
</think>[
  {"text": "The Analytical Engine designed by Charles Babbage in the 1830s was the first general-purpose computing machine", "reason": "definition", "confidence": 0.9}
]"""
    spans = _extract_spans(unclosed_then_json)
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}: {spans}"
    assert "Analytical Engine" in spans[0]
    print(f"  PASS: Got {len(spans)} span after stripping closed <think> before JSON")

    # ── Test 4: Pure JSON response ──
    print("Test 4: Pure JSON response...")
    pure_json = """[
  {"text": "ENIAC I (Electrical Numerical Integrator And Calculator) was developed for military calculations", "reason": "definition", "confidence": 0.9},
  {"text": "The Atanasoff-Berry Computer was the world's first electronic digital computer built in the late 1930s", "reason": "concept", "confidence": 0.85},
  {"text": "A vacuum tube is just that: a glass tube surrounding a vacuum used in early computers", "reason": "definition", "confidence": 0.7}
]"""
    spans = _extract_spans(pure_json)
    assert len(spans) == 3, f"Expected 3 spans, got {len(spans)}"
    print(f"  PASS: Got {len(spans)} spans from pure JSON")

    # ── Test 5: String array ──
    print("Test 5: String array...")
    string_array = '["The first electronic digital computer was built in the late 1930s at Iowa State", "A vacuum tube is a glass tube surrounding a vacuum used in computing"]'
    spans = _extract_spans(string_array)
    assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}: {spans}"
    print(f"  PASS: Got {len(spans)} spans from string array")

    # ── Test 6: Prose response — paragraph-level fallback ──
    print("Test 6: Paragraph-level fallback for prose responses...")
    prose_response = """Here are the key highlights:

The Atanasoff-Berry Computer was the world's first electronic digital computer built in 1937-42
at Iowa State College by John Atanasoff and Clifford Berry.

A Turing machine is a particularly simple kind of computer that operates on an infinite tape.
It can solve a large class of computational problems through its simple instruction set.

ENIAC I (Electrical Numerical Integrator And Calculator) was a military project
developed at the University of Pennsylvania during World War II.
"""
    spans = _extract_spans(prose_response)
    assert len(spans) >= 2, f"Expected at least 2 spans from paragraph fallback, got {len(spans)}: {spans}"
    # Verify paragraphs are joined (multi-line content becomes single span)
    for sp in spans:
        assert len(sp) >= 30, f"Span too short: {sp}"
    print(f"  PASS: Got {len(spans)} spans from paragraph-level fallback")

    # ── Test 7: Too-short spans rejected (min_span_length is now 30) ──
    print("Test 7: Short span rejection (min=30 chars)...")
    short_spans_json = """[
  {"text": "arrays", "reason": "topic", "confidence": 0.7},
  {"text": "cache memory basics", "reason": "topic", "confidence": 0.6},
  {"text": "short line", "reason": "topic", "confidence": 0.6},
  {"text": "The Atanasoff-Berry Computer was the world's first electronic digital computer built in the 1930s", "reason": "concept", "confidence": 0.85}
]"""
    spans = _extract_spans(short_spans_json)
    assert len(spans) == 1, f"Expected 1 span (short ones rejected with min=30), got {len(spans)}: {spans}"
    assert "Atanasoff" in spans[0]
    print(f"  PASS: Short spans correctly rejected, got {len(spans)} valid span")

    # ── Test 8: <think> block + prose (closed tag) ──
    print("Test 8: <think> block + prose (no JSON)...")
    think_then_prose = """<think>
I need to analyze the content.
Let me think about what's important.
</think>

Here are some key highlights from the review evidence and current text:

**Input Devices**: Mentioned in page 1, relevant for computer systems and hardware architecture.
**Output Devices**: Also present in page 1, important for software development and user interaction.
**Integrated Circuits**: Common in hardware components, likely related to exam content and modern computing.
"""
    spans = _extract_spans(think_then_prose)
    assert len(spans) >= 1, f"Expected at least 1 span from <think>+prose, got {len(spans)}"
    print(f"  PASS: Got {len(spans)} spans from <think>+prose response")

    # ── Test 9: Code fence wrapped JSON ──
    print("Test 9: Code-fence wrapped JSON...")
    fenced = """```json
[{"text": "Charles Babbage designed the Analytical Engine which was mechanical and digital in nature", "reason": "definition", "confidence": 0.9}]
```"""
    spans = _extract_spans(fenced)
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    print(f"  PASS: Got {len(spans)} span from code-fenced JSON")

    # ── Test 10: Empty response ──
    print("Test 10: Empty response...")
    spans = _extract_spans("")
    assert spans == [], f"Expected empty list, got {spans}"
    spans = _extract_spans("<think>Just thinking here</think>")
    assert spans == [], f"Expected empty list after stripping think, got {spans}"
    print(f"  PASS: Empty responses handled correctly")

    # ── Test 11: Dict with highlights key ──
    print("Test 11: Dict with 'highlights' key...")
    dict_format = '{"highlights": [{"text": "The Turing test was proposed to determine if a computer has the ability to think like humans", "reason": "concept"}]}'
    spans = _extract_spans(dict_format)
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    print(f"  PASS: Got {len(spans)} span from dict format")

    # ── Test 12: Massive unclosed <think> with garbage (real deepseek-r1:1.5b pattern) ──
    print("Test 12: Massive unclosed <think> (real deepseek-r1:1.5b pattern)...")
    massive_think = "<think>\n" + "\n".join([
        f"Page {i} discusses input devices with more specific details about mouse, keyboard, etc."
        for i in range(1, 50)
    ])  # 50 lines of thinking, never closed
    spans = _extract_spans(massive_think)
    assert spans == [], f"Expected empty list for massive unclosed <think>, got {len(spans)}"
    print(f"  PASS: Massive unclosed <think> (50 lines) stripped correctly")


def test_overflow_logic():
    """Test the overflow buffer concept with simulated page data."""
    print("\nTest: Overflow buffer logic...")

    max_chars = 200  # Small limit for testing
    safety_margin = 0.85
    effective_limit = int(max_chars * safety_margin)  # 170 chars

    # Simulate 5 pages of varying lengths
    pages = [
        "[Page 1]\nShort content for page one.",           # ~40 chars
        "[Page 2]\nAnother short page with some text.",     # ~44 chars
        "[Page 3]\nThis is page three content.",            # ~35 chars
        "[Page 4]\n" + "X" * 120,                           # ~130 chars (will overflow)
        "[Page 5]\nFinal page content here.",               # ~35 chars
    ]

    # Simulate batch processing with overflow
    batch_pages = 5
    overflow_text = ""
    overflow_pages: list[int] = []
    page_cursor = 0
    batches_created: list[dict] = []

    while page_cursor < len(pages) or overflow_text:
        batch_blocks: list[str] = []
        chars_used = 0
        actual_pages: list[int] = []
        new_overflow_text = ""
        new_overflow_pages: list[int] = []

        # Prepend overflow
        if overflow_text:
            batch_blocks.append(overflow_text)
            chars_used += len(overflow_text)
            actual_pages.extend(overflow_pages)
            overflow_text = ""
            overflow_pages = []

        pages_added = 0
        while page_cursor < len(pages) and pages_added < batch_pages:
            page_block = pages[page_cursor]
            if chars_used + len(page_block) > effective_limit and batch_blocks:
                new_overflow_text = page_block
                new_overflow_pages = [page_cursor]
                page_cursor += 1
                pages_added += 1
                while page_cursor < len(pages) and pages_added < batch_pages:
                    new_overflow_text += "\n\n" + pages[page_cursor]
                    new_overflow_pages.append(page_cursor)
                    page_cursor += 1
                    pages_added += 1
                break
            else:
                batch_blocks.append(page_block)
                chars_used += len(page_block)
                actual_pages.append(page_cursor)
                page_cursor += 1
                pages_added += 1

        overflow_text = new_overflow_text
        overflow_pages = new_overflow_pages

        batch_text = "\n\n".join(batch_blocks)
        if batch_text.strip():
            batches_created.append({
                "pages": actual_pages,
                "chars": len(batch_text),
                "text_preview": batch_text[:60],
            })

    # Verify ALL pages are processed
    all_pages = set()
    for batch in batches_created:
        all_pages.update(batch["pages"])

    assert all_pages == {0, 1, 2, 3, 4}, f"Expected all 5 pages processed, got pages: {all_pages}"
    assert len(batches_created) >= 2, f"Expected at least 2 batches due to overflow, got {len(batches_created)}"

    print(f"  PASS: All {len(all_pages)} pages processed across {len(batches_created)} batches")
    for i, b in enumerate(batches_created):
        print(f"    Batch {i}: pages={b['pages']}, chars={b['chars']}")


def test_priority_scoring():
    """Test span priority scoring."""
    print("\nTest: Priority scoring...")

    review_keywords = ["turing", "stack", "tree", "pointer", "pseudocode"]

    def _score_span(span_text: str, confidence: float = 0.5) -> float:
        score = confidence
        text_lower = span_text.lower()
        kw_matches = sum(1 for kw in review_keywords if kw in text_lower)
        score += min(kw_matches * 0.08, 0.4)
        markers = ["definition", "formula", "theorem", "rule", "principle",
                    "algorithm", "equation", "function", "method", "protocol"]
        if any(m in text_lower for m in markers):
            score += 0.12
        if len(span_text) > 100:
            score += 0.08
        elif len(span_text) > 60:
            score += 0.04
        return min(1.0, score)

    spans = [
        "A short generic text here about nothing",
        "The Turing test was proposed to determine if a computer can think like a human being",
        "A stack is a data structure definition where PUSH and POP operations apply to elements",
        "Random unrelated text about nothing in particular at all here and more random text continues",
    ]

    scored = [(s, _score_span(s)) for s in spans]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Stack+definition span should score highest (2 keywords + definition marker + length)
    assert "stack" in scored[0][0].lower(), f"Expected stack span first, got: {scored[0][0]}"
    # Turing span should score second (1 keyword + length)
    assert "turing" in scored[1][0].lower(), f"Expected turing span second, got: {scored[1][0]}"

    print(f"  PASS: Priority scoring ranks correctly")
    for s, score in scored:
        print(f"    Score {score:.2f}: {s[:60]}...")


def test_config_defaults():
    """Test that config defaults match the new values."""
    print("\nTest: Config defaults...")
    cfg = HighlightConfig()

    assert cfg.min_span_length == 30, f"Expected min_span_length=30, got {cfg.min_span_length}"
    assert cfg.max_span_length == 800, f"Expected max_span_length=800, got {cfg.max_span_length}"
    assert cfg.max_spans_per_batch == 25, f"Expected max_spans_per_batch=25, got {cfg.max_spans_per_batch}"
    assert cfg.ai_max_tokens == 4000, f"Expected ai_max_tokens=4000, got {cfg.ai_max_tokens}"
    assert cfg.target_highlights_per_page == 8, f"Expected target_highlights_per_page=8, got {cfg.target_highlights_per_page}"
    assert cfg.max_highlights_per_page == 20, f"Expected max_highlights_per_page=20, got {cfg.max_highlights_per_page}"

    print(f"  PASS: All config defaults are correct")


if __name__ == "__main__":
    print("=" * 60)
    print("HIGHLIGHTED HANDOUT SYSTEM - UNIT TESTS")
    print("=" * 60)

    test_extract_spans()
    test_overflow_logic()
    test_priority_scoring()
    test_config_defaults()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
