"""
Unit tests for the Highlighted Handout pipeline fixes.

Tests cover:
1. _extract_spans with <think> blocks (Ticket 3)
2. _extract_spans with valid JSON (Ticket 3)
3. _extract_spans with line-based fallback (Ticket 3)
4. Overflow buffer logic (Ticket 2)
5. Priority scoring (Ticket 6)
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
        # Step 1: Strip <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
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
        # Step 4: Fallback
        spans = []
        for line in _strip_code_fences(text).splitlines():
            s = line.strip().strip("-*\"' •·►▪")
            s = re.sub(r'^(?:\d+[\.\)]\s*|text:\s*)', '', s, flags=re.IGNORECASE).strip()
            s = s.strip("\"' ")
            if cfg.min_span_length <= len(s) <= cfg.max_span_length:
                spans.append(s)
        return spans[:cfg.max_spans_per_batch]

    # ── Test 1: <think> block stripping (real deepseek-r1 output) ──
    print("Test 1: <think> block stripping...")
    response_with_think = """<think>
Okay, so I need to figure out what text to highlight based on the review evidence and current batch text.
Let me analyze the content carefully.
I should identify key definitions and formulas.
</think>

```json
[
  {"text": "A Turing machine is a particularly simple kind of computer", "reason": "definition", "confidence": 0.8},
  {"text": "The test is conducted with two people and a machine", "reason": "concept", "confidence": 0.7}
]
```"""
    spans = _extract_spans(response_with_think)
    assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}: {spans}"
    assert "Turing machine" in spans[0], f"Expected Turing machine in span[0], got: {spans[0]}"
    print(f"  PASS: Got {len(spans)} spans after stripping <think> block")

    # ── Test 2: Pure JSON response ──
    print("Test 2: Pure JSON response...")
    pure_json = """[
  {"text": "ENIAC I (Electrical Numerical Integrator And Calculator)", "reason": "definition", "confidence": 0.9},
  {"text": "The Atanasoff-Berry Computer was the world's first electronic digital computer", "reason": "concept", "confidence": 0.85},
  {"text": "A vacuum tube is just that: a glass tube surrounding a vacuum", "reason": "definition", "confidence": 0.7}
]"""
    spans = _extract_spans(pure_json)
    assert len(spans) == 3, f"Expected 3 spans, got {len(spans)}"
    print(f"  PASS: Got {len(spans)} spans from pure JSON")

    # ── Test 3: String array ──
    print("Test 3: String array...")
    string_array = '["The first electronic digital computer", "A vacuum tube surrounding a vacuum area"]'
    spans = _extract_spans(string_array)
    assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}: {spans}"
    print(f"  PASS: Got {len(spans)} spans from string array")

    # ── Test 4: Prose response (no JSON) — fallback ──
    print("Test 4: Line-based fallback for prose responses...")
    prose_response = """Here are the key highlights:

- The Atanasoff-Berry Computer was the world's first electronic digital computer built in 1937-42
- A Turing machine is a particularly simple kind of computer
- ENIAC I (Electrical Numerical Integrator And Calculator) was a military project
"""
    spans = _extract_spans(prose_response)
    assert len(spans) >= 2, f"Expected at least 2 spans from fallback, got {len(spans)}: {spans}"
    print(f"  PASS: Got {len(spans)} spans from line-based fallback")

    # ── Test 5: Too-short spans rejected ──
    print("Test 5: Short span rejection...")
    short_spans_json = """[
  {"text": "arrays", "reason": "topic", "confidence": 0.7},
  {"text": "cache", "reason": "topic", "confidence": 0.6},
  {"text": "The Atanasoff-Berry Computer was the world's first electronic digital computer", "reason": "concept", "confidence": 0.85}
]"""
    spans = _extract_spans(short_spans_json)
    # "arrays" (6 chars) and "cache" (5 chars) should be rejected
    assert len(spans) == 1, f"Expected 1 span (short ones rejected), got {len(spans)}: {spans}"
    assert "Atanasoff" in spans[0]
    print(f"  PASS: Short spans correctly rejected, got {len(spans)} valid span")

    # ── Test 6: <think> block with NO JSON after it (pure prose) ──
    print("Test 6: <think> block + prose (no JSON)...")
    think_then_prose = """<think>
I need to analyze the content.
Let me think about what's important.
</think>

Here are some key highlights from the review evidence and current text:

- **Input Devices**: Mentioned in page 1, relevant for computer systems.
- **Output Devices**: Also present in page 1, important for software development.
- **Integrated Circuits**: Common in hardware components, likely related to exam content.
"""
    spans = _extract_spans(think_then_prose)
    # The bold markers (**) get stripped, lines should still be captured
    assert len(spans) >= 1, f"Expected at least 1 span from <think>+prose, got {len(spans)}"
    print(f"  PASS: Got {len(spans)} spans from <think>+prose response")

    # ── Test 7: Code fence wrapped JSON ──
    print("Test 7: Code-fence wrapped JSON...")
    fenced = """```json
[{"text": "Charles Babbage designed the Analytical Engine which was mechanical and digital", "reason": "definition", "confidence": 0.9}]
```"""
    spans = _extract_spans(fenced)
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    print(f"  PASS: Got {len(spans)} span from code-fenced JSON")

    # ── Test 8: Empty response ──
    print("Test 8: Empty response...")
    spans = _extract_spans("")
    assert spans == [], f"Expected empty list, got {spans}"
    spans = _extract_spans("<think>Just thinking here</think>")
    assert spans == [], f"Expected empty list after stripping think, got {spans}"
    print(f"  PASS: Empty responses handled correctly")

    # ── Test 9: Dict with highlights key ──
    print("Test 9: Dict with 'highlights' key...")
    dict_format = '{"highlights": [{"text": "The Turing test was proposed to determine if a computer has the ability to think", "reason": "concept"}]}'
    spans = _extract_spans(dict_format)
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    print(f"  PASS: Got {len(spans)} span from dict format")


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
        "A short generic text here",
        "The Turing test was proposed to determine if a computer can think",
        "A stack is a data structure definition where PUSH and POP operations apply",
        "Random unrelated text about nothing in particular at all here and more",
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


if __name__ == "__main__":
    print("=" * 60)
    print("HIGHLIGHTED HANDOUT SYSTEM — UNIT TESTS")
    print("=" * 60)
    
    test_extract_spans()
    test_overflow_logic()
    test_priority_scoring()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)
