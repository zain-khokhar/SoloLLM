"""
PDF Renderer for VU Academic Auto-Generation.

Renders generated content into clean, print-friendly PDFs
using fpdf2 (lightweight, no external dependencies).
"""

import logging
import os
import sqlite3
import asyncio
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass

from core.config import settings
from core.inference import ollama_client
from rag.vectorstore import VECTORS_DB_PATH

logger = logging.getLogger(__name__)


def _calculate_mids_pages(total_pages: int) -> int:
    """
    Midterm split formula.

    Default ratio is 47.4%, so a 500-page PDF gives 237 pages for mids.
    """
    if total_pages <= 1:
        return total_pages
    mids = int(round(total_pages * 0.474))
    return max(1, min(total_pages - 1, mids))


def _resolve_source_pdf_path(workspace_id: str) -> str | None:
    """Resolve the most recent ingested PDF path for a course workspace."""
    try:
        con = sqlite3.connect(VECTORS_DB_PATH)
        cur = con.cursor()
        cur.execute(
            """
            SELECT filename FROM documents
            WHERE workspace_id = ? AND lower(file_type) = 'pdf'
            ORDER BY created_at DESC LIMIT 1
            """,
            (workspace_id,),
        )
        row = cur.fetchone()
        con.close()
        if not row:
            return None

        filename = row[0]
        candidate = settings.data_dir / "academic_uploads" / filename
        if candidate.exists():
            return str(candidate)
    except Exception:
        return None

    return None

def clean_text(text: str) -> str:
    """Pre-clean text for FPDF built-in latin-1 fonts."""
    if not isinstance(text, str):
        return ""
    replacements = {
        '—': '-', '–': '-', '“': '"', '”': '"', '‘': "'", '’': "'", '…': '...'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')


# Output directory for generated PDFs
ACADEMIC_OUTPUT_DIR = settings.data_dir / "academic_outputs"
ACADEMIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from fpdf import FPDF
    _HAS_FPDF = True
except ImportError:
    _HAS_FPDF = False
    logger.warning("fpdf2 not installed — PDF rendering will produce text files. Install: pip install fpdf2")


class AcademicPDF(FPDF if _HAS_FPDF else object):
    """Custom PDF class with academic styling."""

    def __init__(self, course_code: str = "", version: int = 1):
        if not _HAS_FPDF:
            self._lines: list[str] = []
            self._course_code = course_code
            self._version = version
            return
        super().__init__()
        self._course_code = course_code
        self._version = version
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if not _HAS_FPDF:
            return
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, clean_text(f"{self._course_code} | Academic Auto-Generated"), align="L")
        self.ln(10)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(5)

    def footer(self):
        if not _HAS_FPDF:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.cell(0, 10, clean_text(f"{self._course_code} | v{self._version} | {date_str} | Page {self.page_no()}/{{nb}}"), align="C")


class PDFRenderer:
    """
    Renders academic content into clean PDFs.

    Template types:
    - Highlighted handout
    - Midterm/Final short notes
    - MCQ bank with answer key
    """

    async def render_highlighted_handout(self, course_code: str, items: list[dict],
                                          version: int = 1,
                                          workspace_id: str = "default",
                                          syllabus_scope: str = "all",
                                          split_mode: str = "auto",
                                          manual_midterm_end_page: int | None = None,
                                          progress_callback=None,
                                          pause_check=None,
                                          job_id: str = "") -> str:
        """
        Render highlighted handout to PDF.

        Returns the file path of the generated PDF.
        """
        filename = f"{course_code}_highlighted_v{version}.pdf"
        filepath = str(ACADEMIC_OUTPUT_DIR / filename)

        source_pdf = _resolve_source_pdf_path(workspace_id)
        if not source_pdf or not os.path.exists(source_pdf):
            raise RuntimeError("No source PDF found for this course workspace")

        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise RuntimeError("PyMuPDF is required for real PDF highlighting") from e

        doc = fitz.open(source_pdf)
        total_pages = len(doc)

        mid_end = None
        if split_mode == "manual" and manual_midterm_end_page:
            mid_end = max(1, min(total_pages, int(manual_midterm_end_page)))
        elif split_mode == "auto":
            mid_end = _calculate_mids_pages(total_pages)

        if syllabus_scope == "midterm" and mid_end:
            target_pages = range(0, mid_end)
        elif syllabus_scope == "final" and mid_end:
            target_pages = range(mid_end, total_pages)
        else:
            target_pages = range(0, total_pages)

        snippet_pool: list[str] = []
        for item in items:
            raw = item.get("content", "")
            for line in raw.splitlines():
                snippet = line.strip().strip("-*")
                if 20 <= len(snippet) <= 240 and snippet not in snippet_pool:
                    snippet_pool.append(snippet)

        if not snippet_pool:
            raise RuntimeError("No highlight snippets available to apply")

        live_base = ACADEMIC_OUTPUT_DIR / f"{course_code}_highlighted_live_{job_id or 'tmp'}"
        total_highlights = 0

        def _safe_save(target_path: str) -> str:
            """Save PDF with fallback path on Windows file locks."""
            export_doc = doc
            created_export = False
            if len(target_pages) < total_pages:
                export_doc = fitz.open()
                created_export = True
                for page_idx in target_pages:
                    export_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
            try:
                export_doc.save(target_path, incremental=False, deflate=True)
                return target_path
            except Exception:
                fallback = str(Path(target_path).with_name(
                    f"{Path(target_path).stem}_{datetime.now(timezone.utc).strftime('%H%M%S%f')}{Path(target_path).suffix}"
                ))
                export_doc.save(fallback, incremental=False, deflate=True)
                return fallback
            finally:
                if created_export:
                    export_doc.close()

        def _safe_add_highlight(page, rect) -> bool:
            """Best-effort highlight add: skip invalid/orphan annotation cases."""
            try:
                annot = page.add_highlight_annot(rect)
                if annot is None:
                    return False
                try:
                    annot.update()
                except Exception:
                    # Some MuPDF builds throw orphan/binding errors on update.
                    pass
                return True
            except Exception:
                return False

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

        for pidx in target_pages:
            if pause_check:
                while pause_check():
                    await asyncio.sleep(0.5)

            page = doc[pidx]
            page_hits = 0

            for snippet in snippet_pool:
                # Exact first, then shorter fallback window for robust matching.
                rects = page.search_for(snippet)
                if not rects and len(snippet) > 90:
                    rects = page.search_for(snippet[:90])
                if not rects and len(snippet) > 60:
                    rects = page.search_for(snippet[:60])

                for rect in rects[:3]:
                    if _safe_add_highlight(page, rect):
                        page_hits += 1

            total_highlights += page_hits
            # Write a new preview snapshot per processed page to avoid overwrite locks.
            live_path = str(live_base.with_name(f"{live_base.name}_p{pidx + 1}.pdf"))
            live_path = _safe_save(live_path)

            if progress_callback:
                progress_callback(pidx + 1, page_hits, live_path)

        filepath = _safe_save(filepath)
        doc.close()

        logger.info("Rendered real highlighted handout: %s (highlights=%d)", filepath, total_highlights)
        return filepath

    async def render_highlighted_handout_review_predicted(
        self,
        course_code: str,
        review_texts: list[str],
        version: int = 1,
        workspace_id: str = "default",
        syllabus_scope: str = "all",
        split_mode: str = "auto",
        manual_midterm_end_page: int | None = None,
        model: str | None = None,
        batch_pages: int = 5,
        max_batch_chars: int = 7000,
        max_review_evidence_chars: int = 2500,
        max_spans_per_batch: int = 20,
        ai_logs: list[dict] | None = None,
        progress_callback=None,
        pause_check=None,
        job_id: str = "",
        resume_from_batch: int = 0,
        debug_mode: bool = False,
        continue_on_error: bool = True,
        max_highlights_per_page: int = 15,
    ) -> dict:
        """
        Review-informed sequential highlighting pipeline (v2 — stabilised).

        Key improvements over v1:
        - Overflow buffer: pages that exceed char limit are carried to next batch, never dropped.
        - <think> block stripping: deepseek-r1 thinking tags are removed before parsing.
        - Robust JSON parsing: accepts object-arrays, string-arrays, and line-based fallback.
        - System prompt: strong role-defining prompt sent as system message.
        - Per-batch retry: one retry with a repair prompt on parse failure.
        - Structured logging: every batch is logged via HighlightJobLog.
        - Per-page highlight cap: prevents visual clutter.
        - Priority scoring: spans ranked by review-relevance before application.
        - Deduplication: spans already highlighted are not re-applied.
        """
        from academic.highlight_config import (
            HighlightConfig, HIGHLIGHT_SYSTEM_PROMPT,
            HIGHLIGHT_BATCH_PROMPT_TEMPLATE, HIGHLIGHT_REPAIR_PROMPT,
        )
        from academic.highlight_logger import (
            HighlightJobLog, BatchLogEntry, save_debug_log,
        )
        import time as _time

        cfg = HighlightConfig(
            batch_pages=max(1, min(10, int(batch_pages or 5))),
            max_batch_chars=max(1200, min(30000, int(max_batch_chars or 7000))),
            max_review_evidence_chars=max(500, min(12000, int(max_review_evidence_chars or 2500))),
            max_spans_per_batch=max(5, min(60, int(max_spans_per_batch or 20))),
            debug_mode=debug_mode,
            continue_on_error=continue_on_error,
            max_highlights_per_page=max(3, min(30, int(max_highlights_per_page or 15))),
        )

        filename = f"{course_code}_highlighted_v{version}.pdf"
        filepath = str(ACADEMIC_OUTPUT_DIR / filename)

        source_pdf = _resolve_source_pdf_path(workspace_id)
        if not source_pdf or not os.path.exists(source_pdf):
            raise RuntimeError("No source PDF found for this course workspace")

        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise RuntimeError("PyMuPDF is required for real PDF highlighting") from e

        model_name = model or getattr(settings, "academic_generation_model", None) or settings.default_model

        # ── Model capability check ──
        model_warning: str | None = None
        try:
            model_info = await ollama_client.show_model(model_name)
            param_size_str = (model_info or {}).get("details", {}).get("parameter_size", "")
            if param_size_str:
                size_upper = param_size_str.strip().upper()
                if size_upper.endswith("B"):
                    try:
                        param_size_b = float(size_upper[:-1])
                        if 0 < param_size_b < cfg.min_model_params_b:
                            raise RuntimeError(
                                f"Model '{model_name}' has only {param_size_str} parameters. "
                                f"Models under {cfg.min_model_params_b}B cannot follow JSON format instructions "
                                f"and produce garbage/hallucinated output. "
                                f"Please use a 7B+ model (e.g., qwen2.5:7b, llama3:8b, deepseek-r1:7b)."
                            )
                    except ValueError:
                        pass
        except Exception:
            pass  # Best effort — don't block generation for model info failure

        # ── Build review evidence (once, reused every batch) ──
        review_lines = []
        for t in review_texts or []:
            s = re.sub(r"\s+", " ", str(t).strip())
            if len(s) >= 12:
                review_lines.append(s)
        review_evidence = "\n".join(review_lines)[:cfg.max_review_evidence_chars]

        # ── Open source PDF and resolve page range ──
        doc = fitz.open(source_pdf)
        total_pages = len(doc)

        mid_end = None
        if split_mode == "manual" and manual_midterm_end_page:
            mid_end = max(1, min(total_pages, int(manual_midterm_end_page)))
        elif split_mode == "auto":
            mid_end = _calculate_mids_pages(total_pages)

        if syllabus_scope == "midterm" and mid_end:
            target_pages = list(range(0, mid_end))
        elif syllabus_scope == "final" and mid_end:
            target_pages = list(range(mid_end, total_pages))
        else:
            target_pages = list(range(0, total_pages))

        if not target_pages:
            raise RuntimeError("No pages selected for highlighting")

        # ── Initialise job log ──
        job_log = HighlightJobLog(
            job_id=job_id,
            course_code=course_code,
            model=model_name,
            total_pages=total_pages,
            target_pages=len(target_pages),
        )

        live_base = ACADEMIC_OUTPUT_DIR / f"{course_code}_highlighted_live_{job_id or 'tmp'}"
        total_highlights = 0
        processed_batches = 0
        unmatched_total = 0
        failed_batches: list[int] = []
        skipped_batches: list[int] = []
        # Track highlights per page to enforce cap
        page_highlight_counts: dict[int, int] = {}
        # Track already-applied span texts to deduplicate across batches
        applied_spans: set[str] = set()

        # ── Helper: review keyword extraction (defined before first use) ──
        def _extract_review_keywords(text: str) -> list[str]:
            words = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{3,}", text.lower())
            stop = {
                "that", "this", "with", "from", "into", "about", "there", "their",
                "exam", "topic", "topics", "pages", "highlight", "important", "should",
                "would", "could", "were", "have", "has", "been", "what", "when", "where",
                "also", "very", "just", "some", "only", "more", "most", "then", "than",
                "each", "will", "does", "done", "make", "made", "like", "your", "they",
            }
            freq: dict[str, int] = {}
            for w in words:
                if w in stop:
                    continue
                freq[w] = freq.get(w, 0) + 1
            ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
            return [k for k, _ in ranked[:15]]

        # Review keywords for rescue path + priority scoring
        review_keywords = _extract_review_keywords(review_evidence) if review_evidence else []

        # ── Helper: safe PDF save ──
        def _safe_save(target_path: str) -> str:
            export_doc = doc
            created_export = False
            if len(target_pages) < total_pages:
                export_doc = fitz.open()
                created_export = True
                for page_idx in target_pages:
                    export_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
            try:
                export_doc.save(target_path, incremental=False, deflate=True)
                return target_path
            except Exception:
                fallback = str(Path(target_path).with_name(
                    f"{Path(target_path).stem}_{datetime.now(timezone.utc).strftime('%H%M%S%f')}{Path(target_path).suffix}"
                ))
                export_doc.save(fallback, incremental=False, deflate=True)
                return fallback
            finally:
                if created_export:
                    export_doc.close()

        # ── Helper: safe highlight annotation ──
        def _safe_add_highlight(page, rect) -> bool:
            try:
                annot = page.add_highlight_annot(rect)
                if annot is None:
                    return False
                try:
                    annot.update()
                except Exception:
                    pass
                return True
            except Exception:
                return False

        # ── Helper: strip code fences ──
        def _strip_code_fences(text: str) -> str:
            s = (text or "").strip()
            if s.startswith("```"):
                s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
                s = re.sub(r"\s*```$", "", s)
            return s.strip()

        # ── Helper: extract JSON array body ──
        def _extract_json_array_text(text: str) -> str:
            s = _strip_code_fences(text)
            start = s.find("[")
            end = s.rfind("]")
            if start >= 0 and end > start:
                return s[start:end + 1]
            return s

        # ── TICKET 3: Robust span extraction ──
        def _extract_spans(raw: str) -> list[str]:
            """
            Extract highlight text spans from AI response.

            Handles:
            - <think>...</think> blocks (deepseek-r1 pattern)
            - JSON object arrays [{text: ...}]
            - JSON string arrays ["..."]
            - dict with "highlights" key
            - Rejects hallucinated responses that contain no JSON at all
            """
            if not raw:
                return []

            # Step 1: Strip <think>...</think> and <reasoning>...</reasoning> blocks
            text = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', raw)
            text = re.sub(r'<reasoning>[\s\S]*?(?:</reasoning>|$)', '', text)
            text = text.strip()

            if not text:
                return []

            # Step 2: Reject hallucinated responses — if no '[' found, the AI
            # didn't produce JSON at all (it wrote an essay/summary instead)
            if '[' not in text:
                logger.warning("AI response contains no JSON array — likely hallucinated. Skipping.")
                return []

            # Step 3: Strip code fences and extract JSON body
            json_text = _extract_json_array_text(text)

            # Step 4: Attempt JSON parse
            try:
                parsed = json.loads(json_text)
                spans: list[str] = []
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

            # Step 5: No valid JSON parsed — return empty (don't fall back to
            # line-based extraction which picks up garbage from hallucinated text)
            logger.warning("Could not parse valid JSON spans from AI response (%d chars). Skipping.", len(text))
            return []



        # ── Helper: priority score for a span ──
        def _score_span(span_text: str, confidence: float = 0.5) -> float:
            score = confidence
            text_lower = span_text.lower()
            # Boost for review keyword matches
            kw_matches = sum(1 for kw in review_keywords if kw in text_lower)
            score += min(kw_matches * 0.08, 0.4)
            # Boost for academic content markers
            markers = ["definition", "formula", "theorem", "rule", "principle",
                        "algorithm", "equation", "function", "method", "protocol"]
            if any(m in text_lower for m in markers):
                score += 0.12
            # Boost for longer, more complete spans
            if len(span_text) > 100:
                score += 0.08
            elif len(span_text) > 60:
                score += 0.04
            return min(1.0, score)

        # ── Helper: apply spans to pages with dedup + per-page cap ──
        def _apply_spans_to_pages(
            spans: list[str],
            batch_page_indices: list[int],
            batch_log: BatchLogEntry,
        ) -> int:
            batch_hits = 0
            # Sort spans by priority score (highest first)
            scored = []
            for s in spans:
                # Try to get confidence from parsed data (best-effort)
                scored.append((s, _score_span(s)))
            scored.sort(key=lambda x: x[1], reverse=True)

            for s, _ in scored:
                # Dedup: skip if this exact span was already highlighted
                norm_key = re.sub(r'\s+', ' ', s.lower().strip())
                if norm_key in applied_spans:
                    continue

                matched = False
                for pidx in batch_page_indices:
                    # Enforce per-page highlight cap
                    if page_highlight_counts.get(pidx, 0) >= cfg.max_highlights_per_page:
                        continue

                    rects = doc[pidx].search_for(s)
                    if not rects:
                        # Fallback 1: whitespace-normalized prefix
                        prefix = re.sub(r"\s+", " ", s)[:120]
                        rects = doc[pidx].search_for(prefix) if len(prefix) >= 30 else []
                    if not rects and len(s) > 50:
                        # Fallback 2: alphanumeric-only head for OCR tolerance
                        head = re.sub(r"[^A-Za-z0-9\s]", " ", s)
                        head = re.sub(r"\s+", " ", head).strip()[:70]
                        rects = doc[pidx].search_for(head) if len(head) >= 28 else []
                    if not rects and len(s) > 80:
                        # Fallback 3: head+tail anchor matching
                        import fitz as _fitz
                        head_text = s[:40].strip()
                        tail_text = s[-40:].strip()
                        head_rects = doc[pidx].search_for(head_text) if len(head_text) >= 15 else []
                        tail_rects = doc[pidx].search_for(tail_text) if len(tail_text) >= 15 else []
                        if head_rects and tail_rects:
                            # Only combine if anchors are within ~200pt vertically (same region)
                            hr, tr = head_rects[0], tail_rects[0]
                            if abs(tr.y1 - hr.y0) < 200:
                                combined = _fitz.Rect(hr)
                                combined.include_rect(tr)
                                rects = [combined]
                    if not rects and '. ' in s and len(s) > 60:
                        # Fallback 4: sentence-level decomposition
                        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', s)
                        for sent in sentences:
                            sent = sent.strip()
                            if len(sent) >= 30:
                                sent_rects = doc[pidx].search_for(sent[:120])
                                for rect in sent_rects[:1]:
                                    if _safe_add_highlight(doc[pidx], rect):
                                        matched = True
                                        batch_hits += 1
                                        page_highlight_counts[pidx] = page_highlight_counts.get(pidx, 0) + 1
                                        batch_log.page_highlight_counts[pidx] = page_highlight_counts[pidx]
                    for rect in rects[:3]:
                        if _safe_add_highlight(doc[pidx], rect):
                            matched = True
                            batch_hits += 1
                            page_highlight_counts[pidx] = page_highlight_counts.get(pidx, 0) + 1
                            batch_log.page_highlight_counts[pidx] = page_highlight_counts[pidx]

                if matched:
                    applied_spans.add(norm_key)
                else:
                    batch_log.unmatched_spans.append(s[:120])
                    batch_log.spans_unmatched += 1

            return batch_hits

        # ── TICKET 2: Overflow-aware batch generator ──
        # Instead of hard-truncating page text, we carry overflow pages forward.
        effective_char_limit = cfg.effective_max_batch_chars
        overflow_text: str = ""          # text carried from previous batch
        overflow_pages: list[int] = []   # page indices carried from previous batch
        batch_index = 0
        page_cursor = 0  # index into target_pages

        while page_cursor < len(target_pages) or overflow_text:
            if pause_check:
                while pause_check():
                    await asyncio.sleep(0.5)

            batch_start_time = _time.monotonic()
            batch_log = BatchLogEntry(batch_index=batch_index)

            # Build batch text with overflow awareness
            batch_text_blocks: list[str] = []
            chars_used = 0
            actual_pages: list[int] = []
            new_overflow_text = ""
            new_overflow_pages: list[int] = []

            # Prepend overflow from previous batch
            if overflow_text:
                batch_text_blocks.append(overflow_text)
                chars_used += len(overflow_text)
                actual_pages.extend(overflow_pages)
                batch_log.overflow_pages = list(overflow_pages)
                overflow_text = ""
                overflow_pages = []

            # Add pages until char limit or batch_pages count reached
            pages_added_this_batch = 0
            while page_cursor < len(target_pages) and pages_added_this_batch < cfg.batch_pages:
                pidx = target_pages[page_cursor]
                page_text = re.sub(r"\s+", " ", doc[pidx].get_text("text")).strip()
                if not page_text:
                    page_cursor += 1
                    pages_added_this_batch += 1
                    continue

                page_block = f"[Page {pidx + 1}]\n{page_text}"

                if chars_used + len(page_block) > effective_char_limit and batch_text_blocks:
                    # This page would overflow — carry it forward
                    new_overflow_text = page_block
                    new_overflow_pages = [pidx]
                    # Don't advance page_cursor for subsequent pages
                    page_cursor += 1
                    # Also carry any remaining pages in this nominal batch
                    pages_added_this_batch += 1
                    while page_cursor < len(target_pages) and pages_added_this_batch < cfg.batch_pages:
                        pidx2 = target_pages[page_cursor]
                        pt2 = re.sub(r"\s+", " ", doc[pidx2].get_text("text")).strip()
                        if pt2:
                            new_overflow_text += f"\n\n[Page {pidx2 + 1}]\n{pt2}"
                            new_overflow_pages.append(pidx2)
                        page_cursor += 1
                        pages_added_this_batch += 1
                    break
                else:
                    batch_text_blocks.append(page_block)
                    chars_used += len(page_block)
                    actual_pages.append(pidx)
                    page_cursor += 1
                    pages_added_this_batch += 1

            overflow_text = new_overflow_text
            overflow_pages = new_overflow_pages

            batch_text = "\n\n".join(batch_text_blocks)
            if not batch_text.strip():
                if not overflow_text:
                    break
                continue

            batch_log.pages = [p + 1 for p in actual_pages]
            batch_log.actual_pages_processed = list(actual_pages)

            # Skip already-processed batches (checkpoint resume)
            if batch_index < resume_from_batch:
                batch_index += 1
                skipped_batches.append(batch_index)
                continue

            # ── TICKET 5: Build prompt with system message ──
            user_prompt = HIGHLIGHT_BATCH_PROMPT_TEMPLATE.format(
                max_spans=cfg.max_spans_per_batch,
                review_evidence=review_evidence or "No review evidence provided.",
                batch_text=batch_text,
            )
            batch_log.prompt_chars = len(user_prompt) + len(HIGHLIGHT_SYSTEM_PROMPT)

            # AI log entry
            log_entry: dict | None = None
            if ai_logs is not None:
                log_entry = {
                    "content_type": "highlighted_handout",
                    "topic_name": f"batch_pages_{'_'.join(str(p+1) for p in actual_pages)}",
                    "model": model_name,
                    "request": user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt,
                }
                ai_logs.append(log_entry)

            # ── TICKET 1: Per-batch error handling with retry ──
            raw = ""
            spans: list[str] = []
            try:
                resp = await ollama_client.chat(
                    messages=[
                        {"role": "system", "content": HIGHLIGHT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=model_name,
                    temperature=cfg.ai_temperature,
                    max_tokens=cfg.ai_max_tokens,
                )
                raw = (resp or {}).get("content", "")
                batch_log.response_chars = len(raw)
                batch_log.raw_response = raw
                if log_entry is not None:
                    log_entry["response"] = raw

                spans = _extract_spans(raw)
                batch_log.spans_returned = len(spans)

                # If no valid spans extracted, retry once with repair prompt
                if not spans and raw and cfg.max_retries_per_batch > 0:
                    logger.warning(
                        "Batch %d: No valid spans from AI response (%d chars). Retrying with repair prompt.",
                        batch_index, len(raw),
                    )
                    batch_log.retry_used = True
                    try:
                        retry_resp = await ollama_client.chat(
                            messages=[
                                {"role": "system", "content": HIGHLIGHT_SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": raw},
                                {"role": "user", "content": HIGHLIGHT_REPAIR_PROMPT},
                            ],
                            model=model_name,
                            temperature=cfg.ai_temperature,
                            max_tokens=cfg.ai_max_tokens,
                        )
                        retry_raw = (retry_resp or {}).get("content", "")
                        retry_spans = _extract_spans(retry_raw)
                        if retry_spans:
                            spans = retry_spans
                            raw = retry_raw
                            batch_log.spans_returned = len(spans)
                            batch_log.raw_response = retry_raw
                            logger.info("Batch %d: Retry succeeded with %d spans.", batch_index, len(spans))
                        else:
                            batch_log.retry_error = "Retry also produced no valid spans"
                    except Exception as retry_e:
                        batch_log.retry_error = str(retry_e)
                        logger.warning("Batch %d: Retry failed: %s", batch_index, retry_e)

            except Exception as e:
                batch_log.error = str(e)
                logger.error(
                    "Batch %d failed (pages %s): %s",
                    batch_index, batch_log.pages, e, exc_info=True,
                )
                if not cfg.continue_on_error:
                    job_log.add_batch_log(batch_log)
                    raise RuntimeError(
                        f"Batch {batch_index} failed and continue_on_error is disabled: {e}"
                    ) from e
                failed_batches.append(batch_index)
                batch_index += 1
                processed_batches += 1
                job_log.add_batch_log(batch_log)
                continue

            # ── Apply spans to pages ──
            batch_hits = _apply_spans_to_pages(spans, actual_pages, batch_log)
            batch_log.spans_matched = batch_hits

            # ── Review-informed rescue path (if zero AI hits) ──
            if batch_hits == 0 and review_keywords:
                rescue_hits = 0
                for pidx in actual_pages:
                    if page_highlight_counts.get(pidx, 0) >= cfg.max_highlights_per_page:
                        continue
                    page_text = doc[pidx].get_text("text")
                    if not page_text:
                        continue
                    # Build candidate sections: group consecutive non-empty lines
                    raw_lines = page_text.splitlines()
                    sections: list[str] = []
                    current_section: list[str] = []
                    for ln in raw_lines:
                        stripped = ln.strip()
                        if len(stripped) >= 15:
                            current_section.append(stripped)
                        elif current_section:
                            joined = " ".join(current_section)
                            if 40 <= len(joined) <= 600:
                                sections.append(joined)
                            current_section = []
                    if current_section:
                        joined = " ".join(current_section)
                        if 40 <= len(joined) <= 600:
                            sections.append(joined)

                    # Score and highlight sections that match review keywords
                    scored_sections = []
                    for sec in sections:
                        lower_sec = sec.lower()
                        kw_count = sum(1 for kw in review_keywords if kw in lower_sec)
                        if kw_count >= 2:
                            scored_sections.append((sec, kw_count))
                    scored_sections.sort(key=lambda x: x[1], reverse=True)

                    for sec, _ in scored_sections[:5]:
                        if page_highlight_counts.get(pidx, 0) >= cfg.max_highlights_per_page:
                            break
                        search_text = sec[:180]
                        rects = doc[pidx].search_for(search_text)
                        for rect in rects[:1]:
                            if _safe_add_highlight(doc[pidx], rect):
                                rescue_hits += 1
                                page_highlight_counts[pidx] = page_highlight_counts.get(pidx, 0) + 1
                                batch_log.page_highlight_counts[pidx] = page_highlight_counts[pidx]
                                break
                batch_hits += rescue_hits

            total_highlights += batch_hits
            processed_batches += 1
            batch_log.duration_seconds = _time.monotonic() - batch_start_time

            if batch_hits == 0 and batch_log.error:
                failed_batches.append(batch_index)

            # Check failure rate threshold
            if (
                processed_batches >= 3
                and len(failed_batches) / processed_batches > cfg.max_failed_batch_pct
                and not cfg.continue_on_error
            ):
                logger.error(
                    "Aborting job %s: failure rate %.0f%% exceeds threshold (%.0f%%)",
                    job_id, len(failed_batches) / processed_batches * 100,
                    cfg.max_failed_batch_pct * 100,
                )
                job_log.add_batch_log(batch_log)
                break

            job_log.add_batch_log(batch_log)

            # Live preview snapshot
            live_path = str(live_base.with_name(
                f"{live_base.name}_b{processed_batches}_p{actual_pages[-1] + 1 if actual_pages else 0}.pdf"
            ))
            live_path = _safe_save(live_path)

            if progress_callback:
                progress_callback(
                    actual_pages[-1] + 1 if actual_pages else 0,
                    batch_hits,
                    live_path,
                )

            batch_index += 1

        # ── Finalise ──
        filepath = _safe_save(filepath)
        doc.close()

        # Finalise and persist debug logs
        job_log.total_highlights = total_highlights
        job_log.skipped_batches = len(skipped_batches)
        job_log.finalize()

        if cfg.save_ai_logs or cfg.debug_mode:
            save_debug_log(job_log, include_raw_responses=cfg.debug_mode)

        logger.info(
            "Highlighted handout complete: %s | highlights=%d, batches=%d, "
            "match_rate=%.1f%%, failed=%d, unmatched=%d",
            filepath, total_highlights, processed_batches,
            job_log.overall_match_rate * 100,
            len(failed_batches), job_log.total_spans_unmatched,
        )

        return {
            "file_path": filepath,
            "highlight_count": total_highlights,
            "processed_batches": processed_batches,
            "unmatched_spans": job_log.total_spans_unmatched,
            "failed_batches": len(failed_batches),
            "skipped_batches": len(skipped_batches),
            "match_rate": round(job_log.overall_match_rate, 3),
            "total_spans_returned": job_log.total_spans_returned,
            "model_warning": model_warning,
            "debug_log_path": str(
                settings.data_dir / "academic_outputs" / "debug_logs" / (job_id or "unknown")
            ),
        }

    def render_short_notes(self, course_code: str, items: list[dict],
                            exam_type: str = "midterm", version: int = 1) -> str:
        """Render short notes PDF (midterm or final)."""
        label = "Midterm" if exam_type == "midterm" else "Final"
        filename = f"{course_code}_{exam_type}_notes_v{version}.pdf"
        filepath = str(ACADEMIC_OUTPUT_DIR / filename)

        if not _HAS_FPDF:
            return self._render_text_fallback(filepath, course_code, f"{label} Notes", items)

        pdf = AcademicPDF(course_code, version)
        pdf.alias_nb_pages()
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, clean_text(f"{course_code} - {label} Short Notes"), ln=True, align="C")
        pdf.ln(5)

        for item in items:
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, clean_text(item.get("topic_name", "Topic")), ln=True)

            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, clean_text(item.get("content", "")))
            pdf.ln(4)

        pdf.output(filepath)
        logger.info("Rendered %s notes: %s", exam_type, filepath)
        return filepath

    def render_mcq_bank(self, course_code: str, mcqs: list[dict],
                         version: int = 1) -> str:
        """Render MCQ bank with answer key."""
        filename = f"{course_code}_mcqs_v{version}.pdf"
        filepath = str(ACADEMIC_OUTPUT_DIR / filename)

        if not _HAS_FPDF:
            return self._render_text_fallback_mcq(filepath, course_code, mcqs)

        pdf = AcademicPDF(course_code, version)
        pdf.alias_nb_pages()
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, clean_text(f"{course_code} - MCQ Bank"), ln=True, align="C")
        pdf.ln(5)

        # Questions section
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, clean_text("Questions"), ln=True)
        pdf.ln(3)

        for i, mcq in enumerate(mcqs, 1):
            pdf.set_font("Helvetica", "B", 11)
            diff_tag = f"[{mcq.get('difficulty', 'medium').upper()}]"
            pdf.multi_cell(0, 6, clean_text(f"Q{i}. {diff_tag} {mcq.get('question', '')}"))

            pdf.set_font("Helvetica", "", 11)
            for opt in mcq.get("options", []):
                pdf.cell(10)  # indent
                pdf.multi_cell(0, 5, clean_text(opt))

            pdf.ln(3)

        # Answer key on new page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, clean_text("Answer Key"), ln=True)
        pdf.ln(3)

        for i, mcq in enumerate(mcqs, 1):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 6, clean_text(f"Q{i}: {mcq.get('correct_answer', '?')}"), ln=True)
            if mcq.get("rationale"):
                pdf.set_font("Helvetica", "I", 10)
                pdf.multi_cell(0, 5, clean_text(f"  -> {mcq['rationale']}"))
            pdf.ln(2)

        pdf.output(filepath)
        logger.info("Rendered MCQ bank: %s (%d questions)", filepath, len(mcqs))
        return filepath

    def _render_text_fallback(self, filepath: str, course_code: str,
                               title: str, items: list[dict]) -> str:
        """Fallback: render as text file when fpdf2 is not available."""
        filepath = filepath.replace(".pdf", ".txt")
        lines = [f"{course_code} — {title}", "=" * 50, ""]
        for item in items:
            lines.append(f"## {item.get('topic_name', 'Topic')}")
            lines.append(item.get("content", ""))
            lines.append("")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath

    def _render_text_fallback_mcq(self, filepath: str, course_code: str,
                                    mcqs: list[dict]) -> str:
        """Fallback MCQ render as text."""
        filepath = filepath.replace(".pdf", ".txt")
        lines = [f"{course_code} — MCQ Bank", "=" * 50, ""]
        for i, mcq in enumerate(mcqs, 1):
            lines.append(f"Q{i}. [{mcq.get('difficulty', '')}] {mcq.get('question', '')}")
            for opt in mcq.get("options", []):
                lines.append(f"  {opt}")
            lines.append(f"Answer: {mcq.get('correct_answer', '')}")
            lines.append(f"Rationale: {mcq.get('rationale', '')}")
            lines.append("")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath


# Singleton
pdf_renderer = PDFRenderer()
