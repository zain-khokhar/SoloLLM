"""
Structured Logging for the Highlighted Handout Pipeline.

Captures per-batch AI input/output, span match statistics,
and generates a summary report at job completion.
All logs are persisted to disk when debug_mode is enabled.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import settings

logger = logging.getLogger(__name__)

# Debug log directory
DEBUG_LOG_DIR = settings.data_dir / "academic_outputs" / "debug_logs"
DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BatchLogEntry:
    """Log entry for a single batch processing run."""
    batch_index: int = 0
    pages: list[int] = field(default_factory=list)
    actual_pages_processed: list[int] = field(default_factory=list)
    overflow_pages: list[int] = field(default_factory=list)

    # AI interaction
    prompt_chars: int = 0
    response_chars: int = 0
    raw_response: str = ""

    # Parsing results
    spans_returned: int = 0
    spans_matched: int = 0
    spans_unmatched: int = 0
    unmatched_spans: list[str] = field(default_factory=list)
    unmatched_reasons: list[str] = field(default_factory=list)

    # Per-page highlight counts
    page_highlight_counts: dict[int, int] = field(default_factory=dict)

    # Errors
    error: str = ""
    retry_used: bool = False
    retry_error: str = ""

    # Timing
    duration_seconds: float = 0.0

    @property
    def match_rate(self) -> float:
        if self.spans_returned == 0:
            return 0.0
        return self.spans_matched / self.spans_returned

    def to_dict(self) -> dict:
        return {
            "batch_index": self.batch_index,
            "pages": self.pages,
            "actual_pages_processed": self.actual_pages_processed,
            "overflow_pages": self.overflow_pages,
            "prompt_chars": self.prompt_chars,
            "response_chars": self.response_chars,
            "spans_returned": self.spans_returned,
            "spans_matched": self.spans_matched,
            "spans_unmatched": self.spans_unmatched,
            "unmatched_spans": self.unmatched_spans[:10],
            "match_rate": round(self.match_rate, 3),
            "page_highlight_counts": self.page_highlight_counts,
            "error": self.error,
            "retry_used": self.retry_used,
            "retry_error": self.retry_error,
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass
class HighlightJobLog:
    """Aggregate log for an entire highlight generation job."""
    job_id: str = ""
    course_code: str = ""
    model: str = ""
    total_pages: int = 0
    target_pages: int = 0
    total_batches: int = 0

    batch_logs: list[BatchLogEntry] = field(default_factory=list)

    # Aggregates (computed at finalize)
    total_highlights: int = 0
    total_spans_returned: int = 0
    total_spans_matched: int = 0
    total_spans_unmatched: int = 0
    total_retries: int = 0
    failed_batches: int = 0
    skipped_batches: int = 0
    overall_match_rate: float = 0.0
    total_duration_seconds: float = 0.0

    def add_batch_log(self, entry: BatchLogEntry):
        self.batch_logs.append(entry)

    def finalize(self):
        """Compute aggregate statistics from all batch logs."""
        self.total_batches = len(self.batch_logs)
        self.total_spans_returned = sum(b.spans_returned for b in self.batch_logs)
        self.total_spans_matched = sum(b.spans_matched for b in self.batch_logs)
        self.total_spans_unmatched = sum(b.spans_unmatched for b in self.batch_logs)
        self.total_retries = sum(1 for b in self.batch_logs if b.retry_used)
        self.failed_batches = sum(1 for b in self.batch_logs if b.error and b.spans_matched == 0)
        self.total_duration_seconds = sum(b.duration_seconds for b in self.batch_logs)

        if self.total_spans_returned > 0:
            self.overall_match_rate = self.total_spans_matched / self.total_spans_returned
        else:
            self.overall_match_rate = 0.0

    def to_summary_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "course_code": self.course_code,
            "model": self.model,
            "total_pages": self.total_pages,
            "target_pages": self.target_pages,
            "total_batches": self.total_batches,
            "total_highlights": self.total_highlights,
            "total_spans_returned": self.total_spans_returned,
            "total_spans_matched": self.total_spans_matched,
            "total_spans_unmatched": self.total_spans_unmatched,
            "overall_match_rate": round(self.overall_match_rate, 3),
            "total_retries": self.total_retries,
            "failed_batches": self.failed_batches,
            "skipped_batches": self.skipped_batches,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
        }

    def to_full_dict(self) -> dict:
        d = self.to_summary_dict()
        d["batches"] = [b.to_dict() for b in self.batch_logs]
        return d


def save_debug_log(job_log: HighlightJobLog, *, include_raw_responses: bool = False):
    """Persist job log to disk as JSON."""
    try:
        job_dir = DEBUG_LOG_DIR / (job_log.job_id or "unknown")
        job_dir.mkdir(parents=True, exist_ok=True)

        # Summary
        summary_path = job_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(job_log.to_summary_dict(), f, indent=2, ensure_ascii=False)

        # Full batch details
        batches_path = job_dir / "batches.json"
        with open(batches_path, "w", encoding="utf-8") as f:
            json.dump(job_log.to_full_dict(), f, indent=2, ensure_ascii=False)

        # Raw AI responses (if debug mode)
        if include_raw_responses:
            for entry in job_log.batch_logs:
                if entry.raw_response:
                    resp_path = job_dir / f"batch_{entry.batch_index}_response.txt"
                    with open(resp_path, "w", encoding="utf-8") as f:
                        f.write(entry.raw_response)

        logger.info(
            "Debug logs saved for job %s at %s (batches=%d, match_rate=%.1f%%)",
            job_log.job_id, job_dir,
            job_log.total_batches,
            job_log.overall_match_rate * 100,
        )
    except Exception as e:
        logger.warning("Failed to save debug logs for job %s: %s", job_log.job_id, e)


def load_debug_log(job_id: str) -> dict | None:
    """Load a previously saved debug log summary."""
    summary_path = DEBUG_LOG_DIR / job_id / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_debug_log_full(job_id: str) -> dict | None:
    """Load the full debug log including batch details."""
    batches_path = DEBUG_LOG_DIR / job_id / "batches.json"
    if not batches_path.exists():
        return None
    try:
        with open(batches_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
