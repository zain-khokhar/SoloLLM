"""
Review Ingestion Pipeline for VU Academic Auto-Generation.

Handles CSV/JSON/TXT review uploads, text normalization for
Urdu/Roman/English mix, duplicate detection, and spam filtering.
"""

import csv
import io
import json
import re
import hashlib
import logging
from dataclasses import dataclass, field

from storage.database import (
    acad_create_review_source, acad_add_review, acad_list_reviews,
    acad_update_review_flags, acad_get_review_count,
    acad_get_course_by_code,
)

logger = logging.getLogger(__name__)


# ── Text Normalization ──────────────────────────────────────

# Common Urdu/Roman transliteration patterns
_ROMAN_URDU_PATTERNS = {
    r'\bimportant\b': 'important',
    r'\bzaruri\b': 'important',
    r'\blaazmi\b': 'important',
    r'\bpakka\b': 'certain',
    r'\byaqeenan\b': 'certainly',
    r'\baat?a\s*hai\b': 'comes',
    r'\bhar\s*baar\b': 'every_time',
    r'\brepeat\b': 'repeat',
    r'\bdobara\b': 'repeat',
    r'\bmushkil\b': 'difficult',
    r'\baasan\b': 'easy',
    r'\bsawal\b': 'question',
    r'\bsure\s*question\b': 'sure_question',
}


def normalize_review_text(text: str) -> str:
    """
    Normalize review text for processing.

    Handles:
    - Whitespace normalization
    - Basic Roman Urdu keyword standardization
    - Lowercasing for analysis (preserves original for storage)
    """
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def extract_urgency_score(text: str) -> float:
    """
    Score the urgency/importance language in a review (0.0 - 1.0).

    Looks for words like 'important', 'zaruri', 'sure question',
    'repeat', 'definitely comes', etc.
    """
    text_lower = text.lower()

    urgency_keywords = {
        'important': 0.3,
        'zaruri': 0.3,
        'laazmi': 0.35,
        'must': 0.25,
        'sure question': 0.5,
        'sure shot': 0.5,
        'pakka': 0.4,
        'yaqeenan': 0.35,
        'definitely': 0.35,
        'repeat': 0.2,
        'dobara': 0.2,
        'har baar': 0.3,
        'every time': 0.3,
        'always comes': 0.35,
        'hamesha': 0.3,
        'ata hai': 0.2,
        'aata hai': 0.2,
        'critical': 0.3,
        'focus': 0.2,
        'prepare': 0.15,
    }

    score = 0.0
    for keyword, weight in urgency_keywords.items():
        if keyword in text_lower:
            score += weight

    return min(1.0, score)


def extract_sentiment_score(text: str) -> float:
    """
    Basic sentiment analysis for review text (0.0 = negative, 1.0 = positive).

    Focuses on academic sentiment (helpful, easy, difficult, etc.)
    """
    text_lower = text.lower()

    positive_words = ['easy', 'aasan', 'simple', 'helpful', 'good', 'best', 'clear']
    negative_words = ['difficult', 'mushkil', 'hard', 'confusing', 'bad', 'worst', 'tough']

    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.5  # neutral
    return pos_count / total


def compute_text_hash(text: str) -> str:
    """Compute a content hash for duplicate detection."""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def is_spam_review(text: str) -> bool:
    """
    Basic spam/low-quality detection.

    Flags reviews that are too short, contain only numbers/symbols,
    or are clearly irrelevant.
    """
    stripped = text.strip()

    # Too short to be useful
    if len(stripped) < 10:
        return True

    # All numbers or symbols
    if not re.search(r'[a-zA-Z\u0600-\u06FF]', stripped):
        return True

    # Just repeated characters
    if len(set(stripped.lower())) < 4:
        return True

    return False


# ── Review Ingestion Service ────────────────────────────────

class ReviewIngester:
    """
    Handles ingestion of student reviews for academic courses.

    Supports CSV, JSON, and plain text uploads.
    Reviews are always bound to a specific course_id for isolation.
    """

    async def ingest_csv(self, csv_content: str, course_id: str,
                          filename: str = "upload.csv") -> dict:
        """
        Ingest reviews from CSV content.

        Expected columns: review_text (required), semester (optional),
        reviewer_token (optional), course_code (optional — ignored, uses course_id param).
        """
        source = await acad_create_review_source(filename, "csv")

        reader = csv.DictReader(io.StringIO(csv_content))
        added = 0
        skipped_spam = 0
        skipped_duplicate = 0
        seen_hashes = set()

        for row in reader:
            review_text = (row.get("review_text") or row.get("review")
                           or row.get("text") or row.get("comment") or "").strip()
            if not review_text:
                continue

            # Normalize
            review_text = normalize_review_text(review_text)

            # Spam check
            if is_spam_review(review_text):
                skipped_spam += 1
                continue

            # Duplicate check within this batch
            text_hash = compute_text_hash(review_text)
            if text_hash in seen_hashes:
                skipped_duplicate += 1
                continue
            seen_hashes.add(text_hash)

            # Extract quality signals
            urgency = extract_urgency_score(review_text)
            sentiment = extract_sentiment_score(review_text)

            review = await acad_add_review(
                course_id=course_id,
                review_text=review_text,
                source_id=source["id"],
                reviewer_token=row.get("reviewer_token", row.get("student_id", "")),
                semester=row.get("semester", ""),
            )

            # Update with computed scores
            await acad_update_review_flags(
                review["id"],
                urgency_score=urgency,
                sentiment_score=sentiment,
            )

            added += 1

        # Update source count
        logger.info(
            "Ingested %d reviews from %s (spam=%d, dupe=%d)",
            added, filename, skipped_spam, skipped_duplicate,
        )

        return {
            "source_id": source["id"],
            "course_id": course_id,
            "total_added": added,
            "skipped_spam": skipped_spam,
            "skipped_duplicate": skipped_duplicate,
            "filename": filename,
        }

    async def ingest_json(self, json_content: str, course_id: str,
                           filename: str = "upload.json") -> dict:
        """Ingest reviews from JSON array."""
        source = await acad_create_review_source(filename, "json")

        try:
            data = json.loads(json_content)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format", "total_added": 0}

        if not isinstance(data, list):
            data = [data]

        added = 0
        seen_hashes = set()

        for item in data:
            review_text = ""
            if isinstance(item, str):
                review_text = item
            elif isinstance(item, dict):
                review_text = (item.get("review_text") or item.get("review")
                               or item.get("text") or item.get("comment") or "")

            review_text = normalize_review_text(review_text)
            if not review_text or is_spam_review(review_text):
                continue

            text_hash = compute_text_hash(review_text)
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)

            urgency = extract_urgency_score(review_text)
            sentiment = extract_sentiment_score(review_text)

            semester = item.get("semester", "") if isinstance(item, dict) else ""

            review = await acad_add_review(
                course_id=course_id,
                review_text=review_text,
                source_id=source["id"],
                semester=semester,
            )
            await acad_update_review_flags(
                review["id"],
                urgency_score=urgency,
                sentiment_score=sentiment,
            )
            added += 1

        return {
            "source_id": source["id"],
            "course_id": course_id,
            "total_added": added,
            "filename": filename,
        }

    async def ingest_text(self, text_content: str, course_id: str,
                           filename: str = "upload.txt") -> dict:
        """
        Ingest reviews from plain text (one review per line).
        """
        source = await acad_create_review_source(filename, "text")
        lines = text_content.strip().split("\n")
        added = 0
        seen_hashes = set()

        for line in lines:
            review_text = normalize_review_text(line)
            if not review_text or is_spam_review(review_text):
                continue

            text_hash = compute_text_hash(review_text)
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)

            urgency = extract_urgency_score(review_text)
            sentiment = extract_sentiment_score(review_text)

            review = await acad_add_review(
                course_id=course_id,
                review_text=review_text,
                source_id=source["id"],
            )
            await acad_update_review_flags(
                review["id"],
                urgency_score=urgency,
                sentiment_score=sentiment,
            )
            added += 1

        return {
            "source_id": source["id"],
            "course_id": course_id,
            "total_added": added,
            "filename": filename,
        }

    async def add_manual_review(self, course_id: str, review_text: str,
                                 semester: str = "") -> dict:
        """Add a single review manually."""
        review_text = normalize_review_text(review_text)

        if is_spam_review(review_text):
            return {"error": "Review is too short or appears to be spam"}

        urgency = extract_urgency_score(review_text)
        sentiment = extract_sentiment_score(review_text)

        review = await acad_add_review(
            course_id=course_id,
            review_text=review_text,
            semester=semester,
        )
        await acad_update_review_flags(
            review["id"],
            urgency_score=urgency,
            sentiment_score=sentiment,
        )

        return {"id": review["id"], "course_id": course_id, "added": True}


# Singleton
review_ingester = ReviewIngester()
