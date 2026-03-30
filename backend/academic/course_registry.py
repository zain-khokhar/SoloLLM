"""
Course Registry for VU Academic Auto-Generation.

Handles course code parsing, normalization, registry CRUD,
alias resolution, and unknown-code queue management.
"""

import re
import logging
from dataclasses import dataclass, field

from storage.database import (
    acad_create_course, acad_get_course, acad_get_course_by_code,
    acad_list_courses, acad_update_course, acad_delete_course,
    acad_add_course_alias,
)
from rag.pipeline import rag_pipeline

logger = logging.getLogger(__name__)


# ── Course Code Patterns ────────────────────────────────────

# Matches: CS201, CS-201, CS 201, cs201, MGT301, ENG101, etc.
_CODE_PATTERNS = [
    re.compile(r'(?i)\b([A-Z]{2,5})\s*[-_]?\s*(\d{3,4})\b'),
]

# Known VU department prefixes for higher-confidence matching
_KNOWN_PREFIXES = {
    "CS", "IT", "MGT", "MKT", "ENG", "MTH", "PHY", "STA", "ECO", "FIN",
    "ACC", "BNK", "PAK", "ISL", "EDU", "PSY", "SOC", "BIO", "MCM", "URD",
    "HRM", "BIF", "CIS", "MIS", "IST", "PHI", "POL", "LAW", "LNG",
}


@dataclass
class ParsedCourseCode:
    """Result of parsing a course code."""
    raw: str
    prefix: str  # e.g. "CS"
    number: str  # e.g. "201"
    normalized: str  # e.g. "CS201"
    confidence: float = 0.0  # 0-1, higher = more confident
    source: str = ""  # "filename", "content", "manual"


def extract_course_code(text: str, source: str = "filename") -> ParsedCourseCode | None:
    """
    Extract a course code from text (filename, content, etc.).

    Tries multiple patterns and returns the best match with confidence.
    """
    text = text.strip()

    for pattern in _CODE_PATTERNS:
        match = pattern.search(text)
        if match:
            prefix = match.group(1).upper()
            number = match.group(2)
            normalized = f"{prefix}{number}"

            # Higher confidence if prefix is a known VU department
            confidence = 0.9 if prefix in _KNOWN_PREFIXES else 0.6

            return ParsedCourseCode(
                raw=match.group(0),
                prefix=prefix,
                number=number,
                normalized=normalized,
                confidence=confidence,
                source=source,
            )

    return None


def extract_course_codes_from_filename(filename: str) -> list[ParsedCourseCode]:
    """Extract all course codes found in a filename."""
    results = []
    # Remove extension
    name = re.sub(r'\.[^.]+$', '', filename)

    for pattern in _CODE_PATTERNS:
        for match in pattern.finditer(name):
            prefix = match.group(1).upper()
            number = match.group(2)
            normalized = f"{prefix}{number}"
            confidence = 0.9 if prefix in _KNOWN_PREFIXES else 0.6
            results.append(ParsedCourseCode(
                raw=match.group(0),
                prefix=prefix,
                number=number,
                normalized=normalized,
                confidence=confidence,
                source="filename",
            ))

    return results


# ── Course Registry Service ─────────────────────────────────

class CourseRegistry:
    """
    Manages the VU course registry.

    Provides course CRUD, alias resolution, and auto-registration
    from uploaded PDFs.
    """

    async def register_course(self, code: str, title: str = "",
                               department: str = "") -> dict:
        """Register a new course or return existing one."""
        existing = await acad_get_course_by_code(code)
        if existing:
            logger.info("Course %s already registered (id=%s)", code, existing["id"])
            return existing

        course = await acad_create_course(
            code=code, title=title or code, department=department,
        )
        logger.info("Registered new course: %s (workspace=%s)", code, course["workspace_id"])

        # Create isolated FAISS workspace for this course
        await rag_pipeline.create_workspace(
            name=f"Academic - {code}",
            description=f"Isolated vector store for course {code}",
        )

        return course

    async def register_from_filename(self, filename: str) -> dict | None:
        """
        Auto-detect course code from filename and register if found.

        Returns the course dict or None if code could not be detected.
        """
        parsed = extract_course_code(filename, source="filename")
        if not parsed:
            logger.warning("Could not detect course code from filename: %s", filename)
            return None

        return await self.register_course(
            code=parsed.normalized,
            department=parsed.prefix,
        )

    async def resolve_course(self, code: str) -> dict | None:
        """Resolve a course code (including aliases) to a course record."""
        return await acad_get_course_by_code(code)

    async def list_all(self) -> list[dict]:
        """List all registered courses."""
        return await acad_list_courses()

    async def add_alias(self, course_id: str, alias: str) -> dict:
        """Add an alias for a course."""
        return await acad_add_course_alias(course_id, alias)

    async def get_course(self, course_id: str) -> dict | None:
        """Get course by ID."""
        return await acad_get_course(course_id)

    async def update_course(self, course_id: str, **kwargs) -> bool:
        """Update course fields."""
        return await acad_update_course(course_id, **kwargs)

    async def delete_course(self, course_id: str) -> bool:
        """Delete a course and all associated data (cascade)."""
        return await acad_delete_course(course_id)


# Singleton
course_registry = CourseRegistry()
