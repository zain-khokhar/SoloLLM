"""
Syllabus Segmenter for VU Academic Auto-Generation.

Extracts course structure from handout PDFs: chapters, headings,
subheadings, topic maps with parent-child hierarchy.
Reuses the existing RAG ingest pipeline for PDF parsing.
"""

import re
import logging
from dataclasses import dataclass, field

from rag.ingest import parse_pdf, ParsedDocument, DocumentSection

logger = logging.getLogger(__name__)


@dataclass
class SyllabusTopic:
    """A topic extracted from the course syllabus/handout."""
    name: str
    level: int = 0  # 1=chapter, 2=section, 3=subsection
    page_number: int | None = None
    parent_topic: str = ""
    content_preview: str = ""  # first ~200 chars
    lecture_number: int | None = None
    is_definition: bool = False
    is_formula: bool = False
    is_example: bool = False


@dataclass
class SyllabusMap:
    """Complete structure map of a course handout."""
    course_code: str
    total_pages: int = 0
    topics: list[SyllabusTopic] = field(default_factory=list)
    chapter_count: int = 0
    section_count: int = 0
    errors: list[str] = field(default_factory=list)


class SyllabusSegmenter:
    """
    Extracts hierarchical structure from course handout PDFs.

    Uses the existing rag/ingest.py parse_pdf for initial extraction,
    then applies academic-specific heuristics for topic detection.
    """

    def segment_pdf(self, file_path: str, course_code: str = "") -> SyllabusMap:
        """
        Parse a PDF and extract its topic structure.

        Returns a SyllabusMap with hierarchical topics.
        """
        parsed = parse_pdf(file_path)

        if parsed.errors:
            return SyllabusMap(
                course_code=course_code,
                errors=parsed.errors,
            )

        topics = []
        chapter_count = 0
        section_count = 0

        current_chapter = ""

        for section in parsed.sections:
            topic = self._classify_section(section, current_chapter)
            if topic:
                topics.append(topic)

                if topic.level == 1:
                    chapter_count += 1
                    current_chapter = topic.name
                elif topic.level <= 2:
                    section_count += 1

        # If no structured headings found, create topics from pages
        if not topics and parsed.page_count > 0:
            topics = self._fallback_page_topics(parsed)
            chapter_count = len(topics)

        syllabus_map = SyllabusMap(
            course_code=course_code,
            total_pages=parsed.page_count,
            topics=topics,
            chapter_count=chapter_count,
            section_count=section_count,
        )

        logger.info(
            "Segmented %s: %d pages, %d chapters, %d sections, %d topics",
            course_code, parsed.page_count, chapter_count, section_count, len(topics),
        )

        return syllabus_map

    def _classify_section(self, section: DocumentSection,
                           current_chapter: str) -> SyllabusTopic | None:
        """Classify a document section into a syllabus topic."""
        title = section.title.strip()
        if not title or title.startswith("Page "):
            return None

        # Detect lecture/chapter numbers
        lecture_match = re.search(r'(?:lecture|chapter|unit|lesson)\s*[-:]?\s*(\d+)',
                                   title, re.IGNORECASE)
        lecture_num = int(lecture_match.group(1)) if lecture_match else None

        # Detect content type markers
        content_lower = section.content.lower() if section.content else ""
        is_definition = bool(re.search(r'(?:definition|define|defn)', content_lower))
        is_formula = bool(re.search(r'(?:formula|equation|=)', content_lower))
        is_example = bool(re.search(r'(?:example|e\.g\.|for instance)', content_lower))

        return SyllabusTopic(
            name=title,
            level=section.level if section.level > 0 else 2,
            page_number=section.page_number,
            parent_topic=current_chapter,
            content_preview=(section.content or "")[:200],
            lecture_number=lecture_num,
            is_definition=is_definition,
            is_formula=is_formula,
            is_example=is_example,
        )

    def _fallback_page_topics(self, parsed: ParsedDocument) -> list[SyllabusTopic]:
        """When no headings are found, create one topic per ~10 pages."""
        topics = []
        chunk_size = max(5, parsed.page_count // 10)

        for i in range(0, parsed.page_count, chunk_size):
            end_page = min(i + chunk_size, parsed.page_count)
            topics.append(SyllabusTopic(
                name=f"Section (Pages {i+1}-{end_page})",
                level=1,
                page_number=i + 1,
            ))

        return topics


# Singleton
syllabus_segmenter = SyllabusSegmenter()
