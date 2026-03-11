"""
Citation Tracker for SoloLLM.

Generates inline citations and source references for RAG responses.
Maps retrieved chunks to citation markers [1], [2], etc. so the
user can see exactly where information came from.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A single citation reference."""
    index: int  # [1], [2], etc.
    document_title: str
    section_title: str = ""
    page_number: int | None = None
    document_id: str = ""
    chunk_id: str = ""
    relevance_score: float = 0.0
    excerpt: str = ""  # Short snippet from the source


@dataclass
class CitedContext:
    """Context prepared for LLM with citation markers."""
    context_text: str  # The context with [1], [2] markers
    citations: list[Citation] = field(default_factory=list)
    instruction: str = ""  # Instruction to append to system prompt

    def format_references(self) -> str:
        """Format citations as a references section."""
        if not self.citations:
            return ""

        lines = ["\n---\n**Sources:**"]
        for cite in self.citations:
            parts = [f"[{cite.index}] **{cite.document_title}**"]
            if cite.section_title:
                parts.append(f"§ {cite.section_title}")
            if cite.page_number:
                parts.append(f"(p. {cite.page_number})")
            lines.append(" — ".join(parts))

        return "\n".join(lines)


class CitationTracker:
    """
    Builds cited context for RAG-augmented LLM queries.

    Takes retrieved chunks and creates numbered context blocks
    so the LLM can reference sources in its response.
    """

    def build_cited_context(
        self,
        results: list,
        max_context_chars: int = 4000,
    ) -> CitedContext:
        """
        Build context text with inline citation markers.

        Takes a list of RetrievalResults and produces:
        - A context string with [1], [2], etc. markers
        - A list of Citation objects for the reference section
        - Instructions for the LLM to cite sources
        """
        if not results:
            return CitedContext(
                context_text="",
                citations=[],
                instruction="",
            )

        context_blocks = []
        citations = []
        total_chars = 0

        for i, result in enumerate(results):
            cite_index = i + 1

            # Build context block
            content = result.content.strip()

            # Truncate individual chunks if needed
            remaining = max_context_chars - total_chars
            if remaining <= 0:
                break
            if len(content) > remaining:
                content = content[:remaining - 50] + "..."

            # Create citation marker
            source_label = getattr(result, 'document_title', '') or "Unknown Source"
            section = getattr(result, 'section_title', '') or ""
            page = getattr(result, 'page_number', None)

            header = f"[Source {cite_index}: {source_label}"
            if section:
                header += f" > {section}"
            if page:
                header += f" (p. {page})"
            header += "]"

            context_blocks.append(f"{header}\n{content}")
            total_chars += len(content) + len(header) + 2

            # Create citation
            excerpt = content[:120] + "..." if len(content) > 120 else content
            citations.append(Citation(
                index=cite_index,
                document_title=source_label,
                section_title=section,
                page_number=page,
                document_id=getattr(result, 'document_id', ''),
                chunk_id=getattr(result, 'chunk_id', ''),
                relevance_score=getattr(result, 'score', 0),
                excerpt=excerpt,
            ))

        context_text = "\n\n".join(context_blocks)

        instruction = (
            "Use the provided sources to answer the user's question. "
            "When you use information from a source, cite it by its number, "
            "e.g., [Source 1], [Source 2]. If the sources don't contain "
            "relevant information, say so and answer from your general knowledge."
        )

        return CitedContext(
            context_text=context_text,
            citations=citations,
            instruction=instruction,
        )

    def format_system_context(self, cited_context: CitedContext) -> str:
        """
        Format the cited context as a system message addition.

        This is appended to the system prompt for RAG-augmented queries.
        """
        if not cited_context.context_text:
            return ""

        return (
            f"\n\n--- RETRIEVED CONTEXT ---\n"
            f"{cited_context.instruction}\n\n"
            f"{cited_context.context_text}\n"
            f"--- END CONTEXT ---"
        )


# Singleton
citation_tracker = CitationTracker()
