"""
Hierarchical Multi-Strategy Chunking Engine for SoloLLM.

Implements multiple chunking strategies:
- Structural chunking (by headings/sections)
- Semantic chunking (by meaning shifts)
- Sliding window with overlap
- Parent-child linking
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk of content ready for embedding."""
    id: str = ""
    content: str = ""
    document_id: str = ""
    document_title: str = ""
    section_title: str = ""
    chunk_index: int = 0
    page_number: int | None = None
    parent_chunk_id: str | None = None
    sibling_chunk_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    char_count: int = 0
    estimated_tokens: int = 0

    def __post_init__(self):
        self.char_count = len(self.content)
        self.estimated_tokens = self.char_count // 4


class ChunkingEngine:
    """
    Hierarchical multi-strategy chunking engine.

    Strategies:
    1. Structural: Split by document sections/headings
    2. Sliding window: Fixed-size chunks with overlap
    3. Paragraph-aware: Split on paragraph boundaries, respecting max size
    """

    def __init__(
        self,
        max_chunk_size: int = 800,  # characters
        min_chunk_size: int = 100,
        overlap: int = 150,  # ~20% overlap for 800 char chunks
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap

    def chunk_document(
        self,
        content: str,
        sections: list | None = None,
        document_id: str = "",
        document_title: str = "",
    ) -> list[Chunk]:
        """
        Chunk a document using the best strategy.

        If sections are provided (from ingestion), use structural chunking.
        Otherwise, fall back to paragraph-aware sliding window.
        """
        if not content or not content.strip():
            return []

        if sections and len(sections) > 1:
            chunks = self._structural_chunk(
                sections, document_id, document_title
            )
        else:
            chunks = self._paragraph_chunk(
                content, document_id, document_title
            )

        # Assign IDs and link siblings
        for i, chunk in enumerate(chunks):
            chunk.id = f"{document_id}_chunk_{i}"
            chunk.chunk_index = i

            if i > 0:
                chunk.sibling_chunk_ids.append(chunks[i - 1].id)
            if i < len(chunks) - 1:
                chunk.sibling_chunk_ids.append(f"{document_id}_chunk_{i + 1}")

        logger.info(
            f"Chunked document '{document_title}': {len(chunks)} chunks "
            f"(avg {sum(c.char_count for c in chunks) // max(len(chunks), 1)} chars)"
        )

        return chunks

    def _structural_chunk(
        self,
        sections: list,
        document_id: str,
        document_title: str,
    ) -> list[Chunk]:
        """Chunk by document structure (sections/headings)."""
        chunks = []

        for section in sections:
            section_title = getattr(section, 'title', '')
            section_content = getattr(section, 'content', '')
            page_number = getattr(section, 'page_number', None)

            if not section_content or not section_content.strip():
                continue

            # If section is small enough, keep as one chunk
            if len(section_content) <= self.max_chunk_size:
                chunks.append(Chunk(
                    content=section_content.strip(),
                    document_id=document_id,
                    document_title=document_title,
                    section_title=section_title,
                    page_number=page_number,
                    metadata={"strategy": "structural"},
                ))
            else:
                # Section is too large — split with paragraph awareness
                sub_chunks = self._paragraph_chunk(
                    section_content,
                    document_id,
                    document_title,
                    section_title=section_title,
                    page_number=page_number,
                )
                # Link sub-chunks to parent section
                parent_id = f"{document_id}_section_{len(chunks)}"
                for sc in sub_chunks:
                    sc.parent_chunk_id = parent_id
                chunks.extend(sub_chunks)

        return chunks

    def _paragraph_chunk(
        self,
        text: str,
        document_id: str,
        document_title: str,
        section_title: str = "",
        page_number: int | None = None,
    ) -> list[Chunk]:
        """
        Paragraph-aware chunking with sliding window overlap.

        Splits text at paragraph boundaries, then merges small
        paragraphs and splits large ones to stay within size limits.
        """
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_content = ""

        for para in paragraphs:
            # If this paragraph alone exceeds max size, split it by sentences
            if len(para) > self.max_chunk_size:
                # Save current buffer
                if current_content.strip():
                    chunks.append(Chunk(
                        content=current_content.strip(),
                        document_id=document_id,
                        document_title=document_title,
                        section_title=section_title,
                        page_number=page_number,
                        metadata={"strategy": "paragraph"},
                    ))
                    current_content = ""

                # Split large paragraph by sentences
                sentence_chunks = self._sentence_split(para)
                for sc in sentence_chunks:
                    chunks.append(Chunk(
                        content=sc.strip(),
                        document_id=document_id,
                        document_title=document_title,
                        section_title=section_title,
                        page_number=page_number,
                        metadata={"strategy": "sentence"},
                    ))
                continue

            # Try to add paragraph to current chunk
            test_content = current_content + "\n\n" + para if current_content else para

            if len(test_content) <= self.max_chunk_size:
                current_content = test_content
            else:
                # Current chunk is full — save it
                if current_content.strip():
                    chunks.append(Chunk(
                        content=current_content.strip(),
                        document_id=document_id,
                        document_title=document_title,
                        section_title=section_title,
                        page_number=page_number,
                        metadata={"strategy": "paragraph"},
                    ))

                # Start new chunk with overlap
                if self.overlap > 0 and current_content:
                    overlap_text = current_content[-self.overlap:].strip()
                    current_content = overlap_text + "\n\n" + para
                else:
                    current_content = para

        # Save remaining content
        if current_content.strip() and len(current_content.strip()) >= self.min_chunk_size:
            chunks.append(Chunk(
                content=current_content.strip(),
                document_id=document_id,
                document_title=document_title,
                section_title=section_title,
                page_number=page_number,
                metadata={"strategy": "paragraph"},
            ))
        elif current_content.strip() and chunks:
            # Too small — merge with last chunk
            chunks[-1].content += "\n\n" + current_content.strip()
            chunks[-1].char_count = len(chunks[-1].content)
            chunks[-1].estimated_tokens = chunks[-1].char_count // 4

        return chunks

    def _sentence_split(self, text: str) -> list[str]:
        """Split text into sentence groups that fit within max_chunk_size."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""

        for sentence in sentences:
            test = current + " " + sentence if current else sentence
            if len(test) <= self.max_chunk_size:
                current = test
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks


# Singleton
chunking_engine = ChunkingEngine()
