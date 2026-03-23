"""
BM25 Keyword Index for SoloLLM.

Uses SQLite FTS5 for full-text keyword search.
Complements vector search by catching exact matches
(names, codes, IDs) that embeddings might miss.

Precision upgrade: supports AND/phrase-aware queries and
returns richer lexical metadata (term_coverage, phrase_match,
bm25_normalized) when precision_mode is enabled.
"""

import json
import logging
import re
import aiosqlite
from dataclasses import dataclass, field
from pathlib import Path

from core.config import settings

logger = logging.getLogger(__name__)

KEYWORD_DB_PATH = str(settings.data_dir / "db" / "keyword_index.db")

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id,
    document_id,
    workspace_id,
    content,
    document_title,
    section_title,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS chunks_meta (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    workspace_id TEXT DEFAULT 'default',
    page_number INTEGER,
    chunk_index INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);
"""


@dataclass
class KeywordResult:
    """A single keyword search result."""
    chunk_id: str
    document_id: str
    content: str
    score: float  # BM25 score
    document_title: str = ""
    section_title: str = ""
    page_number: int | None = None
    chunk_index: int = 0
    # Precision metadata
    term_coverage: float = 0.0     # fraction of required terms found
    phrase_match: float = 0.0      # 0.0 or 1.0 (graded phrase hit)
    bm25_normalized: float = 0.0   # BM25 score normalized to 0-1 range


class KeywordIndex:
    """SQLite FTS5-based keyword search index."""

    def __init__(self, db_path: str = KEYWORD_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def _get_db(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        return db

    async def init(self):
        """Initialize the FTS5 tables."""
        db = await self._get_db()
        try:
            await db.executescript(FTS_SCHEMA)
            await db.commit()
            logger.info("Keyword index initialized")
        finally:
            await db.close()

    async def add_chunks(
        self,
        chunks: list[dict],
        workspace_id: str = "default",
    ):
        """
        Index chunks for keyword search.

        Each chunk dict should have: id, document_id, content
        Optional: document_title, section_title, page_number, chunk_index
        """
        db = await self._get_db()
        try:
            for chunk in chunks:
                # Add to FTS index
                await db.execute(
                    """INSERT OR REPLACE INTO chunks_fts
                       (chunk_id, document_id, workspace_id, content,
                        document_title, section_title)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        chunk["id"],
                        chunk["document_id"],
                        workspace_id,
                        chunk["content"],
                        chunk.get("document_title", ""),
                        chunk.get("section_title", ""),
                    ),
                )
                # Add metadata
                await db.execute(
                    """INSERT OR REPLACE INTO chunks_meta
                       (chunk_id, document_id, workspace_id, page_number,
                        chunk_index, metadata)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        chunk["id"],
                        chunk["document_id"],
                        workspace_id,
                        chunk.get("page_number"),
                        chunk.get("chunk_index", 0),
                        json.dumps(chunk.get("metadata", {})),
                    ),
                )
            await db.commit()
            logger.info(f"Indexed {len(chunks)} chunks in keyword index")
        finally:
            await db.close()

    async def search(
        self,
        query: str,
        workspace_id: str = "default",
        top_k: int = 10,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        precision_mode: bool = False,
        required_terms: list[str] | None = None,
        keyphrases: list[str] | None = None,
    ) -> list[KeywordResult]:
        """
        Search using BM25 ranking.

        FTS5's built-in bm25() function provides relevance scoring.

        Args:
            document_id: Filter to a single document (legacy)
            document_ids: Filter to multiple documents (thread-scoped)
            precision_mode: Use AND/phrase-aware query construction
            required_terms: Terms that must appear (precision mode)
            keyphrases: Multi-word phrases for boosted matching
        """
        if not query.strip():
            return []

        # Build FTS query based on mode
        if precision_mode:
            safe_query = self._build_precision_query(
                query, required_terms or [], keyphrases or []
            )
        else:
            safe_query = self._escape_fts_query(query)

        if not safe_query:
            return []

        db = await self._get_db()
        try:
            if document_ids:
                # Thread-scoped: only search within these specific documents
                placeholders = ",".join("?" for _ in document_ids)
                cursor = await db.execute(
                    f"""SELECT f.chunk_id, f.document_id, f.content,
                              f.document_title, f.section_title,
                              m.page_number, m.chunk_index,
                              bm25(chunks_fts, 0, 0, 0, 1, 0, 0) as score
                       FROM chunks_fts f
                       LEFT JOIN chunks_meta m ON f.chunk_id = m.chunk_id
                       WHERE chunks_fts MATCH ?
                       AND f.workspace_id = ?
                       AND f.document_id IN ({placeholders})
                       ORDER BY score
                       LIMIT ?""",
                    (safe_query, workspace_id, *document_ids, top_k * 3 if precision_mode else top_k),
                )
            elif document_id:
                cursor = await db.execute(
                    """SELECT f.chunk_id, f.document_id, f.content,
                              f.document_title, f.section_title,
                              m.page_number, m.chunk_index,
                              bm25(chunks_fts, 0, 0, 0, 1, 0, 0) as score
                       FROM chunks_fts f
                       LEFT JOIN chunks_meta m ON f.chunk_id = m.chunk_id
                       WHERE chunks_fts MATCH ?
                       AND f.workspace_id = ?
                       AND f.document_id = ?
                       ORDER BY score
                       LIMIT ?""",
                    (safe_query, workspace_id, document_id, top_k * 3 if precision_mode else top_k),
                )
            else:
                cursor = await db.execute(
                    """SELECT f.chunk_id, f.document_id, f.content,
                              f.document_title, f.section_title,
                              m.page_number, m.chunk_index,
                              bm25(chunks_fts, 0, 0, 0, 1, 0, 0) as score
                       FROM chunks_fts f
                       LEFT JOIN chunks_meta m ON f.chunk_id = m.chunk_id
                       WHERE chunks_fts MATCH ?
                       AND f.workspace_id = ?
                       ORDER BY score
                       LIMIT ?""",
                    (safe_query, workspace_id, top_k * 3 if precision_mode else top_k),
                )

            rows = await cursor.fetchall()
            results = []
            max_bm25 = 0.0

            # First pass: collect raw results and find max BM25 for normalization
            raw_results = []
            for row in rows:
                r = dict(row)
                bm25_score = abs(r.get("score", 0))  # bm25 returns negative scores
                max_bm25 = max(max_bm25, bm25_score)
                raw_results.append((r, bm25_score))

            # Second pass: build results with precision metadata
            for r, bm25_score in raw_results:
                content_lower = r.get("content", "").lower()

                # Compute precision metadata
                term_cov = 0.0
                phrase_match = 0.0
                bm25_norm = bm25_score / max_bm25 if max_bm25 > 0 else 0.0

                if precision_mode and required_terms:
                    matched = sum(
                        1 for t in required_terms if t.lower() in content_lower
                    )
                    term_cov = matched / max(len(required_terms), 1)

                if precision_mode and keyphrases:
                    for kp in keyphrases:
                        if kp.lower() in content_lower:
                            phrase_match = 1.0
                            break

                results.append(KeywordResult(
                    chunk_id=r["chunk_id"],
                    document_id=r["document_id"],
                    content=r.get("content", ""),
                    score=bm25_score,
                    document_title=r.get("document_title", ""),
                    section_title=r.get("section_title", ""),
                    page_number=r.get("page_number"),
                    chunk_index=r.get("chunk_index", 0),
                    term_coverage=round(term_cov, 3),
                    phrase_match=phrase_match,
                    bm25_normalized=round(bm25_norm, 4),
                ))

            # In precision mode, re-sort by composite lexical score
            if precision_mode:
                for res in results:
                    res.score = (
                        0.55 * res.bm25_normalized
                        + 0.30 * res.phrase_match
                        + 0.15 * res.term_coverage
                    )
                results.sort(key=lambda x: x.score, reverse=True)
                results = results[:top_k]

            return results
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
        finally:
            await db.close()

    async def delete_document(self, document_id: str):
        """Remove all chunks for a document from the keyword index."""
        db = await self._get_db()
        try:
            await db.execute(
                "DELETE FROM chunks_fts WHERE document_id = ?",
                (document_id,),
            )
            await db.execute(
                "DELETE FROM chunks_meta WHERE document_id = ?",
                (document_id,),
            )
            await db.commit()
        finally:
            await db.close()

    def _escape_fts_query(self, query: str) -> str:
        """
        Escape a user query for FTS5 MATCH (legacy OR mode).
        Wraps each word in quotes to avoid syntax errors from special chars.
        """
        words = query.strip().split()
        escaped = []
        for word in words:
            # Remove FTS5 special characters
            clean = ''.join(c for c in word if c.isalnum() or c in "_-")
            if clean:
                escaped.append(f'"{clean}"')
        return " OR ".join(escaped) if escaped else '""'

    def _build_precision_query(
        self,
        query: str,
        required_terms: list[str],
        keyphrases: list[str],
    ) -> str:
        """
        Build a precision-aware FTS5 query.

        Strategy:
        - Keyphrases → NEAR() clauses for phrase proximity matching
        - Required terms → AND clauses (all must appear)
        - Falls back to OR if AND yields nothing constructive
        """
        parts = []

        # 1. Keyphrase NEAR matches (high boost)
        for kp in keyphrases:
            kp_words = [
                ''.join(c for c in w if c.isalnum() or c in "_-")
                for w in kp.split()
            ]
            kp_words = [w for w in kp_words if w]
            if len(kp_words) >= 2:
                near_clause = " ".join(f'"{w}"' for w in kp_words)
                parts.append(f"NEAR({near_clause}, 3)")

        # 2. Required terms (AND)
        and_terms = []
        for term in required_terms:
            clean = ''.join(c for c in term if c.isalnum() or c in "_-")
            if clean:
                and_terms.append(f'"{clean}"')

        if and_terms:
            parts.append(" AND ".join(and_terms))

        # 3. Combine: OR between phrase matches and required-term AND block
        if parts:
            return " OR ".join(parts)

        # Fallback: standard OR query
        return self._escape_fts_query(query)


# Singleton
keyword_index = KeywordIndex()
