"""
BM25 Keyword Index for SoloLLM.

Uses SQLite FTS5 for full-text keyword search.
Complements vector search by catching exact matches
(names, codes, IDs) that embeddings might miss.
"""

import json
import logging
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
    ) -> list[KeywordResult]:
        """
        Search using BM25 ranking.

        FTS5's built-in bm25() function provides relevance scoring.
        """
        if not query.strip():
            return []

        # Escape special FTS5 characters
        safe_query = self._escape_fts_query(query)

        db = await self._get_db()
        try:
            if document_id:
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
                    (safe_query, workspace_id, document_id, top_k),
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
                    (safe_query, workspace_id, top_k),
                )

            rows = await cursor.fetchall()
            results = []
            for row in rows:
                r = dict(row)
                results.append(KeywordResult(
                    chunk_id=r["chunk_id"],
                    document_id=r["document_id"],
                    content=r["content"],
                    score=abs(r.get("score", 0)),  # bm25 returns negative scores
                    document_title=r.get("document_title", ""),
                    section_title=r.get("section_title", ""),
                    page_number=r.get("page_number"),
                    chunk_index=r.get("chunk_index", 0),
                ))
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
        Escape a user query for FTS5 MATCH.
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


# Singleton
keyword_index = KeywordIndex()
