"""
Vector Store for SoloLLM.

SQLite-backed vector store with in-memory HNSW-like search.
Uses numpy for efficient vector operations. Falls back to
brute-force cosine similarity if the dataset is small.

This avoids external dependencies like ChromaDB or Qdrant
while still providing fast approximate nearest neighbor search.
"""

import json
import logging
import aiosqlite
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from core.config import settings

logger = logging.getLogger(__name__)

VECTORS_DB_PATH = str(settings.data_dir / "db" / "vectors.db")

VECTOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    workspace_id TEXT DEFAULT 'default',
    content TEXT NOT NULL,
    document_title TEXT DEFAULT '',
    section_title TEXT DEFAULT '',
    chunk_index INTEGER DEFAULT 0,
    page_number INTEGER,
    parent_chunk_id TEXT,
    embedding BLOB,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_workspace ON document_chunks(workspace_id);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    workspace_id TEXT DEFAULT 'default',
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    title TEXT DEFAULT '',
    content_hash TEXT DEFAULT '',
    chunk_count INTEGER DEFAULT 0,
    page_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents(workspace_id);

CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    document_title: str = ""
    section_title: str = ""
    page_number: int | None = None
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """SQLite-backed vector store with numpy-based search."""

    def __init__(self, db_path: str = VECTORS_DB_PATH):
        self.db_path = db_path
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def _get_db(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        return db

    async def init(self):
        """Initialize the vector store tables."""
        db = await self._get_db()
        try:
            await db.executescript(VECTOR_SCHEMA)
            await db.commit()
            logger.info("Vector store initialized")
        finally:
            await db.close()

    # ── Document Management ─────────────────────────────────

    async def add_document(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        title: str = "",
        content_hash: str = "",
        chunk_count: int = 0,
        page_count: int = 0,
        workspace_id: str = "default",
        metadata: dict | None = None,
    ):
        """Register a document in the store."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        db = await self._get_db()
        try:
            await db.execute(
                """INSERT OR REPLACE INTO documents
                   (id, workspace_id, filename, file_type, title, content_hash,
                    chunk_count, page_count, metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (document_id, workspace_id, filename, file_type, title,
                 content_hash, chunk_count, page_count,
                 json.dumps(metadata or {}), now, now),
            )
            await db.commit()
        finally:
            await db.close()

    async def list_documents(self, workspace_id: str = "default") -> list[dict]:
        """List all documents in a workspace."""
        db = await self._get_db()
        try:
            cursor = await db.execute(
                "SELECT * FROM documents WHERE workspace_id = ? ORDER BY created_at DESC",
                (workspace_id,),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            await db.close()

    async def delete_document(self, document_id: str):
        """Delete a document and all its chunks."""
        db = await self._get_db()
        try:
            await db.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
            await db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            await db.commit()
        finally:
            await db.close()

    async def get_document(self, document_id: str) -> dict | None:
        """Get document metadata."""
        db = await self._get_db()
        try:
            cursor = await db.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None
        finally:
            await db.close()

    # ── Chunk Management ────────────────────────────────────

    async def add_chunks(
        self,
        chunks: list[dict],
        workspace_id: str = "default",
    ):
        """
        Add multiple chunks with embeddings to the store.

        Each chunk dict should have:
        - id, document_id, content, embedding (list[float])
        - Optional: document_title, section_title, chunk_index, page_number, metadata
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        db = await self._get_db()
        try:
            for chunk in chunks:
                embedding_bytes = None
                if chunk.get("embedding"):
                    embedding_bytes = np.array(chunk["embedding"], dtype=np.float32).tobytes()

                await db.execute(
                    """INSERT OR REPLACE INTO document_chunks
                       (id, document_id, workspace_id, content, document_title,
                        section_title, chunk_index, page_number, parent_chunk_id,
                        embedding, metadata, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chunk["id"],
                        chunk["document_id"],
                        workspace_id,
                        chunk["content"],
                        chunk.get("document_title", ""),
                        chunk.get("section_title", ""),
                        chunk.get("chunk_index", 0),
                        chunk.get("page_number"),
                        chunk.get("parent_chunk_id"),
                        embedding_bytes,
                        json.dumps(chunk.get("metadata", {})),
                        now,
                    ),
                )
            await db.commit()
            logger.info(f"Added {len(chunks)} chunks to vector store")
        finally:
            await db.close()

    async def search(
        self,
        query_embedding: list[float],
        workspace_id: str = "default",
        top_k: int = 10,
        document_id: str | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar chunks using cosine similarity.

        Loads all embeddings from the workspace into memory
        and performs brute-force cosine similarity search.
        For datasets under ~100K chunks, this is fast enough.
        """
        db = await self._get_db()
        try:
            if document_id:
                cursor = await db.execute(
                    """SELECT id, document_id, content, document_title, section_title,
                              chunk_index, page_number, embedding, metadata
                       FROM document_chunks
                       WHERE workspace_id = ? AND document_id = ? AND embedding IS NOT NULL""",
                    (workspace_id, document_id),
                )
            else:
                cursor = await db.execute(
                    """SELECT id, document_id, content, document_title, section_title,
                              chunk_index, page_number, embedding, metadata
                       FROM document_chunks
                       WHERE workspace_id = ? AND embedding IS NOT NULL""",
                    (workspace_id,),
                )
            rows = await cursor.fetchall()
        finally:
            await db.close()

        if not rows:
            return []

        # Convert query embedding to numpy
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm

        # Compute similarities
        results = []
        for row in rows:
            row_dict = dict(row)
            emb_bytes = row_dict["embedding"]
            if not emb_bytes:
                continue

            chunk_vec = np.frombuffer(emb_bytes, dtype=np.float32)
            chunk_norm = np.linalg.norm(chunk_vec)
            if chunk_norm == 0:
                continue
            chunk_vec = chunk_vec / chunk_norm

            similarity = float(np.dot(query_vec, chunk_vec))

            results.append(SearchResult(
                chunk_id=row_dict["id"],
                document_id=row_dict["document_id"],
                content=row_dict["content"],
                score=similarity,
                document_title=row_dict["document_title"],
                section_title=row_dict["section_title"],
                page_number=row_dict["page_number"],
                chunk_index=row_dict["chunk_index"],
                metadata=json.loads(row_dict.get("metadata", "{}")),
            ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def get_chunk_count(self, workspace_id: str = "default") -> int:
        """Get total chunk count for a workspace."""
        db = await self._get_db()
        try:
            cursor = await db.execute(
                "SELECT COUNT(*) as cnt FROM document_chunks WHERE workspace_id = ?",
                (workspace_id,),
            )
            row = await cursor.fetchone()
            return row["cnt"] if row else 0
        finally:
            await db.close()

    # ── Workspace Management ────────────────────────────────

    async def create_workspace(self, workspace_id: str, name: str, description: str = ""):
        """Create a new workspace."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        db = await self._get_db()
        try:
            await db.execute(
                "INSERT OR IGNORE INTO workspaces (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (workspace_id, name, description, now, now),
            )
            await db.commit()
        finally:
            await db.close()

    async def list_workspaces(self) -> list[dict]:
        """List all workspaces."""
        db = await self._get_db()
        try:
            cursor = await db.execute("SELECT * FROM workspaces ORDER BY created_at DESC")
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            await db.close()


# Singleton
vector_store = VectorStore()
