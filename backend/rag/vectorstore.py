"""FAISS-backed vector store for SoloLLM.

This module implements a real vector database layer by combining:
1) FAISS for persistent ANN similarity search
2) SQLite for document/chunk metadata and vector-id mapping
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import numpy as np

from core.config import settings

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency check
    faiss = None

logger = logging.getLogger(__name__)

VECTORS_DB_PATH = str(settings.data_dir / "db" / "vectors.db")
VECTOR_INDEX_DIR = settings.data_dir / "db" / "vector_indexes"

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

CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    vector_id INTEGER NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunk_vectors_workspace ON chunk_vectors(workspace_id);
CREATE INDEX IF NOT EXISTS idx_chunk_vectors_vector_id ON chunk_vectors(vector_id);

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
    score_normalized: float = 0.0  # Normalized to 0-1 range within the result batch


class VectorStore:
    """FAISS-backed vector store with SQLite metadata."""

    def __init__(self, db_path: str = VECTORS_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.index_dir = VECTOR_INDEX_DIR
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._workspace_indexes: dict[str, "faiss.IndexIDMap2"] = {}
        self._index_lock = asyncio.Lock()

    def _ensure_faiss_available(self):
        if faiss is None:
            raise RuntimeError(
                "FAISS is not installed. Install dependency 'faiss-cpu' to enable vector search."
            )

    def _workspace_index_path(self, workspace_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in workspace_id)
        return self.index_dir / f"{safe}.faiss"

    def _new_index(self, dimension: int) -> "faiss.IndexIDMap2":
        self._ensure_faiss_available()
        base = faiss.IndexFlatIP(dimension)
        return faiss.IndexIDMap2(base)

    async def _get_next_vector_id(self, db: aiosqlite.Connection, workspace_id: str) -> int:
        cursor = await db.execute(
            "SELECT COALESCE(MAX(vector_id), 0) AS max_id FROM chunk_vectors WHERE workspace_id = ?",
            (workspace_id,),
        )
        row = await cursor.fetchone()
        return int(row["max_id"] if row and row["max_id"] is not None else 0) + 1

    async def _get_workspace_dimension(self, db: aiosqlite.Connection, workspace_id: str) -> int | None:
        cursor = await db.execute(
            "SELECT dimension FROM chunk_vectors WHERE workspace_id = ? ORDER BY vector_id DESC LIMIT 1",
            (workspace_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return int(row["dimension"])

    def _load_workspace_index(self, workspace_id: str, dimension: int) -> "faiss.IndexIDMap2":
        if workspace_id in self._workspace_indexes:
            return self._workspace_indexes[workspace_id]

        index_path = self._workspace_index_path(workspace_id)
        if index_path.exists():
            index = faiss.read_index(str(index_path))
            if not isinstance(index, faiss.IndexIDMap2):
                index = faiss.IndexIDMap2(index)
            if index.d != dimension:
                raise ValueError(
                    f"Workspace '{workspace_id}' index dimension mismatch: "
                    f"index={index.d}, embedding={dimension}"
                )
        else:
            index = self._new_index(dimension)

        self._workspace_indexes[workspace_id] = index
        return index

    def _save_workspace_index(self, workspace_id: str):
        index = self._workspace_indexes.get(workspace_id)
        if index is None:
            return
        faiss.write_index(index, str(self._workspace_index_path(workspace_id)))

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        return vectors / norms

    async def _get_db(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        return db

    async def init(self):
        """Initialize the vector store tables."""
        self._ensure_faiss_available()
        db = await self._get_db()
        try:
            await db.executescript(VECTOR_SCHEMA)
            await self._migrate_legacy_embeddings(db)
            await db.commit()
            logger.info("Vector store initialized")
        finally:
            await db.close()

    async def _migrate_legacy_embeddings(self, db: aiosqlite.Connection):
        """Migrate legacy SQLite BLOB embeddings into FAISS indexes once."""
        cursor = await db.execute(
            """SELECT dc.id AS chunk_id, dc.workspace_id, dc.embedding
               FROM document_chunks dc
               LEFT JOIN chunk_vectors cv ON cv.chunk_id = dc.id
               WHERE dc.embedding IS NOT NULL AND cv.chunk_id IS NULL"""
        )
        rows = await cursor.fetchall()
        if not rows:
            return

        by_workspace: dict[str, list[dict]] = {}
        for row in rows:
            r = dict(row)
            by_workspace.setdefault(r["workspace_id"], []).append(r)

        migrated_total = 0
        async with self._index_lock:
            for workspace_id, workspace_rows in by_workspace.items():
                vectors: list[np.ndarray] = []
                chunk_ids: list[str] = []
                for row in workspace_rows:
                    emb_bytes = row.get("embedding")
                    if not emb_bytes:
                        continue
                    vec = np.frombuffer(emb_bytes, dtype=np.float32)
                    if vec.size == 0:
                        continue
                    vectors.append(vec)
                    chunk_ids.append(row["chunk_id"])

                if not vectors:
                    continue

                first_dim = int(vectors[0].shape[0])
                filtered_pairs = [
                    (cid, vec)
                    for cid, vec in zip(chunk_ids, vectors)
                    if int(vec.shape[0]) == first_dim
                ]
                if not filtered_pairs:
                    continue

                chunk_ids = [p[0] for p in filtered_pairs]
                vectors_np = np.stack([p[1] for p in filtered_pairs]).astype(np.float32)
                vectors_np = self._normalize_vectors(vectors_np)

                next_vector_id = await self._get_next_vector_id(db, workspace_id)
                vector_ids = np.arange(next_vector_id, next_vector_id + len(chunk_ids), dtype=np.int64)

                index = self._load_workspace_index(workspace_id, first_dim)
                index.add_with_ids(vectors_np, vector_ids)
                self._save_workspace_index(workspace_id)

                now = datetime.now(timezone.utc).isoformat()
                for i, chunk_id in enumerate(chunk_ids):
                    await db.execute(
                        """INSERT OR REPLACE INTO chunk_vectors
                           (chunk_id, workspace_id, vector_id, dimension, created_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (chunk_id, workspace_id, int(vector_ids[i]), first_dim, now),
                    )
                    # Clear legacy blob to reduce DB size after migration.
                    await db.execute(
                        "UPDATE document_chunks SET embedding = NULL WHERE id = ?",
                        (chunk_id,),
                    )

                migrated_total += len(chunk_ids)

        if migrated_total:
            logger.info("Migrated %d legacy chunk embeddings into FAISS indexes", migrated_total)

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
        """Delete a document and all its vectors/chunks."""
        db = await self._get_db()
        try:
            cursor = await db.execute("SELECT workspace_id FROM documents WHERE id = ?", (document_id,))
            doc_row = await cursor.fetchone()
            if not doc_row:
                return

            workspace_id = doc_row["workspace_id"]

            cursor = await db.execute(
                """SELECT cv.vector_id
                   FROM chunk_vectors cv
                   JOIN document_chunks dc ON dc.id = cv.chunk_id
                   WHERE dc.document_id = ?""",
                (document_id,),
            )
            rows = await cursor.fetchall()
            vector_ids = [int(r["vector_id"]) for r in rows]

            async with self._index_lock:
                if vector_ids:
                    # Resolve dimension from existing index when possible.
                    index = self._workspace_indexes.get(workspace_id)
                    if index is None and self._workspace_index_path(workspace_id).exists():
                        dimension = await self._get_workspace_dimension(db, workspace_id)
                        if dimension is not None:
                            index = self._load_workspace_index(workspace_id, dimension)

                    if index is not None:
                        remove_ids = np.array(vector_ids, dtype=np.int64)
                        index.remove_ids(remove_ids)
                        self._save_workspace_index(workspace_id)

            await db.execute(
                "DELETE FROM chunk_vectors WHERE chunk_id IN (SELECT id FROM document_chunks WHERE document_id = ?)",
                (document_id,),
            )
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
        self._ensure_faiss_available()

        now = datetime.now(timezone.utc).isoformat()
        if not chunks:
            return

        embeddings = []
        valid_chunks = []
        for chunk in chunks:
            emb = chunk.get("embedding")
            if not emb:
                continue
            embeddings.append(emb)
            valid_chunks.append(chunk)

        if not valid_chunks:
            logger.warning("No chunks with embeddings to add")
            return

        vectors = np.asarray(embeddings, dtype=np.float32)
        vectors = self._normalize_vectors(vectors)
        dimension = int(vectors.shape[1])

        db = await self._get_db()
        try:
            next_vector_id = await self._get_next_vector_id(db, workspace_id)
            vector_ids = np.arange(next_vector_id, next_vector_id + len(valid_chunks), dtype=np.int64)

            for i, chunk in enumerate(valid_chunks):
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
                        None,
                        json.dumps(chunk.get("metadata", {})),
                        now,
                    ),
                )

                await db.execute(
                    """INSERT OR REPLACE INTO chunk_vectors
                       (chunk_id, workspace_id, vector_id, dimension, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (chunk["id"], workspace_id, int(vector_ids[i]), dimension, now),
                )

            async with self._index_lock:
                index = self._load_workspace_index(workspace_id, dimension)
                index.add_with_ids(vectors, vector_ids)
                self._save_workspace_index(workspace_id)

            await db.commit()
            logger.info(
                "Added %d chunks to FAISS index (%s), dimension=%d",
                len(valid_chunks),
                workspace_id,
                dimension,
            )
        finally:
            await db.close()

    async def search(
        self,
        query_embedding: list[float],
        workspace_id: str = "default",
        top_k: int = 10,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        score_threshold: float | None = None,
        fetch_k: int | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks using FAISS cosine similarity (IP on normalized vectors)."""
        self._ensure_faiss_available()

        query_vec = np.asarray([query_embedding], dtype=np.float32)
        if query_vec.size == 0:
            return []
        query_vec = self._normalize_vectors(query_vec)
        query_dim = int(query_vec.shape[1])

        async with self._index_lock:
            index = self._load_workspace_index(workspace_id, query_dim)
            if index.ntotal == 0:
                return []

            # Over-fetch when metadata filters are applied or explicit fetch_k given.
            filtered = bool(document_id or document_ids)
            effective_fetch_k = fetch_k if fetch_k is not None else (
                top_k if not filtered else max(top_k * 10, 100)
            )
            effective_fetch_k = min(effective_fetch_k, int(index.ntotal))

            scores, ids = index.search(query_vec, effective_fetch_k)
            candidate_pairs = [
                (int(vector_id), float(score))
                for vector_id, score in zip(ids[0].tolist(), scores[0].tolist())
                if int(vector_id) >= 0
            ]

        if not candidate_pairs:
            return []

        vector_ids = [p[0] for p in candidate_pairs]
        score_by_vector_id = {p[0]: p[1] for p in candidate_pairs}

        db = await self._get_db()
        try:
            placeholders = ",".join("?" for _ in vector_ids)
            sql = (
                """SELECT cv.vector_id, dc.id AS chunk_id, dc.document_id, dc.content,
                          dc.document_title, dc.section_title, dc.chunk_index,
                          dc.page_number, dc.metadata
                   FROM chunk_vectors cv
                   JOIN document_chunks dc ON dc.id = cv.chunk_id
                   WHERE cv.workspace_id = ?
                   AND cv.vector_id IN ("""
                + placeholders
                + ")"
            )
            params: list = [workspace_id, *vector_ids]

            if document_ids:
                doc_placeholders = ",".join("?" for _ in document_ids)
                sql += f" AND dc.document_id IN ({doc_placeholders})"
                params.extend(document_ids)
            elif document_id:
                sql += " AND dc.document_id = ?"
                params.append(document_id)

            cursor = await db.execute(sql, tuple(params))
            rows = await cursor.fetchall()
        finally:
            await db.close()

        row_by_vector_id = {int(r["vector_id"]): dict(r) for r in rows}
        results: list[SearchResult] = []

        # Determine max score for normalization
        all_scores = [score_by_vector_id.get(vid, 0.0) for vid in vector_ids if row_by_vector_id.get(vid) is not None]
        max_score = max(all_scores) if all_scores else 1.0
        if max_score <= 0:
            max_score = 1.0

        for vector_id in vector_ids:
            row = row_by_vector_id.get(vector_id)
            if row is None:
                continue

            raw_score = score_by_vector_id.get(vector_id, 0.0)
            normalized = max(0.0, raw_score) / max_score if max_score > 0 else 0.0

            # Apply score threshold gate (precision mode)
            if score_threshold is not None and normalized < score_threshold:
                continue

            raw_metadata = row.get("metadata", "{}")
            try:
                metadata = json.loads(raw_metadata) if raw_metadata else {}
            except Exception:
                metadata = {}

            results.append(
                SearchResult(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    score=raw_score,
                    document_title=row.get("document_title", ""),
                    section_title=row.get("section_title", ""),
                    page_number=row.get("page_number"),
                    chunk_index=row.get("chunk_index", 0),
                    metadata=metadata,
                    score_normalized=round(normalized, 4),
                )
            )
            if len(results) >= top_k:
                break

        return results

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

    async def get_backend_info(self) -> dict:
        """Return vector backend details for diagnostics/UI."""
        return {
            "backend": "faiss",
            "index_dir": str(self.index_dir),
            "available": faiss is not None,
        }

    # ── Workspace Management ────────────────────────────────

    async def create_workspace(self, workspace_id: str, name: str, description: str = ""):
        """Create a new workspace."""
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
