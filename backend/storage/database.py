import aiosqlite
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.config import settings


DB_PATH = str(settings.db_path)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model TEXT NOT NULL,
    system_prompt TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    is_continuation INTEGER DEFAULT 0,
    continuation_of TEXT,
    thread_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_profile (
    id INTEGER PRIMARY KEY DEFAULT 1,
    gpu_name TEXT,
    vram_mb INTEGER,
    ram_mb INTEGER,
    cpu_name TEXT,
    cpu_cores INTEGER,
    os_info TEXT,
    profiled_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    message_count INTEGER DEFAULT 0,
    compressed_from INTEGER DEFAULT 0,
    compressed_to INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS distillation_metrics (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    query TEXT NOT NULL,
    compression_ratio REAL DEFAULT 0,
    confidence_score REAL DEFAULT 0,
    confidence_level TEXT DEFAULT '',
    retrieval_quality REAL DEFAULT 0,
    coverage REAL DEFAULT 0,
    source_diversity REAL DEFAULT 0,
    query_type TEXT DEFAULT '',
    sub_queries TEXT DEFAULT '',
    hops_used INTEGER DEFAULT 1,
    verified INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    answer TEXT DEFAULT '',
    model TEXT DEFAULT '',
    total_steps INTEGER DEFAULT 0,
    tools_used TEXT DEFAULT '[]',
    steps_json TEXT DEFAULT '[]',
    success INTEGER DEFAULT 1,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT 'Main',
    system_prompt TEXT DEFAULT '',
    context_mode TEXT DEFAULT 'isolated',
    is_default INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS thread_settings (
    thread_id TEXT PRIMARY KEY,
    max_tokens INTEGER,
    temperature REAL,
    rag_enabled INTEGER DEFAULT 1,
    rag_top_k INTEGER DEFAULT 5,
    compression_enabled INTEGER DEFAULT 0,
    memory_layers INTEGER DEFAULT 1,
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS thread_documents (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    attached_at TEXT NOT NULL,
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE,
    UNIQUE(thread_id, document_id)
);

CREATE TABLE IF NOT EXISTS context_pages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);
CREATE INDEX IF NOT EXISTS idx_conversation_summaries ON conversation_summaries(conversation_id);
CREATE INDEX IF NOT EXISTS idx_distillation_metrics ON distillation_metrics(conversation_id);
CREATE INDEX IF NOT EXISTS idx_agent_memories_category ON agent_memories(category);
CREATE INDEX IF NOT EXISTS idx_agent_runs_created ON agent_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_threads_conversation ON threads(conversation_id);
CREATE INDEX IF NOT EXISTS idx_thread_documents_thread ON thread_documents(thread_id);
CREATE INDEX IF NOT EXISTS idx_context_pages_thread ON context_pages(thread_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    """Create tables if they don't exist."""
    db = await get_db()
    try:
        await db.executescript(SCHEMA_SQL)
        await db.commit()
        # Migration: add thread_id column to messages if missing
        cursor = await db.execute("PRAGMA table_info(messages)")
        columns = [row[1] for row in await cursor.fetchall()]
        if "thread_id" not in columns:
            await db.execute("ALTER TABLE messages ADD COLUMN thread_id TEXT")
            await db.commit()
    finally:
        await db.close()


# ── Conversations ───────────────────────────────────────────

async def create_conversation(title: str, model: str, system_prompt: str = "") -> dict:
    db = await get_db()
    try:
        cid = new_id()
        now = _now()
        await db.execute(
            "INSERT INTO conversations (id, title, model, system_prompt, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (cid, title, model, system_prompt, now, now),
        )
        # Auto-create first thread — no "Main" thread concept, all threads are equal
        tid = new_id()
        await db.execute(
            """INSERT INTO threads (id, conversation_id, title, system_prompt, context_mode, is_default, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (tid, cid, "Thread 1", "", "isolated", 0, now, now),
        )
        await db.execute(
            "INSERT INTO thread_settings (thread_id) VALUES (?)",
            (tid,),
        )
        await db.commit()
        return {
            "id": cid, "title": title, "model": model, "system_prompt": system_prompt,
            "created_at": now, "updated_at": now, "default_thread_id": tid,
        }
    finally:
        await db.close()


async def list_conversations(limit: int = 50, offset: int = 0) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_conversation(conversation_id: str) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def update_conversation(conversation_id: str, **kwargs) -> bool:
    db = await get_db()
    try:
        allowed = {"title", "model", "system_prompt"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        fields["updated_at"] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [conversation_id]
        await db.execute(f"UPDATE conversations SET {set_clause} WHERE id = ?", values)
        await db.commit()
        return True
    finally:
        await db.close()


async def delete_conversation(conversation_id: str) -> bool:
    db = await get_db()
    try:
        cursor = await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


# ── Messages ────────────────────────────────────────────────

async def add_message(
    conversation_id: str,
    role: str,
    content: str,
    token_count: int = 0,
    is_continuation: bool = False,
    continuation_of: str | None = None,
    thread_id: str | None = None,
) -> dict:
    db = await get_db()
    try:
        mid = new_id()
        now = _now()
        # Ensure thread_id column exists (added for thread support)
        await db.execute(
            """INSERT INTO messages (id, conversation_id, role, content, token_count, is_continuation, continuation_of, thread_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (mid, conversation_id, role, content, token_count, int(is_continuation), continuation_of, thread_id, now),
        )
        # Update conversation timestamp
        await db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        await db.commit()
        return {
            "id": mid, "conversation_id": conversation_id, "role": role,
            "content": content, "token_count": token_count,
            "is_continuation": is_continuation, "continuation_of": continuation_of,
            "thread_id": thread_id, "created_at": now,
        }
    finally:
        await db.close()


async def get_messages(conversation_id: str, thread_id: str | None = None) -> list[dict]:
    db = await get_db()
    try:
        if thread_id:
            cursor = await db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? AND thread_id = ? ORDER BY created_at ASC",
                (conversation_id, thread_id),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def update_message(message_id: str, content: str, token_count: int | None = None) -> bool:
    db = await get_db()
    try:
        if token_count is not None:
            await db.execute(
                "UPDATE messages SET content = ?, token_count = ? WHERE id = ?",
                (content, token_count, message_id),
            )
        else:
            await db.execute(
                "UPDATE messages SET content = ? WHERE id = ?",
                (content, message_id),
            )
        await db.commit()
        return True
    finally:
        await db.close()


async def get_message(message_id: str) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM messages WHERE id = ?", (message_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


# ── Settings ────────────────────────────────────────────────

async def get_setting(key: str) -> str | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = await cursor.fetchone()
        return row["value"] if row else None
    finally:
        await db.close()


async def set_setting(key: str, value: str):
    db = await get_db()
    try:
        now = _now()
        await db.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?",
            (key, value, now, value, now),
        )
        await db.commit()
    finally:
        await db.close()


async def get_all_settings() -> dict[str, str]:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT key, value FROM settings")
        rows = await cursor.fetchall()
        return {r["key"]: r["value"] for r in rows}
    finally:
        await db.close()


# ── System Profile ──────────────────────────────────────────

async def save_system_profile(profile: dict):
    db = await get_db()
    try:
        now = _now()
        await db.execute(
            """INSERT INTO system_profile (id, gpu_name, vram_mb, ram_mb, cpu_name, cpu_cores, os_info, profiled_at)
               VALUES (1, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
               gpu_name=?, vram_mb=?, ram_mb=?, cpu_name=?, cpu_cores=?, os_info=?, profiled_at=?""",
            (
                profile.get("gpu_name"), profile.get("vram_mb"), profile.get("ram_mb"),
                profile.get("cpu_name"), profile.get("cpu_cores"), profile.get("os_info"), now,
                profile.get("gpu_name"), profile.get("vram_mb"), profile.get("ram_mb"),
                profile.get("cpu_name"), profile.get("cpu_cores"), profile.get("os_info"), now,
            ),
        )
        await db.commit()
    finally:
        await db.close()


async def get_system_profile() -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM system_profile WHERE id = 1")
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


# ── Conversation Summaries ──────────────────────────────────

async def save_conversation_summary(
    conversation_id: str,
    summary: str,
    message_count: int = 0,
    compressed_from: int = 0,
    compressed_to: int = 0,
) -> dict:
    db = await get_db()
    try:
        sid = new_id()
        now = _now()
        await db.execute(
            "INSERT INTO conversation_summaries (id, conversation_id, summary, message_count, compressed_from, compressed_to, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sid, conversation_id, summary, message_count, compressed_from, compressed_to, now),
        )
        await db.commit()
        return {"id": sid, "conversation_id": conversation_id, "summary": summary, "created_at": now}
    finally:
        await db.close()


async def get_conversation_summary(conversation_id: str) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM conversation_summaries WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 1",
            (conversation_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def delete_conversation_summaries(conversation_id: str):
    db = await get_db()
    try:
        await db.execute("DELETE FROM conversation_summaries WHERE conversation_id = ?", (conversation_id,))
        await db.commit()
    finally:
        await db.close()


# ── Distillation Metrics ───────────────────────────────────

async def save_distillation_metric(metric: dict) -> dict:
    db = await get_db()
    try:
        mid = new_id()
        now = _now()
        await db.execute(
            """INSERT INTO distillation_metrics
               (id, conversation_id, query, compression_ratio, confidence_score,
                confidence_level, retrieval_quality, coverage, source_diversity,
                query_type, sub_queries, hops_used, verified, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mid, metric.get("conversation_id", ""),
                metric.get("query", ""), metric.get("compression_ratio", 0),
                metric.get("confidence_score", 0), metric.get("confidence_level", ""),
                metric.get("retrieval_quality", 0), metric.get("coverage", 0),
                metric.get("source_diversity", 0), metric.get("query_type", ""),
                metric.get("sub_queries", ""), metric.get("hops_used", 1),
                int(metric.get("verified", False)), now,
            ),
        )
        await db.commit()
        return {"id": mid, **metric, "created_at": now}
    finally:
        await db.close()


async def get_distillation_metrics(conversation_id: str | None = None, limit: int = 50) -> list[dict]:
    db = await get_db()
    try:
        if conversation_id:
            cursor = await db.execute(
                "SELECT * FROM distillation_metrics WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM distillation_metrics ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


# ── Agent Memory ────────────────────────────────────────────

async def save_agent_memory(content: str, category: str = "general") -> dict:
    db = await get_db()
    try:
        mid = new_id()
        now = _now()
        await db.execute(
            "INSERT INTO agent_memories (id, content, category, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (mid, content, category, now, now),
        )
        await db.commit()
        return {"id": mid, "content": content, "category": category, "created_at": now, "updated_at": now}
    finally:
        await db.close()


async def search_agent_memories(query: str, limit: int = 10) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM agent_memories WHERE content LIKE ? ORDER BY updated_at DESC LIMIT ?",
            (f"%{query}%", limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def list_agent_memories(limit: int = 50, category: str | None = None) -> list[dict]:
    db = await get_db()
    try:
        if category:
            cursor = await db.execute(
                "SELECT * FROM agent_memories WHERE category = ? ORDER BY updated_at DESC LIMIT ?",
                (category, limit),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM agent_memories ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def delete_agent_memory(memory_id: str) -> bool:
    db = await get_db()
    try:
        cursor = await db.execute("DELETE FROM agent_memories WHERE id = ?", (memory_id,))
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def clear_agent_memories() -> int:
    db = await get_db()
    try:
        cursor = await db.execute("DELETE FROM agent_memories")
        await db.commit()
        return cursor.rowcount
    finally:
        await db.close()


# ── Agent Runs ──────────────────────────────────────────────

async def save_agent_run(
    query: str, answer: str, model: str,
    total_steps: int, tools_used: list[str],
    steps_json: str, success: bool, error: str | None = None,
) -> dict:
    db = await get_db()
    try:
        rid = new_id()
        now = _now()
        await db.execute(
            """INSERT INTO agent_runs
               (id, query, answer, model, total_steps, tools_used, steps_json, success, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (rid, query, answer, model, total_steps, json.dumps(tools_used), steps_json, int(success), error, now),
        )
        await db.commit()
        return {"id": rid, "query": query, "answer": answer, "created_at": now}
    finally:
        await db.close()


async def list_agent_runs(limit: int = 20) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM agent_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


# ── Threads ─────────────────────────────────────────────────

async def create_thread(
    conversation_id: str,
    title: str = "Main",
    system_prompt: str = "",
    context_mode: str = "isolated",
    is_default: bool = False,
) -> dict:
    db = await get_db()
    try:
        tid = new_id()
        now = _now()
        await db.execute(
            """INSERT INTO threads (id, conversation_id, title, system_prompt, context_mode, is_default, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (tid, conversation_id, title, system_prompt, context_mode, int(is_default), now, now),
        )
        # Create default thread_settings row
        await db.execute(
            "INSERT INTO thread_settings (thread_id) VALUES (?)",
            (tid,),
        )
        await db.commit()
        return {
            "id": tid, "conversation_id": conversation_id, "title": title,
            "system_prompt": system_prompt, "context_mode": context_mode,
            "is_default": is_default, "created_at": now, "updated_at": now,
        }
    finally:
        await db.close()


async def list_threads(conversation_id: str) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM threads WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_thread(thread_id: str) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM threads WHERE id = ?", (thread_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def update_thread(thread_id: str, **kwargs) -> bool:
    db = await get_db()
    try:
        allowed = {"title", "system_prompt", "context_mode"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        fields["updated_at"] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        await db.execute(
            f"UPDATE threads SET {set_clause} WHERE id = ?",
            (*fields.values(), thread_id),
        )
        await db.commit()
        return True
    finally:
        await db.close()


async def delete_thread(thread_id: str) -> bool:
    """Delete a thread. Any thread can be deleted."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT id FROM threads WHERE id = ?", (thread_id,))
        row = await cursor.fetchone()
        if not row:
            return False
        await db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        await db.commit()
        return True
    finally:
        await db.close()


async def get_default_thread(conversation_id: str) -> dict | None:
    """Get the first thread for a conversation (no 'Main' thread concept)."""
    db = await get_db()
    try:
        # First try is_default=1 for backward compatibility, then fall back to first thread
        cursor = await db.execute(
            "SELECT * FROM threads WHERE conversation_id = ? AND is_default = 1 LIMIT 1",
            (conversation_id,),
        )
        row = await cursor.fetchone()
        if not row:
            cursor = await db.execute(
                "SELECT * FROM threads WHERE conversation_id = ? ORDER BY created_at ASC LIMIT 1",
                (conversation_id,),
            )
            row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


# ── Thread Settings ─────────────────────────────────────────

async def get_thread_settings(thread_id: str) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM thread_settings WHERE thread_id = ?", (thread_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def update_thread_settings(thread_id: str, **kwargs) -> bool:
    db = await get_db()
    try:
        allowed = {"max_tokens", "temperature", "rag_enabled", "rag_top_k", "compression_enabled", "memory_layers"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        await db.execute(
            f"UPDATE thread_settings SET {set_clause} WHERE thread_id = ?",
            (*fields.values(), thread_id),
        )
        await db.commit()
        return True
    finally:
        await db.close()


# ── Thread Documents ────────────────────────────────────────

async def attach_document_to_thread(thread_id: str, document_id: str) -> bool:
    db = await get_db()
    try:
        did = new_id()
        now = _now()
        await db.execute(
            "INSERT OR IGNORE INTO thread_documents (id, thread_id, document_id, attached_at) VALUES (?, ?, ?, ?)",
            (did, thread_id, document_id, now),
        )
        await db.commit()
        return True
    finally:
        await db.close()


async def get_thread_documents(thread_id: str) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM thread_documents WHERE thread_id = ? ORDER BY attached_at ASC",
            (thread_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def detach_document_from_thread(thread_id: str, document_id: str) -> bool:
    db = await get_db()
    try:
        cursor = await db.execute(
            "DELETE FROM thread_documents WHERE thread_id = ? AND document_id = ?",
            (thread_id, document_id),
        )
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


# ── Context Pages ───────────────────────────────────────────

async def get_active_context_pages(thread_id: str) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM context_pages WHERE thread_id = ? AND is_active = 1 ORDER BY page_number ASC",
            (thread_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_all_context_pages(thread_id: str) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM context_pages WHERE thread_id = ? ORDER BY page_number ASC",
            (thread_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()
