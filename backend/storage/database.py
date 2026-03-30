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
    documents_used TEXT DEFAULT '[]',
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

CREATE TABLE IF NOT EXISTS finetuned_models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    base_model TEXT NOT NULL,
    base_model_hf TEXT,
    training_examples INTEGER DEFAULT 0,
    final_loss REAL,
    model_path TEXT NOT NULL,
    is_registered INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    registered_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_finetuned_models_name ON finetuned_models(name);

-- ── Academic Auto-Generation Tables ────────────────────────
CREATE TABLE IF NOT EXISTS academic_courses (
    id TEXT PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    title TEXT DEFAULT '',
    department TEXT DEFAULT '',
    total_lectures INTEGER DEFAULT 0,
    workspace_id TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_acad_courses_code ON academic_courses(code);

CREATE TABLE IF NOT EXISTS academic_course_aliases (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    alias TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    FOREIGN KEY (course_id) REFERENCES academic_courses(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_acad_aliases_course ON academic_course_aliases(course_id);

CREATE TABLE IF NOT EXISTS academic_review_sources (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_type TEXT DEFAULT 'csv',
    review_count INTEGER DEFAULT 0,
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS academic_student_reviews (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    source_id TEXT,
    review_text TEXT NOT NULL,
    reviewer_token TEXT DEFAULT '',
    semester TEXT DEFAULT '',
    urgency_score REAL DEFAULT 0.0,
    sentiment_score REAL DEFAULT 0.0,
    is_duplicate INTEGER DEFAULT 0,
    is_spam INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (course_id) REFERENCES academic_courses(id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES academic_review_sources(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_acad_reviews_course ON academic_student_reviews(course_id);
CREATE INDEX IF NOT EXISTS idx_acad_reviews_source ON academic_student_reviews(source_id);

CREATE TABLE IF NOT EXISTS academic_review_topic_labels (
    id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL,
    course_id TEXT NOT NULL,
    topic_name TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (review_id) REFERENCES academic_student_reviews(id) ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES academic_courses(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_acad_topic_labels_review ON academic_review_topic_labels(review_id);
CREATE INDEX IF NOT EXISTS idx_acad_topic_labels_course ON academic_review_topic_labels(course_id);

CREATE TABLE IF NOT EXISTS academic_topic_scores (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    topic_name TEXT NOT NULL,
    exam_probability REAL DEFAULT 0.0,
    weight_bucket TEXT DEFAULT 'low',
    midterm_relevance REAL DEFAULT 0.0,
    final_relevance REAL DEFAULT 0.0,
    review_frequency REAL DEFAULT 0.0,
    urgency_signal REAL DEFAULT 0.0,
    consensus_score REAL DEFAULT 0.0,
    syllabus_importance REAL DEFAULT 0.0,
    recency_weight REAL DEFAULT 0.0,
    llm_confidence REAL DEFAULT 0.0,
    evidence_json TEXT DEFAULT '[]',
    scored_at TEXT NOT NULL,
    FOREIGN KEY (course_id) REFERENCES academic_courses(id) ON DELETE CASCADE,
    UNIQUE(course_id, topic_name)
);
CREATE INDEX IF NOT EXISTS idx_acad_scores_course ON academic_topic_scores(course_id);

CREATE TABLE IF NOT EXISTS academic_generation_jobs (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    job_type TEXT NOT NULL DEFAULT 'full',
    output_types TEXT DEFAULT '[]',
    status TEXT DEFAULT 'pending',
    progress REAL DEFAULT 0.0,
    current_stage TEXT DEFAULT '',
    stages_completed TEXT DEFAULT '[]',
    checkpoint_json TEXT DEFAULT '{}',
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (course_id) REFERENCES academic_courses(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_acad_jobs_course ON academic_generation_jobs(course_id);
CREATE INDEX IF NOT EXISTS idx_acad_jobs_status ON academic_generation_jobs(status);

CREATE TABLE IF NOT EXISTS academic_generation_outputs (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    output_type TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    file_path TEXT NOT NULL,
    file_size INTEGER DEFAULT 0,
    topic_count INTEGER DEFAULT 0,
    generation_params TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (course_id) REFERENCES academic_courses(id) ON DELETE CASCADE,
    FOREIGN KEY (job_id) REFERENCES academic_generation_jobs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_acad_outputs_course ON academic_generation_outputs(course_id);
CREATE INDEX IF NOT EXISTS idx_acad_outputs_job ON academic_generation_outputs(job_id);

CREATE TABLE IF NOT EXISTS academic_output_metrics (
    id TEXT PRIMARY KEY,
    output_id TEXT NOT NULL,
    topic_hit_rate REAL DEFAULT 0.0,
    mcq_quality_rate REAL DEFAULT 0.0,
    hallucination_rate REAL DEFAULT 0.0,
    generation_stability REAL DEFAULT 0.0,
    computed_at TEXT NOT NULL,
    FOREIGN KEY (output_id) REFERENCES academic_generation_outputs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_acad_metrics_output ON academic_output_metrics(output_id);

CREATE TABLE IF NOT EXISTS academic_output_feedback (
    id TEXT PRIMARY KEY,
    output_id TEXT NOT NULL,
    rating INTEGER DEFAULT 0,
    comment TEXT DEFAULT '',
    topic_accuracy_pct REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (output_id) REFERENCES academic_generation_outputs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_acad_feedback_output ON academic_output_feedback(output_id);
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
        if "documents_used" not in columns:
            await db.execute("ALTER TABLE messages ADD COLUMN documents_used TEXT DEFAULT '[]'")
            await db.commit()
        # Migration: add context management columns to thread_settings if missing
        cursor = await db.execute("PRAGMA table_info(thread_settings)")
        ts_columns = [row[1] for row in await cursor.fetchall()]
        if "max_history_messages" not in ts_columns:
            await db.execute("ALTER TABLE thread_settings ADD COLUMN max_history_messages INTEGER DEFAULT 20")
            await db.commit()
        if "compression_ratio" not in ts_columns:
            await db.execute("ALTER TABLE thread_settings ADD COLUMN compression_ratio REAL DEFAULT 0.6")
            await db.commit()
        # Migration: add precision mode columns to thread_settings if missing
        if "rag_precision_mode" not in ts_columns:
            await db.execute("ALTER TABLE thread_settings ADD COLUMN rag_precision_mode TEXT DEFAULT 'legacy_rrf'")
            await db.commit()
        if "rag_vector_min_score" not in ts_columns:
            await db.execute("ALTER TABLE thread_settings ADD COLUMN rag_vector_min_score REAL DEFAULT 0.28")
            await db.commit()
        if "rag_lexical_required_coverage" not in ts_columns:
            await db.execute("ALTER TABLE thread_settings ADD COLUMN rag_lexical_required_coverage REAL DEFAULT 0.5")
            await db.commit()
        if "rag_candidate_pool_size" not in ts_columns:
            await db.execute("ALTER TABLE thread_settings ADD COLUMN rag_candidate_pool_size INTEGER DEFAULT 80")
            await db.commit()
        if "rag_per_document_cap" not in ts_columns:
            await db.execute("ALTER TABLE thread_settings ADD COLUMN rag_per_document_cap INTEGER DEFAULT 2")
            await db.commit()
        if "rag_use_mmr" not in ts_columns:
            await db.execute("ALTER TABLE thread_settings ADD COLUMN rag_use_mmr INTEGER DEFAULT 1")
            await db.commit()
        if "rag_mmr_lambda" not in ts_columns:
            await db.execute("ALTER TABLE thread_settings ADD COLUMN rag_mmr_lambda REAL DEFAULT 0.65")
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
            (tid, cid, "Thread 1", "", "isolated", 1, now, now),
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
    documents_used: list[str] | None = None,
) -> dict:
    db = await get_db()
    try:
        mid = new_id()
        now = _now()
        # Ensure thread_id column exists (added for thread support)
        await db.execute(
            """INSERT INTO messages (id, conversation_id, role, content, token_count, is_continuation, continuation_of, thread_id, documents_used, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mid,
                conversation_id,
                role,
                content,
                token_count,
                int(is_continuation),
                continuation_of,
                thread_id,
                json.dumps(documents_used or []),
                now,
            ),
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
            "thread_id": thread_id, "documents_used": documents_used or [], "created_at": now,
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
            # FIXED: When no thread_id, resolve to default thread to prevent cross-thread leakage
            default_thread = await get_default_thread(conversation_id)
            if default_thread:
                cursor = await db.execute(
                    "SELECT * FROM messages WHERE conversation_id = ? AND thread_id = ? ORDER BY created_at ASC",
                    (conversation_id, default_thread["id"]),
                )
            else:
                # No threads exist yet — return all messages (legacy conversations)
                cursor = await db.execute(
                    "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                    (conversation_id,),
                )
        rows = await cursor.fetchall()
        parsed: list[dict] = []
        for row in rows:
            item = dict(row)
            raw_docs = item.get("documents_used")
            if isinstance(raw_docs, str):
                try:
                    item["documents_used"] = json.loads(raw_docs)
                except json.JSONDecodeError:
                    item["documents_used"] = []
            elif raw_docs is None:
                item["documents_used"] = []
            parsed.append(item)
        return parsed
    finally:
        await db.close()


async def update_message(
    message_id: str,
    content: str,
    token_count: int | None = None,
    documents_used: list[str] | None = None,
) -> bool:
    db = await get_db()
    try:
        set_parts = ["content = ?"]
        values: list[object] = [content]
        if token_count is not None:
            set_parts.append("token_count = ?")
            values.append(token_count)
        if documents_used is not None:
            set_parts.append("documents_used = ?")
            values.append(json.dumps(documents_used))

        values.append(message_id)
        await db.execute(
            f"UPDATE messages SET {', '.join(set_parts)} WHERE id = ?",
            values,
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
        if not row:
            return None
        item = dict(row)
        raw_docs = item.get("documents_used")
        if isinstance(raw_docs, str):
            try:
                item["documents_used"] = json.loads(raw_docs)
            except json.JSONDecodeError:
                item["documents_used"] = []
        elif raw_docs is None:
            item["documents_used"] = []
        return item
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
        allowed = {"max_tokens", "temperature", "rag_enabled", "rag_top_k", "compression_enabled", "memory_layers", "max_history_messages", "compression_ratio", "rag_precision_mode", "rag_vector_min_score", "rag_lexical_required_coverage", "rag_candidate_pool_size", "rag_per_document_cap", "rag_use_mmr", "rag_mmr_lambda"}
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


async def save_context_pages(thread_id: str, pages: list) -> None:
    """Persist context pages to the database, replacing existing pages for this thread."""
    db = await get_db()
    try:
        await db.execute("DELETE FROM context_pages WHERE thread_id = ?", (thread_id,))
        now = _now()
        for page in pages:
            pid = new_id()
            content = page.active_content if hasattr(page, 'active_content') else page.get("content", "")
            token_count = page.token_count if hasattr(page, 'token_count') else page.get("token_count", 0)
            page_number = page.page_number if hasattr(page, 'page_number') else page.get("page_number", 0)
            is_active = page.is_active if hasattr(page, 'is_active') else page.get("is_active", True)
            await db.execute(
                """INSERT INTO context_pages (id, thread_id, page_number, content, token_count, is_active, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (pid, thread_id, page_number, content, token_count, int(is_active), now),
            )
        await db.commit()
    finally:
        await db.close()


# ── Fine-tuned Models ────────────────────────────────────────

async def save_finetuned_model(
    name: str,
    display_name: str,
    base_model: str,
    base_model_hf: str,
    model_path: str,
    training_examples: int = 0,
    final_loss: float | None = None,
    is_registered: bool = False,
) -> dict:
    """Save a fine-tuned model to the database."""
    db = await get_db()
    try:
        mid = new_id()
        now = _now()
        registered_at = now if is_registered else None
        await db.execute(
            """INSERT INTO finetuned_models
               (id, name, display_name, base_model, base_model_hf, training_examples, final_loss, model_path, is_registered, created_at, registered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
               display_name=?, base_model=?, base_model_hf=?, training_examples=?, final_loss=?, model_path=?, is_registered=?, registered_at=?""",
            (
                mid, name, display_name, base_model, base_model_hf, training_examples, final_loss,
                model_path, int(is_registered), now, registered_at,
                display_name, base_model, base_model_hf, training_examples, final_loss,
                model_path, int(is_registered), registered_at,
            ),
        )
        await db.commit()
        return {
            "id": mid, "name": name, "display_name": display_name,
            "base_model": base_model, "base_model_hf": base_model_hf,
            "training_examples": training_examples, "final_loss": final_loss,
            "model_path": model_path, "is_registered": is_registered,
            "created_at": now, "registered_at": registered_at,
        }
    finally:
        await db.close()


async def list_finetuned_models() -> list[dict]:
    """List all fine-tuned models."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM finetuned_models ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_finetuned_model(name: str) -> dict | None:
    """Get a fine-tuned model by name."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM finetuned_models WHERE name = ?", (name,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def update_finetuned_model_registration(name: str, is_registered: bool) -> bool:
    """Update the registration status of a fine-tuned model."""
    db = await get_db()
    try:
        now = _now()
        registered_at = now if is_registered else None
        await db.execute(
            "UPDATE finetuned_models SET is_registered = ?, registered_at = ? WHERE name = ?",
            (int(is_registered), registered_at, name),
        )
        await db.commit()
        return True
    finally:
        await db.close()


async def delete_finetuned_model(name: str) -> bool:
    """Delete a fine-tuned model record."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "DELETE FROM finetuned_models WHERE name = ?", (name,)
        )
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


# ── Academic: Courses ───────────────────────────────────────

async def acad_create_course(code: str, title: str = "", department: str = "",
                             total_lectures: int = 0, workspace_id: str = "",
                             metadata: dict | None = None) -> dict:
    db = await get_db()
    try:
        cid = new_id()
        now = _now()
        ws = workspace_id or f"academic_{code.upper().replace(' ', '').replace('-', '')}"
        await db.execute(
            """INSERT INTO academic_courses
               (id, code, title, department, total_lectures, workspace_id, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cid, code.upper().replace(" ", "").replace("-", ""), title, department,
             total_lectures, ws, json.dumps(metadata or {}), now, now),
        )
        await db.commit()
        return {"id": cid, "code": code.upper().replace(" ", "").replace("-", ""),
                "title": title, "department": department, "workspace_id": ws,
                "total_lectures": total_lectures, "created_at": now, "updated_at": now}
    finally:
        await db.close()


async def acad_get_course(course_id: str) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM academic_courses WHERE id = ?", (course_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def acad_get_course_by_code(code: str) -> dict | None:
    normalized = code.upper().replace(" ", "").replace("-", "")
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM academic_courses WHERE code = ?", (normalized,))
        row = await cursor.fetchone()
        if row:
            return dict(row)
        # Try alias lookup
        cursor = await db.execute(
            """SELECT ac.* FROM academic_courses ac
               JOIN academic_course_aliases aca ON aca.course_id = ac.id
               WHERE aca.alias = ?""", (normalized,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def acad_list_courses() -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM academic_courses ORDER BY code ASC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def acad_update_course(course_id: str, **kwargs) -> bool:
    db = await get_db()
    try:
        allowed = {"title", "department", "total_lectures", "metadata"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        if "metadata" in fields and isinstance(fields["metadata"], dict):
            fields["metadata"] = json.dumps(fields["metadata"])
        fields["updated_at"] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        await db.execute(
            f"UPDATE academic_courses SET {set_clause} WHERE id = ?",
            (*fields.values(), course_id),
        )
        await db.commit()
        return True
    finally:
        await db.close()


async def acad_delete_course(course_id: str) -> bool:
    db = await get_db()
    try:
        cursor = await db.execute("DELETE FROM academic_courses WHERE id = ?", (course_id,))
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def acad_add_course_alias(course_id: str, alias: str) -> dict:
    db = await get_db()
    try:
        aid = new_id()
        now = _now()
        normalized = alias.upper().replace(" ", "").replace("-", "")
        await db.execute(
            "INSERT INTO academic_course_aliases (id, course_id, alias, created_at) VALUES (?, ?, ?, ?)",
            (aid, course_id, normalized, now),
        )
        await db.commit()
        return {"id": aid, "course_id": course_id, "alias": normalized}
    finally:
        await db.close()


# ── Academic: Reviews ───────────────────────────────────────

async def acad_create_review_source(filename: str, file_type: str = "csv",
                                     review_count: int = 0) -> dict:
    db = await get_db()
    try:
        sid = new_id()
        now = _now()
        await db.execute(
            "INSERT INTO academic_review_sources (id, filename, file_type, review_count, uploaded_at) VALUES (?, ?, ?, ?, ?)",
            (sid, filename, file_type, review_count, now),
        )
        await db.commit()
        return {"id": sid, "filename": filename, "file_type": file_type,
                "review_count": review_count, "uploaded_at": now}
    finally:
        await db.close()


async def acad_add_review(course_id: str, review_text: str, source_id: str | None = None,
                           reviewer_token: str = "", semester: str = "") -> dict:
    db = await get_db()
    try:
        rid = new_id()
        now = _now()
        await db.execute(
            """INSERT INTO academic_student_reviews
               (id, course_id, source_id, review_text, reviewer_token, semester, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (rid, course_id, source_id, review_text, reviewer_token, semester, now),
        )
        await db.commit()
        return {"id": rid, "course_id": course_id, "review_text": review_text,
                "semester": semester, "created_at": now}
    finally:
        await db.close()


async def acad_list_reviews(course_id: str, include_spam: bool = False,
                             limit: int = 200) -> list[dict]:
    db = await get_db()
    try:
        if include_spam:
            cursor = await db.execute(
                "SELECT * FROM academic_student_reviews WHERE course_id = ? ORDER BY created_at DESC LIMIT ?",
                (course_id, limit),
            )
        else:
            cursor = await db.execute(
                """SELECT * FROM academic_student_reviews
                   WHERE course_id = ? AND is_spam = 0 AND is_duplicate = 0
                   ORDER BY created_at DESC LIMIT ?""",
                (course_id, limit),
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def acad_update_review_flags(review_id: str, **kwargs) -> bool:
    db = await get_db()
    try:
        allowed = {"urgency_score", "sentiment_score", "is_duplicate", "is_spam"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        await db.execute(
            f"UPDATE academic_student_reviews SET {set_clause} WHERE id = ?",
            (*fields.values(), review_id),
        )
        await db.commit()
        return True
    finally:
        await db.close()


async def acad_get_review_count(course_id: str) -> int:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM academic_student_reviews WHERE course_id = ? AND is_spam = 0",
            (course_id,),
        )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0
    finally:
        await db.close()


# ── Academic: Topic Scores ──────────────────────────────────

async def acad_upsert_topic_score(course_id: str, topic_name: str, **kwargs) -> dict:
    db = await get_db()
    try:
        sid = new_id()
        now = _now()
        await db.execute(
            """INSERT INTO academic_topic_scores
               (id, course_id, topic_name, exam_probability, weight_bucket,
                midterm_relevance, final_relevance, review_frequency, urgency_signal,
                consensus_score, syllabus_importance, recency_weight, llm_confidence,
                evidence_json, scored_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(course_id, topic_name) DO UPDATE SET
                exam_probability=?, weight_bucket=?, midterm_relevance=?, final_relevance=?,
                review_frequency=?, urgency_signal=?, consensus_score=?, syllabus_importance=?,
                recency_weight=?, llm_confidence=?, evidence_json=?, scored_at=?""",
            (sid, course_id, topic_name, kwargs.get("exam_probability", 0.0),
             kwargs.get("weight_bucket", "low"), kwargs.get("midterm_relevance", 0.0),
             kwargs.get("final_relevance", 0.0), kwargs.get("review_frequency", 0.0),
             kwargs.get("urgency_signal", 0.0), kwargs.get("consensus_score", 0.0),
             kwargs.get("syllabus_importance", 0.0), kwargs.get("recency_weight", 0.0),
             kwargs.get("llm_confidence", 0.0), json.dumps(kwargs.get("evidence", [])), now,
             # ON CONFLICT update values:
             kwargs.get("exam_probability", 0.0), kwargs.get("weight_bucket", "low"),
             kwargs.get("midterm_relevance", 0.0), kwargs.get("final_relevance", 0.0),
             kwargs.get("review_frequency", 0.0), kwargs.get("urgency_signal", 0.0),
             kwargs.get("consensus_score", 0.0), kwargs.get("syllabus_importance", 0.0),
             kwargs.get("recency_weight", 0.0), kwargs.get("llm_confidence", 0.0),
             json.dumps(kwargs.get("evidence", [])), now),
        )
        await db.commit()
        return {"course_id": course_id, "topic_name": topic_name,
                "exam_probability": kwargs.get("exam_probability", 0.0), "scored_at": now}
    finally:
        await db.close()


async def acad_get_topic_scores(course_id: str) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM academic_topic_scores WHERE course_id = ? ORDER BY exam_probability DESC",
            (course_id,),
        )
        rows = await cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("evidence_json"), str):
                try:
                    d["evidence"] = json.loads(d["evidence_json"])
                except Exception:
                    d["evidence"] = []
            results.append(d)
        return results
    finally:
        await db.close()


# ── Academic: Generation Jobs ───────────────────────────────

async def acad_create_job(course_id: str, job_type: str = "full",
                           output_types: list[str] | None = None) -> dict:
    db = await get_db()
    try:
        jid = new_id()
        now = _now()
        await db.execute(
            """INSERT INTO academic_generation_jobs
               (id, course_id, job_type, output_types, status, created_at)
               VALUES (?, ?, ?, ?, 'pending', ?)""",
            (jid, course_id, job_type, json.dumps(output_types or []), now),
        )
        await db.commit()
        return {"id": jid, "course_id": course_id, "job_type": job_type,
                "output_types": output_types or [], "status": "pending", "created_at": now}
    finally:
        await db.close()


async def acad_update_job(job_id: str, **kwargs) -> bool:
    db = await get_db()
    try:
        allowed = {"status", "progress", "current_stage", "stages_completed",
                    "checkpoint_json", "error", "started_at", "completed_at"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        if "stages_completed" in fields and isinstance(fields["stages_completed"], list):
            fields["stages_completed"] = json.dumps(fields["stages_completed"])
        if "checkpoint_json" in fields and isinstance(fields["checkpoint_json"], dict):
            fields["checkpoint_json"] = json.dumps(fields["checkpoint_json"])
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        await db.execute(
            f"UPDATE academic_generation_jobs SET {set_clause} WHERE id = ?",
            (*fields.values(), job_id),
        )
        await db.commit()
        return True
    finally:
        await db.close()


async def acad_get_job(job_id: str) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM academic_generation_jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        for json_field in ("output_types", "stages_completed"):
            if isinstance(d.get(json_field), str):
                try:
                    d[json_field] = json.loads(d[json_field])
                except Exception:
                    d[json_field] = []
        if isinstance(d.get("checkpoint_json"), str):
            try:
                d["checkpoint_json"] = json.loads(d["checkpoint_json"])
            except Exception:
                d["checkpoint_json"] = {}
        return d
    finally:
        await db.close()


async def acad_list_jobs(course_id: str | None = None, status: str | None = None,
                          limit: int = 50) -> list[dict]:
    db = await get_db()
    try:
        sql = "SELECT * FROM academic_generation_jobs WHERE 1=1"
        params: list = []
        if course_id:
            sql += " AND course_id = ?"
            params.append(course_id)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor = await db.execute(sql, tuple(params))
        rows = await cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            for json_field in ("output_types", "stages_completed"):
                if isinstance(d.get(json_field), str):
                    try:
                        d[json_field] = json.loads(d[json_field])
                    except Exception:
                        d[json_field] = []
            if isinstance(d.get("checkpoint_json"), str):
                try:
                    d["checkpoint_json"] = json.loads(d["checkpoint_json"])
                except Exception:
                    d["checkpoint_json"] = {}
            results.append(d)
        return results
    finally:
        await db.close()


async def acad_delete_job(job_id: str) -> bool:
    db = await get_db()
    try:
        cursor = await db.execute("DELETE FROM academic_generation_jobs WHERE id = ?", (job_id,))
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


# ── Academic: Outputs ───────────────────────────────────────

async def acad_create_output(course_id: str, job_id: str, output_type: str,
                              file_path: str, file_size: int = 0,
                              topic_count: int = 0, version: int = 1,
                              generation_params: dict | None = None) -> dict:
    db = await get_db()
    try:
        oid = new_id()
        now = _now()
        await db.execute(
            """INSERT INTO academic_generation_outputs
               (id, course_id, job_id, output_type, version, file_path, file_size,
                topic_count, generation_params, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (oid, course_id, job_id, output_type, version, file_path, file_size,
             topic_count, json.dumps(generation_params or {}), now),
        )
        await db.commit()
        return {"id": oid, "course_id": course_id, "output_type": output_type,
                "version": version, "file_path": file_path, "created_at": now}
    finally:
        await db.close()


async def acad_list_outputs(course_id: str | None = None, output_type: str | None = None,
                             limit: int = 100) -> list[dict]:
    db = await get_db()
    try:
        sql = "SELECT * FROM academic_generation_outputs WHERE 1=1"
        params: list = []
        if course_id:
            sql += " AND course_id = ?"
            params.append(course_id)
        if output_type:
            sql += " AND output_type = ?"
            params.append(output_type)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor = await db.execute(sql, tuple(params))
        rows = await cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("generation_params"), str):
                try:
                    d["generation_params"] = json.loads(d["generation_params"])
                except Exception:
                    d["generation_params"] = {}
            results.append(d)
        return results
    finally:
        await db.close()


async def acad_get_output(output_id: str) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM academic_generation_outputs WHERE id = ?", (output_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        if isinstance(d.get("generation_params"), str):
            try:
                d["generation_params"] = json.loads(d["generation_params"])
            except Exception:
                d["generation_params"] = {}
        return d
    finally:
        await db.close()


async def acad_delete_output(output_id: str) -> bool:
    db = await get_db()
    try:
        cursor = await db.execute("DELETE FROM academic_generation_outputs WHERE id = ?", (output_id,))
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


# ── Academic: Feedback ──────────────────────────────────────

async def acad_add_feedback(output_id: str, rating: int = 0, comment: str = "",
                             topic_accuracy_pct: float | None = None) -> dict:
    db = await get_db()
    try:
        fid = new_id()
        now = _now()
        await db.execute(
            """INSERT INTO academic_output_feedback
               (id, output_id, rating, comment, topic_accuracy_pct, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (fid, output_id, rating, comment, topic_accuracy_pct, now),
        )
        await db.commit()
        return {"id": fid, "output_id": output_id, "rating": rating, "created_at": now}
    finally:
        await db.close()


async def acad_get_feedback(output_id: str) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM academic_output_feedback WHERE output_id = ? ORDER BY created_at DESC",
            (output_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


# ── Academic: Metrics Overview ──────────────────────────────

async def acad_get_overview_stats() -> dict:
    """Get aggregate stats across all academic data."""
    db = await get_db()
    try:
        c1 = await db.execute("SELECT COUNT(*) as cnt FROM academic_courses")
        r1 = await c1.fetchone()
        c2 = await db.execute("SELECT COUNT(*) as cnt FROM academic_student_reviews WHERE is_spam = 0")
        r2 = await c2.fetchone()
        c3 = await db.execute("SELECT COUNT(*) as cnt FROM academic_generation_jobs WHERE status = 'running'")
        r3 = await c3.fetchone()
        c4 = await db.execute("SELECT COUNT(*) as cnt FROM academic_generation_jobs WHERE status = 'completed'")
        r4 = await c4.fetchone()
        c5 = await db.execute("SELECT COUNT(*) as cnt FROM academic_generation_jobs WHERE status = 'failed'")
        r5 = await c5.fetchone()
        c6 = await db.execute("SELECT COUNT(*) as cnt FROM academic_generation_outputs")
        r6 = await c6.fetchone()
        return {
            "total_courses": r1["cnt"] if r1 else 0,
            "total_reviews": r2["cnt"] if r2 else 0,
            "active_jobs": r3["cnt"] if r3 else 0,
            "completed_jobs": r4["cnt"] if r4 else 0,
            "failed_jobs": r5["cnt"] if r5 else 0,
            "total_outputs": r6["cnt"] if r6 else 0,
        }
    finally:
        await db.close()

