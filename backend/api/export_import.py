"""
Export / Import API for SoloLLM — Phase 6.

Handles exporting and importing:
- Conversations (JSON format)
- Documents and workspace metadata
- Settings
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response

from storage.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/export", tags=["Export/Import"])


@router.get("/conversations")
async def export_conversations():
    """Export all conversations and messages as JSON."""
    db = await get_db()
    try:
        # Get all conversations
        cursor = await db.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC"
        )
        conversations = [dict(row) for row in await cursor.fetchall()]

        # Get all messages
        for conv in conversations:
            msg_cursor = await db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
                (conv["id"],),
            )
            conv["messages"] = [dict(row) for row in await msg_cursor.fetchall()]

        export_data = {
            "version": "1.0",
            "app": "SoloLLM",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "conversation_count": len(conversations),
            "conversations": conversations,
        }
    finally:
        await db.close()

    return Response(
        content=json.dumps(export_data, indent=2, default=str),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="solollm_conversations_{datetime.now().strftime("%Y%m%d")}.json"'
        },
    )


@router.post("/conversations/import")
async def import_conversations(file: UploadFile = File(...)):
    """Import conversations from a JSON export."""
    try:
        content = await file.read()
        data = json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    if "conversations" not in data:
        raise HTTPException(status_code=400, detail="Missing 'conversations' key")

    db = await get_db()
    imported = 0
    try:
        for conv in data["conversations"]:
            # Insert conversation
            await db.execute(
                """INSERT OR IGNORE INTO conversations
                   (id, title, model, system_prompt, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (conv["id"], conv.get("title", "Imported"),
                 conv.get("model", ""), conv.get("system_prompt", ""),
                 conv.get("created_at", datetime.now(timezone.utc).isoformat()),
                 conv.get("updated_at", datetime.now(timezone.utc).isoformat())),
            )
            # Insert messages
            for msg in conv.get("messages", []):
                await db.execute(
                    """INSERT OR IGNORE INTO messages
                       (id, conversation_id, role, content, token_count,
                        is_continuation, continuation_of, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (msg["id"], msg["conversation_id"], msg["role"],
                     msg["content"], msg.get("token_count", 0),
                     msg.get("is_continuation", False),
                     msg.get("continuation_of"),
                     msg.get("created_at", datetime.now(timezone.utc).isoformat())),
                )
            imported += 1
        await db.commit()
    finally:
        await db.close()

    return {"success": True, "imported_count": imported}


@router.get("/settings")
async def export_settings():
    """Export all settings as JSON."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT key, value FROM settings")
        rows = await cursor.fetchall()
        settings_dict = {row["key"]: row["value"] for row in rows}
    finally:
        await db.close()

    return Response(
        content=json.dumps({
            "version": "1.0",
            "app": "SoloLLM",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "settings": settings_dict,
        }, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": 'attachment; filename="solollm_settings.json"'
        },
    )


@router.post("/settings/import")
async def import_settings(file: UploadFile = File(...)):
    """Import settings from a JSON export."""
    try:
        content = await file.read()
        data = json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    if "settings" not in data:
        raise HTTPException(status_code=400, detail="Missing 'settings' key")

    db = await get_db()
    try:
        for key, value in data["settings"].items():
            await db.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )
        await db.commit()
    finally:
        await db.close()

    return {"success": True, "imported_keys": list(data["settings"].keys())}
