"""
Threads API — CRUD for conversation threads with isolated context.

Every conversation can have multiple threads.
Each thread has its own:
  - Message history (no cross-contamination)
  - System prompt
  - Attached documents (scoped RAG)
  - Advanced settings (context mode, compression, memory layers)
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from storage.database import (
    create_thread, list_threads, get_thread, update_thread, delete_thread,
    get_default_thread, get_thread_settings, update_thread_settings,
    attach_document_to_thread, get_thread_documents, detach_document_from_thread,
    get_messages, get_conversation,
    get_active_context_pages, get_all_context_pages,
)
from storage.schemas import ThreadCreate, ThreadUpdate, ThreadSettingsUpdate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/conversations/{conversation_id}/threads")
async def list_conversation_threads(conversation_id: str):
    """List all threads for a conversation."""
    conversation = await get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    threads = await list_threads(conversation_id)
    return {"threads": threads}


@router.post("/conversations/{conversation_id}/threads")
async def create_conversation_thread(conversation_id: str, request: ThreadCreate):
    """Create a new thread in a conversation."""
    conversation = await get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    thread = await create_thread(
        conversation_id=conversation_id,
        title=request.title,
        system_prompt=request.system_prompt,
        context_mode=request.context_mode,
    )
    return {"thread": thread}


@router.get("/threads/{thread_id}")
async def get_thread_detail(thread_id: str):
    """Get a thread with its messages and settings."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Get only this thread's messages (isolated context)
    messages = await get_messages(thread["conversation_id"], thread_id=thread_id)
    thread_settings = await get_thread_settings(thread_id)

    return {
        "thread": thread,
        "messages": messages,
        "settings": thread_settings,
    }


@router.put("/threads/{thread_id}")
async def update_thread_endpoint(thread_id: str, request: ThreadUpdate):
    """Update a thread's title, system prompt, or context mode."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    update_data = request.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    await update_thread(thread_id, **update_data)
    updated = await get_thread(thread_id)
    return {"thread": updated}


@router.delete("/threads/{thread_id}")
async def delete_thread_endpoint(thread_id: str):
    """Delete a thread (cannot delete the default thread)."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    deleted = await delete_thread(thread_id)
    if not deleted:
        raise HTTPException(status_code=400, detail="Cannot delete the default thread")
    return {"status": "deleted", "id": thread_id}


# ── Thread Settings ─────────────────────────────────────────

@router.get("/threads/{thread_id}/settings")
async def get_thread_settings_endpoint(thread_id: str):
    """Get advanced settings for a thread."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    ts = await get_thread_settings(thread_id)
    return {"settings": ts}


@router.put("/threads/{thread_id}/settings")
async def update_thread_settings_endpoint(thread_id: str, request: ThreadSettingsUpdate):
    """Update advanced settings for a thread."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    update_data = request.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    await update_thread_settings(thread_id, **update_data)
    updated = await get_thread_settings(thread_id)
    return {"settings": updated}


# ── Thread Documents ────────────────────────────────────────

@router.post("/threads/{thread_id}/documents/{document_id}")
async def attach_document(thread_id: str, document_id: str):
    """Attach a document to a thread (scoped RAG)."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    result = await attach_document_to_thread(thread_id, document_id)
    return {"attached": result}


@router.get("/threads/{thread_id}/documents")
async def list_thread_documents(thread_id: str):
    """List documents attached to a thread."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    docs = await get_thread_documents(thread_id)
    return {"documents": docs}


@router.delete("/threads/{thread_id}/documents/{document_id}")
async def detach_document(thread_id: str, document_id: str):
    """Remove a document from a thread."""
    detached = await detach_document_from_thread(thread_id, document_id)
    if not detached:
        raise HTTPException(status_code=404, detail="Document not attached to this thread")
    return {"status": "detached"}


# ── Thread Context Pages (Memory Management) ───────────────

@router.get("/threads/{thread_id}/context")
async def get_thread_context(thread_id: str, active_only: bool = Query(True)):
    """Get context pages for a thread (memory paging visualization)."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if active_only:
        pages = await get_active_context_pages(thread_id)
    else:
        pages = await get_all_context_pages(thread_id)

    return {"pages": pages, "thread_id": thread_id}
