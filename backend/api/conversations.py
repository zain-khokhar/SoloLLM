import logging

from fastapi import APIRouter, HTTPException, Query

from storage.database import (
    list_conversations, get_conversation, update_conversation,
    delete_conversation, get_messages,
)
from storage.schemas import ConversationUpdate

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/conversations")
async def get_conversations(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    """List all conversations, newest first."""
    convos = await list_conversations(limit=limit, offset=offset)
    return {"conversations": convos}


@router.get("/conversations/{conversation_id}")
async def get_conversation_detail(conversation_id: str):
    """Get a conversation with its threads and default thread info."""
    conversation = await get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    from storage.database import list_threads, get_default_thread

    threads = await list_threads(conversation_id)
    default_thread = await get_default_thread(conversation_id)

    # Get messages for the default thread (or all if no thread)
    if default_thread:
        messages = await get_messages(conversation_id, thread_id=default_thread["id"])
    else:
        messages = await get_messages(conversation_id)

    return {
        "conversation": conversation,
        "messages": messages,
        "threads": threads,
        "default_thread_id": default_thread["id"] if default_thread else None,
    }


@router.put("/conversations/{conversation_id}")
async def update_conversation_endpoint(conversation_id: str, request: ConversationUpdate):
    """Update conversation title, model, or system prompt."""
    conversation = await get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    update_data = request.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    await update_conversation(conversation_id, **update_data)
    updated = await get_conversation(conversation_id)
    return {"conversation": updated}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    """Delete a conversation and all its messages."""
    deleted = await delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "id": conversation_id}
