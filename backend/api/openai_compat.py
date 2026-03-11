"""
OpenAI-Compatible API for SoloLLM — Phase 6.

Provides `/v1/chat/completions` endpoint so SoloLLM can serve as a
drop-in replacement for the OpenAI API. Supports both streaming
and non-streaming modes.
"""

import json
import time
import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["OpenAI-Compatible"])


# ── Request / Response Models ───────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "llama3.2:latest"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


# ── Endpoints ───────────────────────────────────────────────

@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Proxies to Ollama and reformats the response.
    """
    from core.inference import ollama_client

    if not await ollama_client.is_available():
        raise HTTPException(status_code=503, detail="Ollama not available")

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Build messages for Ollama
    ollama_messages = [
        {"role": m.role, "content": m.content}
        for m in request.messages
    ]

    if request.stream:
        return StreamingResponse(
            _stream_response(
                completion_id, created, request, ollama_messages, ollama_client
            ),
            media_type="text/event-stream",
        )

    # Non-streaming
    try:
        import httpx
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": request.model,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens or -1,
                        "top_p": request.top_p,
                    },
                },
            )
            data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")

    content = data.get("message", {}).get("content", "")
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[Choice(
            message=ChatMessage(role="assistant", content=content),
            finish_reason="stop",
        )],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _stream_response(
    completion_id: str,
    created: int,
    request: ChatCompletionRequest,
    messages: list[dict],
    ollama_client,
):
    """Generate SSE stream in OpenAI format."""
    import httpx

    # Initial chunk with role
    initial = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
    )
    yield f"data: {initial.model_dump_json()}\n\n"

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": request.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens or -1,
                        "top_p": request.top_p,
                    },
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        done = data.get("done", False)

                        if content:
                            chunk = ChatCompletionChunk(
                                id=completion_id,
                                created=created,
                                model=request.model,
                                choices=[StreamChoice(
                                    delta=DeltaMessage(content=content),
                                    finish_reason=None,
                                )],
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"

                        if done:
                            final = ChatCompletionChunk(
                                id=completion_id,
                                created=created,
                                model=request.model,
                                choices=[StreamChoice(
                                    delta=DeltaMessage(),
                                    finish_reason="stop",
                                )],
                            )
                            yield f"data: {final.model_dump_json()}\n\n"
                            yield "data: [DONE]\n\n"
                            break
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        error_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@router.get("/models")
async def list_models():
    """List available models in OpenAI format."""
    from core.inference import ollama_client
    try:
        models = await ollama_client.list_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": m.get("name", "unknown"),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama",
                }
                for m in models
            ],
        }
    except Exception:
        return {"object": "list", "data": []}
