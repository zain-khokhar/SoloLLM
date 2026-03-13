import json
import logging

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from core.inference import ollama_client
from storage.schemas import PullModelRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/models")
async def list_models():
    """List all locally available Ollama models."""
    if not await ollama_client.is_available():
        raise HTTPException(status_code=503, detail="Ollama is not running. Please start Ollama first.")

    try:
        models = await ollama_client.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/pull")
async def pull_model(request: PullModelRequest):
    """Pull a model from Ollama registry. Streams progress via SSE."""
    if not await ollama_client.is_available():
        raise HTTPException(status_code=503, detail="Ollama is not running.")

    async def _stream_pull():
        try:
            async for event in ollama_client.pull_model(request.name):
                status = event.get("status", "")
                total = event.get("total", 0)
                completed = event.get("completed", 0)

                progress = 0
                if total and total > 0:
                    progress = round((completed / total) * 100, 1)

                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "status": status,
                        "total": total,
                        "completed": completed,
                        "progress": progress,
                    }),
                }

            yield {
                "event": "done",
                "data": json.dumps({"status": "success", "model": request.name}),
            }
        except Exception as e:
            logger.error(f"Failed to pull model {request.name}: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(_stream_pull(), media_type="text/event-stream")


@router.delete("/models/{model_name:path}")
async def delete_model(model_name: str):
    """Delete a locally downloaded model."""
    if not await ollama_client.is_available():
        raise HTTPException(status_code=503, detail="Ollama is not running.")

    success = await ollama_client.delete_model(model_name)
    if success:
        return {"status": "deleted", "model": model_name}
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found or could not be deleted")


@router.get("/models/{model_name:path}/info")
async def get_model_info(model_name: str):
    """Get detailed model info including context window size."""
    if not await ollama_client.is_available():
        raise HTTPException(status_code=503, detail="Ollama is not running.")
    try:
        info = await ollama_client.show_model(model_name)
        return {"info": info}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
