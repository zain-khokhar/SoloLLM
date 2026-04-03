"""
Quantize API endpoints for GGUF model quantization.

Provides REST + SSE endpoints for:
- Tool setup (downloading llama.cpp binaries)
- Starting quantization jobs (HuggingFace or local GGUF)
- Job management (list, status, cancel, delete)
- Ollama import of completed outputs
"""

import asyncio
import json
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional
from sse_starlette.sse import EventSourceResponse

from core.config import settings

from quantize.quantize_manager import quantize_manager, QUANT_TYPES

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/quantize")


# ── Request models ─────────────────────────────────────────

class StartQuantizeRequest(BaseModel):
    source_type: str = Field(..., description="'local_gguf' or 'huggingface'")
    source: str = Field(..., min_length=1, description="Local GGUF path or HuggingFace model ID")
    quant_level: str = Field(default="Q4_K_M", description="Quantization level")
    output_name: str = Field(..., min_length=1, description="Output model name")
    import_to_ollama: bool = Field(default=True, description="Auto-import to Ollama after quantization")


class ImportGGUFRequest(BaseModel):
    gguf_path: str = Field(..., min_length=1, description="Absolute path to local .gguf file")
    model_name: str = Field(..., min_length=1, description="Name to register in Ollama")


# ── Tool management ────────────────────────────────────────

@router.get("/tools-status")
async def get_tools_status():
    """Check if quantization tools are ready."""
    return quantize_manager.get_tools_status()


@router.post("/setup-tools")
async def setup_tools():
    """Download and set up llama.cpp quantization tools. Returns SSE stream."""
    async def event_stream():
        try:
            async for event in quantize_manager.setup_tools():
                yield {"event": "progress", "data": json.dumps(event)}
            yield {"event": "done", "data": json.dumps({"message": "Setup complete"})}
        except Exception as e:
            logger.exception("setup-tools SSE error")
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    return EventSourceResponse(event_stream())


# ── Quant types ────────────────────────────────────────────

@router.get("/quant-types")
async def get_quant_types():
    """Return available quantization types with descriptions."""
    return QUANT_TYPES


# ── Job management ─────────────────────────────────────────

@router.post("/start")
async def start_quantize(req: StartQuantizeRequest):
    """Start a quantization job."""
    try:
        job_id = quantize_manager.start_job(
            source_type=req.source_type,
            source=req.source,
            quant_level=req.quant_level,
            output_name=req.output_name,
            import_to_ollama=req.import_to_ollama,
        )
        job = quantize_manager.get_job(job_id)
        return {"job_id": job_id, "job": job.to_dict() if job else None}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to start quantization")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def list_jobs():
    """List all quantization jobs."""
    return quantize_manager.list_jobs()


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get details of a specific job."""
    job = quantize_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    """SSE stream that polls a job's state every 0.5s until it finishes."""
    job = quantize_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_stream():
        while True:
            j = quantize_manager.get_job(job_id)
            if not j:
                yield {"event": "error", "data": json.dumps({"message": "Job not found"})}
                break
            yield {"event": "status", "data": json.dumps(j.to_dict())}
            if j.status in ("complete", "error", "cancelled"):
                break
            await asyncio.sleep(0.5)
        yield {"event": "done", "data": json.dumps({"message": "Stream ended"})}

    return EventSourceResponse(event_stream())


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    job = quantize_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "running":
        raise HTTPException(status_code=400, detail="Job is not running")
    await quantize_manager.cancel_job(job_id)
    return {"message": "Job cancelled"}


@router.post("/import-ollama/{job_id}")
async def import_to_ollama(job_id: str):
    """Import a completed quantization output into Ollama."""
    job = quantize_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "complete":
        raise HTTPException(status_code=400, detail="Job is not complete")
    if not job.output_path:
        raise HTTPException(status_code=400, detail="No output file found")

    try:
        from pathlib import Path
        await quantize_manager._import_to_ollama(job, Path(job.output_path), job.output_name)
        return {"message": f"Imported as '{job.output_name}' in Ollama"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a completed/failed job and its output file."""
    job = quantize_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "running":
        raise HTTPException(status_code=400, detail="Cannot delete a running job. Cancel it first.")
    quantize_manager.delete_job(job_id)
    return {"message": "Job deleted"}


# ── Direct GGUF import ────────────────────────────────────

@router.post("/import-gguf")
async def import_gguf_direct(req: ImportGGUFRequest):
    """Import a local GGUF file directly into Ollama (no quantization)."""
    try:
        result = await quantize_manager.import_gguf_to_ollama(
            gguf_path=req.gguf_path.strip(),
            model_name=req.model_name.strip(),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to import GGUF")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-gguf")
async def upload_gguf(
    file: UploadFile = File(...),
    model_name: str = Form(""),
):
    """Upload a .gguf file and import it into Ollama."""
    if not file.filename or not file.filename.lower().endswith(".gguf"):
        raise HTTPException(status_code=400, detail="Only .gguf files are accepted")

    # Auto-generate model name from filename if not provided
    if not model_name.strip():
        import re
        base = file.filename.rsplit(".", 1)[0]
        model_name = re.sub(r"[^a-z0-9_-]", "-", base.lower())

    # Save uploaded file to a temp location
    upload_dir = settings.data_dir / "quantize_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = await quantize_manager.import_gguf_to_ollama(
            gguf_path=str(dest),
            model_name=model_name.strip(),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to upload and import GGUF")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploaded file after import
        try:
            if dest.exists():
                dest.unlink()
        except Exception:
            pass
