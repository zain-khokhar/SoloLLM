"""
RAG API Endpoints for SoloLLM.

Handles document upload, management, workspace management,
and document search/query operations.
"""

import os
import shutil
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from core.config import settings
from rag.pipeline import rag_pipeline
from rag.ingest import detect_file_type, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Upload directory
UPLOAD_DIR = settings.data_dir / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Request/Response Models ─────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    workspace_id: str = "default"
    top_k: int = 5
    document_id: str | None = None
    rerank: bool = True


class WorkspaceCreate(BaseModel):
    name: str
    description: str = ""


# ── Document Upload & Management ────────────────────────────

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    workspace_id: str = Form("default"),
):
    """
    Upload and ingest a document.

    Supports: PDF, DOCX, TXT, MD, HTML, CSV, and source code files.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_type = detect_file_type(file.filename)
    if not file_type:
        ext = Path(file.filename).suffix
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {supported}",
        )

    # Save uploaded file
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = str(UPLOAD_DIR / safe_name)

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Ingest
    try:
        result = await rag_pipeline.ingest_document(
            file_path=file_path,
            workspace_id=workspace_id,
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        # Clean up file on failure
        try:
            os.remove(file_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    if not result.get("success"):
        try:
            os.remove(file_path)
        except OSError:
            pass
        raise HTTPException(
            status_code=422,
            detail=result.get("errors", ["Unknown error"])[0],
        )

    return result


@router.get("/list")
async def list_documents(workspace_id: str = "default"):
    """List all documents in a workspace."""
    docs = await rag_pipeline.list_documents(workspace_id)
    return {"documents": docs, "workspace_id": workspace_id}


@router.get("/{document_id}")
async def get_document(document_id: str):
    """Get document details."""
    doc = await rag_pipeline.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    doc = await rag_pipeline.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    await rag_pipeline.delete_document(document_id)
    return {"success": True, "document_id": document_id}


# ── Search / Query ──────────────────────────────────────────

@router.post("/query")
async def query_documents(request: QueryRequest):
    """
    Query documents using hybrid search (vector + keyword).

    Returns matched chunks with citations.
    """
    cited = await rag_pipeline.query(
        query=request.query,
        workspace_id=request.workspace_id,
        top_k=request.top_k,
        document_id=request.document_id,
        rerank=request.rerank,
    )

    return {
        "citations": [
            {
                "index": c.index,
                "document_title": c.document_title,
                "section_title": c.section_title,
                "page_number": c.page_number,
                "document_id": c.document_id,
                "chunk_id": c.chunk_id,
                "relevance_score": c.relevance_score,
                "excerpt": c.excerpt,
            }
            for c in cited.citations
        ],
        "context_text": cited.context_text,
        "source_count": len(cited.citations),
    }


@router.get("/stats/{workspace_id}")
async def get_stats(workspace_id: str = "default"):
    """Get RAG stats for a workspace."""
    return await rag_pipeline.get_stats(workspace_id)


# ── Workspace Management ────────────────────────────────────

@router.post("/workspaces")
async def create_workspace(request: WorkspaceCreate):
    """Create a new document workspace."""
    return await rag_pipeline.create_workspace(request.name, request.description)


@router.get("/workspaces/list")
async def list_workspaces():
    """List all workspaces."""
    workspaces = await rag_pipeline.list_workspaces()
    return {"workspaces": workspaces}


# ── Supported File Types ────────────────────────────────────

@router.get("/supported-types")
async def get_supported_types():
    """Get list of supported file types."""
    return {
        "extensions": sorted(SUPPORTED_EXTENSIONS.keys()),
        "types": sorted(set(SUPPORTED_EXTENSIONS.values())),
    }
