"""
Academic API Endpoints for SoloLLM.

Provides REST API for the VU Academic Auto-Generation system:
- Course management with bulk PDF upload and vector storage
- Review upload and categorization
- Content generation pipeline (highlights, notes, MCQs)
- Job management (queue, retry, cancel)
- Output archive and download
- Quality metrics and feedback

CRITICAL: All operations are strictly course-scoped. No cross-course data access.
"""

import os
import uuid
import json
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

from core.config import settings
from rag.pipeline import rag_pipeline
from rag.ingest import detect_file_type
from storage.database import (
    acad_list_courses, acad_get_course, acad_get_course_by_code,
    acad_list_reviews, acad_get_review_count,
    acad_get_job, acad_list_jobs, acad_list_outputs, acad_get_output,
    acad_add_feedback, acad_get_feedback, acad_get_overview_stats,
    acad_get_topic_scores, acad_delete_job, acad_delete_output,
)
from academic.course_registry import course_registry, extract_course_code
from academic.review_ingest import review_ingester
from academic.exam_signal_engine import exam_signal_engine
from academic.job_orchestrator import job_orchestrator
from academic.evaluator import quality_evaluator
from core.inference import ollama_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/academic", tags=["Academic"])

# Upload directory for academic files
ACADEMIC_UPLOAD_DIR = settings.data_dir / "academic_uploads"
ACADEMIC_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Request/Response Models ─────────────────────────────────

class ManualReviewRequest(BaseModel):
    course_code: str = Field(..., min_length=2)
    review_text: str = Field(..., min_length=10)
    semester: str = ""


class GenerateRequest(BaseModel):
    course_code: str = Field(..., min_length=2)
    output_types: list[str] = ["highlighted_handout", "midterm_notes", "final_notes", "mcqs"]
    generation_model: str | None = None
    max_context_chars: int = Field(2200, ge=600, le=12000)
    syllabus_scope: str = Field("all", pattern="^(all|midterm|final)$")
    split_mode: str = Field("auto", pattern="^(auto|manual|none)$")
    manual_midterm_end_page: int | None = Field(None, ge=1)
    retrieval_top_k: int = Field(5, ge=1, le=20)
    batch_pages: int = Field(5, ge=1, le=10)
    max_batch_chars: int = Field(7000, ge=1200, le=30000)
    max_review_evidence_chars: int = Field(2500, ge=500, le=12000)
    max_spans_per_batch: int = Field(25, ge=5, le=60)
    debug_mode: bool = False
    continue_on_error: bool = True
    max_highlights_per_page: int = Field(20, ge=3, le=30)


class BulkGenerateRequest(BaseModel):
    course_codes: list[str] = Field(..., min_items=1)
    output_types: list[str] = ["highlighted_handout", "midterm_notes", "final_notes", "mcqs"]
    generation_model: str | None = None
    max_context_chars: int = Field(2200, ge=600, le=12000)
    syllabus_scope: str = Field("all", pattern="^(all|midterm|final)$")
    split_mode: str = Field("auto", pattern="^(auto|manual|none)$")
    manual_midterm_end_page: int | None = Field(None, ge=1)
    retrieval_top_k: int = Field(5, ge=1, le=20)
    batch_pages: int = Field(5, ge=1, le=10)
    max_batch_chars: int = Field(7000, ge=1200, le=30000)
    max_review_evidence_chars: int = Field(2500, ge=500, le=12000)
    max_spans_per_batch: int = Field(25, ge=5, le=60)
    debug_mode: bool = False
    continue_on_error: bool = True
    max_highlights_per_page: int = Field(20, ge=3, le=30)


class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""
    topic_accuracy_pct: Optional[float] = None


class ScoringWeightsUpdate(BaseModel):
    review_frequency: Optional[float] = None
    urgency_signal: Optional[float] = None
    consensus: Optional[float] = None
    syllabus_importance: Optional[float] = None
    recency: Optional[float] = None
    llm_confidence: Optional[float] = None


# ── Course & PDF Upload Endpoints ───────────────────────────

@router.post("/courses/bulk-import")
async def bulk_import_courses(
    files: list[UploadFile] = File(...),
):
    """
    Bulk import course PDFs. Auto-detects course codes from filenames.

    Each PDF is:
    1. Parsed for course code
    2. Registered in the course registry
    3. Ingested into the course's ISOLATED FAISS workspace
    """
    results = []

    for file in files:
        if not file.filename:
            results.append({"filename": "unknown", "error": "No filename"})
            continue

        file_type = detect_file_type(file.filename)
        if file_type != "pdf":
            results.append({
                "filename": file.filename,
                "error": f"Not a PDF file (detected: {file_type or 'unknown'})",
            })
            continue

        # Save uploaded file
        safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = str(ACADEMIC_UPLOAD_DIR / safe_name)

        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        except Exception as e:
            results.append({"filename": file.filename, "error": f"Save failed: {e}"})
            continue

        # Extract course code from filename
        parsed_code = extract_course_code(file.filename, source="filename")

        if not parsed_code:
            results.append({
                "filename": file.filename,
                "error": "Could not detect course code from filename",
                "file_path": file_path,
                "needs_manual_mapping": True,
            })
            continue

        # Register course (idempotent)
        course = await course_registry.register_course(
            code=parsed_code.normalized,
            department=parsed_code.prefix,
        )

        # Ingest PDF into course's ISOLATED workspace
        try:
            ingest_result = await rag_pipeline.ingest_document(
                file_path=file_path,
                workspace_id=course["workspace_id"],
            )

            results.append({
                "filename": file.filename,
                "course_code": course["code"],
                "course_id": course["id"],
                "workspace_id": course["workspace_id"],
                "confidence": parsed_code.confidence,
                "ingest": ingest_result,
            })

        except Exception as e:
            logger.error("Ingestion failed for %s: %s", file.filename, e)
            results.append({
                "filename": file.filename,
                "course_code": parsed_code.normalized,
                "error": f"Ingestion failed: {e}",
            })

    success_count = sum(1 for r in results if "error" not in r)
    return {
        "total_files": len(files),
        "success_count": success_count,
        "failed_count": len(files) - success_count,
        "results": results,
    }


@router.post("/courses/upload-pdf")
async def upload_single_pdf(
    file: UploadFile = File(...),
    course_code: str = Form(...),
):
    """
    Upload a single PDF for a specific course.

    The PDF is ingested into the course's isolated FAISS workspace.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    file_type = detect_file_type(file.filename)
    if file_type != "pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Register course
    course = await course_registry.register_course(code=course_code)

    # Save file
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = str(ACADEMIC_UPLOAD_DIR / safe_name)

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {e}")

    # Ingest into isolated workspace
    try:
        result = await rag_pipeline.ingest_document(
            file_path=file_path,
            workspace_id=course["workspace_id"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {
        "course_code": course["code"],
        "course_id": course["id"],
        "workspace_id": course["workspace_id"],
        "ingest": result,
    }


@router.get("/courses")
async def list_courses():
    """List all registered courses."""
    courses = await acad_list_courses()

    # Enrich with review counts
    enriched = []
    for c in courses:
        count = await acad_get_review_count(c["id"])
        enriched.append({**c, "review_count": count})

    return {"courses": enriched, "total": len(enriched)}


@router.get("/courses/{course_code}")
async def get_course_detail(course_code: str):
    """Get detailed info for a specific course."""
    course = await acad_get_course_by_code(course_code)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    review_count = await acad_get_review_count(course["id"])
    scores = await acad_get_topic_scores(course["id"])
    outputs = await acad_list_outputs(course_id=course["id"])

    return {
        **course,
        "review_count": review_count,
        "topic_scores": scores,
        "outputs": outputs,
    }


# ── Review Endpoints ────────────────────────────────────────

@router.post("/reviews/upload")
async def upload_reviews(
    file: UploadFile = File(...),
    course_code: str = Form(...),
):
    """
    Upload reviews for a specific course (CSV/JSON/TXT).

    Reviews are strictly bound to the specified course.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    course = await course_registry.register_course(code=course_code)

    content = (await file.read()).decode("utf-8", errors="replace")

    ext = Path(file.filename).suffix.lower()
    if ext == ".csv":
        result = await review_ingester.ingest_csv(content, course["id"], file.filename)
    elif ext == ".json":
        result = await review_ingester.ingest_json(content, course["id"], file.filename)
    elif ext == ".txt":
        result = await review_ingester.ingest_text(content, course["id"], file.filename)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}. Use .csv, .json, or .txt")

    return result


@router.post("/reviews/manual")
async def add_manual_review(request: ManualReviewRequest):
    """Add a single review manually for a course."""
    
    # Auto-register course if new
    course = await course_registry.register_course(code=request.course_code)

    return await review_ingester.add_manual_review(
        course_id=course["id"],
        review_text=request.review_text,
        semester=request.semester,
    )


@router.get("/reviews")
async def list_reviews(
    course_code: str = Query(...),
    include_spam: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
):
    """List reviews for a specific course."""
    course = await acad_get_course_by_code(course_code)
    if not course:
        raise HTTPException(status_code=404, detail=f"Course '{course_code}' not found")

    reviews = await acad_list_reviews(course["id"], include_spam=include_spam, limit=limit)
    return {"course_code": course_code, "reviews": reviews, "total": len(reviews)}


@router.post("/reviews/reprocess")
async def reprocess_reviews(course_code: str = Query(...)):
    """Re-extract features from a course's reviews."""
    course = await acad_get_course_by_code(course_code)
    if not course:
        raise HTTPException(status_code=404, detail=f"Course '{course_code}' not found")

    scored = await exam_signal_engine.score_course(course["id"])
    return {
        "course_code": course_code,
        "topics_scored": len(scored),
        "top_topics": [
            {"name": s.topic_name, "probability": s.exam_probability, "bucket": s.weight_bucket}
            for s in scored[:10]
        ],
    }


# ── Generation Endpoints ────────────────────────────────────

@router.post("/generate")
async def generate_content(request: GenerateRequest):
    """
    Generate academic content for a SINGLE selected course.

    All generation is scoped to this course's vectors and reviews only.
    """
    course = await acad_get_course_by_code(request.course_code)
    if not course:
        raise HTTPException(status_code=404, detail=f"Course '{request.course_code}' not found")

    job = await job_orchestrator.submit_job(
        course_id=course["id"],
        output_types=request.output_types,
        options={
            "generation_model": request.generation_model,
            "max_context_chars": request.max_context_chars,
            "syllabus_scope": request.syllabus_scope,
            "split_mode": request.split_mode,
            "manual_midterm_end_page": request.manual_midterm_end_page,
            "retrieval_top_k": request.retrieval_top_k,
            "batch_pages": request.batch_pages,
            "max_batch_chars": request.max_batch_chars,
            "max_review_evidence_chars": request.max_review_evidence_chars,
            "max_spans_per_batch": request.max_spans_per_batch,
            "debug_mode": request.debug_mode,
            "continue_on_error": request.continue_on_error,
            "max_highlights_per_page": request.max_highlights_per_page,
        },
    )

    if "error" in job:
        raise HTTPException(status_code=400, detail=job["error"])

    return job


@router.post("/generate/bulk")
async def bulk_generate(request: BulkGenerateRequest):
    """Generate content for multiple courses (each fully isolated)."""
    jobs = []
    for code in request.course_codes:
        course = await acad_get_course_by_code(code)
        if not course:
            jobs.append({"course_code": code, "error": "Not found"})
            continue

        job = await job_orchestrator.submit_job(
            course_id=course["id"],
            output_types=request.output_types,
            options={
                "generation_model": request.generation_model,
                "max_context_chars": request.max_context_chars,
                "syllabus_scope": request.syllabus_scope,
                "split_mode": request.split_mode,
                "manual_midterm_end_page": request.manual_midterm_end_page,
                "retrieval_top_k": request.retrieval_top_k,
                "batch_pages": request.batch_pages,
                "max_batch_chars": request.max_batch_chars,
                "max_review_evidence_chars": request.max_review_evidence_chars,
                "max_spans_per_batch": request.max_spans_per_batch,
                "debug_mode": request.debug_mode,
                "continue_on_error": request.continue_on_error,
                "max_highlights_per_page": request.max_highlights_per_page,
            },
        )
        jobs.append({"course_code": code, **job})

    return {"jobs": jobs, "total": len(jobs)}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a generation job."""
    job = await acad_get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{job_id}/preview")
async def get_job_preview(job_id: str):
    """Get live preview/checkpoint info for a job."""
    job = await acad_get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    checkpoint = job.get("checkpoint_json") or {}
    preview = checkpoint.get("preview") or {}

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "current_stage": job.get("current_stage", ""),
        "progress": job.get("progress", 0.0),
        "preview": preview,
        "error": job.get("error"),
    }


@router.get("/jobs/{job_id}/preview/download")
async def download_job_preview(job_id: str):
    """Download the live in-progress preview PDF for a job if available."""
    job = await acad_get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    checkpoint = job.get("checkpoint_json") or {}
    preview = checkpoint.get("preview") or {}
    live_file_path = preview.get("live_file_path") or ""

    if not live_file_path or not os.path.exists(live_file_path):
        raise HTTPException(status_code=404, detail="Preview file not ready yet")

    filename = os.path.basename(live_file_path)
    return FileResponse(path=live_file_path, filename=filename, media_type="application/pdf")


@router.get("/jobs")
async def list_jobs(
    course_code: str = Query(None),
    status: str = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    """List generation jobs, optionally filtered by course or status."""
    course_id = None
    if course_code:
        course = await acad_get_course_by_code(course_code)
        if course:
            course_id = course["id"]

    jobs = await acad_list_jobs(course_id=course_id, status=status, limit=limit)
    return {"jobs": jobs, "total": len(jobs)}


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str):
    """Retry a failed generation job."""
    result = await job_orchestrator.retry_job(job_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause a running generation job."""
    result = await job_orchestrator.pause_job(job_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused generation job."""
    result = await job_orchestrator.resume_job(job_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    """Stop a running or paused generation job."""
    return await job_orchestrator.cancel_job(job_id)


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a failed/cancelled job and its DB record."""
    job = await acad_get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") not in ("failed", "cancelled", "completed"):
        raise HTTPException(status_code=400, detail="Only failed, cancelled, or completed jobs can be deleted")

    deleted = await acad_delete_job(job_id)
    return {"job_id": job_id, "deleted": bool(deleted)}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running generation job."""
    return await job_orchestrator.cancel_job(job_id)


# ── Output Endpoints ────────────────────────────────────────

@router.get("/outputs")
async def list_outputs(
    course_code: str = Query(None),
    output_type: str = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """List generated outputs, optionally filtered by course."""
    course_id = None
    if course_code:
        course = await acad_get_course_by_code(course_code)
        if course:
            course_id = course["id"]

    outputs = await acad_list_outputs(course_id=course_id, output_type=output_type, limit=limit)
    return {"outputs": outputs, "total": len(outputs)}


@router.get("/outputs/{output_id}/download")
async def download_output(output_id: str):
    """Download a generated output file (PDF)."""
    output = await acad_get_output(output_id)
    if not output:
        raise HTTPException(status_code=404, detail="Output not found")

    file_path = output["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Output file not found on disk")

    filename = os.path.basename(file_path)
    media_type = "application/pdf" if filename.endswith(".pdf") else "text/plain"

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type,
    )


@router.delete("/outputs/{output_id}")
async def delete_output(output_id: str):
    """Delete a generated output file and its DB row."""
    output = await acad_get_output(output_id)
    if not output:
        raise HTTPException(status_code=404, detail="Output not found")

    file_path = output.get("file_path", "")
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning("Could not delete output file %s: %s", file_path, e)

    deleted = await acad_delete_output(output_id)
    return {"output_id": output_id, "deleted": bool(deleted)}


@router.get("/models")
async def list_generation_models():
    """List local Ollama models for academic generation model selection."""
    if not await ollama_client.is_available():
        raise HTTPException(status_code=503, detail="Ollama is not running")
    return {"models": await ollama_client.list_models()}


@router.post("/outputs/{output_id}/feedback")
async def submit_feedback(output_id: str, request: FeedbackRequest):
    """Submit feedback on a generated output."""
    output = await acad_get_output(output_id)
    if not output:
        raise HTTPException(status_code=404, detail="Output not found")

    return await acad_add_feedback(
        output_id=output_id,
        rating=request.rating,
        comment=request.comment,
        topic_accuracy_pct=request.topic_accuracy_pct,
    )


# ── Evaluation/Admin Endpoints ──────────────────────────────

@router.get("/metrics/overview")
async def metrics_overview():
    """Get system-wide academic metrics."""
    return await quality_evaluator.get_system_overview()


@router.get("/metrics/course/{course_code}")
async def course_metrics(course_code: str):
    """Get quality metrics for a specific course."""
    course = await acad_get_course_by_code(course_code)
    if not course:
        raise HTTPException(status_code=404, detail=f"Course '{course_code}' not found")

    metrics = await quality_evaluator.evaluate_course(course["id"])
    return {
        "course_code": course_code,
        "course_id": course["id"],
        "metrics": {
            "topic_count": metrics.topic_count,
            "high_confidence": metrics.high_confidence_topics,
            "medium_confidence": metrics.medium_confidence_topics,
            "low_confidence": metrics.low_confidence_topics,
            "avg_exam_probability": metrics.avg_exam_probability,
            "output_count": metrics.output_count,
            "avg_feedback_rating": metrics.avg_feedback_rating,
            "feedback_count": metrics.feedback_count,
        },
    }


@router.put("/scoring-weights")
async def update_scoring_weights(request: ScoringWeightsUpdate):
    """Update the exam topic scoring weights."""
    weights = {}
    for field_name in ["review_frequency", "urgency_signal", "consensus",
                       "syllabus_importance", "recency", "llm_confidence"]:
        val = getattr(request, field_name, None)
        if val is not None:
            weights[field_name] = val

    if not weights:
        raise HTTPException(status_code=400, detail="No weights provided")

    updated = await exam_signal_engine.update_weights(weights)
    return {"weights": updated}


@router.get("/scoring-weights")
async def get_scoring_weights():
    """Get current exam topic scoring weights."""
    return {"weights": await exam_signal_engine.get_weights()}


# ── Debug / AI Log Endpoints ────────────────────────────────

@router.get("/jobs/{job_id}/ai-logs")
async def get_job_ai_logs(job_id: str):
    """Get AI input/output logs for a highlight generation job."""
    from academic.highlight_logger import load_debug_log_full

    full_log = load_debug_log_full(job_id)
    if full_log:
        return full_log

    # Fallback: return ai_logs from checkpoint
    job = await acad_get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    checkpoint = job.get("checkpoint_json") or {}
    ai_logs = checkpoint.get("ai_logs", [])
    return {
        "job_id": job_id,
        "source": "checkpoint",
        "ai_logs": ai_logs,
        "total": len(ai_logs),
    }


@router.get("/jobs/{job_id}/match-report")
async def get_job_match_report(job_id: str):
    """Get span match/unmatch statistics for a highlight generation job."""
    from academic.highlight_logger import load_debug_log, load_debug_log_full

    summary = load_debug_log(job_id)
    if not summary:
        # Fallback: compute from checkpoint render_stats
        job = await acad_get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        checkpoint = job.get("checkpoint_json") or {}
        render_stats = checkpoint.get("render_stats", {})
        if render_stats:
            return {"job_id": job_id, "source": "checkpoint", **render_stats}
        raise HTTPException(status_code=404, detail="No match report available for this job")

    return {"job_id": job_id, "source": "debug_log", **summary}
