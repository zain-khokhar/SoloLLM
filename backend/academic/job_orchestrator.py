"""
Job Orchestrator for VU Academic Auto-Generation.

Manages the async generation pipeline: parse → score → generate → render.
Supports checkpointing, resumability, retry with backoff, and per-stage caching.
"""

import asyncio
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field

from storage.database import (
    acad_create_job, acad_update_job, acad_get_job, acad_list_jobs,
    acad_get_course, acad_create_output, acad_list_reviews,
)
from academic.topic_ranker import topic_ranker
from academic.content_generators import content_generator
from academic.mcq_generator import mcq_generator
from academic.pdf_renderer import pdf_renderer

logger = logging.getLogger(__name__)

# Active job tracking
_active_jobs: dict[str, asyncio.Task] = {}
_paused_jobs: set[str] = set()


@dataclass
class PipelineStage:
    name: str
    status: str = "pending"  # pending, running, completed, failed
    error: str | None = None


PIPELINE_STAGES = ["score_topics", "generate_content", "render_pdfs"]


class JobOrchestrator:
    """
    Orchestrates the full academic content generation pipeline.

    Pipeline stages:
    1. score_topics — run exam signal engine for the course
    2. generate_content — generate highlights/notes/MCQs
    3. render_pdfs — render to PDF files

    All operations are course-scoped via course_id and workspace_id.
    """

    async def submit_job(self, course_id: str, output_types: list[str] | None = None,
                          job_type: str = "full", options: dict | None = None) -> dict:
        """Submit a new generation job for a course."""
        course = await acad_get_course(course_id)
        if not course:
            return {"error": "Course not found"}

        valid_types = {"highlighted_handout", "midterm_notes", "final_notes", "mcqs"}
        if output_types:
            output_types = [t for t in output_types if t in valid_types]
        else:
            output_types = list(valid_types)

        job = await acad_create_job(course_id, job_type, output_types)
        job_id = job["id"]

        checkpoint = {
            "options": options or {},
            "ai_logs": [],
            "generated_outputs": [],
            "preview": {
                "processed_pages": [],
                "processed_topics": 0,
                "highlights_applied": 0,
                "live_file_path": "",
                "last_updated": "",
            },
        }
        await acad_update_job(job_id, checkpoint_json=checkpoint)

        # Launch async worker
        task = asyncio.create_task(self._run_pipeline(job_id, course, output_types, options or {}))
        _active_jobs[job_id] = task

        logger.info("Job %s submitted for course %s (types=%s)", job_id, course["code"], output_types)
        return job

    async def retry_job(self, job_id: str) -> dict:
        """Retry a failed job."""
        job = await acad_get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        if job["status"] not in ("failed", "cancelled"):
            return {"error": f"Cannot retry job with status '{job['status']}'"}

        course = await acad_get_course(job["course_id"])
        if not course:
            return {"error": "Course not found"}

        await acad_update_job(job_id, status="pending", progress=0.0,
                               error=None, current_stage="")

        checkpoint = job.get("checkpoint_json") or {}
        options = checkpoint.get("options") or {}
        task = asyncio.create_task(self._run_pipeline(job_id, course, job.get("output_types", []), options))
        _active_jobs[job_id] = task

        return {"job_id": job_id, "status": "retrying"}

    async def pause_job(self, job_id: str) -> dict:
        """Pause a running job (cooperative pause between topic/page steps)."""
        job = await acad_get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        if job.get("status") != "running":
            return {"error": f"Cannot pause job with status '{job.get('status')}'"}

        _paused_jobs.add(job_id)
        await acad_update_job(job_id, status="paused")
        return {"job_id": job_id, "status": "paused"}

    async def resume_job(self, job_id: str) -> dict:
        """Resume a paused job from persisted checkpoint."""
        job = await acad_get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        if job.get("status") != "paused":
            return {"error": f"Cannot resume job with status '{job.get('status')}'"}

        _paused_jobs.discard(job_id)
        await acad_update_job(job_id, status="running")

        if job_id not in _active_jobs:
            course = await acad_get_course(job["course_id"])
            if not course:
                return {"error": "Course not found"}
            checkpoint = job.get("checkpoint_json") or {}
            options = checkpoint.get("options") or {}
            task = asyncio.create_task(self._run_pipeline(job_id, course, job.get("output_types", []), options))
            _active_jobs[job_id] = task

        return {"job_id": job_id, "status": "resumed"}

    async def cancel_job(self, job_id: str) -> dict:
        """Cancel a running job."""
        if job_id in _active_jobs:
            _active_jobs[job_id].cancel()
            del _active_jobs[job_id]

        _paused_jobs.discard(job_id)
        await acad_update_job(job_id, status="cancelled")
        return {"job_id": job_id, "status": "cancelled"}

    async def _run_pipeline(self, job_id: str, course: dict,
                             output_types: list[str], options: dict | None = None):
        """Execute the full generation pipeline for a job."""
        course_id = course["id"]
        course_code = course["code"]
        workspace_id = course["workspace_id"]
        options = options or {}
        stages_completed: list[str] = []
        ai_logs: list[dict] = []

        try:
            async def _wait_if_paused():
                while job_id in _paused_jobs:
                    await asyncio.sleep(0.5)

            now_str = datetime.now(timezone.utc).isoformat()
            await acad_update_job(job_id, status="running", started_at=now_str,
                                   progress=0.0)

            existing_job = await acad_get_job(job_id)
            existing_checkpoint = (existing_job or {}).get("checkpoint_json") or {}
            ai_logs = existing_checkpoint.get("ai_logs") or []

            # Stage 1: Score topics
            await _wait_if_paused()
            await acad_update_job(job_id, current_stage="score_topics", progress=0.1)
            ranked_topics = await topic_ranker.rank_topics(course_id)
            stages_completed.append("score_topics")
            await acad_update_job(job_id, progress=0.3,
                                   stages_completed=stages_completed)

            if not ranked_topics:
                await acad_update_job(
                    job_id, status="completed", progress=1.0,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    stages_completed=stages_completed,
                    error="No topics found — upload reviews or check course data",
                )
                return

            # Stage 2: Generate content
            await _wait_if_paused()
            await acad_update_job(job_id, current_stage="generate_content", progress=0.4)
            generated_outputs: list[dict] = []

            if "highlighted_handout" in output_types:
                # Highlighted flow uses sequential page batches + reviews (no top-k retrieval).
                result = {"mode": "review_batch_prediction"}
                generated_outputs.append({
                    "type": "highlighted_handout", "result": result,
                })

            if "midterm_notes" in output_types:
                result = await content_generator.generate_short_notes(
                    course_code, workspace_id, ranked_topics, exam_type="midterm",
                    model=options.get("generation_model"),
                    ai_logs=ai_logs,
                )
                generated_outputs.append({
                    "type": "midterm_notes", "result": result,
                })

            if "final_notes" in output_types:
                result = await content_generator.generate_short_notes(
                    course_code, workspace_id, ranked_topics, exam_type="final",
                    model=options.get("generation_model"),
                    ai_logs=ai_logs,
                )
                generated_outputs.append({
                    "type": "final_notes", "result": result,
                })

            if "mcqs" in output_types:
                result = await mcq_generator.generate_mcqs(
                    course_code, workspace_id, ranked_topics,
                    model=options.get("generation_model"),
                    ai_logs=ai_logs,
                )
                generated_outputs.append({
                    "type": "mcqs", "result": result,
                })

            job_after_generation = await acad_get_job(job_id)
            checkpoint_after_generation = (job_after_generation or {}).get("checkpoint_json") or {}
            checkpoint_after_generation["options"] = options
            checkpoint_after_generation["ai_logs"] = ai_logs
            await acad_update_job(job_id, checkpoint_json=checkpoint_after_generation)

            stages_completed.append("generate_content")
            await acad_update_job(job_id, progress=0.7,
                                   stages_completed=stages_completed)

            # Stage 3: Render PDFs
            await _wait_if_paused()
            await acad_update_job(job_id, current_stage="render_pdfs", progress=0.8)

            # Get next version number
            from storage.database import acad_list_outputs
            existing = await acad_list_outputs(course_id=course_id)
            version = len(existing) + 1

            for gen_output in generated_outputs:
                await _wait_if_paused()
                output_type = gen_output["type"]
                result = gen_output["result"]

                # Quality gate handled per output type.
                topic_count = 0

                if output_type == "highlighted_handout":
                    reviews = await acad_list_reviews(course_id, include_spam=False, limit=500)
                    review_texts = [r.get("review_text", "") for r in reviews if r.get("review_text")]

                    def _progress_cb(page_number: int, highlights_added: int, live_file_path: str = ""):
                        async def _update():
                            job = await acad_get_job(job_id)
                            checkpoint = (job or {}).get("checkpoint_json") or {}
                            preview = checkpoint.get("preview") or {}
                            processed_pages = preview.get("processed_pages") or []
                            if page_number not in processed_pages:
                                processed_pages.append(page_number)
                                processed_pages.sort()
                            preview["processed_pages"] = processed_pages
                            preview["highlights_applied"] = int(preview.get("highlights_applied", 0)) + int(highlights_added)
                            preview["processed_topics"] = len(processed_pages)
                            if live_file_path:
                                preview["live_file_path"] = live_file_path
                            preview["last_updated"] = datetime.now(timezone.utc).isoformat()
                            checkpoint["preview"] = preview
                            checkpoint["options"] = options
                            checkpoint["ai_logs"] = ai_logs
                            await acad_update_job(job_id, checkpoint_json=checkpoint)
                        asyncio.create_task(_update())

                    def _pause_check() -> bool:
                        return job_id in _paused_jobs

                    # Calculate resume point from checkpoint if available
                    resume_batch = 0
                    existing_preview = existing_checkpoint.get("preview", {})
                    if existing_preview.get("last_completed_batch_index"):
                        resume_batch = int(existing_preview["last_completed_batch_index"]) + 1

                    render_out = await pdf_renderer.render_highlighted_handout_review_predicted(
                        course_code,
                        review_texts,
                        version,
                        workspace_id=workspace_id,
                        syllabus_scope=str(options.get("syllabus_scope", "all")),
                        split_mode=str(options.get("split_mode", "auto")),
                        manual_midterm_end_page=options.get("manual_midterm_end_page"),
                        model=options.get("generation_model"),
                        batch_pages=int(options.get("batch_pages", 5)),
                        max_batch_chars=int(options.get("max_batch_chars", 7000)),
                        max_review_evidence_chars=int(options.get("max_review_evidence_chars", 2500)),
                        max_spans_per_batch=int(options.get("max_spans_per_batch", 20)),
                        ai_logs=ai_logs,
                        progress_callback=_progress_cb,
                        pause_check=_pause_check,
                        job_id=job_id,
                        resume_from_batch=resume_batch,
                        debug_mode=bool(options.get("debug_mode", False)),
                        continue_on_error=bool(options.get("continue_on_error", True)),
                        max_highlights_per_page=int(options.get("max_highlights_per_page", 15)),
                    )
                    file_path = render_out.get("file_path", "")
                    topic_count = int(render_out.get("highlight_count", 0))
                    if topic_count == 0:
                        logger.warning(
                            "Highlighted handout generated with zero highlights for job %s (course=%s)",
                            job_id,
                            course_code,
                        )
                        job = await acad_get_job(job_id)
                        checkpoint = (job or {}).get("checkpoint_json") or {}
                        checkpoint["quality_warning"] = "Zero highlights produced for highlighted_handout"
                        await acad_update_job(job_id, checkpoint_json=checkpoint)

                    # Store render statistics in checkpoint
                    job_after_highlights = await acad_get_job(job_id)
                    checkpoint_after_highlights = (job_after_highlights or {}).get("checkpoint_json") or {}
                    checkpoint_after_highlights["options"] = options
                    checkpoint_after_highlights["ai_logs"] = ai_logs
                    checkpoint_after_highlights["render_stats"] = {
                        "highlight_count": render_out.get("highlight_count", 0),
                        "processed_batches": render_out.get("processed_batches", 0),
                        "unmatched_spans": render_out.get("unmatched_spans", 0),
                        "failed_batches": render_out.get("failed_batches", 0),
                        "match_rate": render_out.get("match_rate", 0),
                        "total_spans_returned": render_out.get("total_spans_returned", 0),
                        "debug_log_path": render_out.get("debug_log_path", ""),
                    }
                    if render_out.get("model_warning"):
                        checkpoint_after_highlights["model_warning"] = render_out["model_warning"]
                    await acad_update_job(job_id, checkpoint_json=checkpoint_after_highlights)
                elif output_type in ("midterm_notes", "final_notes"):
                    items = [{"topic_name": i.topic_name, "content": i.content}
                             for i in result.items]
                    exam_type = "midterm" if output_type == "midterm_notes" else "final"
                    file_path = pdf_renderer.render_short_notes(
                        course_code, items, exam_type, version,
                    )
                    topic_count = getattr(result, "total_topics", 0) or getattr(result, "total_count", 0)
                    if topic_count == 0:
                        raise ValueError(f"Quality gate failed: Zero valid items generated for {output_type}")
                elif output_type == "mcqs":
                    mcqs_data = [{"question": m.question, "options": m.options,
                                  "correct_answer": m.correct_answer, "rationale": m.rationale,
                                  "difficulty": m.difficulty}
                                 for m in result.mcqs]
                    file_path = pdf_renderer.render_mcq_bank(
                        course_code, mcqs_data, version,
                    )
                    topic_count = getattr(result, "total_topics", 0) or getattr(result, "total_count", 0)
                    if topic_count == 0:
                        raise ValueError(f"Quality gate failed: Zero valid items generated for {output_type}")
                else:
                    continue

                # Record output in DB
                import os
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                await acad_create_output(
                    course_id=course_id, job_id=job_id,
                    output_type=output_type, file_path=file_path,
                    file_size=file_size, version=version,
                    topic_count=topic_count,
                    generation_params=options,
                )

            stages_completed.append("render_pdfs")
            await acad_update_job(
                job_id, status="completed", progress=1.0,
                completed_at=datetime.now(timezone.utc).isoformat(),
                stages_completed=stages_completed,
                current_stage="done",
            )

            logger.info("Job %s completed for %s (outputs=%d)", job_id, course_code, len(generated_outputs))

        except asyncio.CancelledError:
            await acad_update_job(job_id, status="cancelled",
                                   stages_completed=stages_completed)
            logger.info("Job %s cancelled", job_id)

        except Exception as e:
            logger.error("Job %s failed: %s", job_id, e, exc_info=True)
            await acad_update_job(
                job_id, status="failed", error=str(e),
                stages_completed=stages_completed,
            )

        finally:
            _active_jobs.pop(job_id, None)


# Singleton
job_orchestrator = JobOrchestrator()
