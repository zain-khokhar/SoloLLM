"""
Content Generators for VU Academic Auto-Generation.

Generates highlighted handouts, midterm/final short notes using
the local Ollama LLM. All generation is scoped to a single course's
topics and RAG context.
"""

import logging
from dataclasses import dataclass, field

from core.inference import ollama_client
from core.config import settings
from rag.pipeline import rag_pipeline
from academic.topic_ranker import RankedTopic

logger = logging.getLogger(__name__)


@dataclass
class GeneratedContent:
    """Result of content generation for a single topic."""
    topic_name: str
    content_type: str  # "highlight", "midterm_notes", "final_notes"
    content: str = ""
    importance_rank: int = 0
    source_reference: str = ""
    reason_tag: str = ""  # "review-driven" / "syllabus-core" / "mixed"


@dataclass
class GenerationResult:
    """Complete generation result for a course."""
    course_code: str
    content_type: str
    items: list[GeneratedContent] = field(default_factory=list)
    total_topics: int = 0
    model_used: str = ""


class ContentGenerator:
    """
    Generates academic content using local LLM via Ollama.

    All RAG queries are scoped to the course's isolated workspace.
    """

    async def generate_highlighted_handout(
        self, course_code: str, workspace_id: str,
        ranked_topics: list[RankedTopic], model: str | None = None,
        max_context_chars: int = 2200,
        retrieval_top_k: int = 5,
    ) -> GenerationResult:
        """
        Generate highlighted handout excerpts for top-ranked topics.

        For each high/medium topic, retrieves relevant chunks from the
        course's isolated vector store and asks the LLM to extract exact
        source snippets that can be highlighted in the original PDF.
        """
        model = model or getattr(settings, "academic_generation_model", None) or settings.default_model
        items = []

        for topic in ranked_topics:
            if not topic.generate_highlight:
                continue

            # Retrieve context from this course's isolated workspace only
            cited = await rag_pipeline.query(
                query=topic.topic_name,
                workspace_id=workspace_id,
                top_k=retrieval_top_k,
                rerank=True,
            )

            if not cited.context_text:
                continue

            context_text = cited.context_text[:max_context_chars]

            prompt = (
                f"You are a strict PDF highlighter planner for exams.\n"
                f"Topic: {topic.topic_name}\n\n"
                f"From the SOURCE_CONTEXT below, extract 4 to 10 exact snippets that should be yellow-highlighted.\n"
                f"Rules:\n"
                f"- Return only snippets that appear verbatim in SOURCE_CONTEXT\n"
                f"- Keep each snippet between 25 and 220 characters\n"
                f"- Prefer definitions, formulas, key rules, and exam-critical statements\n"
                f"- No paraphrasing, no commentary, no markdown\n"
                f"- Output one snippet per line\n\n"
                f"SOURCE_CONTEXT:\n{context_text}\n"
            )

            try:
                response = await ollama_client.chat(
                    messages=[{"role": "user", "content": prompt}], model=model,
                    temperature=0.1, max_tokens=700,
                )
                content = response.get("content", "").strip()
                if not content:
                    raise ValueError("Empty response from LLM")

                # Keep only plausible snippet lines and cap volume for fast page search.
                snippet_lines = []
                for line in content.splitlines():
                    clean = line.strip().strip("-*")
                    if 20 <= len(clean) <= 240:
                        snippet_lines.append(clean)
                if not snippet_lines:
                    raise ValueError("No valid highlight snippets extracted")
                content = "\n".join(snippet_lines[:12])
            except Exception as e:
                logger.error("LLM generation failed for topic %s: %s", topic.topic_name, e)
                raise RuntimeError(f"Generation failed for topic '{topic.topic_name}': {str(e)}") from e

            reason_tag = "review-driven" if topic.evidence else "syllabus-core"

            items.append(GeneratedContent(
                topic_name=topic.topic_name,
                content_type="highlight",
                content=content,
                importance_rank=topic.rank,
                reason_tag=reason_tag,
            ))

        return GenerationResult(
            course_code=course_code,
            content_type="highlighted_handout",
            items=items,
            total_topics=len(items),
            model_used=model,
        )

    async def generate_short_notes(
        self, course_code: str, workspace_id: str,
        ranked_topics: list[RankedTopic], exam_type: str = "midterm",
        model: str | None = None,
        ai_logs: list[dict] | None = None,
    ) -> GenerationResult:
        """
        Generate short notes for midterm or final exam.

        Only uses topics with matching relevance (midterm_relevance or final_relevance).
        """
        model = model or getattr(settings, "academic_generation_model", None) or settings.default_model
        items = []

        for topic in ranked_topics:
            if not topic.generate_notes:
                continue

            # Filter by exam type
            relevance = (topic.midterm_relevance if exam_type == "midterm"
                         else topic.final_relevance)
            if relevance < 0.1:
                continue

            # Retrieve from isolated workspace
            cited = await rag_pipeline.query(
                query=topic.topic_name,
                workspace_id=workspace_id,
                top_k=5,
                rerank=True,
            )

            if not cited.context_text:
                continue

            prompt = (
                f"You are an exam preparation assistant. Create concise {exam_type} short notes "
                f"about '{topic.topic_name}' based on the following course material.\n\n"
                f"Course Material:\n{cited.context_text}\n\n"
                f"Generate short notes that include:\n"
                f"1. High-probability exam topics\n"
                f"2. Concise exam-focused explanation\n"
                f"3. Likely conceptual traps to watch for\n"
                f"4. Quick revision bullet points\n\n"
                f"Be precise and exam-oriented. No fluff."
            )

            try:
                if ai_logs is not None:
                    ai_logs.append({
                        "content_type": f"{exam_type}_notes",
                        "topic_name": topic.topic_name,
                        "model": model,
                        "request": prompt,
                    })
                response = await ollama_client.chat(
                    messages=[{"role": "user", "content": prompt}], model=model,
                    temperature=0.3, max_tokens=1000,
                )
                content = response.get("content", "").strip()
                if not content:
                    raise ValueError("Empty response from LLM")
                if ai_logs is not None:
                    ai_logs[-1]["response"] = content
            except Exception as e:
                logger.error("Short notes generation failed for %s: %s", topic.topic_name, e)
                if ai_logs is not None and ai_logs:
                    ai_logs[-1]["error"] = str(e)
                raise RuntimeError(f"Generation failed for topic '{topic.topic_name}': {str(e)}") from e

            items.append(GeneratedContent(
                topic_name=topic.topic_name,
                content_type=f"{exam_type}_notes",
                content=content,
                importance_rank=topic.rank,
                reason_tag="mixed",
            ))

        return GenerationResult(
            course_code=course_code,
            content_type=f"{exam_type}_notes",
            items=items,
            total_topics=len(items),
            model_used=model,
        )


# Singleton
content_generator = ContentGenerator()
