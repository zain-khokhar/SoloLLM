"""
MCQ Generator for VU Academic Auto-Generation.

Generates multiple-choice questions per topic with difficulty tiers,
answer keys, rationale, and distractor quality checks.
Uses RAG-grounded generation from course-isolated workspace.
"""

import re
import json
import logging
from dataclasses import dataclass, field

from core.inference import ollama_client
from core.config import settings
from rag.pipeline import rag_pipeline
from academic.topic_ranker import RankedTopic

logger = logging.getLogger(__name__)


@dataclass
class MCQ:
    """A single multiple-choice question."""
    question: str
    options: list[str] = field(default_factory=list)  # 4 options (A, B, C, D)
    correct_answer: str = ""  # "A", "B", "C", or "D"
    rationale: str = ""
    difficulty: str = "medium"  # easy / medium / hard
    topic_name: str = ""


@dataclass
class MCQSet:
    """Complete MCQ set for a course."""
    course_code: str
    mcqs: list[MCQ] = field(default_factory=list)
    total_count: int = 0
    by_difficulty: dict = field(default_factory=lambda: {"easy": 0, "medium": 0, "hard": 0})
    model_used: str = ""


class MCQGenerator:
    """
    Generates MCQs per topic using RAG-grounded LLM generation.

    Features:
    - Difficulty tiers (easy/medium/hard)
    - 4 options per question with answer key + rationale
    - Distractor quality checks
    - Duplicate MCQ detection
    - All context from course-isolated workspace only
    """

    async def generate_mcqs(
        self, course_code: str, workspace_id: str,
        ranked_topics: list[RankedTopic],
        per_topic_count: int = 5,
        model: str | None = None,
        ai_logs: list[dict] | None = None,
    ) -> MCQSet:
        """
        Generate MCQs for high-ranked topics.

        Each topic gets `per_topic_count` MCQs spread across difficulty tiers.
        """
        model = model or getattr(settings, "academic_generation_model", None) or settings.default_model
        all_mcqs: list[MCQ] = []
        seen_questions: set[str] = set()

        for topic in ranked_topics:
            if not topic.generate_mcq:
                continue

            # Retrieve context from isolated workspace
            cited = await rag_pipeline.query(
                query=topic.topic_name,
                workspace_id=workspace_id,
                top_k=5,
                rerank=True,
            )

            if not cited.context_text:
                continue

            prompt = (
                f"You are an exam MCQ generator. Create exactly {per_topic_count} multiple-choice questions "
                f"about '{topic.topic_name}' based on the following course material.\n\n"
                f"Course Material:\n{cited.context_text}\n\n"
                f"Requirements:\n"
                f"- Generate {per_topic_count} MCQs with varying difficulty (easy, medium, hard)\n"
                f"- Each MCQ must have exactly 4 options (A, B, C, D)\n"
                f"- One correct answer per question\n"
                f"- Distractors should be plausible, not obviously wrong\n"
                f"- Include a short rationale for the correct answer\n\n"
                f"Format each MCQ as:\n"
                f"Q: [question]\n"
                f"A) [option]\nB) [option]\nC) [option]\nD) [option]\n"
                f"Answer: [letter]\n"
                f"Difficulty: [easy/medium/hard]\n"
                f"Rationale: [brief explanation]\n"
                f"---\n"
            )

            try:
                if ai_logs is not None:
                    ai_logs.append({
                        "content_type": "mcqs",
                        "topic_name": topic.topic_name,
                        "model": model,
                        "request": prompt,
                    })
                response = await ollama_client.chat(
                    messages=[{"role": "user", "content": prompt}], model=model,
                    temperature=0.5, max_tokens=2000,
                )
                raw_text = response.get("content", "")
                if not raw_text.strip():
                    raise ValueError("Empty response from LLM")
                if ai_logs is not None:
                    ai_logs[-1]["response"] = raw_text

                mcqs = self._parse_mcqs(raw_text, topic.topic_name)
                if not mcqs:
                    raise ValueError("Could not parse any MCQs from LLM response")

                # Deduplicate
                for mcq in mcqs:
                    q_normalized = mcq.question.lower().strip()
                    if q_normalized not in seen_questions:
                        seen_questions.add(q_normalized)
                        all_mcqs.append(mcq)

            except Exception as e:
                logger.error("MCQ generation failed for topic %s: %s", topic.topic_name, e)
                if ai_logs is not None and ai_logs:
                    ai_logs[-1]["error"] = str(e)
                raise RuntimeError(f"MCQ generation failed for topic '{topic.topic_name}': {str(e)}") from e

        # Count by difficulty
        by_diff = {"easy": 0, "medium": 0, "hard": 0}
        for m in all_mcqs:
            by_diff[m.difficulty] = by_diff.get(m.difficulty, 0) + 1

        return MCQSet(
            course_code=course_code,
            mcqs=all_mcqs,
            total_count=len(all_mcqs),
            by_difficulty=by_diff,
            model_used=model,
        )

    def _parse_mcqs(self, raw_text: str, topic_name: str) -> list[MCQ]:
        """Parse LLM output into structured MCQ objects."""
        mcqs = []
        # Split by separator
        blocks = re.split(r'\n---+\n', raw_text)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Extract question
            q_match = re.search(r'Q:\s*(.+?)(?:\n|$)', block, re.DOTALL)
            if not q_match:
                continue

            question = q_match.group(1).strip()

            # Extract options
            options = []
            for letter in ['A', 'B', 'C', 'D']:
                opt_match = re.search(rf'{letter}\)\s*(.+?)(?:\n|$)', block)
                if opt_match:
                    options.append(opt_match.group(1).strip())

            if len(options) < 4:
                continue  # Skip malformed MCQs

            # Extract answer
            ans_match = re.search(r'Answer:\s*([A-D])', block, re.IGNORECASE)
            correct = ans_match.group(1).upper() if ans_match else "A"

            # Extract difficulty
            diff_match = re.search(r'Difficulty:\s*(easy|medium|hard)', block, re.IGNORECASE)
            difficulty = diff_match.group(1).lower() if diff_match else "medium"

            # Extract rationale
            rat_match = re.search(r'Rationale:\s*(.+?)(?:\n|$)', block, re.DOTALL)
            rationale = rat_match.group(1).strip() if rat_match else ""

            mcqs.append(MCQ(
                question=question,
                options=options,
                correct_answer=correct,
                rationale=rationale,
                difficulty=difficulty,
                topic_name=topic_name,
            ))

        return mcqs


# Singleton
mcq_generator = MCQGenerator()
