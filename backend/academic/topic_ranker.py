"""
Topic Ranker for VU Academic Auto-Generation.

Combines syllabus segmentation with ExamSignalEngine scoring
to produce the final ranked topic list for content generation.
All operations scoped to a single course.
"""

import logging
from dataclasses import dataclass, field

from academic.exam_signal_engine import exam_signal_engine, ScoredTopic
from academic.syllabus_segmenter import syllabus_segmenter, SyllabusMap
from storage.database import acad_get_topic_scores

logger = logging.getLogger(__name__)


@dataclass
class RankedTopic:
    """A fully ranked topic ready for content generation."""
    topic_name: str
    rank: int = 0
    exam_probability: float = 0.0
    weight_bucket: str = "low"
    midterm_relevance: float = 0.0
    final_relevance: float = 0.0
    page_reference: int | None = None
    content_preview: str = ""
    evidence: list[dict] = field(default_factory=list)
    # Flags for generation
    generate_highlight: bool = False
    generate_notes: bool = False
    generate_mcq: bool = False


class TopicRanker:
    """
    Produces the final ranked topic list for a course.

    Merges exam signal scores with syllabus structure data.
    """

    async def rank_topics(self, course_id: str,
                           min_probability: float = 0.1) -> list[RankedTopic]:
        """
        Get ranked topics for a course, filtered by minimum probability.

        If scores haven't been computed yet, triggers scoring first.
        """
        # Get or compute scores
        scores = await acad_get_topic_scores(course_id)

        if not scores:
            # Trigger scoring
            scored = await exam_signal_engine.score_course(course_id)
            if not scored:
                return []
            scores = await acad_get_topic_scores(course_id)

        # Filter and rank
        ranked = []
        for rank_idx, score in enumerate(scores):
            prob = score.get("exam_probability", 0.0)
            if prob < min_probability:
                continue

            bucket = score.get("weight_bucket", "low")

            ranked.append(RankedTopic(
                topic_name=score["topic_name"],
                rank=rank_idx + 1,
                exam_probability=prob,
                weight_bucket=bucket,
                midterm_relevance=score.get("midterm_relevance", 0.0),
                final_relevance=score.get("final_relevance", 0.0),
                evidence=score.get("evidence", []),
                generate_highlight=bucket in ("high", "medium"),
                generate_notes=bucket in ("high", "medium"),
                generate_mcq=bucket == "high",
            ))

        logger.info(
            "Ranked %d topics for course %s (min_prob=%.2f)",
            len(ranked), course_id, min_probability,
        )

        return ranked


# Singleton
topic_ranker = TopicRanker()
