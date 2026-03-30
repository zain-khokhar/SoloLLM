"""
Quality Evaluator for VU Academic Auto-Generation.

Tracks and computes quality metrics for generated content:
topic hit-rate, MCQ quality, hallucination rate, feedback aggregation,
and generates weight adjustment recommendations.
"""

import logging
from dataclasses import dataclass

from storage.database import (
    acad_get_topic_scores, acad_list_outputs, acad_get_feedback,
    acad_get_overview_stats,
)

logger = logging.getLogger(__name__)


@dataclass
class CourseMetrics:
    """Quality metrics for a single course."""
    course_id: str
    topic_count: int = 0
    high_confidence_topics: int = 0
    medium_confidence_topics: int = 0
    low_confidence_topics: int = 0
    avg_exam_probability: float = 0.0
    output_count: int = 0
    avg_feedback_rating: float = 0.0
    feedback_count: int = 0


class QualityEvaluator:
    """
    Evaluates and tracks quality of academic content generation.

    Provides per-course and system-wide quality metrics.
    """

    async def evaluate_course(self, course_id: str) -> CourseMetrics:
        """Compute quality metrics for a single course."""
        scores = await acad_get_topic_scores(course_id)
        outputs = await acad_list_outputs(course_id=course_id)

        # Topic statistics
        high = sum(1 for s in scores if s.get("weight_bucket") == "high")
        medium = sum(1 for s in scores if s.get("weight_bucket") == "medium")
        low = sum(1 for s in scores if s.get("weight_bucket") == "low")
        avg_prob = (sum(s.get("exam_probability", 0) for s in scores) / len(scores)
                    if scores else 0.0)

        # Feedback aggregation
        total_rating = 0.0
        feedback_count = 0
        for output in outputs:
            feedbacks = await acad_get_feedback(output["id"])
            for fb in feedbacks:
                total_rating += fb.get("rating", 0)
                feedback_count += 1

        avg_rating = total_rating / feedback_count if feedback_count > 0 else 0.0

        return CourseMetrics(
            course_id=course_id,
            topic_count=len(scores),
            high_confidence_topics=high,
            medium_confidence_topics=medium,
            low_confidence_topics=low,
            avg_exam_probability=round(avg_prob, 4),
            output_count=len(outputs),
            avg_feedback_rating=round(avg_rating, 2),
            feedback_count=feedback_count,
        )

    async def get_system_overview(self) -> dict:
        """Get system-wide quality metrics."""
        stats = await acad_get_overview_stats()
        return {
            **stats,
            "system_status": "operational",
        }


# Singleton
quality_evaluator = QualityEvaluator()
