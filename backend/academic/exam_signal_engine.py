"""
Exam Signal Engine for VU Academic Auto-Generation.

The core prediction algorithm that computes exam-topic probability scores
using a weighted blend of signals from reviews, syllabus analysis,
and LLM confidence. All scoring is strictly course-scoped.
"""

import logging
from dataclasses import dataclass, field

from storage.database import (
    acad_upsert_topic_score, acad_get_topic_scores,
    acad_get_course, get_setting, set_setting,
)
from academic.review_features import review_feature_extractor

logger = logging.getLogger(__name__)


# ── Default Scoring Weights ─────────────────────────────────

DEFAULT_WEIGHTS = {
    "review_frequency": 0.30,
    "urgency_signal": 0.20,
    "consensus": 0.15,
    "syllabus_importance": 0.15,
    "recency": 0.10,
    "llm_confidence": 0.10,
}


@dataclass
class ScoredTopic:
    """A topic with computed exam probability and evidence."""
    topic_name: str
    exam_probability: float = 0.0
    weight_bucket: str = "low"  # high / medium / low
    midterm_relevance: float = 0.0
    final_relevance: float = 0.0
    evidence: list[dict] = field(default_factory=list)
    # Component scores for transparency
    review_frequency: float = 0.0
    urgency_signal: float = 0.0
    consensus_score: float = 0.0
    syllabus_importance: float = 0.0
    recency_weight: float = 0.0
    llm_confidence: float = 0.0


def _bucket(score: float) -> str:
    """Convert probability to weight bucket."""
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "medium"
    return "low"


def _midterm_final_split(topic_name: str, total_topics: int,
                          topic_rank: int) -> tuple[float, float]:
    """
    Estimate midterm vs final relevance based on topic position.

    Early topics (first half) → midterm; later topics → final.
    Both can overlap for important topics.
    """
    if total_topics == 0:
        return 0.5, 0.5

    position_ratio = topic_rank / total_topics

    if position_ratio < 0.45:
        return 0.8, 0.3
    elif position_ratio < 0.55:
        return 0.5, 0.5
    else:
        return 0.2, 0.8


# ── Exam Signal Engine ──────────────────────────────────────

class ExamSignalEngine:
    """
    Computes exam-topic probability scores using weighted signal blending.

    Formula:
        topic_exam_score = W1*review_freq + W2*urgency + W3*consensus
                         + W4*syllabus_importance + W5*recency + W6*llm_confidence

    All scoring is strictly scoped to a single course.
    """

    async def get_weights(self) -> dict:
        """Get current scoring weights from settings or defaults."""
        raw = await get_setting("academic_scoring_weights")
        if raw:
            try:
                import json
                weights = json.loads(raw)
                if isinstance(weights, dict):
                    return {**DEFAULT_WEIGHTS, **weights}
            except Exception:
                pass
        return DEFAULT_WEIGHTS.copy()

    async def update_weights(self, weights: dict) -> dict:
        """Update scoring weights in settings."""
        import json
        merged = {**DEFAULT_WEIGHTS, **weights}
        # Normalize to sum to 1.0
        total = sum(merged.values())
        if total > 0:
            merged = {k: v / total for k, v in merged.items()}
        await set_setting("academic_scoring_weights", json.dumps(merged))
        return merged

    async def score_course(self, course_id: str) -> list[ScoredTopic]:
        """
        Score all topics for a course based on review features.

        This is the main prediction entry point.
        Steps:
        1. Extract features from course reviews
        2. Apply weighted scoring formula
        3. Persist scores to DB
        4. Return ranked topics
        """
        weights = await self.get_weights()

        # Step 1: Extract features from this course's reviews only
        features = await review_feature_extractor.extract_features(course_id)

        if not features:
            logger.info("No topic features found for course %s", course_id)
            return []

        # Find max values for normalization
        max_mentions = max(f.mention_count for f in features) if features else 1
        max_reviews = max(f.review_count for f in features) if features else 1

        # Step 2: Score each topic
        scored_topics = []
        for rank, feature in enumerate(features):
            # Normalize frequency to 0-1
            norm_freq = feature.mention_count / max_mentions if max_mentions > 0 else 0
            norm_consensus = feature.consensus_score

            # Compute weighted score
            exam_score = (
                weights["review_frequency"] * norm_freq
                + weights["urgency_signal"] * feature.avg_urgency
                + weights["consensus"] * norm_consensus
                + weights["syllabus_importance"] * 0.5  # placeholder until syllabus analysis
                + weights["recency"] * feature.recency_weight
                + weights["llm_confidence"] * 0.5  # placeholder until LLM scoring
            )

            exam_score = min(1.0, max(0.0, exam_score))

            # Midterm/final split
            mid_rel, fin_rel = _midterm_final_split(
                feature.topic_name, len(features), rank,
            )

            # Build evidence
            evidence = [{
                "type": "review_mentions",
                "count": feature.mention_count,
                "review_ids": feature.evidence_reviews[:5],
            }]

            scored = ScoredTopic(
                topic_name=feature.topic_name,
                exam_probability=round(exam_score, 4),
                weight_bucket=_bucket(exam_score),
                midterm_relevance=round(mid_rel * exam_score, 4),
                final_relevance=round(fin_rel * exam_score, 4),
                review_frequency=round(norm_freq, 4),
                urgency_signal=round(feature.avg_urgency, 4),
                consensus_score=round(norm_consensus, 4),
                syllabus_importance=0.5,  # placeholder
                recency_weight=round(feature.recency_weight, 4),
                llm_confidence=0.5,  # placeholder
                evidence=evidence,
            )
            scored_topics.append(scored)

            # Step 3: Persist to DB
            await acad_upsert_topic_score(
                course_id=course_id,
                topic_name=feature.topic_name,
                exam_probability=scored.exam_probability,
                weight_bucket=scored.weight_bucket,
                midterm_relevance=scored.midterm_relevance,
                final_relevance=scored.final_relevance,
                review_frequency=scored.review_frequency,
                urgency_signal=scored.urgency_signal,
                consensus_score=scored.consensus_score,
                syllabus_importance=scored.syllabus_importance,
                recency_weight=scored.recency_weight,
                llm_confidence=scored.llm_confidence,
                evidence=evidence,
            )

        # Sort by exam probability descending
        scored_topics.sort(key=lambda s: s.exam_probability, reverse=True)

        logger.info(
            "Scored %d topics for course %s (top: %s @ %.2f)",
            len(scored_topics), course_id,
            scored_topics[0].topic_name if scored_topics else "N/A",
            scored_topics[0].exam_probability if scored_topics else 0.0,
        )

        return scored_topics

    async def get_cached_scores(self, course_id: str) -> list[dict]:
        """Get previously computed scores from DB."""
        return await acad_get_topic_scores(course_id)


# Singleton
exam_signal_engine = ExamSignalEngine()
