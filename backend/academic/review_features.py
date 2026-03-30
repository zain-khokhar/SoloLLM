"""
Review Feature Extraction for VU Academic Auto-Generation.

Extracts structured signals from student reviews for use
in the exam topic prediction engine. Each feature is computed
per-topic within a single course's review set (strict isolation).
"""

import re
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from storage.database import acad_list_reviews, acad_get_topic_scores

logger = logging.getLogger(__name__)


@dataclass
class TopicFeatures:
    """Extracted feature signals for a single topic within a course."""
    topic_name: str
    mention_count: int = 0
    review_count: int = 0  # how many distinct reviews mention this topic
    avg_urgency: float = 0.0
    max_urgency: float = 0.0
    recency_weight: float = 0.0  # higher = mentioned in more recent reviews
    consensus_score: float = 0.0  # agreement across reviewers
    evidence_reviews: list[str] = field(default_factory=list)  # review IDs that mention it


# ── Topic Extraction Heuristics ─────────────────────────────

# Academic topic keywords commonly found in VU reviews
_TOPIC_INDICATORS = [
    r'(?:chapter|lecture|topic|unit)\s*(\d+)',
    r'(?:ch|lec|lect)\s*[-.]?\s*(\d+)',
]

# Keywords that often precede topic names
_TOPIC_PREFIX_RE = re.compile(
    r'(?:focus\s+on|prepare|read|study|cover|learn|skip)\s+(.{5,60}?)(?:\.|,|$)',
    re.IGNORECASE,
)


def extract_mentioned_topics(review_text: str) -> list[str]:
    """
    Extract topic references from a single review.

    Returns a list of topic identifiers found in the text.
    """
    topics = []
    text_lower = review_text.lower()

    # Pattern 1: chapter/lecture numbers
    for pattern in _TOPIC_INDICATORS:
        for match in re.finditer(pattern, text_lower):
            topics.append(f"lecture_{match.group(1)}")

    # Pattern 2: "focus on <topic>" / "prepare <topic>"
    for match in _TOPIC_PREFIX_RE.finditer(review_text):
        topic_text = match.group(1).strip()
        if len(topic_text) > 3:
            # Normalize to lowercase key
            key = re.sub(r'\s+', '_', topic_text.lower().strip('.,;:'))
            topics.append(key)

    return topics


# ── Feature Extraction Engine ───────────────────────────────

class ReviewFeatureExtractor:
    """
    Extracts per-topic feature vectors from a course's reviews.

    All operations are scoped to a single course_id — no cross-course data access.
    """

    async def extract_features(self, course_id: str) -> list[TopicFeatures]:
        """
        Analyze all reviews for a course and extract topic-level features.

        Returns a list of TopicFeatures sorted by mention frequency.
        """
        reviews = await acad_list_reviews(course_id, include_spam=False, limit=1000)

        if not reviews:
            logger.info("No reviews found for course %s", course_id)
            return []

        # Track per-topic signals
        topic_mentions: Counter = Counter()
        topic_review_ids: defaultdict[str, set] = defaultdict(set)
        topic_urgency_scores: defaultdict[str, list] = defaultdict(list)
        topic_recency: defaultdict[str, list] = defaultdict(list)

        total_reviews = len(reviews)

        for idx, review in enumerate(reviews):
            review_text = review.get("review_text", "")
            review_id = review.get("id", "")
            urgency = review.get("urgency_score", 0.0) or 0.0

            # Recency: later reviews get higher weight (0.0 to 1.0)
            recency = (idx + 1) / total_reviews

            # Extract topics mentioned in this review
            mentioned_topics = extract_mentioned_topics(review_text)

            for topic in mentioned_topics:
                topic_mentions[topic] += 1
                topic_review_ids[topic].add(review_id)
                topic_urgency_scores[topic].append(urgency)
                topic_recency[topic].append(recency)

        # Build feature objects
        features = []
        for topic_name, count in topic_mentions.most_common():
            review_ids = list(topic_review_ids[topic_name])
            urgency_scores = topic_urgency_scores[topic_name]
            recency_scores = topic_recency[topic_name]

            avg_urgency = sum(urgency_scores) / len(urgency_scores) if urgency_scores else 0.0
            max_urgency = max(urgency_scores) if urgency_scores else 0.0
            avg_recency = sum(recency_scores) / len(recency_scores) if recency_scores else 0.0

            # Consensus: how many distinct reviews agree on this topic
            # Normalized by total reviews
            consensus = len(review_ids) / total_reviews if total_reviews > 0 else 0.0

            features.append(TopicFeatures(
                topic_name=topic_name,
                mention_count=count,
                review_count=len(review_ids),
                avg_urgency=round(avg_urgency, 4),
                max_urgency=round(max_urgency, 4),
                recency_weight=round(avg_recency, 4),
                consensus_score=round(consensus, 4),
                evidence_reviews=review_ids[:20],  # cap evidence list
            ))

        features.sort(key=lambda f: f.mention_count, reverse=True)
        logger.info(
            "Extracted features for %d topics from %d reviews (course=%s)",
            len(features), total_reviews, course_id,
        )

        return features


# Singleton
review_feature_extractor = ReviewFeatureExtractor()
