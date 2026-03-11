import re
import logging

logger = logging.getLogger(__name__)

# Keyword patterns for query type classification
_SUMMARY_PATTERNS = re.compile(
    r"\b(summarize|summary|overview|outline|breakdown|recap|brief)\b", re.IGNORECASE
)
_DETAIL_PATTERNS = re.compile(
    r"\b(explain|detail|elaborate|describe|in[-\s]?depth|thorough|comprehensive|step[-\s]?by[-\s]?step)\b",
    re.IGNORECASE,
)
_LIST_PATTERNS = re.compile(
    r"\b(list|enumerate|bullet|points|all the|every)\b", re.IGNORECASE
)
_COMPARE_PATTERNS = re.compile(
    r"\b(compare|comparison|difference|versus|vs\.?|contrast)\b", re.IGNORECASE
)
_SHORT_PATTERNS = re.compile(
    r"\b(what is|who is|when|define|yes or no|true or false)\b", re.IGNORECASE
)


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for text.
    Uses a fast heuristic: ~4 chars per token for English.
    This avoids loading tiktoken on every call.
    """
    if not text:
        return 0
    # Rough estimate: 1 token ≈ 4 characters, but also count words
    char_estimate = len(text) / 4
    word_estimate = len(text.split()) * 1.3
    return int((char_estimate + word_estimate) / 2)


def classify_query(query: str) -> str:
    """
    Classify the query type to estimate expected output length.

    Returns one of: 'short', 'medium', 'long', 'very_long'
    """
    if _SHORT_PATTERNS.search(query) and len(query.split()) < 15:
        return "short"
    if _SUMMARY_PATTERNS.search(query):
        return "long"
    if _DETAIL_PATTERNS.search(query):
        return "very_long"
    if _LIST_PATTERNS.search(query):
        return "long"
    if _COMPARE_PATTERNS.search(query):
        return "very_long"
    if len(query.split()) > 30:
        return "medium"
    return "medium"


def estimate_required_tokens(query: str) -> int:
    """Estimate the number of output tokens needed based on query type."""
    query_type = classify_query(query)
    estimates = {
        "short": 200,
        "medium": 600,
        "long": 1200,
        "very_long": 2500,
    }
    return estimates.get(query_type, 600)


def will_likely_truncate(query: str, max_tokens: int) -> bool:
    """Check if the response is likely to exceed max_tokens."""
    estimated = estimate_required_tokens(query)
    return estimated > max_tokens


class TokenTracker:
    """Tracks token count during streaming generation."""

    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
        self.tokens_generated = 0
        self.content = ""

    def add_token(self, token_text: str):
        self.content += token_text
        # Approximate: each streamed chunk is roughly 1 token
        self.tokens_generated += 1

    def set_final_count(self, eval_count: int):
        """Set the actual token count from Ollama's response."""
        if eval_count > 0:
            self.tokens_generated = eval_count

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.tokens_generated)

    @property
    def utilization(self) -> float:
        if self.max_tokens == 0:
            return 0.0
        return self.tokens_generated / self.max_tokens
