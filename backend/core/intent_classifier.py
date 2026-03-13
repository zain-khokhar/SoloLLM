"""
Intent-based context classifier for smart context management.

Classifies user prompts to determine the optimal history depth and
whether RAG retrieval is needed. Uses zero-latency regex patterns
instead of LLM calls to avoid per-message inference costs.

Intent categories:
  - standalone: No history needed (greetings, general knowledge, creative)
  - recent_context: Last 4-6 messages (follow-ups, pronoun refs)
  - full_history: All messages (explicit references to earlier chat)
  - rag_retrieval: Document search needed (references to uploaded files)
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Pattern sets ────────────────────────────────────────────────

FULL_HISTORY_PATTERNS = [
    r"\b(first|earliest|initial|original)\s+(question|message|prompt|thing)\b",
    r"\bearlier\s+(you|i|we)\s+(said|wrote|mentioned|asked|discussed)\b",
    r"\bremember\s+when\b",
    r"\bwhat\s+(did|was)\s+(i|we|you)\s+(say|ask|write|discuss|mention)\b",
    r"\bgo\s+back\s+to\b",
    r"\bprevious(ly)?\b",
    r"\b(beginning|start)\s+of\s+(this|our|the)\s+(conversation|chat|session)\b",
    r"\ball\s+(of\s+)?(our|the)\s+(conversation|chat|messages)\b",
    r"\bsummar(y|ize)\s+(of\s+)?(our|this|the)\s+(conversation|chat)\b",
    r"\bhow\s+many\s+(times|questions|messages)\b",
    r"\bwhat\s+has\b.+\b(so\s+far|until\s+now)\b",
    r"\brecap\b",
]

RAG_PATTERNS = [
    r"\b(in|from|according\s+to)\s+the\s+(document|file|pdf|paper|upload|attachment)\b",
    r"\bthe\s+(document|file|pdf)\s+(says?|mentions?|states?|contains?)\b",
    r"\bwhat\s+does\s+the\s+(document|file|pdf|paper)\b",
    r"\b(search|look|find)\s+(in|through|from)\s+(the\s+)?(document|file|pdf|upload)\b",
    r"\bpage\s+\d+\b",
    r"\b(cited?|reference|source)\b.*\b(document|file)\b",
    r"\buploaded?\s+(file|document|pdf)\b",
    r"\baccording\s+to\b.*\buploaded?\b",
]

STANDALONE_PATTERNS = [
    r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|howdy|what'?s?\s+up)\s*[!.?]*$",
    r"^(thanks?|thank\s+you|ty|thx)\s*[!.?]*$",
    r"^(bye|goodbye|see\s+you|later|quit|exit)\s*[!.?]*$",
    r"\b(define|explain|what\s+is|what\s+are|tell\s+me\s+about)\b(?!.*(earlier|before|previous|mentioned|said))",
    r"\b(write|compose|draft|generate|create)\s+(a|an|me)\s+(poem|story|essay|email|letter|song|code|function)\b",
    r"\b(translate|convert)\s+.+\s+(to|into)\s+\w+\b",
    r"\b(how\s+to|how\s+do\s+(i|you))\b(?!.*(earlier|before|previous|mentioned|said|context))",
]

RECENT_CONTEXT_INDICATORS = [
    r"^(yes|no|yeah|nah|sure|okay|ok|yep|nope|exactly|right|correct)\b",
    r"^(this|that|it|these|those)\b",
    r"\b(what\s+about|how\s+about|and\s+also)\b",
    r"\b(instead|also|too|as\s+well)\b",
    r"\b(more|another|else|other)\b.*\?",
    r"\b(why|how|when|where)\b.*\?$",
    r"\b(can\s+you|could\s+you|please)\b.*\b(fix|change|modify|update|redo|try\s+again)\b",
    r"\b(same|similar)\s+(thing|way|approach)\b",
]


def _match_any(text: str, patterns: list[str]) -> bool:
    """Check if text matches any pattern in the list."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def classify(message: str, history_length: int = 0) -> dict:
    """
    Classify a user message to determine context retrieval strategy.

    Returns dict with:
        intent: str - one of "standalone", "recent_context", "full_history", "rag_retrieval"
        history_depth: int - how many messages to include (0 = none, -1 = all)
        needs_rag: bool - whether to run RAG retrieval
        confidence: float - confidence in classification (0-1)
    """
    text = message.strip()

    # Check for RAG-related intent first (highest priority)
    if _match_any(text, RAG_PATTERNS):
        return {
            "intent": "rag_retrieval",
            "history_depth": 4,
            "needs_rag": True,
            "confidence": 0.9,
        }

    # Check for full history references
    if _match_any(text, FULL_HISTORY_PATTERNS):
        return {
            "intent": "full_history",
            "history_depth": -1,  # -1 means ALL messages
            "needs_rag": False,
            "confidence": 0.85,
        }

    # Check standalone patterns
    if _match_any(text, STANDALONE_PATTERNS):
        # If it's a truly new conversation, confidence is higher
        if history_length <= 2:
            return {
                "intent": "standalone",
                "history_depth": 0,
                "needs_rag": False,
                "confidence": 0.9,
            }
        # With existing history, we might still want some context
        return {
            "intent": "standalone",
            "history_depth": 0,
            "needs_rag": False,
            "confidence": 0.7,
        }

    # Check for recent context indicators
    if _match_any(text, RECENT_CONTEXT_INDICATORS):
        return {
            "intent": "recent_context",
            "history_depth": 6,
            "needs_rag": False,
            "confidence": 0.8,
        }

    # Short messages with history are likely follow-ups
    if len(text.split()) <= 5 and history_length > 0:
        return {
            "intent": "recent_context",
            "history_depth": 6,
            "needs_rag": False,
            "confidence": 0.6,
        }

    # Default: include recent context for safety
    if history_length > 0:
        return {
            "intent": "recent_context",
            "history_depth": 10,
            "needs_rag": False,
            "confidence": 0.5,
        }

    return {
        "intent": "standalone",
        "history_depth": 0,
        "needs_rag": False,
        "confidence": 0.5,
    }
