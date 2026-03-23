"""
Query Understanding Module for SoloLLM.

Lightweight, deterministic query analyzer using regex + heuristics.
Extracts query type, keyphrases, required/optional terms, and
ambiguity score — no LLM call needed.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Common stop-words that should never be treated as required terms
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "about", "also", "and", "but", "or", "if", "while", "this",
    "that", "these", "those", "it", "its", "i", "me", "my", "we", "us",
    "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "what", "which", "who", "whom", "whose",
})

# Generic terms that inflate recall without adding precision
_GENERIC_TERMS = frozenset({
    "function", "method", "class", "object", "type", "data", "value",
    "system", "process", "model", "example", "result", "output", "input",
    "file", "code", "error", "list", "set", "get", "use", "make",
    "thing", "way", "part", "point", "case", "work", "run", "call",
    "step", "level", "form", "note", "info", "text", "line", "name",
})


@dataclass
class QueryAnalysis:
    """Result of query understanding."""
    query_type: str = "default"           # factual / definition / procedural / code / comparison
    keyphrases: list[str] = field(default_factory=list)  # multi-word phrases
    required_terms: list[str] = field(default_factory=list)
    optional_terms: list[str] = field(default_factory=list)
    ambiguity_score: float = 0.0          # 0.0 = precise, 1.0 = very ambiguous


class QueryAnalyzer:
    """
    Deterministic query analyzer.

    Extracts structured information from a raw query string
    to guide retrieval precision.
    """

    # ── Query-type classification signals ───────────────────

    _TYPE_SIGNALS: dict[str, list[str]] = {
        "code": [
            "code", "implement", "bug", "error", "syntax", "api",
            "debug", "script", "library", "import", "module", "package",
            "compile", "runtime", "exception", "stack trace",
        ],
        "comparison": [
            "compare", "difference", "versus", "vs", "better",
            "pros and cons", "advantages", "contrast", "tradeoff",
        ],
        "definition": [
            "what is", "define", "meaning of", "definition",
            "what does", "what are",
        ],
        "factual": [
            "who is", "when", "where", "how many", "how much",
            "name the", "list the", "list all",
        ],
        "procedural": [
            "how to", "how do", "steps to", "guide", "tutorial",
            "walkthrough", "instructions", "setup", "install", "configure",
        ],
    }

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a query and return structured metadata."""
        q = query.strip()
        if not q:
            return QueryAnalysis()

        query_type = self._classify_type(q)
        keyphrases = self._extract_keyphrases(q)
        required, optional = self._split_terms(q, keyphrases)
        ambiguity = self._compute_ambiguity(required, optional)

        analysis = QueryAnalysis(
            query_type=query_type,
            keyphrases=keyphrases,
            required_terms=required,
            optional_terms=optional,
            ambiguity_score=round(ambiguity, 3),
        )

        logger.debug(
            "QueryAnalysis: type=%s, keyphrases=%s, required=%s, "
            "optional=%s, ambiguity=%.3f",
            analysis.query_type,
            analysis.keyphrases,
            analysis.required_terms,
            analysis.optional_terms,
            analysis.ambiguity_score,
        )
        return analysis

    # ── Internal helpers ────────────────────────────────────

    def _classify_type(self, query: str) -> str:
        q = query.lower()
        for qtype, signals in self._TYPE_SIGNALS.items():
            for signal in signals:
                if signal in q:
                    return qtype
        return "default"

    def _extract_keyphrases(self, query: str) -> list[str]:
        """
        Extract multi-word keyphrases from the query.

        Sources:
        1. Explicitly quoted phrases ("function calling")
        2. Bigrams/trigrams of non-stop-words
        """
        phrases: list[str] = []

        # 1. Quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        phrases.extend(p.strip().lower() for p in quoted if len(p.strip().split()) >= 2)

        # 2. Build bigrams/trigrams from remaining text
        clean = re.sub(r'"[^"]*"', "", query)  # remove quoted parts
        words = [w.lower() for w in re.findall(r"\w+", clean) if w.lower() not in _STOP_WORDS]

        # Bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if words[i] not in _GENERIC_TERMS or words[i+1] not in _GENERIC_TERMS:
                phrases.append(bigram)

        # Trigrams
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            phrases.append(trigram)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for p in phrases:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def _split_terms(
        self, query: str, keyphrases: list[str]
    ) -> tuple[list[str], list[str]]:
        """
        Split query tokens into required vs optional.

        Required: non-stop, non-generic content words (or words that
        are part of a keyphrase).
        Optional: generic single words that are not part of any keyphrase.
        """
        words = [w.lower() for w in re.findall(r"\w+", query)]
        keyphrase_words = set()
        for kp in keyphrases:
            keyphrase_words.update(kp.split())

        required: list[str] = []
        optional: list[str] = []

        for word in words:
            if word in _STOP_WORDS:
                continue
            if len(word) <= 1:
                continue

            if word in keyphrase_words:
                # Part of a keyphrase → required
                if word not in required:
                    required.append(word)
            elif word in _GENERIC_TERMS:
                if word not in optional:
                    optional.append(word)
            else:
                if word not in required:
                    required.append(word)

        return required, optional

    def _compute_ambiguity(
        self, required: list[str], optional: list[str]
    ) -> float:
        """
        Ambiguity score: high when query is dominated by generic terms.

        0.0 = all terms are precise
        1.0 = all terms are generic/optional
        """
        total = len(required) + len(optional)
        if total == 0:
            return 1.0
        generic_ratio = len(optional) / total
        # Boost ambiguity if very few required terms
        shortness_penalty = max(0.0, 1.0 - len(required) / 3)
        return min(1.0, generic_ratio * 0.6 + shortness_penalty * 0.4)


# Singleton
query_analyzer = QueryAnalyzer()
