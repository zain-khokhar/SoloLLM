import re
import logging

from core.config import settings

logger = logging.getLogger(__name__)

# Sentence ending markers
_SENTENCE_ENDS = re.compile(r'[.!?…]\s*$')
# Code block markers
_CODE_BLOCK_OPEN = re.compile(r'```\w*\s*$')
_CODE_BLOCK_CLOSE = re.compile(r'```\s*$')
# Incomplete list item
_LIST_INCOMPLETE = re.compile(r'^\s*[-*\d]+[.)]\s+\S+.*[^.!?]\s*$', re.MULTILINE)
# Heading pattern (markdown)
_HEADING = re.compile(r'^#{1,6}\s+', re.MULTILINE)


def detect_truncation(content: str, done_reason: str, eval_count: int, max_tokens: int) -> dict:
    """
    Analyze if a response was truncated and needs continuation.

    Returns:
        {
            "is_truncated": bool,
            "reason": str,          # "token_limit", "incomplete_sentence", "incomplete_code", "incomplete_list"
            "confidence": float,    # 0.0-1.0
            "last_content": str,    # last N chars for continuation context
        }
    """
    result = {
        "is_truncated": False,
        "reason": "complete",
        "confidence": 0.0,
        "last_content": content[-settings.continuation_overlap_chars:] if content else "",
    }

    if not content or not content.strip():
        return result

    content_stripped = content.rstrip()

    # 1. Ollama explicitly says it hit the token limit
    if done_reason == "length" or (max_tokens > 0 and eval_count >= max_tokens - 5):
        result["is_truncated"] = True
        result["reason"] = "token_limit"
        result["confidence"] = 0.95
        return result

    # 2. Check for incomplete code blocks (opened but not closed)
    open_blocks = len(re.findall(r'```\w*', content))
    close_blocks = len(re.findall(r'```\s*$', content, re.MULTILINE))
    # Count standalone ``` on their own line as close blocks too
    close_blocks += content.count('\n```\n')
    if open_blocks > close_blocks:
        result["is_truncated"] = True
        result["reason"] = "incomplete_code"
        result["confidence"] = 0.9
        return result

    # 3. Check if it ends mid-sentence
    if not _SENTENCE_ENDS.search(content_stripped):
        # Doesn't end with sentence-ending punctuation
        # But check if it ends with a valid structural end (heading, list end, etc.)
        last_line = content_stripped.split('\n')[-1].strip()

        # If the last line looks like a heading or structural element, probably truncated
        if _HEADING.match(last_line):
            result["is_truncated"] = True
            result["reason"] = "incomplete_section"
            result["confidence"] = 0.85
            return result

        # If it's just a short answer that doesn't need punctuation, don't flag
        if len(content_stripped.split()) < 10:
            return result

        result["is_truncated"] = True
        result["reason"] = "incomplete_sentence"
        result["confidence"] = 0.7
        return result

    # 4. Check if we used a high percentage of token budget (suggests model wanted to say more)
    if max_tokens > 0 and eval_count > 0:
        utilization = eval_count / max_tokens
        if utilization > 0.9:
            # Used >90% of budget — might have self-truncated
            result["is_truncated"] = True
            result["reason"] = "near_limit"
            result["confidence"] = 0.5
            return result

    return result


def build_continuation_messages(
    original_messages: list[dict],
    assistant_content_so_far: str,
    overlap_chars: int | None = None,
) -> list[dict]:
    """
    Build the message list for a continuation request.

    Strategy:
    - Keep the original conversation (system + user messages)
    - Add the assistant's partial response as an assistant message
    - Add a user message instructing the model to continue
    """
    overlap = overlap_chars or settings.continuation_overlap_chars
    last_chunk = assistant_content_so_far[-overlap:] if assistant_content_so_far else ""

    messages = []

    # Copy original messages (system prompt, user messages, previous assistant messages)
    for msg in original_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add the truncated assistant response
    messages.append({"role": "assistant", "content": assistant_content_so_far})

    # Add continuation instruction
    continuation_instruction = (
        "Continue your response from exactly where you stopped. "
        "Do not repeat any content you have already written. "
        "Do not add any preamble or acknowledgement. "
        "Continue directly from this point:\n\n"
        f"...{last_chunk}"
    )
    messages.append({"role": "user", "content": continuation_instruction})

    return messages


def stitch_content(original: str, continuation: str) -> str:
    """
    Stitch the original content and continuation together,
    removing any overlap or repeated content at the boundary.
    """
    if not original:
        return continuation
    if not continuation:
        return original

    continuation = continuation.lstrip()

    # Check if the continuation starts with repeated content from the end of original
    # Try different overlap lengths
    original_end = original[-300:] if len(original) > 300 else original

    best_overlap = 0
    for i in range(min(len(original_end), len(continuation)), 0, -1):
        if original_end.endswith(continuation[:i]):
            best_overlap = i
            break

    if best_overlap > 0:
        continuation = continuation[best_overlap:]

    # Ensure clean join
    if original.endswith(" ") or continuation.startswith(" "):
        return original + continuation
    else:
        return original + continuation
