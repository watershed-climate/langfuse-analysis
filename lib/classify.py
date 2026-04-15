"""Trace classification: user type, prompt type, rejection detection.

Classification uses signals from Langfuse trace data:
- `tags` (top-level): e.g. ['customer', 'customer-org'] or ['watershed-employee', 'demo-org']
- `environment`: 'production', 'development', 'evals'
- `metadata.isDemoOrg` / `metadata.isTestOrg`: boolean flags
- `metadata.orgName`: real company name vs test org names
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict


class TraceDict(TypedDict, total=False):
    """Shape of the dict produced by langfuse_client._trace_to_dict."""

    id: str
    name: str | None
    session_id: str | None
    user_id: str | None
    tags: list[str]
    environment: str | None
    metadata: dict[str, Any]
    input: Any
    output: Any
    timestamp: str | None


# ---------------------------------------------------------------------------
# User / trace classification
# ---------------------------------------------------------------------------


class TraceCategory(str, Enum):
    CUSTOMER = "customer"
    INTERNAL = "internal"
    EVAL = "eval"
    UNKNOWN = "unknown"


def classify_trace(trace: TraceDict) -> TraceCategory:
    """Classify a trace based on tags, environment, and metadata."""
    tags = set(trace.get("tags") or [])
    meta = trace.get("metadata") or {}
    env = trace.get("environment") or ""
    user_id = trace.get("user_id") or ""

    if env == "evals" or user_id == "eval_user":
        return TraceCategory.EVAL

    if "watershed-employee" in tags:
        return TraceCategory.INTERNAL
    if "demo-org" in tags or "test-org" in tags:
        return TraceCategory.INTERNAL
    if meta.get("isDemoOrg") is True or meta.get("isTestOrg") is True:
        return TraceCategory.INTERNAL
    if env == "development":
        return TraceCategory.INTERNAL

    if "customer" in tags or "customer-org" in tags:
        return TraceCategory.CUSTOMER
    if env == "production" and meta.get("isDemoOrg") is False:
        return TraceCategory.CUSTOMER

    return TraceCategory.UNKNOWN


def is_customer_trace(trace: TraceDict) -> bool:
    """Returns True only for confirmed customer traces."""
    return classify_trace(trace) == TraceCategory.CUSTOMER


def is_non_customer_trace(trace: TraceDict) -> bool:
    """Returns True for internal, eval, or unknown traces."""
    return classify_trace(trace) != TraceCategory.CUSTOMER


# Keep old name as alias for backward compatibility
is_internal_trace = is_non_customer_trace


# ---------------------------------------------------------------------------
# Pre-canned suggestion matching
# ---------------------------------------------------------------------------

PRECANNED_SUGGESTIONS = {
    "clean_data_simple": (
        "I will provide you a raw file in the next message that I want you to "
        "transform into the required format based on the schemas you have access "
        "to. Use your templates skill to pull out schemas and then use the "
        "plan-data-transform and execute-data-transform skills to perform the "
        "transformation. Make reasonable assumptions as you go. Go end-to-end, "
        "no need to confirm plan or assumptions with me. Do not stop until you "
        "have produced a transformed file."
    ),
    "clean_data_complex": (
        "I will provide you a raw file in the next message that I want you to "
        "transform into the required format based on the schemas you have access "
        "to. Use your templates skill to pull out schemas and then use the "
        "plan-data-transform and execute-data-transform skills to perform the "
        "transformation. Make reasonable assumptions as you go, but do confirm "
        "key assumptions with me before executing. Ask me one question at a time."
    ),
    "emissions_hotspots": (
        "What are the biggest contributors to emissions in my latest footprint? "
        "Start with a high-level breakdown by scope, GHG category, and one other "
        "important dimension, suggest 2-3 important next deep dives"
    ),
    "yoy_analysis": (
        "Can you compare the emission results from the last two years of data "
        "and break down the impact of the year over year change into impact "
        "driven by activity data change, emissions factor change, and changes "
        "in the mix of categories or activities? Do it by sub-category "
        "(e.g. scope 3.1), and confirm footprint selection with me"
    ),
}

PRECANNED_MATCH_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# System / greeting patterns
# ---------------------------------------------------------------------------

SYSTEM_PATTERNS = [
    "# AskPrint Reviewer",
    "--- name: askprint:review",
    "You are a GHG accounting specialist reviewer",
    "Please review this AskPrint response",
]

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|hiya|yo|test|hi again)[\s!.,]*$", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Rejection patterns
#
# Patterns are grouped by confidence. High-confidence patterns are
# unambiguous rejection phrases. Low-confidence single-word patterns
# (no, wrong, stop) require sentence-initial position or explicit negation
# context to avoid false positives on normal conversational inputs like
# "No, I want scope 3 too" or "What's wrong with this data?".
# ---------------------------------------------------------------------------

REJECTION_PATTERNS = [
    # High confidence: multi-word phrases that are unambiguous rejections
    r"\bthat'?s not (what|right|correct)",
    r"\bnot what i (want|need|mean|asked)",
    r"\bdon'?t do that\b",
    r"\bthat'?s incorrect\b",
    r"\bi (don'?t|didn'?t) (want|need|ask|mean)",
    r"\bplease don'?t\b",
    r"\bnot (right|correct|helpful)\b",
    r"\bforget (that|it|about)\b",
    r"\btry again\b",
    r"\bstart over\b",
    r"\bnevermind\b",
    r"\bnever mind\b",
    # Medium confidence: require sentence-initial or standalone position
    r"^no[,.\s!]",
    r"^nope\b",
    r"^stop\b",
    r"^cancel\b",
    r"^redo\b",
    r"^actually[,\s]",
    r"^instead[,\s]",
    r"^wrong\b",
    r"\bi meant\b",
]

_REJECTION_RE = re.compile("|".join(REJECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CONTEXT_WINDOW = 40


def extract_input_text(inp: Any, max_len: int = 2000) -> str:
    """Flatten various input formats to a plain string."""
    if inp is None:
        return ""
    if isinstance(inp, str):
        return inp[:max_len]
    if isinstance(inp, dict):
        content = inp.get("content", "")
        if isinstance(content, str):
            return content[:max_len]
        if isinstance(content, list):
            parts = [
                item.get("text", str(item))[:1000]
                if isinstance(item, dict)
                else str(item)[:1000]
                for item in content
            ]
            return " ".join(parts)[:max_len]
        return str(inp)[:max_len]
    return str(inp)[:max_len]


def strip_file_upload_prefix(text: str) -> str:
    """Remove '[User has uploaded the following files: ...]' prefix."""
    return re.sub(r"\[User has uploaded the following files:[^\]]*\]\s*", "", text).strip()


def classify_prompt(text: str) -> tuple[str, str | None]:
    """Classify a user prompt.

    Returns (type, suggestion_name).
    Types: 'pre_canned', 'custom', 'system_auto', 'greeting_test'.
    """
    stripped = strip_file_upload_prefix(text).strip()
    lower = stripped.lower()

    for pattern in SYSTEM_PATTERNS:
        if pattern.lower() in lower:
            return "system_auto", None

    if _GREETING_RE.match(lower):
        return "greeting_test", None

    match_name, _ = _match_precanned(text)
    if match_name:
        return "pre_canned", match_name

    return "custom", None


def detect_rejection(text: str) -> tuple[bool, str | None]:
    """Check if a user message contains rejection/redirection signals.

    Returns (is_rejection, context_snippet).
    """
    stripped = strip_file_upload_prefix(text).strip()
    if not stripped:
        return False, None
    match = _REJECTION_RE.search(stripped)
    if match:
        start = max(0, match.start() - _CONTEXT_WINDOW)
        end = min(len(stripped), match.end() + _CONTEXT_WINDOW)
        context = stripped[start:end].replace("\n", " ")
        return True, context
    return False, None


# ---------------------------------------------------------------------------
# Session-level analysis
# ---------------------------------------------------------------------------

_POSITIVE_RE = re.compile(
    r"\b(thanks|thank you|great|perfect|helpful"
    r"|that works|looks good|awesome|excellent)\b",
    re.IGNORECASE,
)


class SessionOutcome(str, Enum):
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    REDIRECTED = "redirected"


@dataclass
class SessionAnalysis:
    outcome: SessionOutcome
    rejection_turns: list[int] = field(default_factory=list)
    total_rejections: int = 0
    has_positive_close: bool = False


def classify_session_outcome(
    traces: list[TraceDict],
) -> SessionAnalysis:
    """Analyze all user messages in a session.

    Scans every trace's input for rejection signals and checks the
    last user message for positive-close signals.
    """
    rejection_turns: list[int] = []

    for i, t in enumerate(traces):
        text = extract_input_text(t.get("input"))
        if not text:
            continue
        is_rej, _ = detect_rejection(text)
        if is_rej:
            rejection_turns.append(i)

    has_positive = False
    for t in reversed(traces):
        text = extract_input_text(t.get("input"))
        if text:
            has_positive = bool(_POSITIVE_RE.search(text))
            break

    if rejection_turns:
        outcome = SessionOutcome.REDIRECTED
    elif has_positive:
        outcome = SessionOutcome.COMPLETED
    else:
        outcome = SessionOutcome.ABANDONED

    return SessionAnalysis(
        outcome=outcome,
        rejection_turns=rejection_turns,
        total_rejections=len(rejection_turns),
        has_positive_close=has_positive,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MIN_SUBSTRING_RATIO = 0.5


_MIN_WORDS_FOR_OVERLAP = 10


def _word_overlap_ratio(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    if min(len(words_a), len(words_b)) < _MIN_WORDS_FOR_OVERLAP:
        return 0.0
    overlap = len(words_a & words_b)
    return overlap / min(len(words_a), len(words_b))


def _match_precanned(text: str) -> tuple[str | None, float]:
    stripped = strip_file_upload_prefix(text).strip()
    best_match = None
    best_score = 0.0

    for name, pattern in PRECANNED_SUGGESTIONS.items():
        stripped_lower = stripped.lower()
        pattern_lower = pattern.lower()

        if pattern_lower in stripped_lower:
            return name, 1.0
        if (
            stripped_lower in pattern_lower
            and len(stripped) > len(pattern) * _MIN_SUBSTRING_RATIO
        ):
            return name, 1.0

        score = _word_overlap_ratio(stripped, pattern)
        if score > best_score:
            best_score = score
            best_match = name

    if best_score >= PRECANNED_MATCH_THRESHOLD:
        return best_match, best_score
    return None, 0.0
