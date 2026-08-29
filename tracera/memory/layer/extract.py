"""
Memory Layer Extraction — the write path executed asynchronously after each
LLM call (Advanced Augmentation).

Given a full turn (user message + assistant message), a cheap/fast model is
asked to return structured memory items as JSON:

    [
      {"kind": "preference", "subject": "user",
       "predicate": "favorite_color", "object": "blue",
       "text": "User's favorite color is blue."},
      ...
    ]

Only durable, memory-worthy content is extracted — never small talk or the
literal question being asked. The downstream store embeds and dedups each item.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from tracera.logging import get_logger
from tracera.memory.layer.store import ALL_KINDS

log = get_logger("memory.layer.extract")

#: Fully-typed call signature for the extraction LLM: prompt → raw text reply.
LLMCallFn = Callable[[str], Awaitable[str]]


EXTRACTION_PROMPT = """You extract structured memory from a conversation turn.
Given the user message and assistant message below, return a JSON array of
objects, each with:
- kind: one of "fact", "preference", "skill", "attribute", "relationship", "event", "rule", "decision", "goal", "constraint", "experience"
- subject: usually "user" unless clearly about someone or something else
- predicate: short snake_case relation name
- object: the value
- text: a short natural-language restatement of the fact
- confidence: 0.0-1.0 (how confident you are this is correct and durable)
- importance: 0.0-1.0 (how important this memory is for future recall)

Only extract durable, memory-worthy information — not small talk, not the
literal question being asked. Return [] if nothing is worth remembering.

User: {user_message}
Assistant: {assistant_message}
"""


@dataclass(frozen=True)
class ExtractedMemory:
    """One validated memory item produced by the extraction model."""

    kind: str
    subject: str
    predicate: str
    object: str
    text: str
    confidence: float = 0.8
    importance: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "text": self.text,
            "confidence": self.confidence,
            "importance": self.importance,
        }


# ── Memory Worthiness Filter ──────────────────────────────────────────────────

# Patterns that indicate NON-durable content (should NOT be stored as memory)
NON_DURABLE_PATTERNS = [
    # Greetings/small talk
    r"^(hi|hello|hey|thanks|thank you|ok|okay|sure|got it|understood)[\s\.\!]*$",
    # Questions (the literal question being asked)
    r"^\s*(what|how|why|when|where|who|which|can you|could you|would you)\s",
    # Temporary calculations/one-off
    r"^(the answer is|the result is|calculation|compute)[\s\:]",
    # Conversational filler
    r"^(let me |i think |i believe |it seems |basically |actually )",
    # Tool output dumps
    r"(^|\n)(stdout|stderr|exit code|duration|memory usage)\:",
    # Generic acknowledgments
    r"^(sounds good|looks good|makes sense|fair enough)[\s\.\!]*$",
]

# Patterns that indicate HIGH durability (should be stored)
DURABLE_PATTERNS = [
    # Explicit instructions
    r"(remember|always|never|don't|do not|please use|please don't)",
    # Preferences
    r"(i prefer|i like|i want|my preference|we use|we always|our convention)",
    # Decisions
    r"(decided|chose|going with|switched to|we'll use|we will use)",
    # Facts about project/tools
    r"(uses?|based on|runs on|configured with|depends on|requires)",
    # Rules/conventions
    r"(always |never |must |should |convention|standard|pattern)",
    # Skills/capabilities
    r"(can |able to |knows how to |experience with |skilled at)",
    # Relationships
    r"(calls|imports|depends on|extends|implements|contains|tests)",
    # Events/fixes
    r"(fixed|resolved|the issue was|the bug was|root cause)",
    # Goals/constraints
    r"(goal|objective|constraint|requirement|must not|cannot)",
]


def is_memory_worthy(text: str, min_score: float = 0.3) -> tuple[bool, float]:
    """
    Determine if a text segment is memory-worthy.

    Returns (is_worthy, confidence_score).
    """
    text_lower = text.lower().strip()

    # Too short to be meaningful
    if len(text_lower) < 10:
        return False, 0.0

    # Check non-durable patterns (negative signals)
    non_durable_score = 0.0
    for pattern in NON_DURABLE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            non_durable_score += 0.3

    # Check durable patterns (positive signals)
    durable_score = 0.0
    for pattern in DURABLE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            durable_score += 0.2

    # Length bonus for substantial content
    length_bonus = min(0.2, len(text_lower) / 1000)

    # Final score
    score = max(0.0, 0.3 + durable_score - non_durable_score + length_bonus)
    score = min(1.0, score)

    return score >= min_score, score


def filter_memory_worthy(
    items: list[ExtractedMemory],
    min_confidence: float = 0.5,
    min_importance: float = 0.3,
    enable_safety: bool = True,
) -> list[ExtractedMemory]:
    """Filter extracted memories by worthiness, confidence, and importance."""
    filtered = []
    for item in items:
        # Check confidence and importance thresholds
        if item.confidence < min_confidence:
            continue
        if item.importance < min_importance:
            continue

        # Check if the text content is memory-worthy
        worthy, worthiness_score = is_memory_worthy(item.text)
        if not worthy:
            log.debug("Filtered out non-worthy memory: %s (score=%.2f)", item.text[:50], worthiness_score)
            continue

        # Safety checks
        if enable_safety:
            # PII detection
            if detect_pii(item.text):
                log.debug("Filtered out memory with potential PII: %s", item.text[:50])
                continue

            # Prompt injection protection
            if detect_prompt_injection(item.text):
                log.debug("Filtered out memory with potential prompt injection: %s", item.text[:50])
                continue

        # Boost confidence by worthiness
        boosted_confidence = min(1.0, item.confidence * (0.8 + 0.2 * worthiness_score))
        filtered.append(ExtractedMemory(
            kind=item.kind,
            subject=item.subject,
            predicate=item.predicate,
            object=item.object,
            text=item.text,
            confidence=boosted_confidence,
            importance=item.importance,
        ))

    return filtered


# ── Safety Features (Phase 11) ──────────────────────────────────────────────────

# PII patterns (common types)
PII_PATTERNS = [
    # Email
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    # Phone (various formats)
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    # SSN
    r"\b\d{3}-\d{2}-\d{4}\b",
    # Credit card (basic)
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    # API keys (common prefixes)
    r"\b(sk|pk)_(live|test)_[A-Za-z0-9]{24,}\b",
    # AWS keys
    r"\bAKIA[0-9A-Z]{16}\b",
    # GitHub tokens
    r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    # Generic secrets
    r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*[A-Za-z0-9_\-]{20,}",
]


def detect_pii(text: str) -> bool:
    """Detect potential PII in text."""
    for pattern in PII_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def sanitize_pii(text: str) -> str:
    """Redact potential PII from text."""
    result = text
    for pattern in PII_PATTERNS:
        result = re.sub(pattern, "[REDACTED]", result, flags=re.IGNORECASE)
    return result


# Prompt injection patterns
PROMPT_INJECTION_PATTERNS = [
    # Direct instruction override
    r"(?i)(ignore|forget|disregard)\s+(previous|all|above)\s+(instructions?|prompt|rules?)",
    r"(?i)(ignore|forget|disregard)\s+all\s+previous\s+(instructions?|prompt|rules?)",
    # Role manipulation
    r"(?i)(you are now|act as|pretend to be|roleplay as)\s+",
    # System prompt extraction
    r"(?i)(show|print|output|reveal)\s+(system|initial|original)\s+(prompt|instructions?)",
    r"(?i)(show|print|output|reveal)\s+(me\s+)?(your|my)\s+(system|initial|original)\s+(prompt|instructions?)",
    # Jailbreak attempts
    r"(?i)(DAN|Do Anything Now|unrestricted|unfiltered|no rules)",
    # Data exfiltration
    r"(?i)(output|print|show|dump)\s+(all\s+)?(your|the|my)\s+(memories?|data|secrets?|keys?)",
    # Chain of thought extraction
    r"(?i)(show|print|output)\s+(your|the|my)\s+(reasoning|thinking|chain of thought)",
    # Encoding bypasses
    r"(?i)(base64|rot13|encode|decode)\s+",
    # Special tokens
    r"(?i)(<\||\|>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>)",
]


def detect_prompt_injection(text: str) -> bool:
    """Detect potential prompt injection attempts in text."""
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def sanitize_prompt_injection(text: str) -> str:
    """Sanitize potential prompt injection patterns."""
    result = text
    for pattern in PROMPT_INJECTION_PATTERNS:
        result = re.sub(pattern, "[FILTERED]", result, flags=re.IGNORECASE)
    return result


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_items(raw: str) -> list[ExtractedMemory]:
    """Parse + validate the model's JSON reply, skipping malformed entries."""
    text = _strip_code_fences(raw)
    if not text:
        return []
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as e:
        log.warning("Extraction model returned invalid JSON: %s", e)
        return []
    if not isinstance(data, list):
        return []

    items: list[ExtractedMemory] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind", "")).strip()
        subject = str(entry.get("subject", "")).strip() or "user"
        predicate = str(entry.get("predicate", "")).strip()
        obj = str(entry.get("object", "")).strip()
        text = str(entry.get("text", "")).strip()
        confidence = float(entry.get("confidence", 0.8))
        importance = float(entry.get("importance", 0.5))
        if kind not in ALL_KINDS or not predicate or not obj or not text:
            continue
        items.append(ExtractedMemory(kind, subject, predicate, obj, text, confidence, importance))
    if items:
        log.debug("Extraction produced %d memory item(s)", len(items))
    return items


class MemoryExtractor:
    """
    Runs the structured extraction prompt through an injected LLM callable.

    The callable is provided by the layer (which wires it to the underlying —
    pre-wrapped — provider so extraction never recurses through the wrapper).
    """

    def __init__(
        self,
        call_llm: LLMCallFn,
        *,
        min_confidence: float = 0.5,
        min_importance: float = 0.3,
        enable_worthiness_filter: bool = True,
        enable_safety: bool = True,
    ) -> None:
        self._call_llm = call_llm
        self._min_confidence = min_confidence
        self._min_importance = min_importance
        self._enable_worthiness_filter = enable_worthiness_filter
        self._enable_safety = enable_safety

    async def extract_turn(
        self,
        user_message: str,
        assistant_message: str,
    ) -> list[ExtractedMemory]:
        """Extract memory-worthy items from one turn."""
        try:
            prompt = EXTRACTION_PROMPT.format(
                user_message=(user_message or "")[:2000],
                assistant_message=(assistant_message or "")[:6000],
            )
            raw = await self._call_llm(prompt)
            items = _parse_items(raw)

            if self._enable_worthiness_filter:
                items = filter_memory_worthy(
                    items,
                    min_confidence=self._min_confidence,
                    min_importance=self._min_importance,
                    enable_safety=self._enable_safety,
                )

            return items
        except Exception as e:  # noqa: BLE001 — extraction must never crash the worker
            log.warning("Memory extraction failed: %s", str(e)[:200])
            return []