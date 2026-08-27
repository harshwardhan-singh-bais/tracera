"""
Conversation Memory Extractor — automatically extracts structured memories
from agent conversations using the LLM.

After each coding session (or periodically during long sessions), the extractor
analyzes the conversation and extracts:
  - Facts about the project or codebase
  - Rules and conventions the user follows
  - Relationships between code entities
  - Skills the agent demonstrated
  - Preferences the user expressed
  - Events (decisions, fixes, discoveries)

This runs in the background after the agent responds — no latency added to
the user-facing response.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tracera.logging import get_logger
from tracera.memory.taxonomy import (
    MemoryEvent,
    MemoryFact,
    MemoryPreference,
    MemoryRelationship,
    MemoryRule,
    MemorySkill,
    StructuredMemory,
    create_event,
    create_fact,
    create_preference,
    create_relationship,
    create_rule,
    create_skill,
)

log = get_logger("memory.extractor")


_EXTRACTION_PROMPT = """You are a memory extraction system for a coding agent.

Analyze the following coding conversation and extract structured memories.
Focus on:

1. FACTS — objective knowledge about the project, codebase, or tools
   Example: "The project uses pytest for testing"

2. RULES — behavioral constraints or conventions
   Example: "Always run tests after editing"

3. RELATIONSHIPS — connections between code entities
   Example: "AuthMiddleware calls UserService"

4. SKILLS — capabilities demonstrated
   Example: "Agent can parse pytest failure output"

5. PREFERENCES — how the user likes things done
   Example: "User prefers concise commit messages"

6. EVENTS — decisions, fixes, discoveries
   Example: "Decided to use LanceDB for vector storage"

Return a JSON array of extracted memories. Each memory object must have:
- "type": one of "fact", "rule", "relationship", "skill", "preference", "event"
- "content": the memory content (clear, concise, self-contained)
- "confidence": 0.0-1.0 (how confident you are this is correct)

For relationships, also include:
- "subject": first entity
- "predicate": relationship type (calls, uses, depends_on, etc.)
- "object": second entity

For events, also include:
- "event_type": one of "decision", "fix", "discovery", "failure", "success"

For rules, also include:
- "priority": 0-9 (0 = most important)

Only extract genuinely useful memories. Skip trivial or obvious statements.
Return ONLY the JSON array, no other text.

Conversation:
{conversation}"""


class ConversationExtractor:
    """
    Extracts structured memories from coding conversations using the LLM.

    Usage:
        extractor = ConversationExtractor(provider)
        memories = await extractor.extract(conversation_text, session_id="...")
        for memory in memories:
            memory_store.add(memory)
    """

    def __init__(self, provider: Any = None) -> None:
        """
        Args:
            provider: LLM provider for extraction. If None, uses rule-based
                      extraction (no LLM calls, lower quality but free).
        """
        self._provider = provider

    async def extract(
        self,
        conversation: str,
        *,
        session_id: str = "",
        entity_id: str = "",
        process_id: str = "tracera",
        max_memories: int = 20,
    ) -> list[StructuredMemory]:
        """
        Extract structured memories from a conversation.

        Args:
            conversation: The conversation text to analyze.
            session_id: The session this conversation belongs to.
            entity_id: The user/entity attribution.
            process_id: The agent/process attribution.
            max_memories: Maximum memories to extract.

        Returns:
            List of StructuredMemory objects.
        """
        if not conversation.strip():
            return []

        if self._provider is not None:
            return await self._extract_with_llm(
                conversation,
                session_id=session_id,
                entity_id=entity_id,
                process_id=process_id,
                max_memories=max_memories,
            )
        else:
            return self._extract_with_rules(
                conversation,
                session_id=session_id,
                entity_id=entity_id,
                process_id=process_id,
                max_memories=max_memories,
            )

    # ── LLM-based extraction ──────────────────────────────────────────────────

    async def _extract_with_llm(
        self,
        conversation: str,
        *,
        session_id: str,
        entity_id: str,
        process_id: str,
        max_memories: int,
    ) -> list[StructuredMemory]:
        """Use the LLM to extract structured memories."""
        from tracera.providers.base import LLMMessage

        prompt = _EXTRACTION_PROMPT.format(conversation=conversation[:8000])

        try:
            response = await self._provider.complete(
                [LLMMessage.user(prompt)],
                temperature=0.1,
                max_tokens=4096,
            )
            raw = response.content or "[]"

            # Parse JSON from response
            raw = raw.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            items = json.loads(raw)
            if not isinstance(items, list):
                return []

            memories = []
            for item in items[:max_memories]:
                memory = self._item_to_memory(
                    item,
                    session_id=session_id,
                    entity_id=entity_id,
                    process_id=process_id,
                )
                if memory:
                    memories.append(memory)

            log.info("LLM extracted %d memories from conversation", len(memories))
            return memories

        except Exception as e:
            log.warning("LLM extraction failed (%s), falling back to rules", e)
            return self._extract_with_rules(
                conversation,
                session_id=session_id,
                entity_id=entity_id,
                process_id=process_id,
                max_memories=max_memories,
            )

    def _item_to_memory(
        self,
        item: dict,
        *,
        session_id: str,
        entity_id: str,
        process_id: str,
    ) -> StructuredMemory | None:
        """Convert a parsed JSON item to a StructuredMemory."""
        mem_type = item.get("type", "").lower()
        content = item.get("content", "").strip()
        if not content:
            return None

        common = {
            "session_id": session_id,
            "entity_id": entity_id,
            "process_id": process_id,
            "confidence": float(item.get("confidence", 0.7)),
        }

        if mem_type == "fact":
            return create_fact(content, **common)
        elif mem_type == "rule":
            return create_rule(
                content,
                priority=int(item.get("priority", 0)),
                **common,
            )
        elif mem_type == "relationship":
            return create_relationship(
                subject=item.get("subject", ""),
                predicate=item.get("predicate", "relates_to"),
                obj=item.get("object", ""),
                **common,
            )
        elif mem_type == "skill":
            return create_skill(content, **common)
        elif mem_type == "preference":
            return create_preference(content, **common)
        elif mem_type == "event":
            return create_event(
                content,
                event_type=item.get("event_type", "decision"),
                **common,
            )
        return None

    # ── Rule-based extraction (no LLM) ───────────────────────────────────────

    def _extract_with_rules(
        self,
        conversation: str,
        *,
        session_id: str,
        entity_id: str,
        process_id: str,
        max_memories: int,
    ) -> list[StructuredMemory]:
        """
        Rule-based extraction — pattern matching without LLM calls.

        Lower quality than LLM extraction but free and instant.
        Catches common patterns in coding conversations.
        """
        memories: list[StructuredMemory] = []
        common = {
            "session_id": session_id,
            "entity_id": entity_id,
            "process_id": process_id,
        }

        lines = conversation.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect user preferences (lines starting with "I prefer", "I like", etc.)
            lower = line.lower()
            if any(lower.startswith(p) for p in ["i prefer", "i like", "i want", "please use", "always use"]):
                memories.append(create_preference(line, strength=0.8, **common))
                continue

            # Detect rules (lines starting with "always", "never", "don't")
            if any(lower.startswith(p) for p in ["always ", "never ", "don't ", "do not "]):
                memories.append(create_rule(line, **common))
                continue

            # Detect facts about the project (lines with "uses", "is based on", "runs on")
            if any(kw in lower for kw in ["uses ", "is based on", "runs on", "configured with"]):
                memories.append(create_fact(line, confidence=0.7, **common))
                continue

            # Detect decisions (lines with "decided", "chose", "going with")
            if any(kw in lower for kw in ["decided", "chose", "going with", "switched to"]):
                memories.append(create_event(line, event_type="decision", **common))
                continue

            # Detect fixes (lines with "fixed", "resolved", "the issue was")
            if any(kw in lower for kw in ["fixed ", "resolved ", "the issue was", "the bug was"]):
                memories.append(create_event(line, event_type="fix", **common))
                continue

        # Deduplicate
        seen = set()
        deduped = []
        for mem in memories:
            key = mem.content.lower()[:100]
            if key not in seen:
                seen.add(key)
                deduped.append(mem)

        result = deduped[:max_memories]
        log.info("Rule-based extraction: %d memories from conversation", len(result))
        return result

    # ── Convenience: extract from a session ───────────────────────────────────

    async def extract_from_session(
        self,
        session: Any,  # Session object
    ) -> list[StructuredMemory]:
        """Extract memories from a closed session."""
        conversation = session.conversation_text()
        return await self.extract(
            conversation,
            session_id=session.id,
            entity_id=session.entity_id,
            process_id=session.process_id,
        )
