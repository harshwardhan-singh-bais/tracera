"""
Memory Layer Recall — the read path executed *before* each LLM call.

1. Embed the outbound user message.
2. Vector-search the entity's stored memories (never cross-entity).
3. Take the top-k most relevant.
4. Inject them into the system prompt in a clearly delimited block.

This is what makes the agent "remember" without any manual RAG code at the
call site — the wrapper handles everything.
"""

from __future__ import annotations

from typing import Any, Callable

from tracera.logging import get_logger
from tracera.memory.layer.attribution import Attribution
from tracera.memory.layer.store import MemoryRecord, MemoryStore
from tracera.providers.base import LLMMessage, Role

log = get_logger("memory.layer.recall")

#: Prompt-injection block header (Memori-style, clearly delimited).
INJECTION_HEADER = "Known context about this user:"

EmbedFn = Callable[[str], list[float]]


def format_memories(results: list[tuple[MemoryRecord, float]]) -> str:
    """Render recalled memories as an injection block for the system prompt."""
    lines = [INJECTION_HEADER]
    for record, _score in results:
        lines.append(f"- {record.to_line()}")
    return "\n".join(lines)


def last_user_text(messages: list[LLMMessage]) -> str:
    """Extract the most recent user-role text from a message list."""
    for message in reversed(messages):
        if message.role == Role.USER and message.content:
            return message.content
    return ""


class RecallInjector:
    """
    Enriches a request's system material with the entity's top-k memories.

    Heavily defensive by design: retrieval problems must never break the LLM
    call, so any exception degrades to the plain (unenriched) request.
    """

    def __init__(
        self,
        store: MemoryStore,
        embed_fn: EmbedFn,
        *,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> None:
        self._store = store
        self._embed = embed_fn
        self._top_k = max(1, top_k)
        self._min_score = min_score

    def inject(
        self,
        messages: list[LLMMessage],
        system: str | None,
        scope: Attribution,
    ) -> tuple[list[LLMMessage], str | None]:
        """
        Return (messages, system) with the memory block appended when any
        memories match the outbound query for this entity.
        """
        query = last_user_text(messages)
        if not query:
            return messages, system
        query_embedding = self._embed(query)
        results = self._store.recall(
            scope.entity_id,
            query_embedding,
            k=self._top_k,
            min_score=self._min_score,
        )
        if not results:
            return messages, system
        block = format_memories(results)
        log.debug(
            "Recalled %d memory(ies) for entity %s into system prompt",
            len(results),
            scope.entity_id,
        )
        return self._attach(messages, system, block)

    @staticmethod
    def _attach(
        messages: list[LLMMessage],
        system: str | None,
        block: str,
    ) -> tuple[list[LLMMessage], str | None]:
        """
        Prefer the ``system`` kwarg; otherwise merge into an existing system
        message; otherwise prepend a new one.
        """
        if system is not None:
            return messages, system + "\n\n" + block
        for i, message in enumerate(messages):
            if message.role == Role.SYSTEM:
                merged = list(messages)
                merged[i] = LLMMessage.system(
                    f"{message.content or ''}\n\n{block}"
                )
                return merged, None
        return [LLMMessage.system(block), *messages], None