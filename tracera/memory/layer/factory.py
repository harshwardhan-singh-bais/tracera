"""
Memory Layer factory — constructs a fully-wired :class:`MemoryLayer` from
TRACERA settings and can install it over an LLM provider.

When the layer is enabled (``TRACERA_MEMORY_ENABLED=true``) the CLI process
defaults to ``entity=TRACERA_MEMORY_ENTITY`` / ``process=tracera-cli`` so that
interactive usage "just works" while every other integration point (MCP,
library consumers) still enforces explicit attribution.
"""

from __future__ import annotations

from pathlib import Path

from tracera.config.settings import Settings, get_settings
from tracera.logging import get_logger
from tracera.memory.layer.attribution import Attribution
from tracera.memory.layer.facade import MemoryLayer
from tracera.memory.layer.store import MemoryPolicy, MemoryStore
from tracera.providers.base import LLMProvider

log = get_logger("memory.layer.factory")


class LocalMemoryEmbedder:
    """
    Lazily loads the local sentence-transformers embedder (with disk cache).

    The model is only loaded on first use so process startup stays fast.
    """

    def __init__(self, model_name: str, device: str, cache_dir: Path | None = None) -> None:
        self._model_name = model_name
        self._device = device
        self._cache_dir = cache_dir
        self._pipeline = None

    def embed(self, text: str) -> list[float]:
        if self._pipeline is None:
            from tracera.retrieval.embedder import EmbeddingPipeline

            self._pipeline = EmbeddingPipeline(
                model_name=self._model_name,
                device=self._device,
                cache_dir=self._cache_dir,
            )
        return self._pipeline.embed_single(text)


def create_memory_layer(settings: Settings | None = None) -> MemoryLayer | None:
    """
    Build a ``MemoryLayer`` from settings, or None when disabled / unavailable.
    """
    if settings is None:
        settings = get_settings()
    if not settings.tracera_memory_enabled:
        return None
    try:
        # One policy, derived once from settings, threaded into both the store
        # and the layer. Settings is the config surface; MemoryPolicy is the
        # object everything reads — so the two can no longer disagree.
        policy = MemoryPolicy.from_settings(settings)
        store = MemoryStore(settings.memory_layer_db, policy=policy)
        embedder = LocalMemoryEmbedder(
            settings.tracera_embedding_model,
            settings.tracera_embedding_device,
            cache_dir=settings.memory_dir / "embed_cache",
        )
        return MemoryLayer(
            store=store,
            embed_fn=embedder.embed,
            top_k=policy.recall_top_k,
            dedup_threshold=policy.dedup_threshold,
            min_recall_score=policy.recall_min_score,
            enabled_processes=settings.memory_layer_processes or None,
            extraction_model=settings.tracera_memory_extraction_model or None,
            worker_enabled=settings.tracera_memory_worker_enabled,
            min_extraction_confidence=policy.extraction_min_confidence,
            min_extraction_importance=policy.extraction_min_importance,
            enable_worthiness_filter=policy.worthiness_filter,
            enable_safety=True,
            # Independently switchable now: an operator may want injection
            # filtering for untrusted tool output while still storing legitimate
            # facts that contain the user's own contact details.
            pii_detection=policy.pii_detection,
            prompt_injection_protection=policy.prompt_injection_protection,
            recall_use_hybrid=True,
            recall_token_budget=policy.recall_token_budget,
            recall_grouped=True,
            recall_graph_expansion=settings.tracera_memory_graph_expansion,
            recall_graph_hops=settings.tracera_memory_graph_hops,
            # Reconciliation (Mem0-style ADD/UPDATE/DELETE) is a distinct
            # mechanism from consolidation (merging near-duplicates), so it
            # keeps its own setting rather than borrowing the policy's flag.
            enable_reconciliation=settings.tracera_memory_reconciliation,
            reconciliation_candidates=settings.tracera_memory_reconciliation_candidates,
            decay_half_life_days=policy.decay_half_life_days,
            retention_days=policy.retention_days,
        )
    except Exception as e:  # noqa: BLE001 — never break startup over memory
        log.warning("Memory layer unavailable (disabled): %s", e)
        return None


def register_memory_layer(
    provider: LLMProvider,
    settings: Settings | None = None,
    *,
    entity: str | None = None,
    process: str = "tracera-cli",
) -> tuple[LLMProvider, MemoryLayer | None]:
    """
    Wrap ``provider`` with the memory layer when enabled.

    Returns ``(provider, layer)`` where layer is None when the memory layer is
    disabled or fails to install (the app always continues without it).

    The CLI's default attribution is applied so interactive flows get memory
    out of the box; explicit callers can override per request via
    ``layer.attribution(entity_id, process_id)``.
    """
    if settings is None:
        settings = get_settings()
    layer = create_memory_layer(settings)
    if layer is None:
        return provider, None
    try:
        wrapped = layer.register(provider)
        if layer._worker_enabled:
            layer.start()
        default_entity = entity or settings.tracera_memory_entity
        layer._default_attribution = Attribution(default_entity, process)
        log.info(
            "Memory layer enabled (entity=%s, process=%s, db=%s)",
            default_entity,
            process,
            settings.memory_layer_db,
        )
        return wrapped, layer
    except Exception as e:  # noqa: BLE001
        log.warning("Memory layer failed to install (%s) — continuing without it", e)
        return provider, None
