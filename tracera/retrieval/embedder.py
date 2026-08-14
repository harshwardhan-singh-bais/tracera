"""
Phase 17 — Local Embedding Pipeline.

Generates vector embeddings from text using a local sentence-transformers model.
Supports batching and a file-based cache to avoid re-embedding identical content.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tracera.logging import get_logger

log = get_logger("retrieval.embedder")


class EmbeddingPipeline:
    """
    Wraps a local sentence-transformers model for generating embeddings.

    The model is loaded lazily on first use to avoid startup overhead.
    A disk cache (SHA256 → embedding) prevents re-embedding identical text.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_dir: Path | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._cache_dir = cache_dir
        self._model: Any = None
        self._memory_cache: dict[str, list[float]] = {}

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            log.info("Loading embedding model: %s on %s", self._model_name, self._device)
            self._model = SentenceTransformer(self._model_name, device=self._device)
            log.info("Embedding model loaded.")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. Run: uv add sentence-transformers"
            )

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()  # type: ignore

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self._model_name}:{text}".encode()).hexdigest()

    def _load_from_cache(self, key: str) -> list[float] | None:
        if key in self._memory_cache:
            return self._memory_cache[key]
        if self._cache_dir:
            cache_file = self._cache_dir / f"{key}.json"
            if cache_file.exists():
                embedding = json.loads(cache_file.read_text())
                self._memory_cache[key] = embedding
                return embedding
        return None

    def _save_to_cache(self, key: str, embedding: list[float]) -> None:
        self._memory_cache[key] = embedding
        if self._cache_dir:
            cache_file = self._cache_dir / f"{key}.json"
            cache_file.write_text(json.dumps(embedding))

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed_single(self, text: str) -> list[float]:
        """Embed a single string. Uses cache when possible."""
        key = self._cache_key(text)
        cached = self._load_from_cache(key)
        if cached is not None:
            return cached

        self._load_model()
        vec = self._model.encode(text, normalize_embeddings=True)  # type: ignore
        embedding = vec.tolist()
        self._save_to_cache(key, embedding)
        return embedding

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed a list of strings in batches.
        Hits cache for each text individually before batching uncached ones.
        """
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached = self._load_from_cache(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            self._load_model()
            log.debug("Embedding %d uncached texts in batches of %d", len(uncached_texts), batch_size)
            
            all_embeddings: list[list[float]] = []
            for start in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[start : start + batch_size]
                vecs = self._model.encode(batch, normalize_embeddings=True, show_progress_bar=False)  # type: ignore
                all_embeddings.extend(vecs.tolist())

            for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                embedding = all_embeddings[i]
                results[idx] = embedding
                key = self._cache_key(text)
                self._save_to_cache(key, embedding)

        return [r for r in results if r is not None]

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two normalized embeddings."""
        va = np.array(a)
        vb = np.array(b)
        return float(np.dot(va, vb))
