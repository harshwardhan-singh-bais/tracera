"""
Phase 45 — Retrieval evaluation dataset.

Benchmark queries with ground-truth relevance judgments, so retrieval
quality can be measured reproducibly.

A ground-truth entry can be one of:
- a **file path** (matched against each result's file_path),
- a **doc ID** (matched against each result's doc_id — chunk ids are the
  deterministic md5 of ``path:start-end`` from the chunker),
- a **symbol name** (matched against the result's symbol field).

Example entries:

    {
      "query": "Where is JWT authentication implemented?",
      "files": ["auth/middleware.py", "auth/token.py"],
      "symbols": ["validate_token"],
      "category": "auth"
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tracera.logging import get_logger

log = get_logger("evaluation.dataset")


@dataclass
class EvalQuery:
    """A single retrieval benchmark query with relevance judgments."""

    query: str
    #: Ground-truth file paths (relative to the workspace root).
    files: list[str] = field(default_factory=list)
    #: Ground-truth chunk/doc ids.
    docs: list[str] = field(default_factory=list)
    #: Ground-truth symbol names.
    symbols: list[str] = field(default_factory=list)
    category: str = "general"

    @property
    def ground_truth(self) -> set[str]:
        """All ground-truth tokens (files + docs + symbols), lower-cased."""
        return {
            *(f.lower() for f in self.files),
            *(d.lower() for d in self.docs),
            *(s.lower() for s in self.symbols),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalQuery":
        return cls(
            query=str(data.get("query", "")),
            files=list(data.get("files") or []),
            docs=list(data.get("docs") or []),
            symbols=list(data.get("symbols") or []),
            category=str(data.get("category", "general")),
        )


class EvaluationDataset:
    """A named collection of EvalQuery objects, persisted as JSON."""

    def __init__(
        self,
        name: str,
        queries: list[EvalQuery] | None = None,
    ) -> None:
        self.name = name
        self.queries: list[EvalQuery] = list(queries or [])

    # ── Construction ──────────────────────────────────────────────────────────

    def add(self, query: EvalQuery) -> None:
        self.queries.append(query)

    def add_query(
        self,
        query: str,
        *,
        files: list[str] | None = None,
        docs: list[str] | None = None,
        symbols: list[str] | None = None,
        category: str = "general",
    ) -> "EvalQuery":
        q = EvalQuery(
            query=query,
            files=list(files or []),
            docs=list(docs or []),
            symbols=list(symbols or []),
            category=category,
        )
        self.add(q)
        return q

    # ── Access ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)

    def __getitem__(self, index: int) -> EvalQuery:
        return self.queries[index]

    def filter_category(self, category: str) -> "EvaluationDataset":
        return EvaluationDataset(
            name=f"{self.name}:{category}",
            queries=[q for q in self.queries if q.category == category],
        )

    @property
    def categories(self) -> list[str]:
        return sorted({q.category for q in self.queries})

    # ── Persistence ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "queries": [q.to_dict() for q in self.queries],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )
        log.info("Saved eval dataset (%d queries) → %s", len(self.queries), path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationDataset":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=str(data.get("name", Path(path).stem)),
            queries=[EvalQuery.from_dict(d) for d in data.get("queries", [])],
        )


# ── Example datasets ──────────────────────────────────────────────────────────

def example_queries() -> list[EvalQuery]:
    """Generic example benchmark queries (edit to match a real codebase)."""
    return [
        EvalQuery(
            query="Where is JWT authentication implemented?",
            files=["auth/middleware.py", "auth/token.py"],
            symbols=["validate_token", "AuthMiddleware"],
            category="auth",
        ),
        EvalQuery(
            query="Which function handles database retries?",
            files=["db/retry.py"],
            symbols=["with_retries"],
            category="database",
        ),
        EvalQuery(
            query="Where is the payment timeout configured?",
            files=["config/settings.py", "payments/timeout.py"],
            symbols=["payment_timeout"],
            category="payments",
        ),
        EvalQuery(
            query="How are API routes registered?",
            files=["app/routes.py", "api/router.py"],
            category="api",
        ),
        EvalQuery(
            query="Where is the logging setup?",
            files=["logging/setup.py", "tracera/logging.py"],
            category="infra",
        ),
    ]


def example_dataset(name: str = "example") -> EvaluationDataset:
    return EvaluationDataset(name=name, queries=example_queries())
