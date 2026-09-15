# ── TRACERA runtime image ────────────────────────────────────────────────────
# Multi-stage: builder installs deps with uv into a venv, runtime copies the
# venv + source only. The embedding model downloads at first `/index`, not at
# build time — keep the image generic and let each workspace bring its own
# .tracera/ data (mounted as a volume).

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first (cached layer — only invalidated by lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Now the source
COPY tracera ./tracera
COPY README.md LICENSE pyproject.toml ./
RUN uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm AS runtime

# git is required for change mapping / provenance / regression baselines
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user
RUN useradd -m -u 1000 tracera

WORKDIR /workspace
COPY --from=builder --chown=tracera:tracera /app/.venv /opt/tracera/.venv
COPY --from=builder --chown=tracera:tracera /app/tracera /opt/tracera/tracera
COPY --from=builder --chown=tracera:tracera /app/pyproject.toml /opt/tracera/pyproject.toml

ENV PATH="/opt/tracera/.venv/bin:$PATH" \
    PYTHONUTF8=1 \
    TRACERA_WORKSPACE=/workspace

USER tracera

ENTRYPOINT ["tracera"]

# Default: MCP over stdio (the primary container use-case).
# Override for TUI/CLI: docker compose run tracera tui
#                       docker compose run tracera ask "explain the pipeline"
CMD ["mcp", "serve", "--workspace", "/workspace"]
