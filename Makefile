# ── TRACERA developer tasks ──────────────────────────────────────────────────
# uv-native. `make help` shows everything.

.DEFAULT_GOAL := help
UV := uv

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Sync the venv from the lockfile
	$(UV) sync --frozen

.PHONY: lint
lint: ## Ruff check (no autofix)
	$(UV) run ruff check tracera tests scripts

.PHONY: fmt
fmt: ## Ruff autofix (imports, style)
	$(UV) run ruff check --fix tracera tests scripts
	$(UV) run ruff format tracera tests scripts

.PHONY: type
type: ## Mypy strict over tracera/
	$(UV) run mypy tracera

.PHONY: test
test: ## Fast unit tests
	$(UV) run pytest tests/ -q -x --ignore=tests/test_mcp.py --ignore=tests/test_mcp_server.py

.PHONY: test-mcp
test-mcp: ## MCP unit + server tests
	$(UV) run pytest tests/test_mcp.py tests/test_mcp_server.py -q

.PHONY: test-all
test-all: ## Full test suite
	$(UV) run pytest tests/ -q

.PHONY: e2e-mcp
e2e-mcp: ## End-to-end MCP stdio test (initialize → list → call)
	$(UV) run python scripts/test_mcp_e2e.py

.PHONY: verify-commands
verify-commands: ## Honest slash-command sweep (works/empty/crash per command)
	$(UV) run python scripts/verify_all_commands.py

.PHONY: verify-commands-fast
verify-commands-fast: ## Sweep without LLM-heavy commands
	$(UV) run python scripts/verify_all_commands.py --fast

.PHONY: audit
audit: ## Dependency vulnerability scan
	$(UV) run pip-audit || echo "pip-audit not installed — run: uv add --dev pip-audit"

.PHONY: build
build: ## Build sdist + wheel into dist/
	rm -rf dist
	$(UV) build
	@python -c "import zipfile,glob; w=glob.glob('dist/*.whl')[0]; bad=[n for n in zipfile.ZipFile(w).namelist() if '__pycache__' in n]; print(f'{w}: {len(bad)} junk entries'); exit(1 if bad else 0)"

.PHONY: publish
publish: build ## Publish to PyPI (requires UV_PUBLISH_TOKEN)
	$(UV) publish

.PHONY: docker-build
docker-build: ## Build the Docker image
	docker build -t tracera:latest .

.PHONY: docker-mcp
docker-mcp: ## MCP stdio container smoke test
	docker compose build tracera
	echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' | docker compose run -T tracera | head -c 200

.PHONY: docker-sse
docker-sse: ## Start MCP over SSE on :8000
	docker compose up mcp-sse

.PHONY: check
check: lint type test ## What CI runs (lint + types + fast tests)

.PHONY: ci-local
ci-local: check test-mcp e2e-mcp ## Full local CI equivalent
