# TRACERA — CodePilotX

> **Agentic Code Intelligence & Autonomous Coding Engine**
>
> A terminal-native AI coding agent with hybrid retrieval, symbol-aware code understanding, and a futuristic TUI.

```
 ████████╗██████╗  █████╗  ██████╗███████╗██████╗  █████╗
 ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗
    ██║   ██████╔╝███████║██║     █████╗  ██████╔╝███████║
    ██║   ██╔══██╗██╔══██║██║     ██╔══╝  ██╔══██╗██╔══██║
    ██║   ██║  ██║██║  ██║╚██████╗███████╗██║  ██║██║  ██║
    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
```

---

## Features (Phases 1–38)

| Phase | Feature |
|-------|---------|
| 1 | Project architecture, config system, Rich logging, Typer CLI |
| 2 | Workspace sandbox — path validation, traversal protection |
| 3 | Git integration — status, diff, log, branch (+ `git` agent tool) |
| 4 | LLM provider abstraction — OpenAI, Anthropic, Gemini, Groq, Ollama, … |
| 5 | Provider-neutral conversation state |
| 6 | Tool abstraction + registry with JSON schema |
| 7 | Basic coding tools — read, write, edit, grep, run |
| 8 | Core ReAct agent loop with streaming |
| 9 | Planning system — task decomposition, TODO tracking |
| 10 | Persistent agent memory — injected into agent context |
| 11–15 | Repository scanner, tree-sitter parsing, symbol extraction, chunking, schema |
| 16–20 | BM25, local embeddings, LanceDB vector index, dense + hybrid retrieval |
| 21–24 | Symbol-aware retrieval, context expansion, cross-encoder rerank, incremental indexing |
| 25–26 | Symbol relationship graph + dependency-aware graph retrieval |
| 27–28 | Code-search agent tools (`search_code`, `find_symbol`, `find_references`, `get_dependencies`, `get_context`) |
| 29–31 | Context assembly, context compression, repository-aware agent |
| 32–34 | Test discovery, safe test execution, failure analysis |
| 35–38 | Retrieval-driven debugging, autonomous fix loop, self-review, regression protection |
| 39–41 | MCP server (7 capabilities as MCP tools), MCP client, unified tool registry |
| TUI | Futuristic Textual terminal UI with full scrolling (PgUp/PgDn + mouse) |

---

## Installation

```bash
# Recommended: use uv
uv tool install .

# Or with pip
pip install -e ".[dev]"
```

## Quick Start

```bash
# Open the interactive TUI
tracera

# Ask the agent something
tracera ask "Explain this codebase"

# Index the repository
tracera index .

# Search code
tracera search "authentication middleware"

# Show status
tracera status

# Manage memory
tracera memory list

# List the tools exposed by the MCP server (Phase 39)
tracera mcp serve --check

# Run the MCP server on stdio (connect from Claude Desktop, Cursor, ...)
tracera mcp serve
```

## MCP Integration (Phases 39–41)

TRACERA speaks the Model Context Protocol in both directions:

**As an MCP server** — your existing capabilities become MCP tools:

```text
search_code · find_symbol · find_references · get_context
· get_dependencies · run_tests · inspect_repository
```

```bash
# List them without serving
tracera mcp serve --check

# Serve on stdio (default transport)
tracera mcp serve
```

Any MCP client (Claude Desktop, Cursor, custom agents) can now call these tools.
The retrieval pipeline is loaded lazily only when a code index exists.

**As an MCP client** — connect to external MCP servers and merge their tools
into the unified registry alongside native tools:

```json
// mcp_servers.json
[
  {"name": "filesystem", "command": "npx",
   "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}
]
```

```bash
tracera mcp connect mcp_servers.json
```

Remote tools are registered as `{server}_{tool}` (e.g. `filesystem_read_file`)
so multiple servers and native tools coexist without name collisions.
Programmatically, `MCPClient` and `MCPManager` (in `tracera/mcp/`) give the
same unified-registry behaviour inside the agent.

**What credentials each external MCP server needs** (GitHub token, Postgres
connection string, Slack tokens, ...) is documented in
[`MCP_CONNECTIONS.md`](MCP_CONNECTIONS.md).

## Multi-Agent Delegation (Phases 42–44)

Specialized sub-agents — **Researcher, Coder, Tester, Reviewer, Debugger** —
built on the same ReAct loop with role-specific prompts and tool subsets.
The orchestrator decomposes a task, assigns each step a role, runs the
sub-agents, and aggregates the results (conflict detection + shared task
state).

```bash
# Delegate a task across the sub-agent fleet
tracera delegate "Implement login and review the diff"

tracera delegate --parallel "..."   # run independent steps concurrently
```

Code: `tracera/agent/subagents.py` (roles, tool filtering) +
`tracera/agent/orchestrator.py` (delegation, shared state, aggregation).

## Evaluation & Benchmarking (Phases 45–50)

```bash
# 45 · write the example retrieval-eval dataset (edit ground truth first!)
tracera eval dataset -o .tracera/eval/dataset.json

# 46-48 · compare grep / BM25 / dense / hybrid / hybrid+reranker
#         (Recall@k, MRR, nDCG@k, latency, context size)
tracera eval retrieval .tracera/eval/dataset.json -o .tracera/eval/report.md

# 49 · end-to-end agent benchmark (success, tests, tokens, latency, cost)
tracera eval agent --tasks "task one,task two"

# 50 · ablation study (Agent → +BM25 → +Dense → +Hybrid → +Reranker → +Graph)
tracera eval ablation
```

Code: `tracera/evaluation/` — `dataset.py` (45), `metrics.py` (46),
`strategies.py` (47-48), `retrieval_benchmark.py` (46-48),
`agent_benchmark.py` (49), `ablation.py` (50).

## Security (Phases 51–55)

| Phase | Defense | Module |
|-------|---------|--------|
| 51 | Prompt-injection detection/neutralization (repo, retrieval, web, MCP) | `tracera/security/injection.py` |
| 52 | Secret detection & redaction (API keys, tokens, .env) | `tracera/security/secrets.py` |
| 53 | Command safety (blocklist, allowlist, confirmation, sandbox cwd guard) | `tracera/security/command_safety.py` |
| 54 | MCP trust model, tool permissions, output validation | `tracera/security/mcp_security.py` |
| 55 | Resource-limit monitoring (iterations, tool calls, tokens) | `tracera/security/resources.py` |

## Terminal UI (Phases 56–59)

`tracera tui` (or just `tracera`) now includes:

- **56 · REPL commands:** `/code` `/search` `/debug` `/index` `/test` `/review`
  `/tools` `/mcp` `/cost` `/inspect` `/deps` `/plan` `/memory` `/model` …
- **57 · Rich execution display:** live phases (`Searching…`, `Running tests…`,
  `✓ N passed`) in the thinking trace and status bar.
- **58 · Repository inspection:** Repo tab — files, symbols, dependency chains
  (`/inspect`, `/deps <symbol>`).
- **59 · Retrieval debugging:** Debug tab — per-strategy results for any query
  (`/debug <query>` shows BM25 / Dense / Hybrid / Reranker side by side).

## Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

Minimum required: one LLM provider key (e.g. `OPENAI_API_KEY`).

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12+ |
| CLI | Typer |
| TUI | Textual |
| Terminal rendering | Rich |
| Config | Pydantic Settings |
| Code parsing (phase 11+) | Tree-sitter |
| Embeddings (phase 17+) | Sentence Transformers |
| Vector DB (phase 18+) | LanceDB |
| Keyword retrieval (phase 16+) | BM25 |
| Reranking (phase 23+) | Cross-Encoder |
| MCP (phase 39+) | Python MCP SDK |
| Testing | Pytest |
| Packaging | uv + pyproject.toml |

## Architecture

```
tracera/
├── config/        # Pydantic Settings + profiles
├── logging/       # Rich-powered structured logging
├── errors/        # Typed exception hierarchy
├── workspace/     # Sandboxed filesystem operations
├── git/           # Git repository operations
├── providers/     # LLM provider adapters
├── conversation/  # Conversation state management
├── tools/         # Tool registry + coding tools
├── agent/         # ReAct loop + planner + memory
└── tui/           # Textual TUI application
```

## License

MIT
