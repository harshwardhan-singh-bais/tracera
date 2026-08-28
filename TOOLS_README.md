# TRACERA Intelligence Tools — jCodeMunch Integration

## What was done

22 new tools were added to TRACERA, inspired by the [jCodeMunch MCP](https://github.com/jgravelle/jcodemunch-mcp) project. These tools give the agent **structural code analysis** capabilities that go far beyond simple file reading and grep — the same kind of queries jCodeMunch provides, but natively integrated into TRACERA's existing architecture.

### Why not just install jCodeMunch?

jCodeMunch is a standalone MCP server (~50k+ lines) with its own indexing engine, protocol server, watcher daemons, and license system. TRACERA already has its own:
- Tree-sitter-based code indexing pipeline (Phases 16-24)
- Symbol relationship graph (NetworkX, Phase 25-26)
- Hybrid BM25 + Dense retrieval (Phases 19-21)
- Context assembly and compression (Phases 29-30)

Rather than running two separate systems, these tools **plug into TRACERA's existing infrastructure** — reusing the same symbol graph, retrieval pipeline, and workspace sandbox that already powers `search_code`, `find_symbol`, and the autonomous fix loop.

---

## Architecture

```
tracera/tools/
├── ast_tools.py          # 8 structural analysis tools
├── refactor_tools.py     # 4 refactoring & safety tools
├── session_tools.py      # 5 context assembly & session tools
├── provenance_tools.py   # 5 provenance & coupling tools
├── registry.py           # Updated: extend_registry_with_ast_tools()
├── __init__.py           # Updated: exports all new tools
└── (existing files)      # read_file, write_file, grep, etc.
```

All tools are registered automatically when the agent starts via `extend_registry_with_ast_tools()` in `tracera/main.py`. No manual setup needed — they appear as available tools in the agent's toolkit.

---

## The 22 Tools

### Structural Analysis (`ast_tools.py`)

These answer questions that **grep cannot answer** — they traverse the symbol graph, not just text.

| Tool | What it does |
|------|-------------|
| `find_importers` | What files/symbols import a given file. Shows the dependency entry points that would break. |
| `get_blast_radius` | BFS traversal of all downstream dependents. Groups by depth with risk labels (🔴 HIGH / 🟡 MEDIUM / 🟢 LOW). |
| `get_call_hierarchy` | Trace callers (who calls this) and callees (what this calls) N levels deep through the call graph. |
| `find_dead_code` | BFS from entry points (main, app, cli, test files). Everything unreachable = potentially dead. |
| `get_changed_symbols` | Runs `git diff`, then maps changed files to their symbols in the graph. Shows exactly what changed. |
| `get_hotspots` | Multiplies cyclomatic complexity (line span) by git churn (commit frequency). Hotspots = complex + frequently changed. |
| `search_ast` | Pattern matching across all files. 6 preset detectors (empty_catch, bare_except, hardcoded_secret, eval_exec, todo_fixme, magic_number) + custom DSL queries (`call:*.unwrap`, `string:/password/i`). |
| `get_class_hierarchy` | Traverse inheritance: base classes, subclasses, and methods of a class. |

### Refactoring & Safety (`refactor_tools.py`)

Preflight checks **before** you make risky changes.

| Tool | What it does |
|------|-------------|
| `plan_refactoring` | Generates edit-ready `{old_text, new_text}` blocks for rename/move/extract/signature-change. Includes collision detection and import rewrite guidance. |
| `check_edit_safe` | Scores risk 0.0-1.0 based on caller count, complexity, entry-point proximity, and has-methods. Returns SAFE / CAUTION / DANGEROUS verdict. |
| `check_delete_safe` | Similar to check_edit_safe but for deletion. Checks if symbol has callers or is an entry point. |
| `get_pr_risk_profile` | Composite risk score for uncommitted changes or a branch. Fuses file count, churn, blast radius, config changes, and test coverage. |

### Context Assembly (`session_tools.py`)

One-call orchestration — the agent doesn't have to chain 5 tools manually.

| Tool | What it does |
|------|-------------|
| `assemble_task_context` | Auto-classifies task intent (explore/debug/refactor/extend/audit), extracts anchor symbols from natural language, runs the appropriate retrieval sequence, and returns a source-attributed context capsule under a token budget. |
| `plan_turn` | Probes the index for confidence before the first read. Returns recommended tool route and estimated token consumption. Low confidence = "this probably doesn't exist." |
| `get_ranked_context` | Packs the most relevant symbols into a fixed token budget. Deduplicates, compresses, and ranks by relevance score. |
| `get_session_stats` | Reports tokens served, files read, estimated naive cost, and tool usage breakdown. Shows token savings vs. naive full-file reading. |
| `get_repo_map` | Cold-start orientation: PageRank-ranked repository overview showing all files and their most important symbols. No query needed. |

### Provenance & Analysis (`provenance_tools.py`)

Git archaeology and code health metrics.

| Tool | What it does |
|------|-------------|
| `get_symbol_provenance` | Traces every git commit that touched a symbol. Classifies commits (bugfix, refactor, feature, perf, rename, revert) and generates an evolution narrative. |
| `audit_agent_config` | Scans CLAUDE.md, .cursorrules, and other agent config files for stale file references, token bloat, and redundancy. |
| `get_endpoint_impact` | Given an HTTP endpoint or handler symbol, shows the blast radius: all callers and callees that would be affected. |
| `get_dependency_cycles` | Detects circular import chains using NetworkX cycle detection. Suggests how to break them. |
| `get_coupling_metrics` | Per-module afferent (Ca) and efferent (Ce) coupling, instability ratio (Ce/(Ca+Ce)), and symbol counts. Highlights god modules. |

---

## How it works

All tools share the same infrastructure:

1. **Symbol Graph** (`tracera/graph/symbol_graph.py`) — NetworkX directed graph where nodes are symbols and edges are relationships (imports, calls, inherits, implements, contains). Built during `tracera index`.

2. **Retrieval Pipeline** (`tracera/main.py::_build_retrieval_pipeline`) — The full 10-tuple pipeline: `(indexer, symbol_retriever, expander, reranker, context_engine, compressor, embedder, vector_store, bm25, graph_retriever)`.

3. **Tool Registration** (`tracera/tools/registry.py::extend_registry_with_ast_tools`) — Creates all 22 tool instances, wires them to the pipeline/workspace, and registers them in the agent's tool registry.

4. **Agent Integration** (`tracera/main.py::_build_agent`) — After building the base agent and memory tools, calls `extend_registry_with_ast_tools()` to add all intelligence tools. Also updates the system prompt to document the new capabilities.

---

## Usage

No extra setup required. If you've already run `tracera index`, all tools are available immediately:

```bash
# Index your codebase (if not already done)
tracera index

# Start the TUI — all 22+ tools are available to the agent
tracera

# Or ask a question that uses the new tools
tracera ask "What would break if I change the AuthService class?"
tracera ask "Find dead code in the project"
tracera ask "Show me the dependency cycles"
tracera ask "What are the riskiest files to change?"
```

The agent will automatically use these tools when appropriate. You can also explicitly ask it to use them:

> "Use get_blast_radius on the UserSerializer class"
> "Run search_ast with the 'all' preset"
> "Check if it's safe to delete the legacy_auth module"

---

## What's NOT included (vs. jCodeMunch)

jCodeMunch has additional capabilities that are not replicated here:

- **MCP server protocol** — TRACERA uses its own agent loop, not MCP
- **GitHub repo indexing** — these tools work on local repos only
- **SCIP compiler-verified references** — requires external SCIP index files
- **MUNCH compact wire format** — TRACERA has its own context compression
- **Enforcement hooks** (PreToolUse/PostToolUse) — agent-level, not tool-level
- **Watch/daemon mode** — live re-indexing on file changes
- **License system** — these tools are MIT, no commercial restrictions
- **90+ tools** — we implemented the 22 most impactful ones

The full jCodeMunch tool set covers ~90 tools. The 22 implemented here cover the core structural analysis, safety, context assembly, and provenance features that provide the most value for an autonomous coding agent.

---

## MCP Integration (Phases 39-41)

TRACERA also exposes its capabilities over the **Model Context Protocol (MCP)**, so external agents (Claude Desktop, Cursor, etc.) can consume TRACERA's tools:

- **`tracera mcp serve`** — Starts the MCP server exposing the 7 core capabilities
- **`tracera mcp connect <config.json>`** — Connects to external MCP servers and merges their tools into the unified registry

See `MCP_CONNECTIONS.md` for credential references and server configurations (GitHub, Postgres, Slack, filesystem, etc.).
