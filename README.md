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

## Features (Phases 1–10)

| Phase | Feature |
|-------|---------|
| 1 | Project architecture, config system, Rich logging, Typer CLI |
| 2 | Workspace sandbox — path validation, traversal protection |
| 3 | Git integration — status, diff, log, branch |
| 4 | LLM provider abstraction — OpenAI, Anthropic, Gemini, Ollama |
| 5 | Provider-neutral conversation state |
| 6 | Tool abstraction + registry with JSON schema |
| 7 | Basic coding tools — read, write, edit, grep, run |
| 8 | Core ReAct agent loop with streaming |
| 9 | Planning system — task decomposition, TODO tracking |
| 10 | Persistent agent memory — project facts, decisions |
| TUI | Futuristic Textual terminal UI |

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
```

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
