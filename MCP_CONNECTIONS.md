# MCP Connections — Credentials Reference

**Status: Phases 39–41 are complete and verified** (see `tests/test_mcp.py` — 12/12 passing):

| Phase | What | Where | Status |
|-------|------|-------|--------|
| 39 | TRACERA's own MCP server (7 tools: `search_code`, `find_symbol`, `find_references`, `get_context`, `get_dependencies`, `run_tests`, `inspect_repository`) | `tracera/mcp/server.py` | ✅ done |
| 40 | MCP client (connect → initialize → tools/list → register → tools/call) | `tracera/mcp/client.py` | ✅ done |
| 41 | Unified tool registry (native + MCP tools side by side) | `tracera/mcp/manager.py`, `tracera/tools/registry.py` | ✅ done |

This document lists **what credentials you need to connect TRACERA to external
MCP servers**, in the order the roadmap says to add them.

---

## How credentials reach an MCP server

Two ways — both end up in the server subprocess environment:

1. **Process environment** — put keys in `.env` (copy from `.env.example`).
   The MCP server subprocess inherits the environment TRACERA was launched
   with.
2. **Per-server `env` field** — declare keys directly in `mcp_servers.json`
   for that server only:

```json
[
  {
    "name": "github",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {
      "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
    }
  }
]
```

Connect with:

```bash
tracera mcp connect mcp_servers.json
```

Remote tools are registered as `{server}_{tool}` (e.g. `github_search_code`)
in the unified registry.

---

## Credential cheat-sheet (priority order)

| # | Server | Credentials needed? | What to provide |
|---|--------|--------------------|-----------------|
| 1 | **Filesystem** | ❌ None | A directory path (the server's root) |
| 2 | **Git / GitHub** | ⚠️ Git: none · GitHub: **Personal Access Token** | `GITHUB_PERSONAL_ACCESS_TOKEN` (or `GITHUB_TOKEN`) |
| 3 | **PostgreSQL** | ✅ **Connection string** | `DATABASE_URL` / `POSTGRES_URL` |
| 4 | **Web / Documentation** | ⚠️ Depends on server | `BRAVE_API_KEY` or `TAVILY_API_KEY` (search); none for fetch |
| 5 | **Playwright** | ❌ None | Local browser (Chromium) |
| 6 | **Docker** | ⚠️ Local daemon: none · registry: **token** | Docker Hub token for private images / rate limits |
| 7 | **Sentry** | ✅ **Org Auth Token** | `SENTRY_AUTH_TOKEN` (`sntrys_...`); `SENTRY_DSN` for self-hosted base URL |
| 8 | **Jira / Linear** | ✅ **Jira:** API token · **Linear:** API key | `JIRA_BASE_URL` + `JIRA_EMAIL` + `JIRA_API_TOKEN` · `LINEAR_API_KEY` |
| 9 | **Slack** | ✅ **Bot + App tokens** | `SLACK_BOT_TOKEN` (`xoxb-...`) + `SLACK_APP_TOKEN` (`xapp-...`) + `SLACK_TEAM_ID` |
| 10 | **Kubernetes** | ⚠️ None to create — use existing **kubeconfig** | `~/.kube/config` (or `KUBECONFIG`) |

Legend: ✅ always required · ⚠️ only in some setups · ❌ none

---

## 1. Filesystem — no credentials

Local file read/write/move/search. **No auth required** — you just pick the
root directory the server is allowed to touch.

```json
[
  {"name": "filesystem", "command": "npx",
   "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/workspace"]}
]
```

## 2. Git / GitHub

**Local Git** needs nothing beyond your existing `git` config/SSH keys — the
server shells out to git on the same machine.

**GitHub** needs a Personal Access Token:

- **Where:** GitHub → *Settings → Developer settings → Personal access tokens*.
- **Recommended:** *Fine-grained token*, scope it to the repos you need with
  read/write on Contents, Issues, Pull requests.
- **Classic token** also works (requires `repo` scope for private repos).
- **Env var:** the official reference server reads `GITHUB_PERSONAL_ACCESS_TOKEN`
  (GitHub's managed server accepts `GITHUB_TOKEN`).

```json
[
  {"name": "github", "command": "npx",
   "args": ["-y", "@modelcontextprotocol/server-github"],
   "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."}}
]
```

## 3. PostgreSQL

Requires a **connection string** to the database, including user + password:

```
postgresql://user:password@host:5432/dbname
```

- **Where:** from your DB host (Supabase, Neon, RDS, local `pg`) — create a
  dedicated read-only role if you only need to inspect.
- **Env var:** `DATABASE_URL` (server default) or `POSTGRES_URL` (project
  convention, see `.env.example`).

```json
[
  {"name": "postgres", "command": "npx",
   "args": ["-y", "@modelcontextprotocol/server-postgres"],
   "env": {"DATABASE_URL": "postgresql://user:password@localhost:5432/dbname"}}
]
```

## 4. Web / Documentation

Depends on which server you pick:

- **Fetch server** (`mcp-server-fetch`, npx) — plain HTTP fetch, **no
  credentials**.
- **Search servers** need an API key:
  - **Brave Search** (`@modelcontextprotocol/server-brave-search`) →
    `BRAVE_API_KEY` (get at brave.com/search/api)
  - **Tavily** (`tavily-mcp`) → `TAVILY_API_KEY` (get at tavily.com)
  - **Perplexity** → `PERPLEXITY_API_KEY`

```json
[
  {"name": "web-search", "command": "npx",
   "args": ["-y", "@modelcontextprotocol/server-brave-search"],
   "env": {"BRAVE_API_KEY": "BSA..."}}
]
```

## 5. Playwright — no credentials

Browser automation / UI testing / screenshots. **No API keys.** Requires a
local browser install (the `playwright` CLI downloads Chromium on first use).

```json
[
  {"name": "playwright", "command": "npx",
   "args": ["-y", "@playwright/mcp"]}
]
```

## 6. Docker

**Local daemon:** none — talks to the Docker socket your user already has
access to (verify with `docker ps`).

**Registries:** a Docker Hub (or GHCR/ECR) token is only needed for private
images or to raise anonymous pull rate limits. Stored via `docker login`,
not passed to the MCP server.

```json
[
  {"name": "docker", "command": "npx",
   "args": ["-y", "docker-mcp"]}
]
```

## 7. Sentry

Requires an **Organization Auth Token** (not just the DSN — the DSN is for
error *ingestion*, the token is for the *API*):

- **Where:** sentry.io → *Settings → Developer Settings → Auth Tokens*
  (tokens start with `sntrys_`). Scope: read access to issues/projects.
- **Env vars:** `SENTRY_AUTH_TOKEN`; for self-hosted Sentry also set
  `SENTRY_BASE_URL` (e.g. `https://sentry.yourcompany.com`).

```json
[
  {"name": "sentry", "command": "npx",
   "args": ["-y", "sentry-mcp"],
   "env": {"SENTRY_AUTH_TOKEN": "sntrys_...", "SENTRY_BASE_URL": "https://sentry.io"}}
]
```

## 8. Jira / Linear

**Jira** (Atlassian):
- **Where:** https://id.atlassian.com/manage-profile/security/api-tokens →
  *Create API token*.
- **Env vars:** `JIRA_BASE_URL` (e.g. `https://yourorg.atlassian.net`),
  `JIRA_EMAIL` (the account email), `JIRA_API_TOKEN`.

**Linear**:
- **Where:** Linear → *Settings → Security & access → Personal API keys*.
- **Env var:** `LINEAR_API_KEY` (starts with `lin_api_`).

```json
[
  {"name": "jira", "command": "npx",
   "args": ["-y", "@modelcontextprotocol/server-atlassian"],
   "env": {"JIRA_BASE_URL": "https://yourorg.atlassian.net",
           "JIRA_EMAIL": "you@example.com",
           "JIRA_API_TOKEN": "..."}},
  {"name": "linear", "command": "npx",
   "args": ["-y", "mcp-linear"],
   "env": {"LINEAR_API_KEY": "lin_api_..."}}
]
```

## 9. Slack

Requires a Slack **app** with two tokens:

- **Where:** api.slack.com → *Create app* (or use an existing app in your
  workspace) → install it to your workspace with the scopes you need
  (channels:read, channels:history, chat:write, ...).
- **Bot token:** `xoxb-...` (OAuth & Permissions tab)
- **App-level token:** `xapp-...` (needed for Socket Mode; App Manifest →
  enable *Socket Mode*)
- **Env vars:** `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_TEAM_ID`.

```json
[
  {"name": "slack", "command": "npx",
   "args": ["-y", "mcp-server-slack"],
   "env": {"SLACK_BOT_TOKEN": "xoxb-...", "SLACK_APP_TOKEN": "xapp-...",
           "SLACK_TEAM_ID": "T..."}}
]
```

## 10. Kubernetes

**Nothing to create.** The server reads your existing **kubeconfig** — the
same credentials `kubectl` already uses:

- Default: `~/.kube/config`
- Override: `KUBECONFIG` env var pointing at another file
- For production safety, create a dedicated **ServiceAccount + limited Role**
  and point the server at a kubeconfig that only contains that account.

```json
[
  {"name": "kubernetes", "command": "npx",
   "args": ["-y", "mcp-server-kubernetes"],
   "env": {"KUBECONFIG": "~/.kube/config"}}
]
```

---

## Security rules

- **Never commit tokens.** `.env` is gitignored; keep `mcp_servers.json`
  with inline `env` blocks out of version control too (add it to `.gitignore`
  or only commit a template).
- **Least privilege:** scope tokens to the minimum (e.g. fine-grained GitHub
  tokens limited to specific repos; read-only DB roles; Sentry read-only).
- **Rotate regularly** and revoke any token pasted into a chat/log.
- If you use per-server `env` in `mcp_servers.json`, prefer referencing
  environment variables already set in `.env` over hardcoding secrets.
