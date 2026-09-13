import Link from 'next/link';
import {
  Boxes,
  CircleDot,
  GitBranch,
  Layers,
  Network,
  Search,
  ShieldCheck,
  Terminal,
} from 'lucide-react';
import { CopyButton } from '@/components/copy-button';
import { appName } from '@/lib/shared';

const installCmd = 'uv tool install .';

/** Feature files shown in the tabbed code showcase. */
const codeTabs = [
  {
    id: 'tui',
    label: 'TUI',
    code: `$ uv run tracera
◆ TRACERA · groq/llama-3.3-70b · indexed: 186 files · memory: on

> /search "where do we embed documents"
  retrieval/pipeline.py · indexer/extractor.py · 12 more

> /blast WorkspaceSandbox
  158 symbols affected across 31 files

> /fix "pytest tests/test_indexer.py fails on empty repo"
  plan → retrieve → edit → test ✓  (3/3 checks passed)`,
  },
  {
    id: 'fix',
    label: 'Fix loop',
    code: `# Autonomous loop: plan → retrieve → edit → test
tracera fix "pytest tests/test_sandbox.py::test_temp_dir fails"

# iteration 1
  plan:  3 steps · retrieve: 12 symbols · edit: 2 files
  test:  FAILED (1/12) — retrying
# iteration 2
  test:  12/12 passed ✓
# regression gate: 334/334 before, 334/334 after ✓`,
  },
  {
    id: 'delegate',
    label: 'Delegate',
    code: `> /delegate "add type hints to tracera/retrieval/symbols.py"
  researcher: located module + 4 dependents
  coder:     applied hints (3 functions)
  tester:    12/12 tests pass
  reviewer:  approved with 1 nit

# sub-agents are sandboxed — no mutating git commands`,
  },
  {
    id: 'blast',
    label: 'Blast radius',
    code: `> /blast WorkspaceSandbox
  158 symbols affected across 31 files
  direct callers: 23
  transitive: 135 (depth ≤ 4)
  entry points touched: CLI (3), TUI (2), MCP (1)`,
  },
];

/** Capability grid — the platform pillars. */
const pillars = [
  {
    icon: Search,
    title: 'Hybrid Retrieval',
    description:
      'BM25 keyword search fused with dense local embeddings over a LanceDB index — symbols, chunks, and files, ranked with score breakdowns.',
    href: '/docs/features/retrieval',
  },
  {
    icon: Network,
    title: 'Structural Analysis',
    description:
      'A symbol graph built by tree-sitter powers references, call hierarchy, blast radius, dead code, hotspots and PageRank.',
    href: '/docs/features/structural',
  },
  {
    icon: Terminal,
    title: 'Autonomous Fix Loop',
    description:
      '/fix runs plan → retrieve → edit → test and iterates until checks pass, with a regression gate before every commit.',
    href: '/docs/features/quality',
  },
  {
    icon: Boxes,
    title: 'Multi-Agent Fleet',
    description:
      'Researcher, Coder, Tester, Reviewer and Debugger sub-agents — hard-sandboxed so they can never run mutating git commands.',
    href: '/docs/features/agent',
  },
  {
    icon: Layers,
    title: 'Persistent Memory',
    description:
      'Facts, decisions and learned skills stored in a knowledge graph, recalled across sessions to steer future work.',
    href: '/docs/features/memory',
  },
  {
    icon: GitBranch,
    title: 'Git-Native',
    description:
      'Change mapping, provenance and risk scoring from real repository state — not just diffs.',
    href: '/docs/features/quality',
  },
  {
    icon: ShieldCheck,
    title: 'Safety Controls',
    description:
      'Shell allowlists, confirmation policies, workspace boundaries, and command timeouts — configurable end to end.',
    href: '/docs/configuration',
  },
  {
    icon: CircleDot,
    title: 'MCP Native',
    description:
      'Serve the full tool registry over the Model Context Protocol or consume external MCP servers as first-class tools.',
    href: '/docs/mcp',
  },
];

/** Secondary strip — cross-cutting platform features. */
const platform = [
  {
    title: '11 LLM providers',
    description:
      'OpenAI, Anthropic, Gemini, Groq, Cerebras, Mistral, Together, Cohere, SambaNova, NVIDIA and Ollama — with automatic failover.',
    href: '/docs/configuration',
  },
  {
    title: 'Built-in evaluation',
    description:
      'Benchmark grep / BM25 / dense / hybrid retrieval with Recall@k, MRR and nDCG. Ablate agent arms on the same tasks.',
    href: '/docs/cli#eval',
  },
  {
    title: 'Live observability',
    description:
      'Token usage, latency and estimated cost for every LLM, tool and retrieval call — in the TUI status line or the CLI.',
    href: '/docs/cli#observability',
  },
  {
    title: 'Headless CLI',
    description:
      'The same engine scripted: ask, index, search, fix, review, delegate — for CI and automation.',
    href: '/docs/cli',
  },
];

const faqs = [
  {
    q: 'What is TRACERA?',
    a: 'TRACERA is an open-source agentic code intelligence engine that lives in your terminal. It indexes your codebase into a hybrid retrieval index, answers architecture questions, and can autonomously plan, edit and verify code changes — all driven by slash commands in a Textual TUI, or scripted through a headless CLI.',
  },
  {
    q: 'How is this different from an editor copilot?',
    a: 'Copilots autocomplete inside one file. TRACERA works at the repository level: it builds a structural index (symbol graph, call hierarchy, blast radius), retrieves across the whole workspace, runs your test suite, and gates its own changes with regression checks before you commit.',
  },
  {
    q: 'Does my code leave my machine?',
    a: 'No. Indexing, embedding (sentence-transformers), BM25, the symbol graph and the memory store are all local — the LanceDB index lives under .tracera/. The only external calls are LLM chat requests to the provider you configure, and Ollama keeps even those local.',
  },
  {
    q: 'Which languages are supported?',
    a: 'Python, JavaScript and TypeScript via tree-sitter. The tree-sitter runtime is pinned to a known-good version to avoid a verified native crash on Windows during symbol extraction.',
  },
  {
    q: 'How do I get started?',
    a: 'Clone the repo, run uv sync, set any provider API key (GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, ...), and launch with uv run tracera. Then /index your workspace and type / for the full command list. The Quickstart takes under a minute.',
  },
  {
    q: 'Is it safe to let it edit my code?',
    a: 'Sub-agents are hard-sandboxed — the git tool is removed and mutating git operations are rejected. The main agent can run /review, /selfreview and /regression before anything is committed, and shell commands are gated by an allowlist plus confirmation policy you control.',
  },
];

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col">
      {/* ── Hero ─────────────────────────────────────────────── */}
      <section className="flex flex-col items-center px-4 pb-16 pt-24 text-center md:pt-32">
        <h1 className="max-w-3xl text-balance text-4xl font-bold tracking-tight md:text-5xl">
          Code intelligence that ships working changes
        </h1>
        <p className="mt-4 max-w-2xl text-balance text-lg text-fd-muted-foreground">
          {appName} indexes your codebase into a hybrid retrieval and symbol
          graph, then plans, edits and <strong>verifies</strong> its own code
          changes — from your terminal.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/docs/quickstart"
            className="rounded-lg bg-fd-primary px-5 py-2.5 font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
          >
            Get Started
          </Link>
          <Link
            href="/docs"
            className="rounded-lg border border-fd-border px-5 py-2.5 font-medium transition-colors hover:bg-fd-muted"
          >
            Read the Docs
          </Link>
        </div>

        {/* Install command */}
        <div className="mt-10 w-full max-w-md">
          <div className="flex items-center justify-between gap-3 rounded-lg border border-fd-border bg-fd-card px-4 py-3">
            <code className="overflow-x-auto font-mono text-sm">
              <span className="select-none text-fd-muted-foreground">$ </span>
              {installCmd}
            </code>
            <CopyButton text={installCmd} />
          </div>
          <p className="mt-2 text-xs text-fd-muted-foreground">
            Python 3.12+ · MIT licensed · runs 100% locally
          </p>
        </div>
      </section>

      {/* ── Product shot ─────────────────────────────────────── */}
      <section className="flex flex-col items-center px-4">
        <div className="w-full max-w-3xl overflow-hidden rounded-xl border border-fd-border bg-fd-card text-left shadow-sm">
          <div className="flex items-center gap-1.5 border-b border-fd-border px-4 py-2.5">
            <span className="size-2.5 rounded-full bg-red-400" />
            <span className="size-2.5 rounded-full bg-yellow-400" />
            <span className="size-2.5 rounded-full bg-green-400" />
            <span className="ml-2 font-mono text-xs text-fd-muted-foreground">
              tracera tui
            </span>
          </div>
          <pre className="overflow-x-auto p-4 font-mono text-[13px] leading-relaxed">
            <code>{`$ uv run tracera
◆ TRACERA · groq/llama-3.3-70b · indexed: 186 files · memory: on

> /search "where do we embed documents"
  retrieval/pipeline.py · indexer/extractor.py · 12 more

> /blast WorkspaceSandbox
  158 symbols affected across 31 files

> /fix "pytest tests/test_indexer.py fails on empty repo"
  plan → retrieve → edit → test ✓  (3/3 checks passed)`}</code>
          </pre>
        </div>
      </section>

      {/* ── Capability grid ──────────────────────────────────── */}
      <section className="mx-auto w-full max-w-5xl px-4 pt-24">
        <h2 className="text-center text-2xl font-semibold tracking-tight md:text-3xl">
          The complete toolkit for code intelligence
        </h2>
        <p className="mx-auto mt-3 max-w-2xl text-center text-fd-muted-foreground">
          Retrieval, structural analysis, autonomous agents, memory and
          observability — one engine, TUI and CLI.
        </p>

        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {pillars.map((p) => (
            <Link
              key={p.title}
              href={p.href}
              className="group rounded-xl border border-fd-border bg-fd-card p-5 transition-colors hover:bg-fd-muted/50"
            >
              <p.icon className="size-5 text-fd-primary" />
              <h3 className="mt-3 font-semibold">{p.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-fd-muted-foreground">
                {p.description}
              </p>
            </Link>
          ))}
        </div>
      </section>

      {/* ── Slash command teaser ─────────────────────────────── */}
      <section className="mx-auto w-full max-w-5xl px-4 pt-24">
        <div className="rounded-2xl border border-fd-border bg-fd-card p-8 md:p-10">
          <div className="grid items-center gap-8 md:grid-cols-2">
            <div>
              <h2 className="text-2xl font-semibold tracking-tight">
                Every capability is a slash command
              </h2>
              <p className="mt-3 text-fd-muted-foreground">
                61 commands with autocomplete, grouped from search to
                autonomous fix loops. Every registry tool is also reachable as{' '}
                <code className="rounded bg-fd-muted px-1 py-0.5 font-mono text-[13px]">
                  /tool &lt;name&gt;
                </code>
                .
              </p>
              <Link
                href="/docs/slash-commands"
                className="mt-5 inline-flex items-center gap-1 font-medium text-fd-primary hover:underline"
              >
                Browse the full reference →
              </Link>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {[
                '/search',
                '/blast',
                '/fix',
                '/delegate',
                '/deadcode',
                '/hotspots',
                '/refs',
                '/plan',
                '/review',
                '/regression',
                '/recall',
                '/pagerank',
              ].map((cmd) => (
                <code
                  key={cmd}
                  className="rounded-md border border-fd-border bg-fd-background px-2 py-1 font-mono text-[13px] text-fd-muted-foreground"
                >
                  {cmd}
                </code>
              ))}
              <code className="rounded-md border border-fd-border bg-fd-background px-2 py-1 font-mono text-[13px] text-fd-muted-foreground">
                +49 more
              </code>
            </div>
          </div>
        </div>
      </section>

      {/* ── Platform strip ───────────────────────────────────── */}
      <section className="mx-auto w-full max-w-5xl px-4 pt-24">
        <h2 className="text-center text-2xl font-semibold tracking-tight md:text-3xl">
          Production-grade under the hood
        </h2>
        <div className="mt-10 grid gap-4 sm:grid-cols-2">
          {platform.map((p) => (
            <Link
              key={p.title}
              href={p.href}
              className="rounded-xl border border-fd-border bg-fd-card p-5 transition-colors hover:bg-fd-muted/50"
            >
              <h3 className="font-semibold">{p.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-fd-muted-foreground">
                {p.description}
              </p>
            </Link>
          ))}
        </div>
      </section>

      {/* ── FAQ ──────────────────────────────────────────────── */}
      <section className="mx-auto w-full max-w-3xl px-4 pt-24">
        <h2 className="text-center text-2xl font-semibold tracking-tight md:text-3xl">
          Frequently asked questions
        </h2>
        <div className="mt-8 divide-y divide-fd-border rounded-xl border border-fd-border bg-fd-card">
          {faqs.map((f) => (
            <details key={f.q} className="group px-5 py-4">
              <summary className="flex cursor-pointer list-none items-center justify-between font-medium [&::-webkit-details-marker]:hidden">
                {f.q}
                <span className="ml-4 text-fd-muted-foreground transition-transform group-open:rotate-45">
                  +
                </span>
              </summary>
              <p className="mt-3 text-sm leading-relaxed text-fd-muted-foreground">
                {f.a}
              </p>
            </details>
          ))}
        </div>
      </section>

      {/* ── Final CTA ────────────────────────────────────────── */}
      <section className="mx-auto flex w-full max-w-3xl flex-col items-center px-4 pb-24 pt-24 text-center">
        <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">
          Start shipping verified changes
        </h2>
        <p className="mt-3 max-w-xl text-fd-muted-foreground">
          Install, index your workspace, and run your first autonomous fix in
          under a minute.
        </p>
        <Link
          href="/docs/quickstart"
          className="mt-6 rounded-lg bg-fd-primary px-6 py-2.5 font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
        >
          Get Started
        </Link>
      </section>
    </main>
  );
}
