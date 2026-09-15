import Link from 'next/link';
import {
  ArrowRight,
  Boxes,
  CircleDot,
  Command,
  Cpu,
  GitBranch,
  Layers,
  Lock,
  Network,
  Search,
  ShieldCheck,
  Terminal,
} from 'lucide-react';
import { CopyButton } from '@/components/copy-button';
import { CodeTabs, type CodeTab } from '@/components/code-tabs';
import { Faq, type FaqItem } from '@/components/faq';
import { Reveal } from '@/components/reveal';
import { Terminal as TerminalWindow, type TermLine } from '@/components/terminal';
import { appName } from '@/lib/shared';
import { AnnounceBar } from '@/components/announce-bar';
import { BrowserFrame } from '@/components/browser-frame';
import { LogoStrip } from '@/components/logo-strip';
import { SiteFooter } from '@/components/site-footer';

const installCmd = 'uv tool install .';

/** Hero transcript — tokenised so the terminal can colour it properly. */
const heroTranscript: TermLine[] = [
  {
    tokens: [
      { text: '$ ', tone: 'accent' },
      { text: 'uv run tracera', tone: 'cmd' },
    ],
  },
  {
    tokens: [
      { text: '◆ ', tone: 'accent' },
      { text: 'TRACERA', tone: 'cmd' },
      { text: ' · groq/llama-3.3-70b · indexed: ', tone: 'dim' },
      { text: '186 files', tone: 'accent' },
      { text: ' · memory: ', tone: 'dim' },
      { text: 'on', tone: 'ok' },
    ],
  },
  { tokens: [{ text: ' ', tone: 'dim' }], pause: 140 },
  {
    tokens: [
      { text: '> ', tone: 'accent' },
      { text: '/search "where do we embed documents"', tone: 'cmd' },
    ],
  },
  {
    tokens: [
      { text: '  retrieval/pipeline.py', tone: 'path' },
      { text: ' · ', tone: 'dim' },
      { text: 'indexer/extractor.py', tone: 'path' },
      { text: ' · 12 more', tone: 'dim' },
    ],
  },
  { tokens: [{ text: ' ', tone: 'dim' }], pause: 140 },
  {
    tokens: [
      { text: '> ', tone: 'accent' },
      { text: '/blast WorkspaceSandbox', tone: 'cmd' },
    ],
  },
  {
    tokens: [
      { text: '  ', tone: 'dim' },
      { text: '158', tone: 'accent' },
      { text: ' symbols affected across ', tone: 'dim' },
      { text: '31', tone: 'accent' },
      { text: ' files', tone: 'dim' },
    ],
  },
  { tokens: [{ text: ' ', tone: 'dim' }], pause: 140 },
  {
    tokens: [
      { text: '> ', tone: 'accent' },
      { text: '/fix "pytest tests/test_indexer.py fails on empty repo"', tone: 'cmd' },
    ],
  },
  {
    tokens: [
      { text: '  plan → retrieve → edit → test ', tone: 'dim' },
      { text: '✓', tone: 'ok' },
      { text: ' 3/3 checks passed', tone: 'ok' },
    ],
  },
];

/** Feature files shown in the tabbed code showcase. */
const codeTabs: CodeTab[] = [
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

/** Headline metrics. Every number here is verified against the source. */
const stats = [
  { value: '86', label: 'slash commands', hint: 'with autocomplete' },
  { value: '44', label: 'MCP tools', hint: 'serve or consume' },
  { value: '13', label: 'LLM providers', hint: 'auto-failover' },
  { value: '100%', label: 'local by default', hint: 'index + memory' },
];

/** Three-step mental model of how the engine works. */
const steps = [
  {
    n: '01',
    icon: Cpu,
    title: 'Index once',
    body: 'tree-sitter parses every file, chunks it, and embeds it locally into LanceDB. Re-runs are incremental — only changed files move.',
  },
  {
    n: '02',
    icon: Command,
    title: 'Ask anything',
    body: 'Hybrid retrieval over symbols, chunks and the call graph answers architecture questions with the exact file and line to look at.',
  },
  {
    n: '03',
    icon: Lock,
    title: 'Ship verified',
    body: 'The agent plans, edits, runs your tests, and gates itself on a regression check — so a green run means green, not "probably fine".',
  },
];

/** Secondary strip — cross-cutting platform features. */
const platform = [
  {
    title: '13 LLM providers',
    description:
      'OpenAI, Anthropic, Gemini, Groq, Cerebras, Mistral, Together, Cohere, SambaNova, NVIDIA, Nemotron, OpenRouter and Ollama — with automatic failover.',
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

const faqs: FaqItem[] = [
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

/** Slash commands surfaced in the teaser; the rest are a link away. */
const teaserCommands = [
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
];

export default function HomePage() {
  return (
    <>
      <AnnounceBar />
      <main className="flex flex-1 flex-col">
        {/* ── Hero ─────────────────────────────────────────────── */}
        <section className="relative isolate overflow-hidden px-4 pb-6 pt-12 md:pt-16">
          {/* Backdrop */}
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0 -z-10 bg-aurora"
          />
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[620px] bg-blueprint"
          />

          <div className="mx-auto grid w-full max-w-6xl grid-cols-1 items-center gap-12 lg:grid-cols-[minmax(0,0.86fr)_minmax(0,1fr)] lg:gap-14">
            {/* Left: copy */}
            <div className="min-w-0 text-center lg:text-left">
              <Reveal>
                <h1 className="text-balance text-4xl font-bold leading-[1.06] tracking-tight sm:text-5xl lg:text-[3.4rem]">
                  Build agents that{' '}
                  <span className="text-gradient">ship verified code</span>
                </h1>
              </Reveal>

              <Reveal delay={90}>
                <p className="mx-auto mt-6 max-w-xl text-balance text-base leading-relaxed text-fd-muted-foreground sm:text-lg lg:mx-0">
                  <span className="font-semibold text-[hsl(142_76%_58%)]">
                    Index, retrieve,
                  </span>{' '}
                  and reason over your whole codebase with {appName} — an
                  agentic code intelligence engine that plans, edits and{' '}
                  <strong className="font-semibold text-fd-foreground">
                    verifies
                  </strong>{' '}
                  its own changes from your terminal.
                </p>
              </Reveal>

              <Reveal delay={170}>
                <div className="mt-8 flex flex-wrap items-center justify-center gap-3 lg:justify-start">
                  <Link
                    href="/docs/quickstart"
                    className="btn-accent group inline-flex items-center gap-2 rounded-lg px-5 py-2.5 font-semibold"
                  >
                    Get Started
                    <ArrowRight className="size-4 transition-transform duration-300 group-hover:translate-x-0.5" />
                  </Link>
                  <Link
                    href="/docs"
                    className="group inline-flex items-center gap-1.5 rounded-lg px-2.5 py-2.5 font-medium text-fd-muted-foreground transition-colors duration-200 hover:text-fd-foreground"
                  >
                    Read the docs
                    <ArrowRight className="size-4 transition-transform duration-300 group-hover:translate-x-0.5" />
                  </Link>
                </div>
              </Reveal>

              <Reveal delay={250}>
                <div className="mx-auto mt-9 w-full max-w-md lg:mx-0">
                  <div className="card-edge flex items-center justify-between gap-3 rounded-lg bg-fd-card/70 px-4 py-3 backdrop-blur-sm">
                    <code className="overflow-x-auto font-mono text-sm">
                      <span className="select-none text-[hsl(142_76%_58%)]">
                        ${' '}
                      </span>
                      {installCmd}
                    </code>
                    <CopyButton text={installCmd} />
                  </div>
                  <p className="mt-2.5 text-xs text-fd-muted-foreground">
                    No telemetry · your index and memory never leave the machine
                  </p>
                </div>
              </Reveal>
            </div>

            {/* Right: browser-framed product shot */}
            <Reveal delay={140}>
              <BrowserFrame url="localhost:3000 · tracera tui" title="live">
                <TerminalWindow
                  title="tracera tui"
                  lines={heroTranscript}
                  chrome={false}
                />
              </BrowserFrame>
            </Reveal>
          </div>
        </section>

        {/* ── Provider strip ───────────────────────────────────── */}
        <Reveal>
          <LogoStrip />
        </Reveal>

      {/* ── Metrics ──────────────────────────────────────────── */}
      <section className="mx-auto w-full max-w-5xl px-4 pt-20">
        <Reveal>
          <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-2xl border border-white/[0.07] bg-white/[0.05] md:grid-cols-4">
            {stats.map((s) => (
              <div
                key={s.label}
                className="bg-fd-background px-5 py-6 text-center transition-colors duration-300 hover:bg-fd-card"
              >
                <dt className="sr-only">{s.label}</dt>
                <dd>
                  <span className="block text-3xl font-bold tracking-tight text-[hsl(142_76%_62%)] md:text-4xl">
                    {s.value}
                  </span>
                  <span className="mt-1.5 block text-sm font-medium">
                    {s.label}
                  </span>
                  <span className="mt-0.5 block text-xs text-fd-muted-foreground">
                    {s.hint}
                  </span>
                </dd>
              </div>
            ))}
          </dl>
        </Reveal>
      </section>

      {/* ── Capability grid ──────────────────────────────────── */}
      <section className="mx-auto w-full max-w-5xl px-4 pt-24">
        <Reveal>
          <h2 className="text-center text-2xl font-semibold tracking-tight md:text-3xl">
            The complete toolkit for code intelligence
          </h2>
          <p className="mx-auto mt-3 max-w-2xl text-center text-fd-muted-foreground">
            Retrieval, structural analysis, autonomous agents, memory and
            observability — one engine, TUI and CLI.
          </p>
        </Reveal>

        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {pillars.map((p, i) => (
            <Reveal key={p.title} delay={(i % 4) * 60}>
              <Link
                href={p.href}
                className="card-edge card-bloom group flex h-full flex-col rounded-xl bg-fd-card/60 p-5 transition-transform duration-300 hover:-translate-y-0.5"
              >
                <span className="icon-tile">
                  <p.icon className="size-[18px]" />
                </span>
                <h3 className="mt-4 font-semibold">{p.title}</h3>
                <p className="mt-1.5 text-sm leading-relaxed text-fd-muted-foreground">
                  {p.description}
                </p>
              </Link>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ── Showcase (tabbed) ────────────────────────────────── */}
      <section className="mx-auto w-full max-w-4xl px-4 pt-24">
        <Reveal>
          <h2 className="text-center text-2xl font-semibold tracking-tight md:text-3xl">
            Four ways to drive it
          </h2>
          <p className="mx-auto mt-3 max-w-2xl text-center text-fd-muted-foreground">
            Every capability is a slash command — or a scriptable CLI call.
          </p>
        </Reveal>
        <Reveal delay={100}>
          <div className="mt-10">
            <CodeTabs tabs={codeTabs} />
          </div>
        </Reveal>
      </section>

      {/* ── How it works ─────────────────────────────────────── */}
      <section className="mx-auto w-full max-w-5xl px-4 pt-24">
        <Reveal>
          <h2 className="text-center text-2xl font-semibold tracking-tight md:text-3xl">
            How it works
          </h2>
        </Reveal>
        <ol className="mt-10 grid gap-4 md:grid-cols-3">
          {steps.map((s, i) => (
            <Reveal key={s.n} as="li" delay={i * 90}>
              <div className="card-edge relative h-full rounded-xl bg-fd-card/60 p-6">
                <span className="font-mono text-xs font-semibold tracking-widest text-[hsl(142_76%_58%)]">
                  {s.n}
                </span>
                <s.icon className="mt-4 size-5 text-fd-muted-foreground" aria-hidden />
                <h3 className="mt-3 font-semibold">{s.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-fd-muted-foreground">
                  {s.body}
                </p>
              </div>
            </Reveal>
          ))}
        </ol>
      </section>

      {/* ── Slash command teaser ─────────────────────────────── */}
      <section className="mx-auto w-full max-w-5xl px-4 pt-24">
        <Reveal>
          <div className="card-edge rounded-2xl bg-fd-card/60 p-8 md:p-10">
            <div className="grid items-center gap-8 md:grid-cols-2">
              <div>
                <h2 className="text-2xl font-semibold tracking-tight">
                  Every capability is a slash command
                </h2>
                <p className="mt-3 leading-relaxed text-fd-muted-foreground">
                  86 commands with autocomplete, from code search to autonomous
                  fix loops. Every registered tool is also reachable as{' '}
                  <code className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[13px] text-[hsl(142_76%_66%)]">
                    /tool &lt;name&gt;
                  </code>
                  .
                </p>
                <Link
                  href="/docs/slash-commands"
                  className="group mt-5 inline-flex items-center gap-1 font-medium text-[hsl(142_76%_58%)] hover:underline"
                >
                  Browse the full reference
                  <ArrowRight className="size-4 transition-transform duration-300 group-hover:translate-x-0.5" />
                </Link>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {teaserCommands.map((cmd) => (
                  <code
                    key={cmd}
                    className="rounded-md border border-white/[0.08] bg-white/[0.03] px-2 py-1 font-mono text-[13px] text-fd-muted-foreground transition-colors duration-200 hover:border-[hsl(142_70%_45%/0.4)] hover:text-[hsl(142_76%_62%)]"
                  >
                    {cmd}
                  </code>
                ))}
                <code className="rounded-md border border-dashed border-white/[0.12] px-2 py-1 font-mono text-[13px] text-fd-muted-foreground">
                  +74 more
                </code>
              </div>
            </div>
          </div>
        </Reveal>
      </section>

      {/* ── Platform strip ───────────────────────────────────── */}
      <section className="mx-auto w-full max-w-5xl px-4 pt-24">
        <Reveal>
          <h2 className="text-center text-2xl font-semibold tracking-tight md:text-3xl">
            Production-grade under the hood
          </h2>
        </Reveal>
        <div className="mt-10 grid gap-4 sm:grid-cols-2">
          {platform.map((p, i) => (
            <Reveal key={p.title} delay={(i % 2) * 70}>
              <Link
                href={p.href}
                className="card-edge card-bloom group block h-full rounded-xl bg-fd-card/60 p-5 transition-transform duration-300 hover:-translate-y-0.5"
              >
                <h3 className="font-semibold">{p.title}</h3>
                <p className="mt-1.5 text-sm leading-relaxed text-fd-muted-foreground">
                  {p.description}
                </p>
              </Link>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ── FAQ ──────────────────────────────────────────────── */}
      <section className="mx-auto w-full max-w-3xl px-4 pt-24">
        <Reveal>
          <h2 className="text-center text-2xl font-semibold tracking-tight md:text-3xl">
            Frequently asked questions
          </h2>
        </Reveal>
        <Reveal delay={100}>
          <div className="mt-8">
            <Faq items={faqs} />
          </div>
        </Reveal>
      </section>

      {/* ── Final CTA ────────────────────────────────────────── */}
      <section className="relative isolate mx-auto flex w-full max-w-3xl flex-col items-center overflow-hidden px-4 pb-28 pt-24 text-center">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 bottom-0 -z-10 h-[420px] bg-aurora opacity-70"
        />
        <Reveal>
          <h2 className="text-balance text-2xl font-semibold tracking-tight md:text-4xl">
            Start shipping verified changes
          </h2>
          <p className="mx-auto mt-4 max-w-xl text-fd-muted-foreground">
            Install, index your workspace, and run your first autonomous fix in
            under a minute.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Link
              href="/docs/quickstart"
              className="btn-accent group inline-flex items-center gap-2 rounded-lg px-6 py-2.5 font-semibold"
            >
              Get Started
              <ArrowRight className="size-4 transition-transform duration-300 group-hover:translate-x-0.5" />
            </Link>
            <Link
              href="/docs/slash-commands"
              className="rounded-lg border border-white/12 bg-white/[0.02] px-6 py-2.5 font-medium transition-colors duration-200 hover:border-white/20 hover:bg-white/[0.05]"
            >
              Browse commands
            </Link>
          </div>
        </Reveal>
      </section>
      </main>
      <SiteFooter />
    </>
  );
}
