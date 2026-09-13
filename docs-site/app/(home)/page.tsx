import Link from 'next/link';
import { Card, Cards } from 'fumadocs-ui/components/card';
import { appName, appTagline } from '@/lib/shared';

const features = [
  {
    title: 'Hybrid Code Search',
    description:
      'BM25 + dense embeddings over a LanceDB index of your entire workspace — symbols, chunks, and files.',
    href: '/docs/features/retrieval',
  },
  {
    title: 'Structural Analysis',
    description:
      'References, call hierarchy, blast radius, importers, dead code, hotspots and PageRank — all via tree-sitter.',
    href: '/docs/features/structural',
  },
  {
    title: 'Autonomous Fix Loop',
    description:
      'Plan → retrieve → edit → test. TRACERA iterates on failing tasks and guards against regressions.',
    href: '/docs/features/agent',
  },
  {
    title: 'Multi-Agent Fleet',
    description:
      'Researcher, Coder, Tester, Reviewer and Debugger sub-agents — sandboxed so they can never touch your git history.',
    href: '/docs/features/agent',
  },
  {
    title: 'Quality & Review',
    description:
      'Dead-code detection, hotspots, self-review and regression guarding on real repository state.',
    href: '/docs/features/quality',
  },
  {
    title: 'Persistent Memory',
    description:
      'A knowledge-graph memory layer with facts, decisions, and skills that survives across sessions.',
    href: '/docs/features/memory',
  },
  {
    title: 'MCP Integration',
    description:
      'Expose the full tool registry over the Model Context Protocol, or consume external MCP servers.',
    href: '/docs/mcp',
  },
];

/** Every slash command the TUI registers — mirrors /docs/slash-commands. */
const commandGroups: { label: string; commands: string[] }[] = [
  {
    label: 'Sessions & help',
    commands: [
      '/help',
      '/features',
      '/clear',
      '/reset',
      '/dashboard',
      '/status',
      '/cost',
      '/observability',
      '/phases',
      '/theme',
      '/files',
    ],
  },
  {
    label: 'Models & providers',
    commands: ['/models', '/model', '/mcp', '/tools', '/agents'],
  },
  {
    label: 'Retrieval & search',
    commands: ['/search', '/debug', '/ranked', '/index'],
  },
  {
    label: 'Structural analysis',
    commands: [
      '/symbol',
      '/symbols',
      '/source',
      '/refs',
      '/callers',
      '/impls',
      '/blast',
      '/importers',
      '/deps',
      '/endpoint',
    ],
  },
  {
    label: 'Quality & risk',
    commands: [
      '/deadcode',
      '/hotspots',
      '/pagerank',
      '/risk',
      '/coupling',
      '/cycles',
      '/audit',
    ],
  },
  {
    label: 'Refactoring',
    commands: ['/refactor', '/editsafe', '/deletesafe'],
  },
  {
    label: 'Git & provenance',
    commands: ['/changed', '/provenance', '/git'],
  },
  {
    label: 'Agent tasks',
    commands: [
      '/plan',
      '/plantask',
      '/code',
      '/fix',
      '/review',
      '/selfreview',
      '/regression',
      '/delegate',
      '/taskcontext',
      '/planturn',
      '/test',
    ],
  },
  {
    label: 'Memory',
    commands: [
      '/memory',
      '/memgraph',
      '/remember',
      '/recall',
      '/forget',
      '/triples',
    ],
  },
];

const totalCommands =
  commandGroups.reduce((n, g) => n + g.commands.length, 0) + 1; // + /tool

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col items-center px-4 pb-24 text-center">
      <p className="mb-5 mt-24 rounded-full border border-fd-border bg-fd-card px-3 py-1 text-xs font-medium text-fd-muted-foreground md:mt-32">
        v0.1.0 · MIT licensed · Python 3.12+
      </p>
      <h1 className="mb-4 max-w-3xl text-4xl font-semibold tracking-tight text-balance md:text-6xl">
        Meet{' '}
        <span className="bg-gradient-to-r from-fd-primary to-fd-primary/60 bg-clip-text text-transparent">
          {appName}
        </span>
      </h1>
      <p className="max-w-2xl text-balance text-lg text-fd-muted-foreground">
        {appTagline}. A terminal-native agent that indexes your codebase,
        answers architecture questions, and writes — and verifies — its own
        code changes.
      </p>

      <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
        <Link
          href="/docs"
          className="rounded-lg bg-fd-primary px-5 py-2.5 font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
        >
          Get Started
        </Link>
        <Link
          href="/docs/slash-commands"
          className="rounded-lg border border-fd-border bg-fd-card px-5 py-2.5 font-medium transition-colors hover:bg-fd-muted"
        >
          Browse all {totalCommands} slash commands
        </Link>
      </div>

      <div className="mt-14 w-full max-w-2xl text-left">
        <div className="overflow-hidden rounded-xl border border-fd-border bg-fd-card shadow-sm">
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
      </div>

      {/* Full slash-command surface */}
      <div className="mt-20 w-full max-w-4xl text-left">
        <h2 className="mb-2 text-center text-sm font-medium uppercase tracking-widest text-fd-muted-foreground">
          Every command, one keystroke away
        </h2>
        <p className="mb-8 text-center text-sm text-fd-muted-foreground">
          All {totalCommands} commands below ship in the TUI — type{' '}
          <code className="rounded bg-fd-muted px-1 py-0.5 font-mono text-[13px]">
            /
          </code>{' '}
          for autocomplete, or <code className="rounded bg-fd-muted px-1 py-0.5 font-mono text-[13px]">/features</code> to
          list them live.
        </p>

        <div className="grid gap-x-8 gap-y-6 sm:grid-cols-2">
          {commandGroups.map((group) => (
            <section key={group.label}>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-fd-primary">
                {group.label}
              </h3>
              <ul className="flex flex-wrap gap-1.5">
                {group.commands.map((cmd) => (
                  <li key={cmd}>
                    <code className="inline-block rounded-md border border-fd-border bg-fd-card px-2 py-0.5 font-mono text-[13px] text-fd-muted-foreground transition-colors hover:border-fd-primary hover:text-fd-foreground">
                      {cmd}
                    </code>
                  </li>
                ))}
              </ul>
            </section>
          ))}
          <section>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-fd-primary">
              Any tool
            </h3>
            <ul className="flex flex-wrap gap-1.5">
              <li>
                <code className="inline-block rounded-md border border-fd-border bg-fd-card px-2 py-0.5 font-mono text-[13px] text-fd-muted-foreground">
                  /tool &lt;name&gt;
                </code>
              </li>
            </ul>
            <p className="mt-2 text-xs text-fd-muted-foreground">
              Every registry tool is slash-addressable — new tools appear in
              autocomplete automatically.
            </p>
          </section>
        </div>

        <p className="mt-6 text-center text-sm text-fd-muted-foreground">
          Full reference with arguments and examples:{' '}
          <Link
            href="/docs/slash-commands"
            className="font-medium text-fd-primary hover:underline"
          >
            Slash Commands →
          </Link>
        </p>
      </div>

      <div className="mt-20 w-full text-left">
        <h2 className="mb-6 text-center text-sm font-medium uppercase tracking-widest text-fd-muted-foreground">
          Everything the agent can do
        </h2>
        <Cards className="mx-auto max-w-4xl">
          {features.map((f) => (
            <Card
              key={f.title}
              title={f.title}
              description={f.description}
              href={f.href}
            />
          ))}
        </Cards>
      </div>
    </main>
  );
}
