import Link from 'next/link';
import { HomeLayout } from 'fumadocs-ui/layouts/home';
import { Card, Cards } from 'fumadocs-ui/components/card';
import { baseOptions } from '@/lib/layout.shared';
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

export default function HomePage() {
  return (
    <HomeLayout {...baseOptions()}>
      <div className="flex flex-1 flex-col items-center justify-center px-4 pb-16 pt-24 text-center md:pt-32">
        <p className="mb-4 rounded-full border border-fd-border bg-fd-muted px-3 py-1 text-xs font-medium text-fd-muted-foreground">
          v0.1.0 · MIT licensed · Python 3.12+
        </p>
        <h1 className="mb-4 text-4xl font-bold tracking-tight md:text-6xl">
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

        <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
          <Link
            href="/docs"
            className="rounded-lg bg-fd-primary px-5 py-2.5 font-medium text-fd-primary-foreground transition-colors hover:opacity-90"
          >
            Get Started
          </Link>
          <Link
            href="/docs/slash-commands"
            className="rounded-lg border border-fd-border px-5 py-2.5 font-medium transition-colors hover:bg-fd-muted"
          >
            Browse 80+ slash commands
          </Link>
        </div>

        <div className="mt-12 w-full max-w-2xl overflow-hidden rounded-xl border border-fd-border bg-fd-card text-left text-sm shadow-sm">
          <div className="flex items-center gap-1.5 border-b border-fd-border px-4 py-2.5">
            <span className="size-2.5 rounded-full bg-red-400" />
            <span className="size-2.5 rounded-full bg-yellow-400" />
            <span className="size-2.5 rounded-full bg-green-400" />
            <span className="ml-2 text-xs text-fd-muted-foreground">
              tracera tui
            </span>
          </div>
          <pre className="overflow-x-auto p-4 leading-relaxed">
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

        <div className="mt-16 text-left">
          <Cards>
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
      </div>
    </HomeLayout>
  );
}
