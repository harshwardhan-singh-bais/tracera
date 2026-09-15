import Link from 'next/link';
import { Mail } from 'lucide-react';
import { appName } from '@/lib/shared';

/** Footer link columns. Mastra-style: grouping by audience, not by page type. */
const COLUMNS: { heading: string; links: { label: string; href: string }[] }[] = [
  {
    heading: 'Engine',
    links: [
      { label: 'Hybrid Retrieval', href: '/docs/features/retrieval' },
      { label: 'Structural Analysis', href: '/docs/features/structural' },
      { label: 'Agent Fleet', href: '/docs/features/agent' },
      { label: 'Memory Layer', href: '/docs/features/memory' },
      { label: 'Quality Gates', href: '/docs/features/quality' },
    ],
  },
  {
    heading: 'Interfaces',
    links: [
      { label: 'TUI', href: '/docs/tui' },
      { label: 'Headless CLI', href: '/docs/cli' },
      { label: 'Slash Commands', href: '/docs/slash-commands' },
      { label: 'MCP Server', href: '/docs/mcp' },
      { label: 'Configuration', href: '/docs/configuration' },
    ],
  },
  {
    heading: 'Get Started',
    links: [
      { label: 'Installation', href: '/docs/installation' },
      { label: 'Quickstart', href: '/docs/quickstart' },
      { label: 'Architecture', href: '/docs/architecture' },
      { label: 'Documentation', href: '/docs' },
      { label: 'LLM Context', href: '/llms.txt' },
    ],
  },
  {
    heading: 'Project',
    links: [
      { label: 'README', href: '/docs' },
      { label: 'Memory System', href: '/docs/features/memory' },
      { label: 'Safety Controls', href: '/docs/configuration' },
      { label: 'Observability', href: '/docs/cli' },
      { label: 'Evaluation', href: '/docs/cli' },
    ],
  },
];

/**
 * Multi-column footer with a newsletter block.
 *
 * The subscribe field is presentational only — there is no backend to post to,
 * so it is a non-submitting form. Wiring it to a real endpoint is a product
 * decision, and silently accepting addresses we then drop would be worse than
 * showing the field.
 */
export function SiteFooter() {
  return (
    <footer className="mt-24 px-4">
      <div className="footer-shell mx-auto w-full max-w-6xl px-6 py-12 md:px-10 md:py-14">
        <div className="relative grid gap-12 lg:grid-cols-[minmax(0,1fr)_auto]">
          {/* Left: wordmark + link columns */}
          <div>
            <Link href="/" className="flex items-center gap-2">
              <span className="text-[hsl(142_76%_58%)]" aria-hidden>
                ◆
              </span>
              <span className="text-base font-semibold tracking-tight [font-variant:small-caps]">
                {appName}
              </span>
            </Link>

            <div className="mt-9 grid grid-cols-2 gap-x-6 gap-y-8 sm:grid-cols-4">
              {COLUMNS.map((col) => (
                <div key={col.heading}>
                  <h3 className="footer-heading">{col.heading}</h3>
                  <ul className="mt-3.5 space-y-2.5">
                    {col.links.map((link) => (
                      <li key={`${col.heading}-${link.label}`}>
                        <Link href={link.href} className="footer-link">
                          {link.label}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          {/* Right: newsletter */}
          <div className="lg:w-80">
            <p className="footer-kicker">
              <Mail className="size-3.5" aria-hidden />
              Get release updates
            </p>
            <form
              className="mt-3 flex items-center gap-2"
              aria-label="Release updates"
            >
              <label htmlFor="footer-email" className="sr-only">
                Email address
              </label>
              <input
                id="footer-email"
                type="email"
                name="email"
                placeholder="you@company.com"
                autoComplete="email"
                className="footer-input"
              />
              <button type="button" className="footer-subscribe">
                Subscribe
              </button>
            </form>
            <p className="mt-2.5 text-[11px] leading-relaxed text-fd-muted-foreground">
              Occasional notes on retrieval, agent reliability and memory. No
              telemetry, no tracking.
            </p>
          </div>
        </div>

        {/* Bottom rule */}
        <div className="rule-fade relative mt-12" />

        <div className="relative mt-5 flex flex-col gap-3 text-[11px] text-fd-muted-foreground sm:flex-row sm:items-center sm:justify-between">
          <span className="inline-flex items-center gap-2">
            <span className="status-dot" aria-hidden />
            {appName} · open source · MIT
          </span>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
            <Link href="/docs" className="footer-link">
              Privacy
            </Link>
            <Link href="/docs" className="footer-link">
              Terms
            </Link>
            <Link href="/llms.txt" className="footer-link">
              llms.txt
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
