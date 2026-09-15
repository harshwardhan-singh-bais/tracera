'use client';

import { useId, useRef, useState, type KeyboardEvent } from 'react';

export type CodeTab = { id: string; label: string; code: string };

type Tone = 'dim' | 'cmd' | 'path' | 'ok' | 'warn' | 'accent' | 'plain';

const TONE_CLASS: Record<Tone, string> = {
  dim: 'text-fd-muted-foreground',
  cmd: 'text-fd-foreground',
  path: 'text-[hsl(172_66%_58%)]',
  ok: 'text-[hsl(142_76%_58%)]',
  warn: 'text-[hsl(38_92%_62%)]',
  accent: 'text-[hsl(142_76%_66%)]',
  plain: 'text-fd-foreground/80',
};

type Segment = { text: string; tone: Tone };

const SEGMENT_PATTERN =
  /(✓|✔|✗|✘)|(\bFAILED\b|\bfailed\b|\bretrying\b)|(\bpassed\b|\bapproved\b|\bOK\b)|([\w./-]+\.(?:py|ts|tsx|js|mjs|md|json|toml|yaml|yml))|(\b\d+(?:\/\d+)?\b)/g;

/**
 * Minimal, dependency-free highlighting for the showcase transcripts.
 *
 * The tab bodies are hand-written transcripts rather than real source, so a
 * full grammar would be overkill. Classifying the handful of shapes that
 * actually appear — comments, prompt prefixes, success/failure markers, file
 * paths and counts — is enough to make them readable at a glance.
 */
function highlight(line: string): Segment[] {
  const trimmed = line.trimStart();

  if (trimmed.startsWith('#')) {
    return [{ text: line, tone: 'dim' }];
  }

  // `$ cmd` / `> cmd` — dim the sigil, brighten the command itself.
  const prompt = line.match(/^(\s*)([$>])(\s*)(.*)$/);
  if (prompt) {
    const [, indent, sigil, gap, rest] = prompt;
    return [
      { text: indent, tone: 'plain' },
      { text: sigil, tone: 'accent' },
      { text: gap, tone: 'plain' },
      { text: rest, tone: 'cmd' },
    ];
  }

  const out: Segment[] = [];
  let cursor = 0;
  SEGMENT_PATTERN.lastIndex = 0;

  for (let m = SEGMENT_PATTERN.exec(line); m; m = SEGMENT_PATTERN.exec(line)) {
    if (m.index > cursor) {
      out.push({ text: line.slice(cursor, m.index), tone: 'dim' });
    }
    const [match, mark, bad, good, file, num] = m;
    const tone: Tone = mark ? 'ok' : bad ? 'warn' : good ? 'ok' : file ? 'path' : num ? 'accent' : 'dim';
    out.push({ text: match, tone });
    cursor = m.index + match.length;
  }

  if (cursor < line.length) {
    out.push({ text: line.slice(cursor), tone: 'dim' });
  }
  return out.length ? out : [{ text: line, tone: 'plain' }];
}

/**
 * Tabbed transcript showcase with full keyboard support.
 *
 * Implements the WAI-ARIA tabs pattern: arrow keys move between tabs,
 * Home/End jump to the ends, and only the active panel is rendered. All
 * panels stay in the DOM-free path (no hidden focus traps) because the
 * transcripts are short and re-rendering is cheap.
 */
export function CodeTabs({ tabs }: { tabs: CodeTab[] }) {
  const [active, setActive] = useState(0);
  const baseId = useId();
  const refs = useRef<Array<HTMLButtonElement | null>>([]);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const last = tabs.length - 1;
    let next: number | null = null;

    if (event.key === 'ArrowRight') next = active === last ? 0 : active + 1;
    else if (event.key === 'ArrowLeft') next = active === 0 ? last : active - 1;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = last;

    if (next === null) return;
    event.preventDefault();
    setActive(next);
    refs.current[next]?.focus();
  };

  const tab = tabs[active];

  return (
    <div className="card-edge overflow-hidden rounded-2xl bg-fd-card/60">
      {/* Tab bar */}
      <div
        role="tablist"
        aria-label="Capability showcase"
        onKeyDown={onKeyDown}
        className="flex flex-wrap gap-1 border-b border-white/[0.07] p-2"
      >
        {tabs.map((t, i) => {
          const selected = i === active;
          return (
            <button
              key={t.id}
              ref={(el) => {
                refs.current[i] = el;
              }}
              role="tab"
              id={`${baseId}-tab-${t.id}`}
              aria-selected={selected}
              aria-controls={`${baseId}-panel-${t.id}`}
              tabIndex={selected ? 0 : -1}
              type="button"
              onClick={() => setActive(i)}
              className={`relative rounded-lg px-3.5 py-1.5 font-mono text-[13px] transition-colors duration-200 outline-none focus-visible:ring-2 focus-visible:ring-[hsl(142_70%_45%)] focus-visible:ring-offset-2 focus-visible:ring-offset-[hsl(0_0%_4%)] ${
                selected
                  ? 'bg-[hsl(142_70%_45%/0.13)] text-[hsl(142_76%_66%)]'
                  : 'text-fd-muted-foreground hover:bg-white/[0.04] hover:text-fd-foreground'
              }`}
            >
              {t.label}
            </button>
          );
        })}
      </div>

      {/* Panel */}
      <div
        role="tabpanel"
        id={`${baseId}-panel-${tab.id}`}
        aria-labelledby={`${baseId}-tab-${tab.id}`}
        tabIndex={0}
        className="outline-none"
      >
        <pre
          key={tab.id}
          className="scroll-fade overflow-x-auto p-4 font-mono text-[12.5px] leading-[1.8] sm:p-5 motion-safe:animate-[tr-fade-up_0.4s_ease-out]"
        >
          <code>
            {tab.code.split('\n').map((line, i) => (
              <span key={i} className="block">
                {highlight(line).map((seg, j) => (
                  <span key={j} className={TONE_CLASS[seg.tone]}>
                    {seg.text}
                  </span>
                ))}
              </span>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
}
