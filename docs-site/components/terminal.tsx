'use client';

import { useEffect, useRef, useState } from 'react';

/** One styled run of text inside a terminal line. */
export type TermToken = {
  text: string;
  /** `dim` prompts/comments · `cmd` typed input · `path` file paths ·
   *  `ok` success · `warn` retries · `accent` highlights. */
  tone?: 'dim' | 'cmd' | 'path' | 'ok' | 'warn' | 'accent';
};

export type TermLine = {
  tokens: TermToken[];
  /** Extra dwell time (ms) before the next line appears. */
  pause?: number;
};

const TONE_CLASS: Record<NonNullable<TermToken['tone']>, string> = {
  dim: 'text-fd-muted-foreground',
  cmd: 'text-fd-foreground',
  path: 'text-[hsl(172_66%_58%)]',
  ok: 'text-[hsl(142_76%_58%)]',
  warn: 'text-[hsl(38_92%_62%)]',
  accent: 'text-[hsl(142_76%_66%)]',
};

/**
 * A terminal window that types its transcript in line by line.
 *
 * The reveal is driven by a single `setTimeout` chain rather than per-line
 * effects, so there is exactly one pending timer at a time and unmounting
 * mid-animation cannot leave stragglers behind. Under
 * `prefers-reduced-motion` every line renders immediately.
 */
export function Terminal({
  title,
  lines,
  lineDelay = 320,
  chrome = true,
}: {
  title: string;
  lines: TermLine[];
  lineDelay?: number;
  /**
   * When false the window traffic lights and title bar are omitted, so the
   * transcript can be nested inside `BrowserFrame` without showing two sets of
   * chrome.
   */
  chrome?: boolean;
}) {
  const [shown, setShown] = useState(0);
  const [reduced, setReduced] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (query.matches) {
      setReduced(true);
      setShown(lines.length);
      return;
    }

    let index = 0;
    const step = () => {
      index += 1;
      setShown(index);
      if (index < lines.length) {
        timer.current = setTimeout(step, lineDelay + (lines[index - 1]?.pause ?? 0));
      }
    };
    timer.current = setTimeout(step, 260);

    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [lines, lineDelay]);

  const complete = shown >= lines.length;

  return (
    <div className="term-shell overflow-hidden rounded-xl">
      {/* Window chrome */}
      {chrome ? (
        <div className="flex items-center gap-2 border-b border-white/[0.07] px-4 py-3">
          <span className="size-2.5 rounded-full bg-[#ff5f57]/80" />
          <span className="size-2.5 rounded-full bg-[#febc2e]/80" />
          <span className="size-2.5 rounded-full bg-[#28c840]/80" />
          <span className="ml-2 font-mono text-xs text-fd-muted-foreground">{title}</span>
          <span className="ml-auto hidden items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-fd-muted-foreground sm:flex">
            <span className="size-1.5 rounded-full bg-[hsl(142_76%_58%)] animate-pulse-soft" />
            live
          </span>
        </div>
      ) : null}

      {/* Transcript */}
      <pre className="scroll-fade overflow-x-auto p-4 font-mono text-[13px] leading-[1.75] sm:p-5">
        <code>
          {lines.map((line, i) => {
            const isVisible = i < shown;
            const isLast = i === shown - 1;
            return (
              <span
                key={i}
                aria-hidden={!isVisible}
                className="block transition-[opacity,transform] duration-500 ease-out motion-reduce:transition-none"
                style={{
                  opacity: isVisible ? 1 : 0,
                  transform: isVisible ? 'none' : 'translateY(4px)',
                }}
              >
                {line.tokens.map((tok, j) => (
                  <span key={j} className={tok.tone ? TONE_CLASS[tok.tone] : undefined}>
                    {tok.text}
                  </span>
                ))}
                {/* Caret trails the line currently being "typed". */}
                {isLast && !complete && !reduced ? (
                  <span className="caret ml-1" aria-hidden />
                ) : null}
                {isLast && complete && !reduced && i === lines.length - 1 ? (
                  <span className="caret ml-1" aria-hidden />
                ) : null}
              </span>
            );
          })}
        </code>
      </pre>
    </div>
  );
}
