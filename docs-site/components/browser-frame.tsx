import type { ReactNode } from 'react';

/**
 * Browser-chrome frame for the hero product shot.
 *
 * Mastra presents its hero as a screenshot inside a window with traffic lights
 * and a URL pill. Rendering the frame ourselves (rather than shipping an image)
 * keeps the content live text — it stays crisp at any zoom, is selectable and
 * screen-reader accessible, and re-themes with the rest of the site.
 */
export function BrowserFrame({
  url = 'localhost:3000',
  title,
  children,
  className = '',
}: {
  url?: string;
  /** Small label pinned to the right of the URL pill. */
  title?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`browser-shell ${className}`}>
      <div className="browser-bar">
        <span className="size-2.5 shrink-0 rounded-full bg-[#ff5f57]/80" />
        <span className="size-2.5 shrink-0 rounded-full bg-[#febc2e]/80" />
        <span className="size-2.5 shrink-0 rounded-full bg-[#28c840]/80" />

        <span className="browser-url">
          <span className="opacity-60" aria-hidden>
            ⌂
          </span>
          <span className="truncate">{url}</span>
        </span>

        {title ? (
          <span className="hidden shrink-0 font-mono text-[10px] uppercase tracking-wider text-fd-muted-foreground sm:inline">
            {title}
          </span>
        ) : (
          <span className="hidden w-[52px] shrink-0 sm:block" aria-hidden />
        )}
      </div>
      {children}
    </div>
  );
}
