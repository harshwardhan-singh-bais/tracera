import Link from 'next/link';
import { ArrowRight } from 'lucide-react';

/**
 * Mastra-style announcement pill.
 *
 * Sits at the very top of the landing page and hangs off the upper edge: the
 * pill is full-width at the top and rounded only at the bottom, so it reads as
 * a tab peeking out of the page rather than a floating banner.
 */
export function AnnounceBar({
  label = 'New',
  text = 'Bi-temporal memory with contradiction resolution',
  href = '/docs/features/memory',
}: {
  label?: string;
  text?: string;
  href?: string;
}) {
  return (
    <div className="flex justify-center pt-3">
      <Link
        href={href}
        className="announce-pill group inline-flex max-w-[92vw] items-center gap-2 rounded-b-xl px-4 py-2 text-[13px] font-medium"
      >
        <span className="hidden shrink-0 opacity-70 sm:inline">{label}</span>
        <span className="hidden shrink-0 opacity-40 sm:inline" aria-hidden>
          ·
        </span>
        <span className="truncate">{text}</span>
        <ArrowRight
          className="size-3.5 shrink-0 transition-transform duration-300 group-hover:translate-x-0.5"
          aria-hidden
        />
      </Link>
    </div>
  );
}
