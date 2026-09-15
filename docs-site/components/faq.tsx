'use client';

import { useId, useState } from 'react';
import { Plus } from 'lucide-react';

export type FaqItem = { q: string; a: string };

/**
 * Accessible FAQ accordion with a smooth open/close.
 *
 * The height animation uses the `grid-template-rows: 0fr → 1fr` technique
 * rather than measuring scrollHeight in JS — it animates correctly without a
 * layout read, needs no resize handling, and degrades to an instant
 * appearance under `prefers-reduced-motion`.
 */
export function Faq({ items }: { items: FaqItem[] }) {
  const [open, setOpen] = useState<number | null>(0);
  const baseId = useId();

  return (
    <div className="card-edge divide-y divide-white/[0.07] overflow-hidden rounded-2xl bg-fd-card/60">
      {items.map((item, i) => {
        const expanded = open === i;
        return (
          <div key={item.q}>
            <h3>
              <button
                type="button"
                id={`${baseId}-btn-${i}`}
                aria-expanded={expanded}
                aria-controls={`${baseId}-panel-${i}`}
                onClick={() => setOpen(expanded ? null : i)}
                className="flex w-full cursor-pointer items-center justify-between gap-4 px-5 py-4 text-left font-medium transition-colors duration-200 outline-none hover:bg-white/[0.03] focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[hsl(142_70%_45%)]"
              >
                <span>{item.q}</span>
                <Plus
                  aria-hidden
                  className={`size-4 shrink-0 text-fd-muted-foreground transition-transform duration-300 motion-reduce:transition-none ${
                    expanded ? 'rotate-45 text-[hsl(142_76%_58%)]' : ''
                  }`}
                />
              </button>
            </h3>
            <div
              id={`${baseId}-panel-${i}`}
              role="region"
              aria-labelledby={`${baseId}-btn-${i}`}
              className={`grid transition-[grid-template-rows,opacity] duration-300 ease-out motion-reduce:transition-none ${
                expanded ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'
              }`}
            >
              <div className="overflow-hidden">
                <p className="px-5 pb-5 text-sm leading-relaxed text-fd-muted-foreground">
                  {item.a}
                </p>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
