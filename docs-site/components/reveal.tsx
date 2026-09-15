'use client';

import { useEffect, useRef, useState, type ReactNode } from 'react';

/**
 * Reveals its children once they scroll into view.
 *
 * The animation itself lives in `global.css` (`.reveal` /
 * `[data-visible='true']`) so it is disabled wholesale under
 * `prefers-reduced-motion` without any JS branching here. If
 * `IntersectionObserver` is unavailable — or the element is already on screen
 * at first paint — we mark it visible immediately rather than leaving content
 * stuck at `opacity: 0`.
 */
export function Reveal({
  children,
  className = '',
  delay = 0,
  as: Tag = 'div',
}: {
  children: ReactNode;
  className?: string;
  /** Stagger offset in milliseconds. */
  delay?: number;
  as?: 'div' | 'section' | 'li';
}) {
  const ref = useRef<HTMLElement | null>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    if (typeof IntersectionObserver === 'undefined') {
      setVisible(true);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setVisible(true);
            observer.disconnect();
          }
        }
      },
      // Fire a little before the element reaches the viewport edge so the
      // motion reads as "already happening" rather than as a late pop-in.
      { rootMargin: '0px 0px -12% 0px', threshold: 0.08 },
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return (
    <Tag
      // `ref` is typed per-tag by React; the union here is narrower than the
      // cast implies but always correct for the three tags we allow.
      ref={ref as never}
      data-visible={visible ? 'true' : 'false'}
      className={`reveal ${className}`}
      style={delay ? { transitionDelay: `${delay}ms` } : undefined}
    >
      {children}
    </Tag>
  );
}
