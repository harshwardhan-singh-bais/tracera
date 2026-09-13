'use client';

import { useState } from 'react';
import { Check, Copy } from 'lucide-react';

export function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <button
      type="button"
      aria-label="Copy to clipboard"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          // clipboard unavailable — ignore
        }
      }}
      className="inline-flex items-center justify-center rounded-md p-1.5 text-fd-muted-foreground transition-colors hover:bg-fd-muted hover:text-fd-foreground"
    >
      {copied ? (
        <Check className="size-4 text-fd-primary" />
      ) : (
        <Copy className="size-4" />
      )}
    </button>
  );
}
