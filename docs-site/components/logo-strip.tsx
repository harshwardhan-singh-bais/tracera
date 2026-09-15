import { appName } from '@/lib/shared';

/**
 * Customer / integration logo strip.
 *
 * These are the LLM providers and runtimes the engine actually ships adapters
 * for — not aspirational logos. Names are rendered as styled wordmarks rather
 * than vendored brand assets, so there is nothing to keep in sync and no
 * trademark asset is redistributed. Each links to the configuration reference
 * where the corresponding adapter is documented.
 */
const LOGOS = [
  { name: 'OpenAI', mark: '◎' },
  { name: 'Anthropic', mark: '✦' },
  { name: 'Gemini', mark: '✧' },
  { name: 'Groq', mark: '⚡' },
  { name: 'Cerebras', mark: '◈' },
  { name: 'Mistral', mark: '⌁' },
  { name: 'Ollama', mark: '⬡' },
];

export function LogoStrip() {
  return (
    <section className="relative mx-auto w-full max-w-6xl px-4 pt-14 pb-16">
      {/* Hairline above, fading at both ends — separates hero from the strip. */}
      <div className="rule-fade" aria-hidden />
      <p className="mt-12 text-center text-xs font-medium uppercase tracking-[0.18em] text-fd-muted-foreground">
        13 providers · including
      </p>
      <div className="mt-8 flex flex-wrap items-center justify-center gap-x-12 gap-y-6">
        {LOGOS.map((logo) => (
          <span key={logo.name} className="logo-mark">
            <span className="text-lg" aria-hidden>
              {logo.mark}
            </span>
            {logo.name}
          </span>
        ))}
      </div>
      <p className="mx-auto mt-8 max-w-xl text-center text-xs leading-relaxed text-fd-muted-foreground">
        One interface across every provider, with automatic failover — or keep
        inference fully local with Ollama.
      </p>
    </section>
  );
}
