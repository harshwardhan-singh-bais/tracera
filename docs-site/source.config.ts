import { defineConfig } from 'fumadocs-mdx/config';

/**
 * Syntax highlighting theme.
 *
 * The fumadocs default is `{ light: 'github-light', dark: 'github-dark' }`.
 * That combination is what made the docs code blocks look almost monochrome:
 * `github-dark` carries only a handful of distinct hues, and it paints comments
 * at `#6A737D` — a 4.2:1 contrast ratio against this near-black canvas, below
 * the WCAG AA 4.5:1 floor for body text. Unclassified identifiers all resolved
 * to one flat colour, so blocks read as grey-on-black mush.
 *
 * The docs site is dark-only (forced in `RootProvider`), so shipping a light
 * palette nobody will ever see earns nothing. Pointing *both* slots at a theme
 * built for a dark canvas gives more distinct, more legible hues.
 *
 * `defaultColor: 'dark'` is the load-bearing detail. In a dual-theme setup shiki
 * writes the primary theme to the inline `color` and the other to a
 * `--shiki-*` variable, then fumadocs' stylesheet switches them under `.dark`:
 *
 *   .dark .shiki code span { color: var(--shiki-dark) }
 *
 * With the default (`'light'`) that means `github-light` is what actually shows,
 * and the theme you asked for is merely the override — the colours look like
 * whichever slot happened to win. Selecting `'dark'` makes the inline colour
 * authoritative and consistent with the forced-dark theme.
 *
 * Measured contrast against `hsl(0 0% 2%)` — every token clears WCAG AA:
 *   plain #f0f3f6 18.3:1 · comment #bdc4cc 11.6:1 · variable #91cbff 11.8:1
 *   constant #ff9492 9.6:1 · entity #ffb757 11.8:1 · function #dbb7ff 11.9:1
 *
 * `rehypeCodeOptions` is fumadocs-mdx's supported hook: it injects `rehypeCode`
 * into the pipeline itself and merges these options, so there is no need to
 * re-declare the plugin (and no risk of dropping the transformers it installs
 * by default).
 */
export default defineConfig({
  mdxOptions: {
    rehypeCodeOptions: {
      themes: {
        light: 'github-dark-high-contrast',
        dark: 'github-dark-high-contrast',
      },
      defaultColor: 'dark',
    },
  },
});
