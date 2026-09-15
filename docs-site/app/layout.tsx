import { RootProvider } from 'fumadocs-ui/provider/next';
import './global.css';
import { Inter } from 'next/font/google';

const inter = Inter({
  subsets: ['latin'],
});

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <head>
        {/*
          Marks the document as JS-capable before first paint. The scroll-reveal
          styles are scoped under `.js-reveal`, so if this script never runs the
          landing page renders fully visible instead of hiding every section
          below the fold at opacity 0.
        */}
        <script
          dangerouslySetInnerHTML={{
            __html: `document.documentElement.classList.add('js-reveal')`,
          }}
        />
      </head>
      <body className="flex flex-col min-h-screen">
        <RootProvider
          theme={{ enabled: false, defaultTheme: 'dark', forcedTheme: 'dark' }}
        >
          {children}
        </RootProvider>
      </body>
    </html>
  );
}
