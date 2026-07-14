import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { defineConfig } from 'vite';

// Resolve workspace packages from live source (mirrors the root tsconfig paths), not built dist.
const alias = {
  '@metael/lang': fileURLToPath(new URL('../../packages/lang/src/index.ts', import.meta.url)),
  '@metael/runtime': fileURLToPath(new URL('../../packages/runtime/src/index.ts', import.meta.url)),
  '@metael/vdom': fileURLToPath(new URL('../../packages/vdom/src/index.ts', import.meta.url)),
};

// RELATIVE base (`./`): emitted asset URLs are relative to the HTML file, so the site works from ANY
// mount point without reconfiguration — the GitHub Pages project sub-path today AND future root custom domain.
// In-app links between pages are likewise relative (`play.html`, `api/index.html`) — see landing-source.ts.
export default defineConfig({
  base: './',
  resolve: { alias },
  build: {
    target: 'esnext',
    rollupOptions: {
      input: {
        index: resolve(fileURLToPath(new URL('.', import.meta.url)), 'index.html'),
        play: resolve(fileURLToPath(new URL('.', import.meta.url)), 'play.html'),
      },
    },
  },
});
