import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

// Resolve workspace packages from live source (mirrors tsconfig paths), not built dist via the
// node_modules symlink. Applied per project — projects resolve independently.
const alias = {
  '@metael/lang': fileURLToPath(new URL('./packages/lang/src/index.ts', import.meta.url)),
  '@metael/runtime': fileURLToPath(new URL('./packages/runtime/src/index.ts', import.meta.url)),
  '@metael/vdom': fileURLToPath(new URL('./packages/vdom/src/index.ts', import.meta.url)),
};

export default defineConfig({
  resolve: { alias },
  test: {
    projects: [
      {
        resolve: { alias },
        test: {
          name: 'node',
          environment: 'node',
          include: ['packages/**/src/**/*.test.ts', 'apps/**/src/**/*.test.ts'],
          exclude: ['**/*.browser.test.ts', '**/dist/**'],
        },
      },
      {
        resolve: { alias },
        test: {
          name: 'browser',
          include: ['packages/**/src/**/*.browser.test.ts', 'apps/**/src/**/*.browser.test.ts'],
          exclude: ['**/dist/**'],
          browser: {
            enabled: true,
            provider: playwright(),
            headless: true,
            instances: [{ browser: 'chromium' }],
          },
        },
      },
    ],
  },
});
