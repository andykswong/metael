import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

// Resolve workspace packages from live source (mirrors tsconfig `paths`),
// not their built dist via the node_modules symlink. Applied per project —
// projects resolve independently, so a top-level alias would not reach them.
const alias = {
  '@metael/lang': fileURLToPath(new URL('./packages/lang/src/index.ts', import.meta.url)),
  '@metael/runtime': fileURLToPath(new URL('./packages/runtime/src/index.ts', import.meta.url)),
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
          include: ['packages/**/src/**/*.test.ts'],
          exclude: ['**/*.browser.test.ts', '**/dist/**'],
        },
      },
    ],
  },
});
