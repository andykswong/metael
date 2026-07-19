import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

// Resolve workspace packages from live source (mirrors tsconfig paths), not built dist via the
// node_modules symlink. Applied per project — projects resolve independently.
const alias = {
  '@metael/lang': fileURLToPath(new URL('./packages/lang/src/index.ts', import.meta.url)),
  '@metael/runtime': fileURLToPath(new URL('./packages/runtime/src/index.ts', import.meta.url)),
  '@metael/vdom': fileURLToPath(new URL('./packages/vdom/src/index.ts', import.meta.url)),
  '@metael/gpu': fileURLToPath(new URL('./packages/gpu/src/index.ts', import.meta.url)),
};

export default defineConfig({
  resolve: { alias },
  test: {
    // Coverage (v8) is produced only for the publishable package SOURCES — the kernel + runtime + the
    // vdom domain. Tests, build output, type-only barrels, dev-only demo/example fixtures, and the
    // showcase app are excluded. Emitted as lcov (for Codecov) + text (local). Run via
    // `npm run test:coverage` / `vitest --coverage`. NOTE: this is the NODE project only — parts of
    // @metael/vdom (DOM reconcile/mount/patch) are exercised by *.browser.test.ts in real Chromium,
    // which this run does not instrument, so their node-only lcov reads low; the Codecov gate accounts
    // for that (informational, non-blocking) rather than forcing browser-coverage collection.
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      include: ['packages/*/src/**/*.ts'],
      exclude: [
        '**/*.test.ts', '**/*.browser.test.ts', '**/index.ts', '**/dist/**', 'apps/**',
        'packages/vdom/src/demo.ts', 'packages/vdom/src/examples.ts',   // dev-only fixtures, not public API
      ],
    },
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
            // WebGPU: the bundled Chromium exposes `navigator.gpu`, but `requestAdapter()` returns null
            // with no flags, so the *.browser.test.ts WGSL paths silently fall back to WebGL2. These flags
            // bring up SwiftShader — a software adapter needing NO GPU hardware, so it works on a headless
            // CI runner — letting those tests actually execute the WGSL-only paths (native `inverse()` in
            // WebGL2 masked a real WGSL bug once). Tests still skip gracefully if no adapter comes up.
            provider: playwright({
              launchOptions: {
                args: [
                  '--enable-unsafe-webgpu',
                  '--use-webgpu-adapter=swiftshader',
                  '--enable-features=Vulkan',
                ],
              },
            }),
            headless: true,
            instances: [{ browser: 'chromium' }],
          },
        },
      },
    ],
  },
});
