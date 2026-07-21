import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

// Resolve workspace packages from live source (mirrors tsconfig paths), not built dist via the
// node_modules symlink. Applied per project — projects resolve independently.
const alias = {
  '@metael/lang': fileURLToPath(new URL('./packages/lang/src/index.ts', import.meta.url)),
  '@metael/runtime': fileURLToPath(new URL('./packages/runtime/src/index.ts', import.meta.url)),
  '@metael/vdom/lang': fileURLToPath(new URL('./packages/vdom/src/lang/index.ts', import.meta.url)),
  '@metael/vdom': fileURLToPath(new URL('./packages/vdom/src/index.ts', import.meta.url)),
  '@metael/gpu/builder': fileURLToPath(new URL('./packages/gpu/src/builder/index.ts', import.meta.url)),
  '@metael/gpu/lang': fileURLToPath(new URL('./packages/gpu/src/lang/index.ts', import.meta.url)),
  '@metael/gpu': fileURLToPath(new URL('./packages/gpu/src/index.ts', import.meta.url)),
  '@metael/math/lang': fileURLToPath(new URL('./packages/math/src/lang/index.ts', import.meta.url)),
  '@metael/math': fileURLToPath(new URL('./packages/math/src/core/index.ts', import.meta.url)),
  '@metael/std': fileURLToPath(new URL('./packages/std/src/index.ts', import.meta.url)),
};

export default defineConfig({
  resolve: { alias },
  test: {
    // Coverage (v8) over the publishable package SOURCES only — tests, build output, test fixtures, and the
    // showcase app are excluded. Emitted as lcov (Codecov) + text (local). This is top-level, so
    // `vitest --coverage` instruments BOTH projects (v8 collects browser coverage over the Chrome DevTools
    // Protocol) and merges them — so vdom's Chromium-only DOM paths are counted. WGSL shader bodies run on
    // the GPU, not in JS, so they stay uncoverable; their JS emitters are covered.
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      include: ['packages/*/src/**/*.ts'],
      exclude: [
        '**/*.test.ts', '**/*.browser.test.ts', '**/dist/**', 'apps/**',
        'packages/*/src/test/**',   // shared test fixtures, not shipped code
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
