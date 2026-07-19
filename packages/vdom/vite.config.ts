import { defineConfig } from 'vite';
import dts from 'vite-plugin-dts';

export default defineConfig({
  // Exclude tests + the example fixtures under src/test/ (not part of the public API; not reachable from
  // index.ts, so no .js is emitted for them — this drops their stray .d.ts from the published tarball).
  plugins: [dts({ tsconfigPath: './tsconfig.build.json', exclude: ['**/*.test.ts', '**/*.browser.test.ts', 'src/test/**'] })],
  build: {
    lib: { entry: 'src/index.ts', formats: ['es'], fileName: 'index' },
    target: 'esnext',
    rollupOptions: {
      // Externalize every bare specifier (@metael/lang, @metael/runtime) — bundle only relative sources.
      external: (id) => !id.startsWith('.') && !id.startsWith('/') && !id.startsWith('\0'),
      output: { preserveModules: true, preserveModulesRoot: 'src', entryFileNames: '[name].js' },
    },
  },
});
