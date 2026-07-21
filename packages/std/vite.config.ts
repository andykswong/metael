import { defineConfig } from 'vite';
import dts from 'vite-plugin-dts';
export default defineConfig({
  plugins: [dts({ tsconfigPath: './tsconfig.build.json', exclude: ['**/*.test.ts'] })],
  build: {
    lib: { entry: { index: 'src/index.ts' }, formats: ['es'] },
    target: 'esnext',
    rollupOptions: {
      external: (id) => !id.startsWith('.') && !id.startsWith('/') && !id.startsWith('\0'),
      output: { preserveModules: true, preserveModulesRoot: 'src', entryFileNames: '[name].js' },
    },
  },
});
