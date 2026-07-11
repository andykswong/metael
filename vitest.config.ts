import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    projects: [
      {
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
