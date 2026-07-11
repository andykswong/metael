import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// Scans this package's own src for dynamic-code escapes. Strips comments+strings,
// then rejects eval/Function/timer-string forms. The eval-free guarantee is load-bearing: the
// interpreter must never reach a dynamic-code construct.
const here = dirname(fileURLToPath(import.meta.url));
const FORBIDDEN = [/\beval\s*\(/, /new\s+Function\b/, /\bFunction\s*\(/, /setTimeout\s*\(\s*['"`]/, /setInterval\s*\(\s*['"`]/, /GeneratorFunction/];

function strip(src: string): string {
  return src.replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/'(?:[^'\\]|\\.)*'/g, "''").replace(/"(?:[^"\\]|\\.)*"/g, '""').replace(/`(?:[^`\\]|\\.)*`/g, '``');
}

describe('eval-free source scan', () => {
  const files = readdirSync(here).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts'));
  it('finds at least 4 source files', () => { expect(files.length).toBeGreaterThanOrEqual(4); });
  for (const f of files) {
    it(`${f} contains no dynamic-code escape`, () => {
      const stripped = strip(readFileSync(join(here, f), 'utf8'));
      for (const pat of FORBIDDEN) expect(stripped).not.toMatch(pat);
    });
  }
});
