import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

describe('@metael/gpu import boundary (self-containment invariant)', () => {
  it('non-test src imports only @metael/lang, @metael/runtime, @metael/math(/lang), and relative paths', () => {
    const srcDir = dirname(fileURLToPath(import.meta.url));
    const walk = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
      e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]);
    const files = walk(srcDir).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts') && !f.endsWith('.browser.test.ts'));
    const offenders: string[] = [];
    const importRe = /\bfrom\s+['"]([^'".][^'"]*)['"]/g;
    const allowed = new Set(['@metael/lang', '@metael/runtime', '@metael/math', '@metael/math/lang']);
    for (const f of files) {
      const text = readFileSync(f, 'utf8');
      for (const m of text.matchAll(importRe)) {
        const spec = m[1]!;
        if (allowed.has(spec)) continue;
        offenders.push(`${f}: ${spec}`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it('the API-first core (src/*.ts, excluding lang/) never pulls the interpreter (evaluateProgram/GpuHostEnv/./lang)', () => {
    const srcDir = dirname(fileURLToPath(import.meta.url));
    // Walk the core, skipping the DSL-binding subdir: the interpreter (evaluateProgram) + the DSL vocabulary
    // (GpuHostEnv) live behind ./lang, so a core file reaching for either would collapse the split.
    const walk = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
      if (e.name === 'lang') return [];
      return e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)];
    });
    const files = walk(srcDir).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts') && !f.endsWith('.browser.test.ts'));
    expect(files.length).toBeGreaterThan(0);
    // Scan CODE, not prose: strip block + line comments so a doc-comment that NAMES the forbidden tokens
    // (to explain the boundary) is not a false positive — only a real import/use trips the guard.
    const stripComments = (t: string): string => t.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1');
    const bad = [/\bfrom\s+['"]\.\/lang\//, /\bfrom\s+['"]\.\.\/lang\//, /\bevaluateProgram\b/, /\bGpuHostEnv\b/];
    const offenders: string[] = [];
    for (const f of files) { const t = stripComments(readFileSync(f, 'utf8')); if (bad.some((re) => re.test(t))) offenders.push(f); }
    expect(offenders).toEqual([]);
  });
});
