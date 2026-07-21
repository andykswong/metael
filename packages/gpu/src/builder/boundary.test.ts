import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// The builder is the JS authoring front-end for kernels: it assembles the SAME AST the parser emits, using
// ONLY @metael/lang (the AST types + `Environment` for the UserFn closure). It must NOT reach the interpreter
// (@metael/runtime), the engine (@metael/gpu), or the DSL text-parse binding (@metael/gpu/lang) — that
// coupling is what keeps a builder-authored kernel a pure data structure the engine then drives. This walk
// enforces the allowlist over the non-test builder sources.
describe('@metael/gpu/builder import boundary', () => {
  it('src/builder/*.ts imports ONLY @metael/lang (AST types + Environment); not ./lang, not the engine, not runtime', () => {
    const dir = dirname(fileURLToPath(import.meta.url));
    const files = readdirSync(dir).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts'));
    expect(files.length).toBeGreaterThan(0); // non-vacuity: there ARE non-test sources to check
    const importRe = /\bfrom\s+['"]([^'"]+)['"]/g;
    const offenders: string[] = [];
    for (const f of files)
      for (const m of readFileSync(join(dir, f), 'utf8').matchAll(importRe)) {
        const spec = m[1] ?? '';
        if (spec.startsWith('.')) continue; // builder-internal siblings
        if (spec === '@metael/lang') continue; // AST types + Environment (the UserFn is built directly)
        offenders.push(`${f}: ${spec}`); // NO @metael/runtime, NO @metael/gpu, NO @metael/gpu/lang, NO @metael/math
      }
    expect(offenders).toEqual([]);
  });
});
