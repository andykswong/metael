import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

describe('@metael/gpu/lang import boundary', () => {
  it('src/lang/*.ts imports only @metael/{lang,runtime,math,math/lang} + core siblings', () => {
    const dir = dirname(fileURLToPath(import.meta.url));
    const files = readdirSync(dir).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts') && !f.endsWith('.browser.test.ts'));
    expect(files.length).toBeGreaterThan(0);
    const importRe = /\bfrom\s+['"]([^'"]+)['"]/g;
    const allowed = new Set(['@metael/lang', '@metael/lang/profile', '@metael/runtime', '@metael/math', '@metael/math/lang']);
    const offenders: string[] = [];
    for (const f of files) for (const m of readFileSync(join(dir, f), 'utf8').matchAll(importRe)) {
      const spec = m[1]!;
      if (spec.startsWith('.')) continue;             // core sibling (../) or lang sibling (./)
      if (allowed.has(spec)) continue;
      offenders.push(`${f}: ${spec}`);
    }
    expect(offenders).toEqual([]);
  });
});
