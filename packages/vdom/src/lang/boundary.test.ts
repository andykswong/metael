import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

describe('@metael/vdom/lang import boundary', () => {
  it('src/lang/*.ts imports only @metael/{lang,runtime} + core siblings', () => {
    const dir = dirname(fileURLToPath(import.meta.url));
    const files = readdirSync(dir).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts'));
    expect(files.length).toBeGreaterThan(0); // non-vacuity: an emptied/renamed lang/ dir must fail, not pass
    const importRe = /\bfrom\s+['"]([^'"]+)['"]/g;
    const offenders: string[] = [];
    for (const f of files) for (const m of readFileSync(join(dir, f), 'utf8').matchAll(importRe)) {
      const spec = m[1]!;
      if (spec.startsWith('.')) continue;                       // core sibling or lang sibling
      if (spec === '@metael/lang' || spec === '@metael/lang/profile' || spec === '@metael/runtime') continue;
      offenders.push(`${f}: ${spec}`);
    }
    expect(offenders).toEqual([]);
  });
});
