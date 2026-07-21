import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
describe('@metael/math/lang import boundary', () => {
  it('src/lang/** imports only @metael/lang + the core sibling; no third-party', () => {
    const dir = dirname(fileURLToPath(import.meta.url));
    const files = readdirSync(dir).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts'));
    const importRe = /\bfrom\s+['"]([^'".][^'"]*)['"]/g;
    const offenders: string[] = [];
    for (const f of files) for (const m of readFileSync(join(dir, f), 'utf8').matchAll(importRe)) {
      if (m[1] === '@metael/lang' || m[1] === '@metael/math') continue;
      offenders.push(`${f}: ${m[1]}`);
    }
    expect(offenders).toEqual([]);
  });
});
