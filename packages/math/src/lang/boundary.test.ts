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
      const spec = m[1]!;
      // A sub-path of an allowed package (e.g. '@metael/lang/profile') resolves to its root package name.
      const pkg = spec.startsWith('@') ? spec.split('/').slice(0, 2).join('/') : spec.split('/')[0]!;
      if (pkg === '@metael/lang' || pkg === '@metael/math') continue;
      offenders.push(`${f}: ${spec}`);
    }
    expect(offenders).toEqual([]);
  });
});
