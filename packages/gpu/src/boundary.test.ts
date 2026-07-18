import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

describe('@metael/gpu import boundary (self-containment invariant)', () => {
  it('non-test src imports only @metael/lang, @metael/runtime, and relative paths', () => {
    const srcDir = dirname(fileURLToPath(import.meta.url));
    const walk = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
      e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]);
    const files = walk(srcDir).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts') && !f.endsWith('.browser.test.ts'));
    const offenders: string[] = [];
    const importRe = /\bfrom\s+['"]([^'".][^'"]*)['"]/g;
    for (const f of files) {
      const text = readFileSync(f, 'utf8');
      for (const m of text.matchAll(importRe)) {
        const spec = m[1]!;
        if (spec === '@metael/lang' || spec === '@metael/runtime') continue;
        offenders.push(`${f}: ${spec}`);
      }
    }
    expect(offenders).toEqual([]);
  });
});
