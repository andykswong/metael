// packages/math/src/core/boundary.test.ts
import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

describe('@metael/math/core import boundary (zero-dependency invariant)', () => {
  it('src/core/** imports only relative paths within core — no bare specifier, no ../lang leak', () => {
    const coreDir = dirname(fileURLToPath(import.meta.url));
    const walk = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
      e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]);
    const files = walk(coreDir).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts') && !f.endsWith('.browser.test.ts'));
    // Non-vacuity guard: prove the walk actually scanned ≥1 core .ts file (else this test could pass trivially).
    expect(files.length).toBeGreaterThan(0);
    // (1) no BARE specifier (a package dep, incl. @metael/lang). (2) no RELATIVE import escaping into a
    // sibling `lang/` dir (../lang/…) — the regex above can't see that, so a second check guards it
    // (guards a core→lang leak via a relative path). Core may only import within core/.
    const bareRe = /\bfrom\s+['"]([^'".][^'"]*)['"]/g;
    const langRe = /\bfrom\s+['"](?:\.\.?\/)+lang\//;
    const offenders: string[] = [];
    for (const f of files) {
      const text = readFileSync(f, 'utf8');
      for (const m of text.matchAll(bareRe)) offenders.push(`${f}: bare '${m[1]}'`);
      if (langRe.test(text)) offenders.push(`${f}: relative import into ../lang`);
    }
    expect(offenders).toEqual([]);
  });
});
