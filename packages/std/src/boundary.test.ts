// packages/std/src/boundary.test.ts
import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// @metael/std may depend on ONE package only: the language kernel. No third-party runtime
// dependency, and no sibling @metael/* package (runtime/vdom/gpu/math). The build/publish
// contract rests on this, so it is asserted as an invariant here rather than by convention.
const ALLOWLIST = new Set(['@metael/lang']);

describe('@metael/std import boundary (depends only on @metael/lang)', () => {
  it('src/** bare imports are limited to the @metael/lang kernel', () => {
    const srcDir = dirname(fileURLToPath(import.meta.url));
    // Recurse (do NOT flat-read) so a future src/ subdirectory can never silently escape this gate.
    const walk = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
      e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]);
    const files = walk(srcDir).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts') && !f.endsWith('.browser.test.ts'));
    // Non-vacuity guard: prove the walk actually scanned ≥1 shipped .ts file (else this test could pass trivially).
    expect(files.length).toBeGreaterThan(0);
    // A BARE specifier (no leading '.' or '/') is a package dep; every one must be in the allowlist.
    const bareRe = /\bfrom\s+['"]([^'".][^'"]*)['"]/g;
    const offenders: string[] = [];
    for (const f of files) {
      const text = readFileSync(f, 'utf8');
      for (const m of text.matchAll(bareRe)) {
        const spec = m[1]!;
        // Sub-path of an allowed package (e.g. '@metael/lang/x') resolves to its root package name.
        const pkg = spec.startsWith('@') ? spec.split('/').slice(0, 2).join('/') : spec.split('/')[0]!;
        if (!ALLOWLIST.has(pkg)) offenders.push(`${f}: bare '${spec}'`);
      }
    }
    expect(offenders).toEqual([]);
  });
});
