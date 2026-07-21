import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// THE TRIPWIRE: @metael/lang is the domain-agnostic language kernel — it imports NOTHING from any other
// package (no @metael/*, no third-party). A `lang → math` edge (or any package edge) would break the
// layering that lets a domain library plug numeric builtins in through the registry seam. If this fails,
// an import was introduced into a lang source file that must not be there.
describe('@metael/lang import boundary (self-containment invariant)', () => {
  it('non-test src imports only relative paths — no bare package specifier', () => {
    const srcDir = dirname(fileURLToPath(import.meta.url));
    const walk = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
      e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]);
    const files = walk(srcDir).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts') && !f.endsWith('.browser.test.ts'));
    const offenders: string[] = [];
    const importRe = /\bfrom\s+['"]([^'".][^'"]*)['"]/g;
    for (const f of files) {
      const text = readFileSync(f, 'utf8');
      for (const m of text.matchAll(importRe)) offenders.push(`${f}: ${m[1]}`);
    }
    expect(offenders).toEqual([]);
  });
});
