import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

describe('@metael/vdom core import boundary', () => {
  it('src/*.ts (core, non-lang) never imports the lang/ subdir or evaluateProgram/derive', () => {
    const dir = dirname(fileURLToPath(import.meta.url));
    const files = readdirSync(dir, { withFileTypes: true })
      .filter((e) => e.isFile() && e.name.endsWith('.ts') && !e.name.endsWith('.test.ts') && !e.name.endsWith('.browser.test.ts'))
      .map((e) => join(dir, e.name));
    expect(files.length).toBeGreaterThan(0);
    const langRe = /\bfrom\s+['"]\.\/lang\//;
    const deriveRe = /\bfrom\s+['"]@metael\/runtime['"][^;]*\bderive\b/;                     // `... derive` after `from '@metael/runtime'`
    const deriveImportRe = /\bimport\b[^;]*\bderive\b[^;]*\bfrom\s+['"]@metael\/runtime['"]/; // `import { derive } from '@metael/runtime'` (clause before from)
    const evalRe = /\bevaluateProgram\b/;
    const offenders: string[] = [];
    for (const f of files) {
      const t = readFileSync(f, 'utf8');
      if (langRe.test(t)) offenders.push(`${f}: imports ./lang/`);
      if (deriveRe.test(t) || deriveImportRe.test(t) || evalRe.test(t)) offenders.push(`${f}: pulls the interpreter (derive/evaluateProgram)`);
    }
    expect(offenders).toEqual([]);
  });

  it('src/*.ts (core, non-lang) imports only @metael/{lang,runtime} + relative paths', () => {
    // The self-containment invariant: vdom core depends ONLY on the kernel (@metael/lang) and the reactive
    // runtime (@metael/runtime) — NEVER @metael/gpu (nor @metael/math / @metael/std, which are dev-only
    // test deps). This bare-import allowlist catches a future `import … from '@metael/gpu'` slipping into a
    // core file. Relative imports (core/lang siblings) are skipped. Mirrors the gpu core boundary guard.
    const dir = dirname(fileURLToPath(import.meta.url));
    const files = readdirSync(dir, { withFileTypes: true })
      .filter((e) => e.isFile() && e.name.endsWith('.ts') && !e.name.endsWith('.test.ts') && !e.name.endsWith('.browser.test.ts'))
      .map((e) => join(dir, e.name));
    expect(files.length).toBeGreaterThan(0);
    const allowed = new Set(['@metael/lang', '@metael/runtime']);
    const importRe = /\bfrom\s+['"]([^'"]+)['"]/g;
    const offenders: string[] = [];
    for (const f of files) {
      const t = readFileSync(f, 'utf8');
      for (const m of t.matchAll(importRe)) {
        const spec = m[1]!;
        if (spec.startsWith('.')) continue;   // relative core/lang sibling
        if (allowed.has(spec)) continue;
        offenders.push(`${f}: ${spec}`);
      }
    }
    expect(offenders).toEqual([]);
  });
});
