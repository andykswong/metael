import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

describe('@metael/lsp/service protocol-free boundary', () => {
  it('the analysis engine never imports vscode-languageserver*', () => {
    const dir = dirname(fileURLToPath(import.meta.url));
    const walk = (d: string): string[] => readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(join(d, e.name)) : [join(d, e.name)]);
    const files = walk(dir).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts') && !f.endsWith('.browser.test.ts'));
    expect(files.length).toBeGreaterThan(0);
    const importRe = /\bfrom\s+['"]([^'"]+)['"]/g;
    const offenders: string[] = [];
    for (const f of files) for (const m of readFileSync(f, 'utf8').matchAll(importRe)) {
      if (m[1]!.startsWith('vscode-languageserver')) offenders.push(`${f}: ${m[1]}`);
    }
    expect(offenders).toEqual([]);
  });
});
