import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

describe('@metael/lang core ↛ profile import boundary', () => {
  it('the language core (src/*.ts, excluding profile/) never imports the profile subpath', () => {
    const coreDir = dirname(dirname(fileURLToPath(import.meta.url))); // .../src
    const walk = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
      if (e.name === 'profile') return [];
      return e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)];
    });
    const files = walk(coreDir).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts') && !f.endsWith('.browser.test.ts'));
    expect(files.length).toBeGreaterThan(0);
    const stripComments = (t: string): string => t.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1');
    const bad = [/\bfrom\s+['"]\.\/profile\//, /\bfrom\s+['"]\.\.\/profile\//];
    const offenders: string[] = [];
    for (const f of files) { const t = stripComments(readFileSync(f, 'utf8')); if (bad.some((re) => re.test(t))) offenders.push(f); }
    expect(offenders).toEqual([]);
  });
});
