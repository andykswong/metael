import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import * as vdom from './index.ts';

describe('@metael/vdom public surface', () => {
  it('exports mount + the vnode helpers + the sanitizer', () => {
    for (const name of ['mount', 'isVNode', 'textVNode', 'FRAGMENT', 'TEXT', 'escapeText', 'safeAttrName', 'safeAttrValue']) {
      expect(vdom[name as keyof typeof vdom]).toBeDefined();
    }
  });
  it('does NOT export examples or the demo harness (a library stays app-free)', () => {
    expect((vdom as Record<string, unknown>).COUNTER).toBeUndefined();
    expect((vdom as Record<string, unknown>).mountDemos).toBeUndefined();
  });
});

describe('@metael/vdom import boundary (self-containment invariant)', () => {
  it('non-test src imports only @metael/lang, @metael/runtime, and relative paths', () => {
    const srcDir = dirname(fileURLToPath(import.meta.url));
    const files = readdirSync(srcDir).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts') && !f.endsWith('.browser.test.ts'));
    const offenders: string[] = [];
    const importRe = /\bfrom\s+['"]([^'".][^'"]*)['"]/g;
    for (const f of files) {
      const text = readFileSync(join(srcDir, f), 'utf8');
      for (const m of text.matchAll(importRe)) {
        const spec = m[1]!;
        if (spec === '@metael/lang' || spec === '@metael/runtime') continue;
        offenders.push(`${f}: ${spec}`);
      }
    }
    expect(offenders).toEqual([]);
  });
});
