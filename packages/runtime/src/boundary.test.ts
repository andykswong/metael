import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import * as runtime from './index.ts';

describe('@metael/runtime public surface', () => {
  it('exports the reactive core, host, keyed diff, and derive', () => {
    for (const name of ['signal', 'memo', 'effect', 'change', 'ReactiveFlushError',
      'RuntimeReactiveHost', 'diffKeyed', 'applyKeyedDiff', 'derive']) {
      expect(runtime[name as keyof typeof runtime]).toBeDefined();
    }
  });
  it('re-exports every lang seam VALUE (single import site for a domain)', () => {
    for (const name of ['lowerEntry', 'region', 'isRegion', 'wrapper', 'isWrapper', 'didYouMean',
      'PlainStorageHost', 'RecordingHostEnv', 'PathKeyMinter']) {
      expect(runtime[name as keyof typeof runtime]).toBeDefined();
    }
  });
});

describe('@metael/runtime import boundary (self-containment invariant)', () => {
  it('non-test src imports only @metael/lang, @vue/reactivity, and relative paths', () => {
    const srcDir = dirname(fileURLToPath(import.meta.url));
    const files = readdirSync(srcDir).filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts'));
    const offenders: string[] = [];
    // Match any bare-specifier import/re-export (`from '<pkg>'` where <pkg> is not './' or '../').
    // Catches BOTH `import ... from` and `export ... from`.
    const importRe = /\bfrom\s+['"]([^'".][^'"]*)['"]/g;
    for (const f of files) {
      const text = readFileSync(join(srcDir, f), 'utf8');
      for (const m of text.matchAll(importRe)) {
        const spec = m[1]!;
        if (spec.startsWith('.')) continue;                       // relative — always allowed
        if (spec === '@metael/lang' || spec === '@vue/reactivity') continue;  // the two permitted deps
        offenders.push(`${f}: ${spec}`);
      }
    }
    expect(offenders).toEqual([]);
  });
});
