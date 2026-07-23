import { describe, it, expect } from 'vitest';
import { LanguageService } from '../index.ts';
import { stdProfile } from '@metael/std';

/** Open a document under stdProfile and return its folding ranges. */
function foldsOf(src: string) {
  const svc = new LanguageService();
  svc.openDocument('a', src, 1);
  svc.setProfile('a', stdProfile);
  return svc.foldingRanges('a');
}

describe('foldingRanges', () => {
  it('yields a fold per block — a component body and its nested if-block — each spanning `{`..`}`', () => {
    const src = 'component App() {\n  let n = 0\n  if (n > 0) {\n    let m = 1\n  }\n}';
    const folds = foldsOf(src);
    expect(folds.length).toBeGreaterThanOrEqual(2);
    // Every fold starts at a `{` and ends just past a `}`.
    for (const f of folds) {
      expect(src[f.start]).toBe('{');
      expect(src[f.end - 1]).toBe('}');
      expect(f.end).toBeGreaterThan(f.start);
    }
    // The component body encloses the nested if-block: an outer fold strictly contains an inner one.
    const outer = folds.find((o) => folds.some((i) => i !== o && i.start > o.start && i.end < o.end));
    expect(outer).toBeDefined();
  });

  it('does not fold object literals (only real blocks fold)', () => {
    const folds = foldsOf('const config = { a: 1, b: 2 }');
    expect(folds).toEqual([]);
  });

  it('returns [] for a document with no blocks', () => {
    expect(foldsOf('const x = 1')).toEqual([]);
  });
});
