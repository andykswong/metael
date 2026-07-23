import { describe, it, expect } from 'vitest';
import { Document } from './index.ts';

describe('Document', () => {
  it('lexes + parses lazily and memoizes per version', () => {
    const d = new Document('const x = 1', 1);
    const p1 = d.parse;
    const p2 = d.parse;
    expect(p1).toBe(p2); // memoized (same object)
    expect(d.parse.program.stmts.length).toBe(1);
  });
  it('a total parse stays queryable on malformed input', () => {
    const d = new Document('const x = ', 1); // incomplete
    expect(() => d.parse).not.toThrow();
    expect(d.parse.diagnostics.length).toBeGreaterThan(0);
  });
  it('update produces a fresh document with the new version', () => {
    const d1 = new Document('const x = 1', 1);
    const d2 = d1.update('const x = 2', 2);
    expect(d2.version).toBe(2);
    expect(d2.text).toBe('const x = 2');
    expect(d2).not.toBe(d1);
  });
});
