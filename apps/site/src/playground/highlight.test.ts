import { describe, it, expect } from 'vitest';
import { tokensToSegments } from './highlight.ts';

describe('tokensToSegments', () => {
  it('reconstructs the full source exactly (no dropped characters)', () => {
    const src = 'component Story() {\n  let n = 0 // hi\n  span(n)\n}';
    const segs = tokensToSegments(src);
    expect(segs.map((s) => s.text).join('')).toBe(src);
  });

  it('classifies keywords, idents, numbers, punctuation', () => {
    const segs = tokensToSegments('let n = 0');
    const kw = segs.find((s) => s.text === 'let');
    const id = segs.find((s) => s.text === 'n');
    const num = segs.find((s) => s.text === '0');
    expect(kw?.kind).toBe('keyword');
    expect(id?.kind).toBe('ident');
    expect(num?.kind).toBe('number');
  });

  it('treats whitespace and // comments as plain gaps', () => {
    const segs = tokensToSegments('n // trailing');
    const tail = segs.slice(segs.findIndex((s) => s.text === 'n') + 1).map((s) => s.kind);
    expect(tail.every((k) => k === 'plain')).toBe(true);
  });

  it('handles empty source', () => {
    expect(tokensToSegments('')).toEqual([]);
  });
});
