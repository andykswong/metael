import { describe, it, expect } from 'vitest';
import { LineIndex } from './index.ts';

describe('LineIndex', () => {
  it('maps offsets to 0-based line/character', () => {
    const li = new LineIndex('ab\ncd\n');
    expect(li.offsetToLineCol(0)).toEqual({ line: 0, character: 0 });
    expect(li.offsetToLineCol(3)).toEqual({ line: 1, character: 0 });
    expect(li.offsetToLineCol(4)).toEqual({ line: 1, character: 1 });
  });
  it('round-trips offset ↔ line/col', () => {
    const li = new LineIndex('one\ntwo\nthree');
    for (const o of [0, 3, 4, 7, 8, 12]) expect(li.lineColToOffset(li.offsetToLineCol(o))).toBe(o);
  });
  it('counts a non-BMP char as two UTF-16 units', () => {
    const li = new LineIndex('a😀b'); // 😀 is 2 UTF-16 code units (offsets 1..3)
    expect(li.offsetToLineCol(3)).toEqual({ line: 0, character: 3 });
  });
  it('handles CRLF (the \\r is part of the line content before \\n)', () => {
    const li = new LineIndex('a\r\nb');
    expect(li.offsetToLineCol(3)).toEqual({ line: 1, character: 0 });
  });
  it('clamps out-of-range offsets to the end', () => {
    const li = new LineIndex('ab');
    expect(li.offsetToLineCol(99)).toEqual({ line: 0, character: 2 });
  });
});
