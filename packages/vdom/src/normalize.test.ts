import { describe, it, expect } from 'vitest';
import { normalizeNodes } from './normalize.ts';
import { h } from './h.ts';

describe('normalizeNodes', () => {
  it('wraps a single node in an array', () => {
    const n = h('div', {}, 'x');
    expect(normalizeNodes(n)).toEqual([n]);
  });
  it('drops conditional holes (null/undefined/false/true) from a list', () => {
    const a = h('p', {}, 'a');
    const b = h('span', {}, 'b');
    expect(normalizeNodes([a, false, null, b, undefined, true])).toEqual([a, b]);
  });
  it('a single hole yields an empty array', () => {
    expect(normalizeNodes(null)).toEqual([]);
    expect(normalizeNodes(false)).toEqual([]);
  });
});
