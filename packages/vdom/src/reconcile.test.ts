import { describe, it, expect } from 'vitest';
import { resolveHandlerKey } from './delegate.ts';
import { flattenFragments } from './reconcile.ts';
import { FRAGMENT, type VNode } from './vnode.ts';

const el = (tag: string, key: string, children: VNode[] = []): VNode => ({ tag, props: {}, children, key });

describe('resolveHandlerKey — nearest ancestor owning the handler', () => {
  it('returns the nearest ancestor key with a handler', () => {
    const chain = [{ key: 'li[a]/button#0', has: false }, { key: 'li[a]', has: true }, { key: 'ul#0', has: false }];
    expect(resolveHandlerKey(chain, (c) => c.has, (c) => c.key)).toBe('li[a]');
  });
  it('returns null when none owns a handler', () => {
    expect(resolveHandlerKey([{ key: 'a', has: false }], (c) => c.has, (c) => c.key)).toBeNull();
  });
});

describe('flattenFragments — a fragment has no DOM node; its children join the parent sequence', () => {
  it('splices fragment children into the parent level', () => {
    const frag = { tag: FRAGMENT, props: {}, children: [el('li', 'a'), el('li', 'b')], key: 'Comp#0' } as VNode;
    const flat = flattenFragments([el('h', 'h'), frag, el('li', 'c')]);
    expect(flat.map((v) => v.key)).toEqual(['h', 'a', 'b', 'c']);
  });
  it('flattens nested fragments', () => {
    const inner = { tag: FRAGMENT, props: {}, children: [el('li', 'x')], key: 'I' } as VNode;
    const outer = { tag: FRAGMENT, props: {}, children: [inner], key: 'O' } as VNode;
    expect(flattenFragments([outer]).map((v) => v.key)).toEqual(['x']);
  });
});
