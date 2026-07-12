import { describe, it, expect } from 'vitest';
import { materialize } from './materialize.ts';
import { wrapper } from '@metael/runtime';
import { makeDiagnostic, type Diagnostic } from '@metael/lang';
import { FRAGMENT, type VNode } from './vnode.ts';

const el = (tag: string, key: string, children: VNode[] = [], handlers?: VNode['handlers']): VNode =>
  ({ tag, props: {}, children, key, ...(handlers ? { handlers } : {}) });

describe('materialize', () => {
  it('an element vnode passes through, recursing children', () => {
    const out = materialize(el('div', 'd', [el('span', 's')]), [], new Map());
    expect(out!.tag).toBe('div');
    expect(out!.children[0]!.tag).toBe('span');
  });

  it("a 'component' wrapper becomes a transparent fragment (children carried, no DOM node)", () => {
    const w = wrapper('component', 'Counter', 'Story#0/Counter#0', [], [el('button', 'b'), el('span', 's')]);
    const out = materialize(w, [], new Map());
    expect(out!.tag).toBe(FRAGMENT);
    expect(out!.key).toBe('Story#0/Counter#0');
    expect(out!.children.map((c) => c.tag)).toEqual(['button', 'span']);
  });

  it("an 'unknown' wrapper yields null + an ML-VDOM-UNKNOWN diagnostic", () => {
    const diags: Diagnostic[] = [];
    expect(materialize(wrapper('unknown', 'sankey', 'k', [], []), diags, new Map())).toBeNull();
    expect(diags.some((d) => d.code === 'ML-VDOM-UNKNOWN')).toBe(true);
  });

  it('handler entries are recorded keyed `${nodeKey}:${event}`', () => {
    const fn = (): void => {};
    const handlers = new Map<string, (arg: unknown) => void>();
    materialize(el('button', 'Story#0/button#0', [], [{ event: 'onClick', fn }]), [], handlers);
    expect(handlers.get('Story#0/button#0:onClick')).toBe(fn);
  });

  it('null / non-object values are dropped', () => {
    expect(materialize(null, [], new Map())).toBeNull();
    expect(materialize(42, [], new Map())).toBeNull();
  });

  it('makeDiagnostic is imported from @metael/lang (sanity: the code path used above resolves)', () => {
    expect(typeof makeDiagnostic).toBe('function');
  });
});
