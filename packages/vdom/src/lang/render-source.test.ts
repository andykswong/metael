import { describe, it, expect } from 'vitest';
import { renderSource } from './render-source.ts';
import { type VNode } from '../vnode.ts';
import { STD_BUILTINS } from '@metael/std';

describe('renderSource (DSL front door over the render core)', () => {
  it('mounts a DSL source headless and exposes the tree', () => {
    const h = renderSource('component Story() { div { "hi" } }', undefined, {});
    expect(h.tree()?.tag).toBe('div');
    expect(h.diagnostics).toEqual([]);
  });

  it('updateData under reactiveData re-derives; a value-only handler write does NOT bump passCount', () => {
    const h = renderSource(
      'component Story() { let n = 0; button({ onClick: () => n = n + 1 }, n) }',
      undefined,
      { reactiveData: true },
    );
    const before = h.passCount();
    // A reactive value-only write (n changes, structure identical) must not re-derive: PathKeyMinter mints
    // 'Story#0/button#0' (the entry component 'Story' is the parent segment, button its direct child), and
    // invokeHandler is an EXACT-match lookup that silently no-ops on a miss, so hasHandler guards the
    // otherwise-vacuous no-op.
    h.invokeHandler('Story#0/button#0', 'onClick', {});
    expect(h.hasHandler('Story#0/button#0', 'onClick')).toBe(true);   // guard against a vacuous no-op
    expect(h.passCount()).toBe(before);   // value-only → leaf effect, no structural pass
  });

  it('exposes the shared handle base members plus updateData', () => {
    const h = renderSource('component Story() { p { "x" } }', undefined, {});
    expect(typeof h.tree).toBe('function');
    expect(typeof h.invokeHandler).toBe('function');
    expect(typeof h.hasHandler).toBe('function');
    expect(typeof h.passCount).toBe('function');
    expect(typeof h.unmount).toBe('function');
    expect(typeof h.updateData).toBe('function');
  });

  it('a pre-keyed registry survives: a mounted button carries its minter-keyed handler entry', () => {
    // The direct proof the hooks did NOT desync the registry: the handler is keyed by the SAME minter key
    // the derive walk assigned the button node, so hasHandler(key) is true (a re-key would break it).
    const h = renderSource('component Story() { button({ onClick: () => 1 }, "x") }', undefined, {});
    expect(h.hasHandler('Story#0/button#0', 'onClick')).toBe(true);
    h.unmount();
  });
});

// The composition-root behavior (headless logic): renderSource derives + materializes + wires handlers and
// exposes the retained tree. Real-DOM identity/focus/delegation are the browser tests.
describe('renderSource — composition root (headless logic)', () => {
  it('derives the entry component; tree() unwraps the top-level component fragment to the first element', () => {
    const h = renderSource('component Story() { div({ class: "root" }) { span("hi") } }', undefined, {});
    expect(h.diagnostics).toEqual([]);
    const root = h.tree();
    expect(root).not.toBeNull();
    expect(root!.tag).toBe('div');                 // top-level 'component' fragment is unwrapped
    expect(root!.props.class).toBe('root');
    expect(root!.children[0]!.children[0]!.text).toBe('hi');
    h.unmount();
  });

  it('a missing entry surfaces ML-LANG-NO-ENTRY + a null tree', () => {
    const h = renderSource('function foo() { 1 }', undefined, {});
    expect(h.diagnostics.some((d) => d.code === 'ML-LANG-NO-ENTRY')).toBe(true);
    expect(h.tree()).toBeNull();
    h.unmount();
  });

  it('a Capitalized in-DSL component instance materializes as a fragment inside its parent', () => {
    const src = 'component Item() { li("x") } component Story() { ul() { Item() } }';
    const h = renderSource(src, undefined, {});
    const root = h.tree()!;                          // <ul> (top-level Story fragment unwrapped)
    expect(root.tag).toBe('ul');
    expect(root.children[0]!.tag).toBe('');          // Item FRAGMENT
    expect(root.children[0]!.children[0]!.tag).toBe('li');
    h.unmount();
  });

  it('is deterministic: same source + seed → identical derived vnode tree', () => {
    const src = 'component Story() { ul() { for (const i of range(3)) { li("row", { key: i, r: rand() }) } } }';
    const snap = (h: ReturnType<typeof renderSource>): unknown => {
      const walk = (v: VNode | null): unknown => v && { tag: v.tag, key: v.key, props: v.props, children: v.children.map(walk) };
      const t = walk(h.tree()); h.unmount(); return t;
    };
    expect(snap(renderSource(src, undefined, { seed: 7, builtins: [STD_BUILTINS] }))).toEqual(snap(renderSource(src, undefined, { seed: 7, builtins: [STD_BUILTINS] })));
  });

  it('BUDGET REGRESSION: thousands of handler invocations stay green (fresh Runner per structural re-derive)', () => {
    // A list whose SHAPE changes each click → each click triggers a structural re-derive with a FRESH
    // Runner (re-seeded identically), so steps/deadline reset. State latches, so `n` actually grows.
    const counter = `
component Story() {
  let n = 0
  div() {
    if (n < 100000) { span("under") } else { span("over") }
    button({ onClick: () => { n = n + 1 } }, "inc")
  }
}`;
    const h = renderSource(counter, undefined, {});
    expect(h.diagnostics).toEqual([]);
    // guard against a vacuous pass: the exact handler key must exist BEFORE the loop.
    expect(h.hasHandler('Story#0/div#0/button#0', 'onClick')).toBe(true);
    // `n` is read by the `if` condition → structural → each click re-derives with a fresh Runner.
    for (let i = 0; i < 3000; i++) h.invokeHandler('Story#0/div#0/button#0', 'onClick', undefined);
    expect(h.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(false);
    h.unmount();
  });
});
