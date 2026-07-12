import { describe, it, expect } from 'vitest';
import { mount } from './mount.ts';
import { type VNode } from './vnode.ts';

// Headless (no container): mount derives + materializes + wires handlers, and exposes the retained tree.
// Real-DOM identity/focus/delegation are the browser tests (a later task).

describe('mount — composition root (headless logic)', () => {
  it('derives the entry component; tree() unwraps the top-level component fragment to the first element', () => {
    const h = mount('component Story() { div({ class: "root" }) { span("hi") } }', undefined, {});
    expect(h.diagnostics).toEqual([]);
    const root = h.tree();
    expect(root).not.toBeNull();
    expect(root!.tag).toBe('div');                 // top-level 'component' fragment is unwrapped
    expect(root!.props.class).toBe('root');
    expect(root!.children[0]!.children[0]!.text).toBe('hi');
    h.unmount();
  });

  it('a missing entry surfaces ML-LANG-NO-ENTRY + a null tree', () => {
    const h = mount('function foo() { 1 }', undefined, {});
    expect(h.diagnostics.some((d) => d.code === 'ML-LANG-NO-ENTRY')).toBe(true);
    expect(h.tree()).toBeNull();
    h.unmount();
  });

  it('a Capitalized in-DSL component instance materializes as a fragment inside its parent', () => {
    const src = 'component Item() { li("x") } component Story() { ul() { Item() } }';
    const h = mount(src, undefined, {});
    const root = h.tree()!;                          // <ul> (top-level Story fragment unwrapped)
    expect(root.tag).toBe('ul');
    expect(root.children[0]!.tag).toBe('');          // Item FRAGMENT
    expect(root.children[0]!.children[0]!.tag).toBe('li');
    h.unmount();
  });

  it('is deterministic: same source + seed → identical derived vnode tree', () => {
    const src = 'component Story() { ul() { for (const i of range(3)) { li("row", { key: i, r: rand() }) } } }';
    const snap = (h: ReturnType<typeof mount>): unknown => {
      const walk = (v: VNode | null): unknown => v && { tag: v.tag, key: v.key, props: v.props, children: v.children.map(walk) };
      const t = walk(h.tree()); h.unmount(); return t;
    };
    expect(snap(mount(src, undefined, { seed: 7 }))).toEqual(snap(mount(src, undefined, { seed: 7 })));
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
    const h = mount(counter, undefined, {});
    expect(h.diagnostics).toEqual([]);
    // guard against a vacuous pass: the exact handler key must exist BEFORE the loop.
    expect(h.hasHandler('Story#0/div#0/button#0', 'onClick')).toBe(true);
    // `n` is read by the `if` condition → structural → each click re-derives with a fresh Runner.
    for (let i = 0; i < 3000; i++) h.invokeHandler('Story#0/div#0/button#0', 'onClick', undefined);
    expect(h.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(false);
    h.unmount();
  });
});
