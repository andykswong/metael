// packages/vdom/src/render.test.ts
import { describe, it, expect } from 'vitest';
import { signal } from '@metael/runtime';
import { h } from './h.ts';
import { render } from './render.ts';

describe('render() — headless (no container): pass driving + handle shape', () => {
  it('builds a tree from a producer and exposes it via tree() with keys assigned', () => {
    const handle = render(() => h('div', { id: 'root' }, 'hi'), undefined);
    const tree = handle.tree();
    expect(tree?.tag).toBe('div');
    expect(tree?.key).toBe('/div#0');       // kind-namespaced (see keying.ts)
    expect(handle.passCount()).toBe(1);
    handle.unmount();
  });

  it('a structural signal read in the producer re-runs the pass (passCount increments)', () => {
    const show = signal(true);
    const handle = render(() => (show.get() ? h('div', {}, 'A') : h('p', {}, 'B')), undefined);
    expect(handle.passCount()).toBe(1);
    expect(handle.tree()?.tag).toBe('div');
    // structural change: the producer reads show.get() at the top level → the tracked pass re-runs
    handle.setState(() => show.set(false));
    expect(handle.passCount()).toBe(2);
    expect(handle.tree()?.tag).toBe('p');
    handle.unmount();
  });

  it('a value-only signal read (inside a thunk) patches the node WITHOUT re-running the pass', () => {
    const label = signal('x');
    const handle = render(() => h('span', {}, () => label.get()), undefined);
    expect(handle.passCount()).toBe(1);
    // the reactive text is a #text child of the span; the leaf effect seeds its .text before build
    expect(handle.tree()?.children[0]?.text).toBe('x');
    handle.setState(() => label.set('y'));
    // POSITIVE assertion: the leaf effect actually fired and updated the value (fails if never bound) ...
    expect(handle.tree()?.children[0]?.text).toBe('y');
    // ... AND it did so without a structural re-derive.
    expect(handle.passCount()).toBe(1);              // leaf effect only — no re-derive
    handle.unmount();
  });

  it('unmount() stops the pass so later writes do not re-run it', () => {
    const s = signal(0);
    const handle = render(() => { s.get(); return h('div', {}, 'x'); }, undefined);
    expect(handle.passCount()).toBe(1);
    handle.unmount();
    handle.setState(() => s.set(1));
    expect(handle.passCount()).toBe(1);              // stopped
  });

  it('tolerates an empty root: a producer returning null yields tree()===null (no crash)', () => {
    const handle = render(() => null, undefined);
    expect(handle.tree()).toBeNull();
    handle.unmount();
  });

  it('drops conditional holes (null/false/undefined) in a producer array, like h() does for children', () => {
    const show = signal(false);
    // The JSX-conditional idiom at the top level: a false branch is a hole, not a crash.
    const handle = render(
      () => [show.get() && h('p', {}, 'banner'), h('span', {}, 'body')],
      undefined,
    );
    // only the span survives; it is the first (and only) real node
    expect(handle.tree()?.tag).toBe('span');
    handle.setState(() => show.set(true));
    // now the p appears ahead of the span
    expect(handle.tree()?.tag).toBe('p');
    handle.unmount();
  });

  it('a handler removed on a surviving element no longer fires after a structural pass', () => {
    const mode = signal(true);
    let fired = 0;
    const fn = (): void => { fired++; };
    // Pass 1: the button carries onClick; pass 2 (mode=false): the same button (stable key) carries NO
    // handlers. The registry must be rebuilt fresh each pass so the stale onClick does not survive.
    const handle = render(() => h('button', mode.get() ? { onClick: fn } : {}, 'x'), undefined);
    expect(handle.hasHandler('/button#0', 'onClick')).toBe(true);
    handle.invokeHandler('/button#0', 'onClick', {});
    expect(fired).toBe(1);
    // structural pass drops the handler from the surviving element
    handle.setState(() => mode.set(false));
    expect(handle.passCount()).toBe(2);
    expect(handle.hasHandler('/button#0', 'onClick')).toBe(false);
    handle.invokeHandler('/button#0', 'onClick', {});
    expect(fired).toBe(1);                            // stale handler did NOT fire
    handle.unmount();
  });
});
