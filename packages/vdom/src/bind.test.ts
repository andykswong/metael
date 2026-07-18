// packages/vdom/src/bind.test.ts
import { describe, it, expect } from 'vitest';
import { signal, change } from '@metael/runtime';
import { h } from './h.ts';
import { assignKeys } from './keying.ts';
import { bindReactive, disposeLeaf } from './bind.ts';
import { TEXT } from './vnode.ts';

describe('bindReactive — thunks → leaf effects, handlers → registry, disposers tracked', () => {
  it('seeds a reactive TEXT vnode from its thunk before build', () => {
    const s = signal('one');
    const tree = [h('span', {}, () => s.get())];
    assignKeys(tree, '');
    const disposers: Array<() => void> = [];
    const registry = new Map<string, (a: unknown) => void>();
    change(() => bindReactive(tree, disposers, registry));
    const textNode = tree[0]!.children[0]!;
    expect(textNode.tag).toBe(TEXT);
    expect(textNode.text).toBe('one');               // seeded synchronously
    expect(disposers.length).toBe(1);                // one leaf effect registered
  });

  it('seeds a reactive prop onto vnode.props before build', () => {
    const c = signal('red');
    const tree = [h('div', { color: () => c.get() })];
    assignKeys(tree, '');
    const disposers: Array<() => void> = [];
    change(() => bindReactive(tree, disposers, new Map()));
    expect(tree[0]!.props.color).toBe('red');
  });

  it('registers onX handlers keyed by `${key}:${event}`', () => {
    const fn = () => {};
    const tree = [h('button', { onClick: fn }, 'Go')];
    assignKeys(tree, '');
    const registry = new Map<string, (a: unknown) => void>();
    change(() => bindReactive(tree, [], registry));
    expect(registry.get('/button#0:onClick')).toBe(fn);
  });

  it('a value-only signal write re-runs only the leaf effect (updates vnode.text) without rebuilding', () => {
    const s = signal('a');
    const tree = [h('span', {}, () => s.get())];
    assignKeys(tree, '');
    const disposers: Array<() => void> = [];
    change(() => bindReactive(tree, disposers, new Map()));
    const textNode = tree[0]!.children[0]!;
    expect(textNode.text).toBe('a');
    change(() => s.set('b'));
    expect(textNode.text).toBe('b');                 // leaf effect patched the vnode field
  });

  it('disposeLeaf() stops a subtree\'s leaf effect so a later write no longer patches it', () => {
    // Isolates the disposeLeaf/onRemove teardown path from render()'s wholesale per-pass dispose: bind an
    // effect, dispose ONLY via disposeLeaf, then write the signal — the thunk must NOT re-run. This fails if
    // disposeLeaf is a no-op (the guarantee render's removal path depends on).
    const s = signal('a');
    const tree = [h('span', {}, () => s.get())];
    assignKeys(tree, '');
    const textNode = tree[0]!.children[0]!;
    change(() => bindReactive(tree, [], new Map()));   // NOTE: disposers array is separate — not swept here
    expect(textNode.text).toBe('a');
    disposeLeaf(tree[0]!);                              // tear down via the subtree path only
    change(() => s.set('b'));
    expect(textNode.text).toBe('a');                   // stopped: the write did NOT patch the vnode
  });
});
