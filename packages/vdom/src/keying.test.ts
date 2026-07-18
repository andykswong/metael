// packages/vdom/src/keying.test.ts
import { describe, it, expect } from 'vitest';
import { h } from './h.ts';
import { assignKeys } from './keying.ts';

describe('assignKeys — kind-namespaced keying (mirrors PathKeyMinter)', () => {
  it('keys elements per-parent, per-tag; text in its own ordinal space', () => {
    const tree = [h('div', {}, 'hello', h('span', {}, 'a'), h('span', {}, 'b'))];
    assignKeys(tree, '');
    expect(tree[0]!.key).toBe('/div#0');
    // children of div: text "hello", span#0, span#1 — text and elements do NOT share an index space
    expect(tree[0]!.children[0]!.key).toBe('/div#0/#text#0');   // "hello"
    expect(tree[0]!.children[1]!.key).toBe('/div#0/span#0');    // first span
    expect(tree[0]!.children[2]!.key).toBe('/div#0/span#1');    // second span
    // the first span's TEXT child:
    expect(tree[0]!.children[1]!.children[0]!.key).toBe('/div#0/span#0/#text#0');
  });

  it('a caller key is tag-namespaced and cannot collide with an element ordinal', () => {
    // li key '0' must NOT collide with an unkeyed li's '#0' ordinal
    const tree = [h('ul', {}, h('li', {}, 'x'), h('li', { key: '0' }, 'y'))];
    assignKeys(tree, '');
    expect(tree[0]!.children[0]!.key).toBe('/ul#0/li#0');     // unkeyed → ordinal
    expect(tree[0]!.children[1]!.key).toBe('/ul#0/li[0]');    // keyed → bracketed, distinct namespace
  });

  it('a distinct tag gets its own ordinal counter (no cross-tag aliasing)', () => {
    const tree = [h('div', {}, h('p', {}, '1'), h('span', {}, '2'), h('p', {}, '3'))];
    assignKeys(tree, '');
    expect(tree[0]!.children.map((c) => c.key)).toEqual(['/div#0/p#0', '/div#0/span#0', '/div#0/p#1']);
  });

  it('the surviving element keeps a STABLE key when an unkeyed different-kind sibling drops out', () => {
    // Pass 1: [p, span]; Pass 2: [span] (the p was conditionally removed). The span's key must be
    // IDENTICAL across both passes so reconcile matches it (a flat index would give the
    // span '/div#0/0' in pass 1 and '/div#0/0' in pass 2 but aliased onto the p's old slot).
    const pass1 = [h('div', {}, h('p', {}, 'x'), h('span', {}, 'y'))];
    assignKeys(pass1, '');
    const pass2 = [h('div', {}, h('span', {}, 'y'))];
    assignKeys(pass2, '');
    const spanKey1 = pass1[0]!.children[1]!.key;   // span in pass 1
    const spanKey2 = pass2[0]!.children[0]!.key;   // span in pass 2
    expect(spanKey1).toBe('/div#0/span#0');
    expect(spanKey2).toBe('/div#0/span#0');
    expect(spanKey1).toBe(spanKey2);               // stable → reconcile reuses the span, never the p
  });

  it('a Fragment has its own namespace, distinct from element slots', () => {
    const tree = [h('div', {}, h('' as never, {}, h('em', {}, 'a')), h('b', {}, 'c'))];
    // Fragment child keyed frag#0; the b element keyed b#0 — no aliasing
    assignKeys(tree, '');
    expect(tree[0]!.children[0]!.key).toBe('/div#0/frag#0');
    expect(tree[0]!.children[1]!.key).toBe('/div#0/b#0');
  });

  it('is deterministic: same structure → identical keys', () => {
    const build = () => [h('div', {}, h('b', {}, 'x'))];
    const a = build(); assignKeys(a, '');
    const b = build(); assignKeys(b, '');
    expect(a[0]!.children[0]!.key).toBe(b[0]!.children[0]!.key);
  });
});
