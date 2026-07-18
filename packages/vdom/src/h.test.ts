// packages/vdom/src/h.test.ts
import { describe, it, expect } from 'vitest';
import { h, Fragment, REACTIVE } from './h.ts';
import { TEXT, FRAGMENT } from './vnode.ts';

describe('h() hyperscript builder', () => {
  it('builds an element vnode with props and string/number children as TEXT vnodes', () => {
    const node = h('div', { class: 'box' }, 'hello', 42);
    expect(node.tag).toBe('div');
    expect(node.props).toEqual({ class: 'box' });
    expect(node.key).toBe('');                       // keys assigned later, in a post-build pass
    expect(node.children.map((c) => c.tag)).toEqual([TEXT, TEXT]);
    expect(node.children.map((c) => c.text)).toEqual(['hello', '42']);
  });

  it('captures onX props as handlers, not attributes', () => {
    const fn = () => {};
    const node = h('button', { onClick: fn, id: 'go' }, 'Go');
    expect(node.props).toEqual({ id: 'go' });
    expect(node.handlers).toEqual([{ event: 'onClick', fn }]);
  });

  it('marks a thunk child as a reactive TEXT vnode carrying the thunk', () => {
    const thunk = () => 'live';
    const node = h('span', {}, thunk);
    const child = node.children[0]!;
    expect(child.tag).toBe(TEXT);
    expect((child as unknown as Record<symbol, unknown>)[REACTIVE]).toBe(thunk);
  });

  it('marks a thunk prop value by stashing it under REACTIVE on the node, not in props', () => {
    const thunk = () => 'red';
    const node = h('div', { color: thunk });
    expect(node.props.color).toBeUndefined();        // not applied statically
    const reactiveProps = (node as unknown as Record<symbol, unknown>)[REACTIVE] as Record<string, () => unknown>;
    expect(reactiveProps.color).toBe(thunk);
  });

  it('stashes a caller key under a symbol and strips it from props', () => {
    const node = h('li', { key: 'item-7', class: 'row' }, 'x');
    expect(node.props.key).toBeUndefined();
    expect((node as unknown as Record<symbol, unknown>)[Symbol.for('metael.vdom.userKey')]).toBe('item-7');
    // sanity: class still applied
    expect(node.props.class).toBe('row');
  });

  it('accepts nested vnode children and a Fragment', () => {
    const inner = h('em', {}, 'hi');
    const node = h(Fragment, {}, inner, h('b', {}, 'bye'));
    expect(node.tag).toBe(FRAGMENT);
    expect(node.children.map((c) => c.tag)).toEqual(['em', 'b']);
  });
});
