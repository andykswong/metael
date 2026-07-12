import { describe, it, expect } from 'vitest';
import { VDomHostEnv } from './host-env.ts';
import { RuntimeReactiveHost, region, change, type Arg } from '@metael/runtime';
import { isVNode, TEXT, type VNode } from './vnode.ts';

const arg = (value: unknown, extra: Partial<Arg> = {}): Arg => ({ value, reactive: false, ...extra });

describe('VDomHostEnv.resolveCall — element vocabulary policy', () => {
  it('a lowercase head builds an element vnode with that tag + key', () => {
    const env = new VDomHostEnv(); env.bindHost(new RuntimeReactiveHost());
    const res = env.resolveCall('div', 'Story#0/div#0', [arg({ id: 'x' })], [], { start: 0, end: 0 });
    expect(res.handled).toBe(true);
    const v = (res as { value: VNode }).value;
    expect(isVNode(v) && v.tag).toBe('div');
    expect(v.key).toBe('Story#0/div#0');
    expect(v.props.id).toBe('x');
  });

  it('a Capitalized head DECLINES (so the walk emits a component wrapper → a fragment)', () => {
    const env = new VDomHostEnv(); env.bindHost(new RuntimeReactiveHost());
    expect(env.resolveCall('Counter', 'k', [], [], { start: 0, end: 0 }).handled).toBe(false);
  });

  it('a static string arg0 becomes a text child', () => {
    const env = new VDomHostEnv(); env.bindHost(new RuntimeReactiveHost());
    const v = (env.resolveCall('span', 'k', [arg('hello')], [], { start: 0, end: 0 }) as { value: VNode }).value;
    expect(v.children.length).toBe(1);
    expect(v.children[0]!.tag).toBe(TEXT);
    expect(v.children[0]!.text).toBe('hello');
  });

  it('a REACTIVE scalar arg0 (a Region) becomes a reactive text child patched by a leaf effect', () => {
    const host = new RuntimeReactiveHost();
    const env = new VDomHostEnv(); env.bindHost(host);
    const cell = host.allocateCell(0);
    const rgn = region(() => host.readCell(cell));
    // span(n): the walk hands { value: <Region>, reactive: true } as arg0.
    const v = (env.resolveCall('span', 'k', [arg(rgn, { reactive: true })], [], { start: 0, end: 0 }) as { value: VNode }).value;
    expect(v.children[0]!.tag).toBe(TEXT);
    expect(v.children[0]!.text).toBe('0');            // initial pipe landed the current value (stringified)
    change(() => host.writeCell(cell, 5));
    expect(v.children[0]!.text).toBe('5');            // leaf effect re-piped onto the SAME text vnode
  });

  it('collected child vnodes are placed as children', () => {
    const env = new VDomHostEnv(); env.bindHost(new RuntimeReactiveHost());
    const child: VNode = { tag: 'span', props: {}, children: [], key: 'c' };
    const v = (env.resolveCall('div', 'k', [], [child], { start: 0, end: 0 }) as { value: VNode }).value;
    expect(v.children).toEqual([child]);
  });

  it('a function-valued prop (onClick) is captured as a handler, not a raw attr', () => {
    const env = new VDomHostEnv(); env.bindHost(new RuntimeReactiveHost());
    const fn = (): void => {};
    const v = (env.resolveCall('button', 'k', [arg({ onClick: fn, class: 'btn' })], [], { start: 0, end: 0 }) as { value: VNode }).value;
    expect(v.handlers).toEqual([{ event: 'onClick', fn }]);
    expect('onClick' in v.props).toBe(false);
    expect(v.props.class).toBe('btn');
  });

  it('a reactive prop ENTRY (a Region in the props object) registers a leaf effect that patches props', () => {
    const host = new RuntimeReactiveHost();
    const env = new VDomHostEnv(); env.bindHost(host);
    const cell = host.allocateCell('on');
    const rgn = region(() => host.readCell(cell));
    const v = (env.resolveCall('div', 'k', [arg({ class: rgn })], [], { start: 0, end: 0 }) as { value: VNode }).value;
    expect(v.props.class).toBe('on');
    change(() => host.writeCell(cell, 'off'));
    expect(v.props.class).toBe('off');
  });

  it('a LEADING props object followed by a text arg (button({…}, "x")) keeps the props AND the text child', () => {
    const env = new VDomHostEnv(); env.bindHost(new RuntimeReactiveHost());
    const fn = (): void => {};
    const v = (env.resolveCall('button', 'k', [arg({ onClick: fn, class: 'btn' }), arg('x')], [], { start: 0, end: 0 }) as { value: VNode }).value;
    expect(v.handlers).toEqual([{ event: 'onClick', fn }]);   // props still parsed from arg0
    expect(v.props.class).toBe('btn');
    expect(v.children.length).toBe(1);                         // the trailing "x" is NOT dropped
    expect(v.children[0]!.tag).toBe(TEXT);
    expect(v.children[0]!.text).toBe('x');
  });

  it('a REACTIVE text arg after a props object binds a leaf effect (button({…}, n))', () => {
    const host = new RuntimeReactiveHost();
    const env = new VDomHostEnv(); env.bindHost(host);
    const cell = host.allocateCell(0);
    const rgn = region(() => host.readCell(cell));
    const v = (env.resolveCall('button', 'k', [arg({ class: 'c' }), arg(rgn, { reactive: true })], [], { start: 0, end: 0 }) as { value: VNode }).value;
    expect(v.props.class).toBe('c');
    expect(v.children[0]!.text).toBe('0');                     // initial pipe
    change(() => host.writeCell(cell, 7));
    expect(v.children[0]!.text).toBe('7');                     // leaf effect re-piped onto the SAME text vnode
  });
});
