/* eslint-disable @typescript-eslint/no-explicit-any -- narrowed HostValue node access in test scaffolding */
import { describe, it, expect } from 'vitest';
import { lowerEntry } from './lower.ts';
import { PlainStorageHost, RecordingHostEnv, PathKeyMinter, isWrapper, isRegion, type LangWrapper } from './ports.ts';
import type { HostEnvironment, HostValue } from './ports.ts';

describe('lowerEntry — entry component + child collection', () => {
  it('a missing entry is ML-LANG-NO-ENTRY', () => {
    const { value, diagnostics } = lowerEntry('function foo() { 1 }', { host: new PlainStorageHost(), env: new RecordingHostEnv(), minter: new PathKeyMinter(), entry: 'Story' });
    expect(value).toBeNull();
    expect(diagnostics.some((d) => d.code === 'ML-LANG-NO-ENTRY')).toBe(true);
  });

  it('the entry component resolves through the host with a parentless key', () => {
    const env = new RecordingHostEnv();
    lowerEntry('component Story() { text("hi") }', { host: new PlainStorageHost(), env, minter: new PathKeyMinter() });
    const heads = env.calls.map((c) => c.head);
    expect(heads).toContain('Story');
    expect(heads).toContain('text');
    expect(env.calls.find((c) => c.head === 'Story')!.key).toBe('Story#0');
    expect(env.calls.find((c) => c.head === 'text')!.key).toBe('Story#0/text#0');
  });

  it('sibling same-kind children get distinct structural ordinals', () => {
    const env = new RecordingHostEnv();
    lowerEntry('component Story() { text("a"); text("b") }', { host: new PlainStorageHost(), env, minter: new PathKeyMinter() });
    const textKeys = env.calls.filter((c) => c.head === 'text').map((c) => c.key);
    expect(textKeys).toEqual(['Story#0/text#0', 'Story#0/text#1']);
  });

  it('a for-of body keys children via listItem (author key → stable)', () => {
    const env = new RecordingHostEnv();
    const src = `component Story() { for (const row of data) { text(row.label, { key: row.id }) } }`;
    lowerEntry(src, { host: new PlainStorageHost(), env, minter: new PathKeyMinter(), data: [{ id: 'x', label: 'X' }, { id: 'y', label: 'Y' }] });
    const textKeys = env.calls.filter((c) => c.head === 'text').map((c) => c.key);
    expect(textKeys).toEqual(['Story#0/text[x]', 'Story#0/text[y]']);
  });

  it('a declined head becomes an unknown-wrapper carrying Arg[] (name/reactive preserved)', () => {
    // Decline every head: the entry (Story) is also offered to resolveCall, so to keep the ROOT a
    // wrapper (isWrapper(root) below) the host must decline Story too — not just sankey.
    const env = new RecordingHostEnv([]);   // host builds nothing → Story + sankey both become wrappers
    const { value } = lowerEntry('component Story() { sankey(data, { title: "T" }) }', {
      host: new PlainStorageHost(), env, minter: new PathKeyMinter(), data: [1, 2, 3],
    });
    const root = value as LangWrapper;
    expect(isWrapper(root)).toBe(true);
    const sankey = (root.children as unknown[]).find((c) => isWrapper(c) && (c as LangWrapper).head === 'sankey') as LangWrapper;
    expect(sankey.__mlWrap).toBe('unknown');
    // args are Arg[] (each { value, name?, reactive? }), NOT a flat value array
    expect(sankey.args.every((a) => typeof a === 'object' && a !== null && 'value' in a)).toBe(true);
    expect(sankey.args[1]!.name).toBe(undefined);
  });

  it('an in-DSL component the host declines becomes a component-wrapper', () => {
    const env = new RecordingHostEnv(['Story']);   // Panel declined
    const { value } = lowerEntry('component Panel() { text("x") } component Story() { Panel() }', { host: new PlainStorageHost(), env, minter: new PathKeyMinter() });
    const root = value as LangWrapper;
    const panel = (root.children as unknown[]).find((c) => isWrapper(c) && (c as LangWrapper).head === 'Panel') as LangWrapper;
    expect(panel.__mlWrap).toBe('component');
  });

  it('a reactive prop (reads a reactive let) lowers to a Region', () => {
    const declining = new RecordingHostEnv(['Story']);   // box declined → wrapper carries the raw Region
    const { value } = lowerEntry('component Story() { let n = 1; box({ count: n }) }', { host: new PlainStorageHost(), env: declining, minter: new PathKeyMinter() });
    const root = value as LangWrapper;
    const box = (root.children as unknown[]).find((c) => isWrapper(c) && (c as LangWrapper).head === 'box') as LangWrapper;
    const countArg = box.args[0]!;
    expect(isRegion((countArg.value as Record<string, unknown>).count)).toBe(true);
  });
});

// Structural coverage of the walk's cases, using generic heads (box/text/Panel/sankey).
describe('lowerEntry — collection structure (source order, control flow, producers, slots)', () => {
  const KNOWN = ['Story', 'Panel', 'box', 'text', 'sankey'];
  const lower = (src: string, data?: unknown, entry = 'Story') =>
    lowerEntry(src, { data, entry, host: new PlainStorageHost(), env: new RecordingHostEnv({ known: KNOWN }), minter: new PathKeyMinter() });

  it('collects children in source order under the entry', () => {
    const r = lower('component Story() { box({ mode: "flex" }) { text("a"); text("b"); } }');
    const root = r.value as any;                          // the Story instance
    expect(root.head).toBe('Story');
    const box = root.children[0];
    expect(box.head).toBe('box');
    expect(box.children.map((c: any) => c.head)).toEqual(['text', 'text']);
  });

  it('passes ORDERED positional args to resolveCall (content is args[0], NOT props)', () => {
    const r = lower('component Story() { text("hi", { size: 48 }) }');
    const text = (r.value as any).children[0];
    expect(text.head).toBe('text');
    expect(text.args).toEqual(['hi', { size: 48 }]);       // raw order, not classified (host flattens Arg.value)
  });

  it('mints a structural key per node using the AUTHORED head (parent + kind + lexical ordinal)', () => {
    const r = lower('component Story() { box({ mode: "flex" }) { text("a") } }');
    const root = r.value as any;
    expect(root.key).toBe('Story#0');
    expect(root.children[0].key).toBe('Story#0/box#0');
    expect(root.children[0].children[0].key).toBe('Story#0/box#0/text#0');
  });

  it('does NOT collect let/const/assign statements as children (they run for effect)', () => {
    const r = lower('component C() { const x = 1; text("only") } component Story() { C() }');
    const c = (r.value as any).children[0];               // the C instance (C not in KNOWN → component wrapper)
    expect(c.children.map((n: any) => n.head)).toEqual(['text']);
  });

  it('flattens for-of in child position; list items get distinct minted keys', () => {
    const r = lower('component Story() { box({ mode: "grid" }) { for (const k of data.items) text(k) } }', { items: ['x', 'y', 'z'] });
    const box = (r.value as any).children[0];
    expect(box.children.map((c: any) => c.head)).toEqual(['text', 'text', 'text']);
    expect(new Set(box.children.map((c: any) => c.key)).size).toBe(3);   // distinct (ordinal tiebreak)
  });

  it('appends only the taken if-branch', () => {
    const r = lower('component Story() { if (data.flag) { text("yes") } else { text("no") } }', { flag: true });
    expect((r.value as any).children.map((c: any) => c.head)).toEqual(['text']);
    expect((r.value as any).children[0].args[0]).toBe('yes');
  });

  it('flattens a while loop in child position (each iteration appends its node)', () => {
    const r = lower('component Story() { let n = 3; while (n > 0) { text("row"); n = n - 1 } }');
    expect((r.value as any).children.map((c: any) => c.head)).toEqual(['text', 'text', 'text']);
  });

  it('functions still use implicit last-expression return (NOT child collection)', () => {
    const r = lower('function two() { 1; 2 } component Story() { text("x", { size: two() }) }');
    const text = (r.value as any).children[0];
    expect(text.args[1].size).toBe(2);                    // function call resolves to 2 in the props arg
  });

  it('lowers a nested shape: Story > [ box(grid) > 3x nested, box > sankey ]', () => {
    const src = `
      component Card(k) { box({ mode: "flex" }) { text(k) } }
      component Story() {
        let step = 0;
        box({ mode: "grid", cols: 3 }) { for (const k of data.cards) Card(k) }
        box({ mode: "stack" }) { sankey({ type: "flow" }) }
      }`;
    const r = lower(src, { cards: ['a', 'b', 'c'] });
    const root = r.value as any;
    expect(root.children[0].head).toBe('box');
    expect(root.children[0].children).toHaveLength(3);     // 3 Cards
    expect(root.children[1].head).toBe('box');
    expect(root.children[1].children[0].head).toBe('sankey');
  });

  it('render-prop (FaaC): an arrow producer passed as a prop is invoked in child position', () => {
    const src = `
      component Card(k) { text(k) }
      component List(p) { box({ mode: "grid" }) { for (const r of p.rows) p.renderItem(r) } }
      component Story() { List({ rows: data.cards, renderItem: (r) => Card(r) }) }`;
    const r = lower(src, { cards: ['a', 'b'] });
    const list = (r.value as any).children[0];             // List instance (component wrapper)
    const grid = list.children[0];
    expect(grid.head).toBe('box');
    expect(grid.children).toHaveLength(2);                 // renderItem invoked per row
    expect(grid.children.every((c: any) => c.head === 'Card')).toBe(true);
  });

  it('slot: an already-instantiated node passed as a prop is placed as a child', () => {
    const src = `
      component Card(k) { text(k) }
      component Panel(p) { box({ mode: "flex" }) { p.body } }
      component Story() { Panel({ body: Card("x") }) }`;
    const r = lower(src, {});
    const panel = (r.value as any).children[0];            // Panel is KNOWN → host node
    expect(panel.children[0].head).toBe('box');
    expect(panel.children[0].children[0].head).toBe('Card');   // the slot node, placed
  });

  it('a bare component reference is a VALUE in value position, a call when applied', () => {
    const src = `
      component Card(k) { text(k) }
      component List(p) { for (const r of p.rows) p.item(r) }
      component Story() { List({ rows: data.cards, item: Card }) }`;   // bare `Card` as a value
    const r = lower(src, { cards: ['a'] });
    expect((r.value as any).children[0].children[0].head).toBe('Card');
  });

  it('a pure function producer invoked in child position yields its implicit-last-expr node', () => {
    const src = `function tile(k) { text(k) } component Story() { box({ mode: "flex" }) { tile("q") } }`;
    const r = lower(src);
    const box = (r.value as any).children[0];
    expect(box.children[0].head).toBe('text');
    expect(box.children[0].args[0]).toBe('q');
  });

  // Regression: a guard-clause `return` inside a component body must STOP that component's collection,
  // keeping children gathered before it — NOT unwind to the top and clobber the whole tree.
  it('a guard-clause return stops the component, keeps prior children, does not clobber the tree', () => {
    const r = lower('component Story() { text("kept"); if (data.stop) { return } text("after") }', { stop: true });
    const root = r.value as any;
    expect(root.head).toBe('Story');                       // root NOT replaced by the return value
    expect(root.children.map((c: any) => c.head)).toEqual(['text']);   // only the pre-return child
    expect(root.children[0].args[0]).toBe('kept');
  });

  it('a return in a NESTED component does not discard the outer tree', () => {
    const r = lower('component C() { return } component Story() { text("a"); C(); text("b") }', {});
    const heads = (r.value as any).children.map((c: any) => c.head);
    expect((r.value as any).head).toBe('Story');
    expect(heads).toEqual(['text', 'C', 'text']);          // C instantiates (empty), siblings intact
  });

  // Regression: a while-body `const` must be re-defined each iteration (fresh block scope),
  // so each row reflects the current loop value — not the stale first-iteration value.
  it('a while-body const is re-evaluated each iteration (no cross-iteration scope leak)', () => {
    const r = lower('component Story() { let y = 0; while (y < 3) { const label = y; text(label); y = y + 1 } }', {});
    const labels = (r.value as any).children.map((c: any) => c.args[0]);
    expect(labels).toEqual([0, 1, 2]);                     // distinct per iteration, not [0,0,0]
  });

  // A bare non-node value statement runs for effect and is NOT appended as a child.
  it('a bare primitive-value statement is not collected as a child', () => {
    const r = lower('component Story() { "just an effect"; 42; text("real") }', {});
    expect((r.value as any).children.map((c: any) => c.head)).toEqual(['text']);
  });
});

describe('reactive data reads — opted-in data is trackable', () => {
  // A host double that records the RAW args (unresolved), so we can observe whether a `data.x` read
  // lowered to a reactive Region (a re-runnable thunk) vs an eager value. RecordingHostEnv deep-resolves
  // Regions, which would erase the very distinction under test — so we capture args verbatim here.
  class RawRecordingHostEnv implements HostEnvironment {
    readonly calls: { head: string; args: HostValue[] }[] = [];
    resolveCall(head: string, key: string, args: { value: HostValue }[], children: HostValue[]):
      { handled: true; value: HostValue } | { handled: false } {
      this.calls.push({ head, args: args.map((a) => a.value) });
      return { handled: true, value: { head, key, args: args.map((a) => a.value), children } };
    }
  }
  const lowerRaw = (src: string, data: unknown, reactiveData: boolean) => {
    const env = new RawRecordingHostEnv();
    lowerEntry(src, { data, entry: 'Story', reactiveData, host: new PlainStorageHost(), env, minter: new PathKeyMinter() });
    return env;
  };
  // `text("hi", { size: data.size })` — the `size` object-entry reads `data.size`.
  const SRC = 'component Story() { text("hi", { size: data.size }) }';

  it('with reactiveData:true, a prop reading data.x is lowered as a reactive Region (not eager)', () => {
    const env = lowerRaw(SRC, { title: 't', size: 20 }, true);
    const textCall = env.calls.find((c) => c.head === 'text')!;
    const props = textCall.args[1] as Record<string, unknown>;
    expect(isRegion(props.size)).toBe(true);            // a re-runnable thunk, NOT the eager 20
    expect((props.size as { run: () => unknown }).run()).toBe(20);   // thunk still yields the current value
  });

  it('with reactiveData:false (default), the same read is eager (frozen value)', () => {
    const env = lowerRaw(SRC, { title: 't', size: 20 }, false);
    const textCall = env.calls.find((c) => c.head === 'text')!;
    const props = textCall.args[1] as Record<string, unknown>;
    expect(isRegion(props.size)).toBe(false);
    expect(props.size).toBe(20);                        // eager, frozen value
  });
});
