import { describe, it, expect } from 'vitest';
import { derive } from './derive.ts';
import { change } from './reactive.ts';   // used by the re-pipe + converge tests below
import type { RuntimeReactiveHost } from './reactive-host.ts';
import { RecordingHostEnv, PathKeyMinter, isRegion, isWrapper, type LangWrapper } from '@metael/lang';

describe('derive — one-shot composition root', () => {
  it('instantiates the entry component and returns its host value + a host', () => {
    const env = new RecordingHostEnv();
    const { value, diagnostics, host } = derive('component Story() { text("hi") }', { env, minter: new PathKeyMinter() });
    expect(diagnostics).toEqual([]);
    expect(value).not.toBeNull();
    expect(host.cells).toBeDefined();
    expect(env.calls.map((c) => c.head)).toEqual(expect.arrayContaining(['Story', 'text']));
  });

  it('a missing entry surfaces ML-LANG-NO-ENTRY (from the walk) and a null value', () => {
    const { value, diagnostics } = derive('function foo() { 1 }', { env: new RecordingHostEnv(), minter: new PathKeyMinter() });
    expect(value).toBeNull();
    expect(diagnostics.some((d) => d.code === 'ML-LANG-NO-ENTRY')).toBe(true);
  });

  it('binds a reactive prop as a leaf effect and re-pipes end-to-end when a handler mutates the cell', () => {
    const piped: number[] = [];
    let tick: ((arg: unknown) => void) | undefined;
    const holder: { host?: RuntimeReactiveHost } = {};
    const env = {
      resolveCall(head: string, key: string, args: { value: unknown; reactive?: boolean }[]) {
        if (head === 'box') {
          const count = (args[0]?.value as Record<string, unknown>)?.count;
          if (isRegion(count)) holder.host!.runLeafEffect((count as { run: () => unknown }).run, (v) => piped.push(v as number));
        }
        if (head === 'onTick') {
          const fn = args[0]?.value;
          if (typeof fn === 'function') tick = fn as (arg: unknown) => void;
        }
        return { handled: true as const, value: { head, key } };
      },
    };
    const minter = new PathKeyMinter();
    const src = 'component Story() { let n = 1; box({ count: n }); onTick(() => { n = n + 1 }) }';
    const res = derive(src, { env, minter, onHost: (h) => { holder.host = h; } });
    expect(res.diagnostics).toEqual([]);
    expect(piped).toEqual([1]);                 // initial pipe at subscription (inside the derive's change())
    change(() => tick!(undefined));
    expect(piped).toEqual([1, 2]);              // the leaf effect re-ran in place — end-to-end reactivity
  });

  it('an unregistered head declines → the walk emits an unknown wrapper (domain materializes later)', () => {
    const env = new RecordingHostEnv(['Story']);   // only Story known
    const { value } = derive('component Story() { sankey() }', { env, minter: new PathKeyMinter() });
    const root = value as LangWrapper;
    const sankey = (root.children as unknown[]).find((c) => (c as LangWrapper).head === 'sankey') as LangWrapper;
    expect(sankey.__mlWrap).toBe('unknown');
  });

  it('a non-converging reactive feedback loop surfaces as an ML-RT-CONVERGE diagnostic (not a throw)', () => {
    const holder: { host?: RuntimeReactiveHost } = {};
    let a: unknown, b: unknown;
    const env = {
      resolveCall(head: string, key: string) {
        const h = holder.host!;
        if (head === 'feedback') {
          a = h.allocateCell(0); b = h.allocateCell(0);
          h.runLeafEffect(() => h.readCell(a), () => h.writeCell(b, (h.readCell(b) as number) + 1));
          h.runLeafEffect(() => h.readCell(b), () => h.writeCell(a, (h.readCell(a) as number) + 1));
        }
        return { handled: true as const, value: { head, key } };
      },
    };
    const { diagnostics } = derive('component Story() { feedback() }', {
      env, minter: new PathKeyMinter(), onHost: (h) => { holder.host = h; },
    });
    expect(diagnostics.some((d) => d.code === 'ML-RT-CONVERGE')).toBe(true);
  });

  it('a non-ReactiveFlushError thrown during the flush is rethrown, not swallowed as converge', () => {
    // A domain leaf-effect sink that throws a plain (non-convergence) error DURING THE FLUSH — the
    // drain() that change() runs in its finally, AFTER the walk has settled. derive must let it
    // propagate: it is a genuine bug, distinct from a ReactiveFlushError (→ ML-RT-CONVERGE).
    //   NOTE: the throw must occur in the DRAIN, not the leaf effect's initial synchronous pipe. The
    //   initial pipe runs inside resolveCall — i.e. inside lowerEntry's own try — so a throw there is
    //   caught by the walk (surfacing as an ML-LANG-INTERNAL diagnostic) and never reaches derive's
    //   catch. So: the sink is safe for its initial value, and a batched write (inside derive's
    //   change()) schedules a drain re-run that throws — escaping change() to derive's rethrow arm.
    const holder: { host?: RuntimeReactiveHost } = {};
    let c: unknown;
    const env = {
      resolveCall(head: string, key: string) {
        const h = holder.host!;
        if (head === 'boom') {
          c = h.allocateCell(1);
          h.runLeafEffect(() => h.readCell(c), (v) => { if (v !== 1) throw new Error('sink boom'); });
          h.writeCell(c, 2);   // batched inside change() → scheduled; the re-run throws during the drain
        }
        return { handled: true as const, value: { head, key } };
      },
    };
    expect(() => derive('component Story() { boom() }', {
      env, minter: new PathKeyMinter(), onHost: (h) => { holder.host = h; },
    })).toThrow('sink boom');
  });

  it('threads priorState so a surviving component instance latches its mutated state across a re-derive', () => {
    // Exercises derive's OWN priorState option end-to-end: pass 1 mutates the component's reactive
    // `let` through a captured handler, snapshots the settled state S via exportState(); pass 2 re-derives
    // the same source with S as priorState → the surviving instance latches the mutated value (n=2), not
    // its initializer (1). Load-bearing: (a) the pass-1 snapshot carried n=2, (b) a pass-2 leaf effect
    // over the latched cell pipes 2.
    // Pass 1: derive, capture the onTick handler, mutate n via change(), snapshot state.
    const holder1: { host?: RuntimeReactiveHost } = {};
    let tick: ((arg: unknown) => void) | undefined;
    const env1 = {
      resolveCall(head: string, key: string, args: { value: unknown }[]) {
        if (head === 'onTick') { const fn = args[0]?.value; if (typeof fn === 'function') tick = fn as (arg: unknown) => void; }
        return { handled: true as const, value: { head, key } };
      },
    };
    derive('component Story() { let n = 1; onTick(() => { n = n + 1 }) }', {
      env: env1, minter: new PathKeyMinter(), onHost: (h) => { holder1.host = h; },
    });
    change(() => tick!(undefined));                    // n: 1 → 2
    const priorState = holder1.host!.exportState();
    expect([...priorState.values()][0]).toBe(2);       // the snapshot carried the mutated n=2

    // Pass 2: derive the SAME instance-key source with priorState. A leaf effect over the reactive prop
    // observes the value the surviving cell latched to at derive time.
    const piped: number[] = [];
    const holder2: { host?: RuntimeReactiveHost } = {};
    const env2 = {
      resolveCall(head: string, key: string, args: { value: unknown }[]) {
        if (head === 'box') {
          const count = (args[0]?.value as Record<string, unknown>)?.count;
          if (isRegion(count)) holder2.host!.runLeafEffect((count as { run: () => unknown }).run, (v) => piped.push(v as number));
        }
        return { handled: true as const, value: { head, key } };
      },
    };
    derive('component Story() { let n = 1; box({ count: n }) }', {
      env: env2, minter: new PathKeyMinter(), priorState, onHost: (h) => { holder2.host = h; },
    });
    expect(piped).toEqual([2]);                         // latched to 2 through derive's priorState — NOT its init 1
  });

  it('reactiveData opts a data read into a reactive Region (vs eager when off)', () => {
    // With reactiveData, a `data.x` read in a prop lowers to a Region (a re-runnable thunk). A declining
    // host (box not registered) preserves the raw arg in the unknown wrapper, so we can observe the Region
    // that reached the host, threaded end-to-end through derive's reactiveData option.
    const env = new RecordingHostEnv(['Story']);   // box declined → wrapper preserves the raw arg
    const { value } = derive('component Story() { box({ v: data.x }) }', {
      env, minter: new PathKeyMinter(), data: { x: 5 }, reactiveData: true,
    });
    const root = value as LangWrapper;
    const box = (root.children as unknown[]).find((c) => isWrapper(c) && (c as LangWrapper).head === 'box') as LangWrapper;
    expect(isRegion((box.args[0]!.value as Record<string, unknown>).v)).toBe(true);
  });
});
