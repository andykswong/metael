import { describe, it, expect } from 'vitest';
import { PlainStorageHost, RecordingHostEnv, PathKeyMinter, didYouMean, frozenClock } from './ports.ts';
import type { Arg, ReactiveHost, HostEnvironment, HostValue, SourceSpan } from './ports.ts';
import type { BindableHostEnv } from './index.ts';

describe('injection ports (test doubles)', () => {
  it('PlainStorageHost stores + reads a cell value (tracking elided, value present)', () => {
    const h = new PlainStorageHost();
    const c = h.allocateCell(5);
    expect(h.readCell(c)).toBe(5);
    h.writeCell(c, 7);
    expect(h.readCell(c)).toBe(7);
  });

  it('PlainStorageHost.runLeafEffect runs the region once synchronously and pipes to sink', () => {
    const h = new PlainStorageHost();
    let sunk: unknown;
    h.runLeafEffect(() => 42, (v) => { sunk = v; });
    expect(sunk).toBe(42);
  });

  it('resolveCall receives head, minted key, ORDERED args, and children — returns an opaque value', () => {
    const env = new RecordingHostEnv();
    const r = env.resolveCall('text', 'story/0/text#0', [{ value: 'hi' }, { value: { size: 48 } }], ['childA'], { start: 0, end: 1 });
    expect(r).toEqual({
      handled: true,
      value: { head: 'text', key: 'story/0/text#0', args: ['hi', { size: 48 }], children: ['childA'] },
    });
  });

  it('resolveCall returns handled:false for an unknown head', () => {
    const env = new RecordingHostEnv({ known: ['layout'] });
    expect(env.resolveCall('sankey', 'k', [], [], { start: 0, end: 1 })).toEqual({ handled: false });
  });

  it('PathKeyMinter mints a structural path key from parent + kind + ordinal', () => {
    const m = new PathKeyMinter();
    expect(m.structural('Story#0', 'layout', 0)).toBe('Story#0/layout#0');
  });
});

// --- The optional host clock capability (replayable time source) ---
describe('ReactiveHost clock capability', () => {
  it('PlainStorageHost exposes a default real clock (now/monotonic are numbers)', () => {
    const h = new PlainStorageHost();
    const clk = h.clock();
    expect(typeof clk.now()).toBe('number');
    expect(typeof clk.monotonic()).toBe('number');
    expect(clk.now()).toBeGreaterThan(0);   // wall-clock ms since epoch
  });

  it('an injected clock overrides the default (replayable time)', () => {
    const h = new PlainStorageHost(() => frozenClock(1234));
    expect(h.clock().now()).toBe(1234);
    expect(h.clock().monotonic()).toBe(1234);
  });

  it('frozenClock(t) reports the same now + monotonic on every read', () => {
    const clk = frozenClock(9000);
    expect(clk.now()).toBe(9000);
    expect(clk.now()).toBe(9000);         // stable across reads (deterministic)
    expect(clk.monotonic()).toBe(9000);
  });
});

// --- resolveCall takes an ordered Arg[] carrying optional name + reactive flag ---
describe('HostEnvironment Arg shape (review fix)', () => {
  it('resolveCall receives args carrying optional name + reactive flag', () => {
    const env = new RecordingHostEnv(['box']);
    const args: Arg[] = [
      { value: 1 },
      { value: 'red', name: 'color' },
      { value: () => 2, name: 'size', reactive: true } as unknown as Arg,
    ];
    const r = env.resolveCall('box', 'box#0', args, [], { start: 0, end: 3 });
    expect(r.handled).toBe(true);
    const rec = env.calls[0]!;
    expect(rec.args.map((a) => a.name)).toEqual([undefined, 'color', 'size']);
    expect(rec.args[2]!.reactive).toBe(true);
  });
});

// --- Native TC39 Disposable teardown + scope() owner boundary ---
// NOTE the CORRECTED expectations: PlainStorageHost pipes the region's initial value to the sink
// SYNCHRONOUSLY at subscription (the load-bearing faithful conformance above), THEN re-pipes on each
// tracked write. So a cell init=1 + one write of 2 yields [1, 2] (initial pipe + one write), not [2].
describe('ReactiveHost disposal (native Disposable) + scope() owner boundary', () => {
  it('runLeafEffect returns a Disposable that stops future sink calls', () => {
    const host = new PlainStorageHost();
    const seen: unknown[] = [];
    const cell = host.allocateCell(1);
    const fx = host.runLeafEffect(() => host.readCell(cell), (v) => { seen.push(v); });
    host.writeCell(cell, 2);
    fx[Symbol.dispose]();
    host.writeCell(cell, 3); // after disposal → does NOT fire
    expect(seen).toEqual([1, 2]);
  });

  it('a Disposable works with `using`', () => {
    const host = new PlainStorageHost();
    const seen: unknown[] = [];
    const cell = host.allocateCell(1);
    {
      using _fx = host.runLeafEffect(() => host.readCell(cell), (v) => { seen.push(v); });
      host.writeCell(cell, 2);
    } // block ends → _fx auto-disposed
    host.writeCell(cell, 3); // after auto-dispose → does NOT fire
    expect(seen).toEqual([1, 2]);
  });

  it('scope() returns a Scope<T> whose disposal tears down every cell + effect inside it', () => {
    const host = new PlainStorageHost();
    const seen: unknown[] = [];
    const s = host.scope(() => {
      const c = host.allocateCell(0);
      host.runLeafEffect(() => host.readCell(c), (v) => { seen.push(v); });
      host.writeCell(c, 10);
      return c;
    });
    expect(s.value).toBeDefined();
    s[Symbol.dispose]();
    const before = seen.length;
    s[Symbol.dispose](); // idempotent — no extra fire
    expect(seen.length).toBe(before);
    host.writeCell(s.value, 20); // scoped cell written after disposal → effect torn down, no growth
    expect(seen.length).toBe(before);
  });
});

// --- disposal hardening: a throwing region/run must not leak a subscription ---
describe('ReactiveHost disposal — error paths do not leak effects', () => {
  it('a region that throws on its initial run leaves no lingering subscription', () => {
    const host = new PlainStorageHost();
    const cell = host.allocateCell(1);
    // The region reads the cell (subscribing) THEN throws — before runLeafEffect can return a handle.
    expect(() => host.runLeafEffect(
      () => { host.readCell(cell); throw new Error('boom'); },
      () => { /* never reached */ },
    )).toThrow('boom');
    // A later write must NOT re-invoke the thrown region (it would re-throw out of writeCell).
    expect(() => host.writeCell(cell, 2)).not.toThrow();
  });

  it('scope() whose run throws tears down effects already registered inside it', () => {
    const host = new PlainStorageHost();
    const seen: unknown[] = [];
    let cell: unknown;
    expect(() => host.scope(() => {
      cell = host.allocateCell(0);
      host.runLeafEffect(() => host.readCell(cell), (v) => { seen.push(v); }); // seen: [0]
      throw new Error('scope-boom');
    })).toThrow('scope-boom');
    const before = seen.length;                 // the initial pipe already ran once
    host.writeCell(cell, 5);                     // effect was torn down by the failed scope → no fire
    expect(seen.length).toBe(before);
  });
});

// --- populate knownHeads on RecordingHostEnv + pure didYouMean helper ---
describe('knownHeads / did-you-mean (review fix)', () => {
  it('RecordingHostEnv exposes its allowlist as knownHeads', () => {
    const env = new RecordingHostEnv(['box', 'sphere', 'union']);
    expect(env.knownHeads?.has('box')).toBe(true);
    expect(env.knownHeads?.has('nope')).toBe(false);
  });
  it('a permissive RecordingHostEnv (no allowlist) has no knownHeads', () => {
    const env = new RecordingHostEnv();
    expect(env.knownHeads).toBeUndefined();
  });
  it('didYouMean suggests the closest known head for a typo', () => {
    expect(didYouMean('bpx', new Set(['box', 'sphere']))).toBe('box');
    expect(didYouMean('zzzzz', new Set(['box']))).toBeUndefined();  // too far → no suggestion
  });
  it('didYouMean returns undefined for an empty candidate set', () => {
    expect(didYouMean('box', new Set())).toBeUndefined();
  });
  it('didYouMean includes distance exactly 2 but excludes distance 3 (boundary)', () => {
    expect(didYouMean('baxx', new Set(['box']))).toBe('box');       // distance 2 → suggested
    expect(didYouMean('baxxx', new Set(['box']))).toBeUndefined();  // distance 3 → excluded
  });
  it('didYouMean tie-break is deterministic — first minimal in set iteration order', () => {
    // 'abcd' and 'xbc' are both distance 1 from 'abc'; the first in insertion order wins.
    expect(didYouMean('abc', new Set(['abcd', 'xbc']))).toBe('abcd');
    expect(didYouMean('abc', new Set(['xbc', 'abcd']))).toBe('xbc');
  });
});

// --- BindableHostEnv: a HostEnvironment plus bindHost port ---
describe('BindableHostEnv', () => {
  it('is a HostEnvironment plus a bindHost(host) method (structural)', () => {
    // A concrete env that satisfies the interface compiles + runs.
    const env: BindableHostEnv = {
      resolveCall(_h: string, _k: string, _a: Arg[], _c: HostValue[], _s: SourceSpan) { return { handled: false as const }; },
      bindHost(_host: ReactiveHost) { /* no-op */ },
    };
    expect(typeof env.bindHost).toBe('function');
    // It is assignable to a plain HostEnvironment (extends relationship).
    const asHost: HostEnvironment = env;
    expect(typeof asHost.resolveCall).toBe('function');
  });
});
