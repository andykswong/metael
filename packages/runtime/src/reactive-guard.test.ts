import { describe, it, expect } from 'vitest';
import { signal, effect, change, ReactiveFlushError } from './reactive.ts';

describe('converge guard (our change() drain, not Vue)', () => {
  it('a cross-effect feedback loop trips the guard instead of hanging', () => {
    const a = signal(0); const b = signal(0);
    effect(() => { b.set(a.get() + 1); });   // A → schedules B
    effect(() => { a.set(b.get() + 1); });   // B → reschedules A
    expect(() => change(() => { a.set(1); })).toThrow(ReactiveFlushError);
  });
  it('a normal batch of independent effects drains without tripping', () => {
    const a = signal(0); let runs = 0;
    effect(() => { a.get(); runs++; });
    expect(() => change(() => { a.set(1); })).not.toThrow();
    expect(runs).toBe(2);   // initial run + one batched flush
  });
  it('nested change() flushes only at the outermost boundary', () => {
    const a = signal(1); let runs = 0;
    effect(() => { a.get(); runs++; });
    change(() => {
      change(() => { a.set(2); });
      // inner change() did NOT flush (still batched by the outer); effect has run only once so far
      expect(runs).toBe(1);
    });
    expect(runs).toBe(2);   // outer boundary flushed exactly once
  });
});

describe('change() exception safety: batch state never leaks', () => {
  it('when fn() throws in the outermost change(), a later change() still batches', () => {
    const a = signal(0); let runs = 0;
    effect(() => { a.get(); runs++; });      // runs === 1 (initial)
    expect(() => change(() => { a.set(1); throw new Error('boom'); })).toThrow('boom');
    // the write before the throw still flushed on the way out (drain runs in finally)
    runs = 0;
    change(() => { a.set(2); });             // MUST batch normally — proves `batched` was reset
    expect(runs).toBe(1);                     // exactly one batched flush, not a leaked-state double/zero
  });

  it('when a scheduled effect throws mid-drain, a later change() still batches', () => {
    const a = signal(0); const b = signal(0);
    let throwNow = false;
    const stop = effect(() => { a.get(); if (throwNow) throw new Error('effect boom'); });
    let bRuns = 0;
    effect(() => { b.get(); bRuns++; });
    throwNow = true;
    expect(() => change(() => { a.set(1); })).toThrow('effect boom');
    stop();                                   // stop the throwing effect so the next change() is clean
    bRuns = 0;
    change(() => { b.set(5); });              // MUST batch — proves batched reset after mid-drain throw
    expect(bRuns).toBe(1);
  });
});
