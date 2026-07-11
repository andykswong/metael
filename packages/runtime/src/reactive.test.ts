import { describe, it, expect } from 'vitest';
import { signal, memo, effect, change } from './reactive.ts';

describe('reactive core (@vue/reactivity + our change())', () => {
  it('effect re-runs when a read signal changes', () => {
    const a = signal(1); const seen: number[] = [];
    effect(() => seen.push(a.get()));
    a.set(2);
    expect(seen).toEqual([1, 2]);
  });
  it('memo caches and recomputes on dep change', () => {
    const a = signal(2); let n = 0;
    const d = memo(() => { n++; return a.get() * 2; });
    expect(d.get()).toBe(4); expect(d.get()).toBe(4); expect(n).toBe(1);
    a.set(3); expect(d.get()).toBe(6); expect(n).toBe(2);
  });
  it('change() batches writes and flushes effects once', () => {
    const a = signal(1); const b = signal(1); let runs = 0;
    effect(() => { a.get(); b.get(); runs++; });
    change(() => { a.set(2); b.set(2); });
    expect(runs).toBe(2); // initial + one batched flush
  });
  it('is glitch-free: an effect never observes a stale memo', () => {
    const a = signal(1); const sum = memo(() => a.get() + 1); const obs: number[] = [];
    effect(() => obs.push(a.get() + sum.get()));
    a.set(10);
    expect(obs).toEqual([3, 21]);
  });
  it('effect stop() prevents further re-runs', () => {
    const a = signal(1); const seen: number[] = [];
    const stop = effect(() => seen.push(a.get()));
    stop();
    a.set(2);
    expect(seen).toEqual([1]);
  });
});
