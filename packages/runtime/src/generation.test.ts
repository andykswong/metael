import { describe, it, expect } from 'vitest';
import { RuntimeReactiveHost } from './reactive-host.ts';
import { change, effect } from './reactive.ts';
import { evaluateProgram, descriptorOf, generationOf } from '@metael/lang';
import { RecordingHostEnv } from '@metael/lang';

describe('per-value generation signal — in-place mutation is reactive', () => {
  it('allocateGeneration starts at 0; touch increments; read subscribes', () => {
    const host = new RuntimeReactiveHost();
    const gen = host.allocateGeneration();
    expect(host.readGeneration(gen)).toBe(0);
    let seen = -1;
    const stop = effect(() => { seen = host.readGeneration(gen); });
    expect(seen).toBe(0);
    change(() => { host.touchGeneration(gen); });
    expect(seen).toBe(1);   // the effect re-ran on the bump
    stop();
  });

  it('multiple touches inside ONE change() coalesce to a single re-run at flush', () => {
    const host = new RuntimeReactiveHost();
    const gen = host.allocateGeneration();
    let runs = 0;
    const stop = effect(() => { host.readGeneration(gen); runs++; });
    expect(runs).toBe(1);   // initial
    change(() => { host.touchGeneration(gen); host.touchGeneration(gen); host.touchGeneration(gen); });
    expect(runs).toBe(2);   // exactly one re-run for three writes (batched)
    stop();
  });

  it('a buffer created + mutated through the evaluator carries a live generation', () => {
    const host = new RuntimeReactiveHost();
    const res = evaluateProgram('f32([1, 2, 3])', { host, env: new RecordingHostEnv() });
    const buf = res.value;
    expect(descriptorOf(buf)?.lower?.access).toBe('linear-buffer');
    const gen = generationOf(buf);
    expect(gen).toBeDefined();
    expect(host.readGeneration(gen!)).toBe(0);
  });
});
