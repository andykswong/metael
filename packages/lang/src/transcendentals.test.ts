import { describe, it, expect } from 'vitest';
import { evaluateProgram, IMPLEMENTED_BUILTINS, BUILTINS } from './index.ts';
import { PlainStorageHost, RecordingHostEnv } from './ports.ts';

const run = (src: string) => evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv() });

describe('transcendental builtins (implemented + f64-exact reference)', () => {
  it('sin / cos', () => {
    expect(run('sin(0)').value).toBe(0);
    expect(run('cos(0)').value).toBe(1);
    expect((run('sin(1)').value as number)).toBeCloseTo(Math.sin(1), 12);
  });
  it('exp / log', () => {
    expect(run('exp(0)').value).toBe(1);
    expect((run('log(2.718281828459045)').value as number)).toBeCloseTo(1, 12);
  });
  it('fract', () => { expect((run('fract(3.25)').value as number)).toBeCloseTo(0.25, 12); });
  it('step (edge, x) → 0 if x < edge else 1', () => {
    expect(run('step(0.5, 0.2)').value).toBe(0);
    expect(run('step(0.5, 0.9)').value).toBe(1);
  });
  it('mix (a, b, t) → a + (b-a)*t', () => { expect((run('mix(0, 10, 0.5)').value as number)).toBeCloseTo(5, 12); });
  it('smoothstep (e0, e1, x)', () => {
    expect(run('smoothstep(0, 1, 0)').value).toBe(0);
    expect(run('smoothstep(0, 1, 1)').value).toBe(1);
    expect((run('smoothstep(0, 1, 0.5)').value as number)).toBeCloseTo(0.5, 12);
  });
  it('a non-numeric arg is fail-loud ML-LANG-BUILTIN-ARG', () => {
    expect(run('sin("x")').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
  it('they are now dispatched (not future) — in IMPLEMENTED_BUILTINS + carry a lowerName', () => {
    for (const name of ['sin', 'cos', 'exp', 'log', 'fract', 'step', 'mix', 'smoothstep']) {
      expect(IMPLEMENTED_BUILTINS.has(name)).toBe(true);
      expect(BUILTINS[name]!.future).toBeFalsy();
      expect(BUILTINS[name]!.lowerName).toBeTruthy();
    }
  });
  it('a user `function sin` still shadows the intrinsic', () => {
    expect(run('function sin() { 42 } sin(0)').value).toBe(42);
  });
});
