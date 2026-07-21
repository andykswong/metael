import { describe, it, expect } from 'vitest';
import 'disposablestack/auto';
import { renderSource } from '@metael/vdom/lang';
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';

it('workspace wiring: app can import and drive both metael targets', () => {
  const h = renderSource('component Story() { span("hi") }', undefined, {});
  expect(h.diagnostics).toEqual([]);
  expect(h.tree()).not.toBeNull();

  const res = evaluateProgram('1 + 2', { host: new PlainStorageHost(), env: new RecordingHostEnv() });
  expect(res.value).toBe(3);
  expect(res.diagnostics).toEqual([]);
});

describe('disposablestack/auto polyfill', () => {
  it('installs Symbol.dispose + DisposableStack globals', () => {
    expect(typeof Symbol.dispose).toBe('symbol');
    expect(typeof (globalThis as { DisposableStack?: unknown }).DisposableStack).toBe('function');
  });
});
