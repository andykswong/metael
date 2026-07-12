import { it, expect } from 'vitest';
import { mount } from '@metael/vdom';
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';

it('workspace wiring: app can import and drive both metael targets', () => {
  const h = mount('component Story() { span("hi") }', undefined, {});
  expect(h.diagnostics).toEqual([]);
  expect(h.tree()).not.toBeNull();

  const res = evaluateProgram('1 + 2', { host: new PlainStorageHost(), env: new RecordingHostEnv() });
  expect(res.value).toBe(3);
  expect(res.diagnostics).toEqual([]);
});
