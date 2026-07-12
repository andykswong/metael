// Target dispatch: the same source, two backends. A UI run mounts a real @metael/vdom app into the given
// container (its live DOM is the preview). A compute run evaluates the source to a pure value and returns
// its pretty-printed string. Both return the diagnostics they produced so the caller can drive the shared
// diagnostics UI. The caller owns the "only swap the preview when diagnostics are empty" policy (see
// create.ts); a run always reports what it found.
import { mount } from '@metael/vdom';
import type { VDomHandle } from '@metael/vdom';
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import type { Diagnostic } from '@metael/lang';
import { prettyValue } from './compute-view.ts';
import type { Target } from './examples.ts';

export interface RunOptions {
  data?: unknown;
  seed?: number;
}

export interface UiRun { kind: 'ui'; handle: VDomHandle; diagnostics: Diagnostic[] }
export interface ComputeRun { kind: 'compute'; text: string; value: unknown; diagnostics: Diagnostic[] }
export type TargetRun = UiRun | ComputeRun;

/** Run `source` against `target`. For 'ui', `container` receives the live mount; for 'compute' it is unused. */
export function runTarget(target: Target, source: string, container: Element | undefined, opts: RunOptions): TargetRun {
  if (target === 'ui') {
    const handle = mount(source, container, { data: opts.data, seed: opts.seed });
    return { kind: 'ui', handle, diagnostics: handle.diagnostics };
  }
  const res = evaluateProgram(source, {
    host: new PlainStorageHost(), env: new RecordingHostEnv(), data: opts.data, seed: opts.seed,
  });
  return { kind: 'compute', text: prettyValue(res.value), value: res.value, diagnostics: res.diagnostics };
}
