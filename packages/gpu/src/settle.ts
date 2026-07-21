// packages/gpu/src/settle.ts
// Free helpers over a `() => GpuResource` re-dispatch thunk — the change()/drain/re-read settle dance,
// lifted off the façade so it works for ANY dispatch kind (map/reduce/histogram). The thunk carries its
// own dispose guard (a disposed engine's dispatch throws), so the loop terminates on teardown with no
// façade-private state to consult.
import { effect } from '@metael/runtime';
import type { GpuResource } from './resource.ts';

/** A macrotask-drain backstop against a never-settling dispatch (far above any real settle latency). */
const MAX_SETTLE_ITERS = 10_000;

/** Type guard: is this resource no longer in flight (`pending === false`)? NOTE `value`/`outputs` stay
 *  `| null` even when settled — a non-core / cost-rejected / emit-errored run also settles with
 *  `pending: false` but no value (it carries `reasons`/`error` instead); `value` is declared independent of
 *  `pending` (no discriminated union). So this narrows "settled vs in-flight", NOT "value present" — a
 *  caller must still null-check `value`. */
export function settled(r: GpuResource): r is GpuResource & { pending: false } {
  return r.pending === false;
}

/** Await a settled resource by re-running `dispatch` (a `() => engine.dispatch(kernel, cfg)` thunk): dispatch
 *  → drain a macrotask → re-dispatch until `!pending` (or an `error`). The thunk carries its own dispose
 *  guard (a disposed engine's dispatch throws), so this loop terminates on teardown. Free of the façade —
 *  works for ANY dispatch kind (map/reduce/histogram) since it only reads `pending`/`error`. */
export async function settle(dispatch: () => GpuResource, opts: { maxIters?: number } = {}): Promise<GpuResource> {
  const maxIters = opts.maxIters ?? MAX_SETTLE_ITERS;
  let r = dispatch();
  let iters = 0;
  while (r.pending && !r.error) {
    if (++iters > maxIters) throw new Error('gpu dispatch did not settle within the iteration bound');
    await new Promise<void>((res) => setTimeout(res, 0));
    r = dispatch();   // a disposed engine throws here → propagates out, ending the loop
  }
  return r;
}

/** Subscribe to a dispatch's lifecycle: `onValue` fires with the pending resource, then again with the
 *  settled one (the engine reads its resource through a reactive cell, so this tracked effect re-fires on
 *  settle). Returns a stop(). `onValue` must sink to NON-reactive targets (a reactive read inside would
 *  self-subscribe). Free of the façade. */
export function subscribe(dispatch: () => GpuResource, onValue: (r: GpuResource) => void): () => void {
  return effect(() => { onValue(dispatch()); });
}
