// Reactive core: @vue/reactivity (vendored) with OUR synchronous change() batch/flush boundary
// (Vue exposes no public synchronous flush; this glue is ours). change() drains scheduled effects;
// a converge guard fails closed (ReactiveFlushError) on cross-effect feedback past a fixed cap.
// Domain-neutral: no @metael/lang import here.
import { shallowRef, computed, effect as vueEffect, type Ref } from '@vue/reactivity';

export interface Signal<T> { get(): T; set(v: T): void }
export interface Memo<T> { get(): T }

export function signal<T>(initial: T): Signal<T> {
  // shallowRef: whole-value-replacement reactivity (set(newRef) fires on Object.is inequality). This is
  // metael's model — a plain value is immutable + rebuilt to update, and a custom value's in-place
  // mutation is tracked by its OWN generation signal (not the cell). A deep `ref` would also fail on a
  // custom value whose non-configurable descriptor Symbol violates a Proxy invariant.
  const r: Ref<T> = shallowRef(initial) as Ref<T>;
  return { get: () => r.value, set: (v: T) => { r.value = v; } };
}
export function memo<T>(compute: () => T): Memo<T> {
  const c = computed(compute);
  return { get: () => c.value };
}

let batched: Set<() => void> | null = null;

export class ReactiveFlushError extends Error {
  constructor() { super('reactive flush did not converge'); this.name = 'ReactiveFlushError'; }
}
const MAX_DRAIN_ITERATIONS = 10_000;

export function effect(fn: () => void): () => void {
  const runner = vueEffect(fn, { scheduler: () => { if (batched) batched.add(runner); else runner(); } });
  return () => { runner.effect.stop(); };
}

/**
 * THE synchronous batch/flush boundary. Writes accumulate; scheduled effects drain. The drain
 * re-collects effects scheduled *during* the drain (where cross-effect feedback shows up) and fails
 * closed with {@link ReactiveFlushError} past MAX_DRAIN_ITERATIONS. Re-entrant: a nested change()
 * joins the outer batch and does NOT flush early (the outermost boundary flushes once).
 */
export function change<T>(fn: () => T): T {
  const outer = batched;
  const local = outer ?? new Set<() => void>();
  batched = local;
  try { return fn(); }
  finally {
    if (!outer) drain(local);   // drain the outermost batch after fn() settles (in a called fn, not inline, to satisfy no-unsafe-finally)
  }
}

function drain(local: Set<() => void>): void {
  let iterations = 0;
  try {
    while (local.size > 0) {
      if (++iterations > MAX_DRAIN_ITERATIONS) throw new ReactiveFlushError();
      const batch = [...local]; local.clear();
      for (const r of batch) r();     // r may re-add to `local` via the scheduler
    }
  } finally {
    batched = null;
  }
}
