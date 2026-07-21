// Reactive core: @vue/reactivity (vendored) with OUR synchronous change() batch/flush boundary
// (Vue exposes no public synchronous flush; this glue is ours). change() drains scheduled effects;
// a converge guard fails closed (ReactiveFlushError) on cross-effect feedback past a fixed cap.
// Domain-neutral: no @metael/lang import here.
import { shallowRef, computed, effect as vueEffect, type Ref } from '@vue/reactivity';

/** A writable reactive cell holding one whole value. Reading through {@link Signal.get} inside an
 *  {@link effect} or {@link memo} subscribes to it; {@link Signal.set} replaces the value and schedules
 *  dependents. Tracking is whole-value (Object.is): a replacement fires, an in-place mutation does not. */
export interface Signal<T> {
  /** Read the current value, registering a dependency when a tracking scope is active. */
  get(): T;
  /** Replace the value; schedules dependents when the new value is Object.is-distinct from the current. */
  set(v: T): void;
}
/** A read-only reactive derivation: a cached value recomputed lazily when a dependency it read changes.
 *  Reading through {@link Memo.get} inside a tracking scope subscribes to the memo. */
export interface Memo<T> {
  /** Read the (lazily recomputed) derived value, registering a dependency when a tracking scope is active. */
  get(): T;
}

/**
 * Create a writable reactive {@link Signal} seeded with `initial`.
 *
 * @param initial - the cell's starting value.
 * @returns a signal with whole-value-replacement reactivity: `set(v)` fires dependents only on Object.is
 *          inequality. This matches the substrate's model — a plain value is immutable and rebuilt to
 *          update, while a custom value's in-place mutation is tracked by its own generation signal, not
 *          the cell.
 */
export function signal<T>(initial: T): Signal<T> {
  // shallowRef: whole-value-replacement reactivity (set(newRef) fires on Object.is inequality). This is
  // metael's model — a plain value is immutable + rebuilt to update, and a custom value's in-place
  // mutation is tracked by its OWN generation signal (not the cell). A deep `ref` would also fail on a
  // custom value whose non-configurable descriptor Symbol violates a Proxy invariant.
  const r: Ref<T> = shallowRef(initial) as Ref<T>;
  return { get: () => r.value, set: (v: T) => { r.value = v; } };
}
/**
 * Create a read-only reactive {@link Memo} that caches `compute()` and recomputes it lazily when a
 * signal/memo it read changes.
 *
 * @param compute - the derivation; it should be a pure function of the reactive values it reads.
 * @returns a memo whose {@link Memo.get} returns the cached value, recomputing on demand after a
 *          dependency changes.
 */
export function memo<T>(compute: () => T): Memo<T> {
  const c = computed(compute);
  return { get: () => c.value };
}

let batched: Set<() => void> | null = null;

/** Thrown by {@link change} when the drain loop fails to converge: effects keep scheduling further
 *  effects past `MAX_DRAIN_ITERATIONS`, signalling a cross-effect feedback cycle. Fails closed rather
 *  than hanging. */
export class ReactiveFlushError extends Error {
  /** Construct the error with the fixed "reactive flush did not converge" message and `name`. */
  constructor() { super('reactive flush did not converge'); this.name = 'ReactiveFlushError'; }
}
const MAX_DRAIN_ITERATIONS = 10_000;

/**
 * Register a reactive effect: run `fn` once immediately (tracking every signal/memo it reads), then
 * re-run it whenever one of those dependencies changes.
 *
 * @param fn - the side-effecting body; its reads become the effect's dependency set on each run.
 * @returns a stop function that tears the effect down (unsubscribes it) when called.
 * @remarks While a {@link change} batch is open, a re-run is deferred and drained at the batch boundary
 *          rather than firing inline.
 */
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
