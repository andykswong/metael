import type { ReactiveHost, CellRef, EffectRegion, Scope, GenerationRef } from '@metael/lang';
import { signal, effect, type Signal } from './reactive.ts';

/**
 * The real ReactiveHost over the @vue/reactivity signal core. One signal per component-scoped reactive
 * `let` (coarse whole-cell tracking). runLeafEffect opens a tracking scope so readCell's dependency is
 * collected; on a later change() only dependent leaf effects re-run.
 *
 *   • runLeafEffect returns a native Disposable (pipes the region's INITIAL value synchronously, then
 *     on each dependent write) — so a keyed-diff `remove` can dispose it instead of leaking it.
 *   • scope<T>(run) is an owner boundary backed by a DisposableStack: disposing it tears down every
 *     leaf effect AND frees every cell allocated inside `run`.
 */
export class RuntimeReactiveHost implements ReactiveHost {
  /** Every reactive-`let` cell allocated during a derive, in allocation order (for a well-formedness
   *  walk if a domain wants one). */
  readonly cells: CellRef[] = [];
  /** cellKey → cell for every KEYED cell (component `let`s). Drives exportState() so state S carries
   *  across a re-derive. Cells with no cellKey are absent → never latched. */
  private readonly keyedCells = new Map<string, CellRef>();
  /** The prior pass's settled state S (cellKey → value), threaded in on a re-derive so a SURVIVING
   *  component instance latches the value its handlers mutated; a NEW instance (unknown key) resets. */
  private readonly priorState?: ReadonlyMap<string, unknown>;
  /** The DisposableStack of the innermost open scope(), or null when not inside a scope. */
  private currentOwner: DisposableStack | null = null;

  constructor(priorState?: ReadonlyMap<string, unknown>) { this.priorState = priorState; }

  allocateCell(initial: unknown, cellKey?: string): CellRef {
    // LATCH: a keyed cell whose key was present in the prior pass's settled S starts from that carried
    // value (surviving instance keeps its state), else from its initializer (fresh / first derive).
    const start = (cellKey !== undefined && this.priorState?.has(cellKey)) ? this.priorState.get(cellKey) : initial;
    const c = signal(start);
    this.cells.push(c);
    if (cellKey !== undefined) this.keyedCells.set(cellKey, c);
    // If allocated inside a scope, free this cell's bookkeeping when the scope is disposed — so a keyed
    // `remove` (which disposes the removed subtree's scope) does NOT leak: exportState() must not carry a
    // gone instance's state, and the keyed store must not grow unbounded. Registering a `defer` on the
    // owner stack ties cell lifetime to the same owner boundary that tears down the leaf effects.
    this.currentOwner?.defer(() => {
      if (cellKey !== undefined) this.keyedCells.delete(cellKey);   // only keyed cells were stored
      const i = this.cells.indexOf(c);
      if (i !== -1) this.cells.splice(i, 1);
    });
    return c;
  }

  readCell(cell: CellRef): unknown { return (cell as Signal<unknown>).get(); }
  writeCell(cell: CellRef, value: unknown): void { (cell as Signal<unknown>).set(value); }

  // A per-VALUE generation signal (a tracked reactive number) backs reactive in-place mutation of a
  // mutable custom value (a typed array). It is NOT a component cell: not pushed to `cells`, not keyed,
  // not latched, not exported in state. A reactive read subscribes; an in-place write bumps it. `touch`
  // writes get()+1 — always Object.is-distinct → always fires (a same-reference writeCell would no-op).
  allocateGeneration(): GenerationRef {
    return signal(0);
  }
  readGeneration(gen: GenerationRef): number { return (gen as Signal<number>).get(); }
  touchGeneration(gen: GenerationRef): void { const s = gen as Signal<number>; s.set(s.get() + 1); }

  runLeafEffect(region: EffectRegion, sink: (v: unknown) => void): Disposable {
    // vue effect() runs `fn` synchronously once (initial pipe), tracking region's cell reads, then on
    // each dependent write (scheduled through change()'s batch). The returned stop() is the teardown.
    // No explicit initial-throw guard is needed: if `fn` throws on this first synchronous run, vue's
    // effect() stops the effect and rethrows, so a throwing region never lingers as a live subscription.
    const stop = effect(() => { sink(region()); });
    const disposable: Disposable = { [Symbol.dispose]: () => { stop(); } };
    // If allocated inside a scope, register teardown on that scope's stack.
    this.currentOwner?.use(disposable);
    return disposable;
  }

  scope<T>(run: () => T): Scope<T> {
    const stack = new DisposableStack();
    const previousOwner = this.currentOwner;
    this.currentOwner = stack;
    let value: T;
    try {
      value = run();
    } catch (err) {
      stack.dispose();   // partial-run cleanup: tear down whatever registered before the throw
      throw err;
    } finally {
      this.currentOwner = previousOwner;
    }
    return {
      value,
      [Symbol.dispose](): void { stack.dispose(); },   // DisposableStack.dispose() is idempotent
    };
  }

  /** Snapshot the settled state S of every keyed cell (cellKey → current value) for carry-forward into
   *  the next re-derive's fresh host. Read AFTER the flush so handler mutations are captured. */
  exportState(): Map<string, unknown> {
    const out = new Map<string, unknown>();
    for (const [k, cell] of this.keyedCells) out.set(k, this.readCell(cell));
    return out;
  }
}
