// The injection ports. `lang` calls these; the domain implements them.
// HostValue is OPAQUE to lang (for a domain it is a domain node). Test doubles here make
// `lang` unit-testable in isolation and prove the boundary (lang imports NOTHING from the domain).
import type { SourceSpan } from './diagnostics.ts';

export type HostValue = unknown;
export type CellRef = unknown;
export type EffectRegion = () => unknown;   // a re-runnable evaluation of an AST region

/** A REACTIVE prop value. An arg — or an object-literal ENTRY inside an arg — whose value
 *  expression reads a reactive `let` is emitted by the derive as a `Region` (a re-runnable thunk)
 *  instead of an eager value, so the runtime can register a per-attribute leaf effect over it. A
 *  static value stays a plain `HostValue`. `isRegion` lets a host tell them apart; test doubles
 *  ignore Regions (treat them as opaque values). */
export interface Region { readonly __region: true; run: EffectRegion }
export function region(run: EffectRegion): Region { return { __region: true, run }; }
export function isRegion(v: unknown): v is Region {
  return typeof v === 'object' && v !== null && (v as { __region?: unknown }).__region === true;
}

/** A DECLINED-CALL wrapper (part of the lang↔host value contract, like Region). When `resolveCall`
 *  returns `{ handled: false }`, lang cannot build a node (it is view-free), so it emits a tagged
 *  wrapper carrying the head + minted key + resolved args + already-built children, and the intent:
 *   • `'component'` — an in-DSL `component` instance the host doesn't recognize as a registered
 *     head → the runtime derive materializes it as a structural group node.
 *   • `'unknown'`   — an unregistered head (not a registered component, not an in-DSL component) →
 *     the runtime derive materializes a fallback node + a diagnostic (the extension-seam fallthrough).
 *  A host that DOES build the node returns `{ handled: true }` and no wrapper is emitted. Test doubles
 *  that answer every head never see one. */
export interface LangWrapper {
  readonly __mlWrap: 'component' | 'unknown';
  readonly head: string;
  readonly key: string;
  readonly args: Arg[];   // keep the parsed name/reactive roles (was flat HostValue[])
  readonly children: HostValue[];
}
export function wrapper(kind: 'component' | 'unknown', head: string, key: string, args: Arg[], children: HostValue[]): LangWrapper {
  return { __mlWrap: kind, head, key, args, children };
}
export function isWrapper(v: unknown): v is LangWrapper {
  return typeof v === 'object' && v !== null && typeof (v as { __mlWrap?: unknown }).__mlWrap === 'string';
}

/** One ordered call argument. lang preserves the parsed name-vs-position info instead of discarding
 *  it into a flat value array: `name` is present iff the author wrote `name: value`, and `reactive`
 *  is true iff `value` is a `Region` (it reads a reactive `let`). lang does NOT classify args beyond
 *  this — the runtime interprets per head. */
export interface Arg {
  value: HostValue;      // static value OR a Region if it reads reactive state
  name?: string;         // present iff the author wrote `name: value`
  reactive?: boolean;    // true iff `value` is a Region
}

/** Builds a node from an already-minted key + the ordered args + children.
 *  `args` is raw source order (e.g. text → [{value:'hi'}, {value:{size:48}}]); the runtime interprets
 *  per head. lang does NOT classify args. An arg's `value` (or an object-entry within it) may be a
 *  `Region` (reactive); a host that wants live reactivity registers a leaf effect per Region, while
 *  `RecordingHostEnv` just reads `region.run()` once. */
export interface HostEnvironment {
  // A resolved result defaults to a NODE (kind absent): an opaque domain value, valid only in child
  // position — a non-array object in expression position is rejected. A host may instead return
  // `kind: 'value'` to declare a PURE scalar/array/record VALUE: it is deep-frozen and allowed in
  // expression position. A `kind:'value'` result MUST be pure + deterministic (no live/mutable node).
  resolveCall(
    head: string,
    key: string,
    args: Arg[],
    children: HostValue[],
    span: SourceSpan,
  ): { handled: true; value: HostValue; kind?: 'value' } | { handled: false };
  // The set of heads this host builds — lang uses it for a did-you-mean on an unknown head.
  // OPTIONAL so test doubles + non-diagnosing hosts still satisfy the contract.
  knownHeads?: ReadonlySet<string>;
}

/** A metael-run scope: disposing it tears down every cell + leaf effect allocated inside `run`.
 *  Backed by a native `DisposableStack`. The standard `Disposable` (`{ [Symbol.dispose](): void }`)
 *  IS the teardown type — no bespoke `() => void` alias. */
export interface Scope<T> extends Disposable { readonly value: T }

export interface ReactiveHost {
  // `cellKey` is a STABLE per-component-instance identity for this reactive `let` (component-instance
  // key + let name + occurrence ordinal, minted by lowering). It lets a host LATCH the cell's settled
  // value across a re-derive (surviving instance keeps its state S; a new instance resets). OPTIONAL so
  // non-component/legacy callers + test doubles still work — when omitted the cell always uses `initial`.
  allocateCell(initial: unknown, cellKey?: string): CellRef; // one per component-scoped reactive `let`
  readCell(cell: CellRef): unknown;                      // value + register dep if a scope is active
  writeCell(cell: CellRef, value: unknown): void;        // value + schedule dependents
  // Opens a tracking scope, pipes the region's value to `sink` synchronously now + on each dependent
  // write, and returns a native Disposable that stops the effect (idempotent). A keyed-diff `remove`
  // disposes this so the leaf effect can't leak.
  runLeafEffect(region: EffectRegion, sink: (v: unknown) => void): Disposable;
  // Owner boundary (Disposable). Every cell + leaf effect allocated inside `run` is registered on the
  // scope's DisposableStack, so disposing the returned Scope tears them all down at once.
  scope<T>(run: () => T): Scope<T>;
}

/** Mints identity keys for reconciliation. The runtime supplies an impl closing over the domain's
 *  own key/hash scheme so lang never imports the domain. `content` (for list items) is opaque. */
export interface KeyMinter {
  structural(parentKey: string, kind: string, lexicalOrdinal: number): string;
  listItem(parentKey: string, kind: string, authorKey: unknown, ordinal: number, content: Record<string, unknown>): string;
}

// --- Test doubles (plain-storage host + path minter make lang a plain evaluator) ---

/** A minimal reactive effect record for PlainStorageHost's dependency tracking. */
interface PlainEffect { disposed: boolean; deps: Set<Set<PlainEffect>>; runOnce(): void }

export class PlainStorageHost implements ReactiveHost {
  private cells: unknown[] = [];
  // Per-cell subscriber sets: cellIndex → the effects that read it during their last run.
  private subscribers: Set<PlainEffect>[] = [];
  // The effect currently running (tracking). null when no effect is active (e.g. evaluator tests).
  private currentEffect: PlainEffect | null = null;
  // The DisposableStack of the innermost open scope(), or null when not inside a scope.
  private currentOwner: DisposableStack | null = null;

  // The plain double ignores cellKey (no latching); it always stores the initializer.
  allocateCell(initial: unknown, _cellKey?: string): CellRef {
    this.cells.push(initial);
    const index = this.cells.length - 1;
    this.subscribers[index] = new Set();
    // The plain array store needs no per-cell teardown — the scope tears down effects, not cells.
    return index;
  }

  readCell(cell: CellRef): unknown {
    // If an effect is tracking, subscribe it to this cell so a later writeCell re-runs it.
    if (this.currentEffect) {
      const subs = this.subscribers[cell as number]!;
      subs.add(this.currentEffect);
      this.currentEffect.deps.add(subs);
    }
    return this.cells[cell as number];
  }

  writeCell(cell: CellRef, value: unknown): void {
    this.cells[cell as number] = value;
    // Re-run every effect currently subscribed to this cell (snapshot — re-runs mutate the set).
    for (const effect of [...this.subscribers[cell as number]!]) {
      effect.runOnce();
    }
  }

  runLeafEffect(region: EffectRegion, sink: (v: unknown) => void): Disposable {
    // `runOnce` is an arrow so it captures the host (`this`) lexically and refers to `effect` by
    // name — no `this`-aliasing, and the tracking scope is set on the host for the region's reads.
    const effect: PlainEffect = {
      disposed: false,
      deps: new Set(),
      runOnce: (): void => {
        if (effect.disposed) return;
        // Clear prior cell-deps so this run's reads reflect the current dependency set.
        for (const subs of effect.deps) subs.delete(effect);
        effect.deps.clear();
        const previous = this.currentEffect;
        this.currentEffect = effect;
        let v: unknown;
        try {
          v = region();
        } finally {
          this.currentEffect = previous;
        }
        sink(v);
      },
    };
    const disposable: Disposable = {
      [Symbol.dispose](): void {
        if (effect.disposed) return; // idempotent
        effect.disposed = true;
        for (const subs of effect.deps) subs.delete(effect);
        effect.deps.clear();
      },
    };
    // Initial synchronous pipe (preserves the faithful conformance test). If the region throws on
    // this first run, unsubscribe the partial dependency set before propagating, so a thrown effect
    // never lingers in a cell's subscriber set (a later writeCell would otherwise re-invoke + re-throw).
    try {
      effect.runOnce();
    } catch (err) {
      disposable[Symbol.dispose]();
      throw err;
    }
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
      // If `run` threw partway, tear down whatever was already registered so no in-scope effect leaks.
      stack.dispose();
      throw err;
    } finally {
      this.currentOwner = previousOwner;
    }
    return {
      value,
      [Symbol.dispose](): void { stack.dispose(); }, // DisposableStack.dispose() is already idempotent
    };
  }
}

/** Nearest known head within Levenshtein distance 2 (else undefined). A domain uses this to turn a
 *  typo'd head into a fail-loud "did you mean X?" diagnostic; metael stays permissive by default
 *  (no knownHeads ⇒ always-wrap, unchanged). Pure — no host state. */
export function didYouMean(head: string, known: ReadonlySet<string>): string | undefined {
  let best: string | undefined;
  let bestDistance = Infinity;
  for (const candidate of known) {
    const d = levenshtein(head, candidate);
    // Keep the FIRST minimal we encounter (do not replace on equal distance) → deterministic tie-break
    // following the set's iteration order.
    if (d < bestDistance) {
      bestDistance = d;
      best = candidate;
    }
  }
  return bestDistance <= 2 ? best : undefined;
}

/** Standard Levenshtein edit distance (insert/delete/substitute, cost 1 each). Eval-free two-row DP. */
function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  let prev = Array.from({ length: n + 1 }, (_, j) => j); // distances for row i-1 (i.e. a[0..0])
  let curr = new Array<number>(n + 1);
  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(
        prev[j]! + 1,        // deletion
        curr[j - 1]! + 1,    // insertion
        prev[j - 1]! + cost, // substitution (or match)
      );
    }
    [prev, curr] = [curr, prev]; // reuse the old row as scratch for the next iteration
  }
  return prev[n]!;
}

export class RecordingHostEnv implements HostEnvironment {
  private readonly known?: string[];
  // The allowlist surfaced as a Set for lang's did-you-mean. Undefined when permissive
  // (no allowlist) so a permissive env exposes no knownHeads — metael stays permissive by default.
  readonly knownHeads?: ReadonlySet<string>;
  // Records each resolved call so tests can assert the Arg shape (name/reactive) reached the host.
  readonly calls: { head: string; key: string; args: Arg[]; children: HostValue[] }[] = [];
  // Union constructor: a positional `string[]` (new Arg-shape tests) OR an options object (faithful tests).
  constructor(known?: string[] | { known?: string[] }) {
    this.known = Array.isArray(known) ? known : known?.known;
    if (this.known) this.knownHeads = new Set(this.known);
  }
  resolveCall(head: string, key: string, args: Arg[], children: HostValue[], _span?: SourceSpan):
    { handled: true; value: HostValue } | { handled: false } {
    if (this.known && !this.known.includes(head)) return { handled: false };
    // Resolve Regions to their current value once (this double has no reactivity) so lang's own
    // lower tests see plain values; a deep resolve also unwraps Regions nested in object-entries.
    const resolve = (v: unknown): unknown =>
      isRegion(v) ? v.run()
      : Array.isArray(v) ? v.map(resolve)
      : (typeof v === 'object' && v !== null) ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, resolve(x)]))
      : v;
    // Preserve each arg's name/reactive via the spread; only resolve its `value`.
    const resolvedArgs = args.map((a) => ({ ...a, value: resolve(a.value) }));
    this.calls.push({ head, key, args: resolvedArgs, children });
    return { handled: true, value: { head, key, args: resolvedArgs.map((a) => a.value), children } };
  }
}

/** A minimal KeyMinter for lang's own tests (mirrors view's structuralKey/listItemKey shape). */
export class PathKeyMinter implements KeyMinter {
  structural(parentKey: string, kind: string, lexicalOrdinal: number): string {
    return `${parentKey}/${kind}#${lexicalOrdinal}`;
  }
  listItem(parentKey: string, kind: string, authorKey: unknown, ordinal: number): string {
    return authorKey != null ? `${parentKey}/${kind}[${String(authorKey)}]` : `${parentKey}/${kind}~${ordinal}`;
  }
}
