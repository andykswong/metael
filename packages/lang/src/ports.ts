// The injection ports. `lang` calls these; the domain implements them.
// HostValue is OPAQUE to lang (for a domain it is a domain node). Test doubles here make
// `lang` unit-testable in isolation and prove the boundary (lang imports NOTHING from the domain).
import type { SourceSpan } from './diagnostics.ts';
export type { SourceSpan } from './diagnostics.ts';

/** The lang↔host value-contract payload — whatever a {@link HostEnvironment.resolveCall} builds. lang
 *  never inspects it: for a consuming domain it is a domain node, for the test doubles a plain record.
 *  Kept `unknown` so the kernel imports nothing from a domain. */
export type HostValue = unknown;
/** An opaque handle to one reactive-`let` cell held by a {@link ReactiveHost}. lang passes it back to
 *  read/write the cell; only the host knows its concrete shape. */
export type CellRef = unknown;
/** An opaque handle to one per-value generation change-signal held by a {@link ReactiveHost} — the
 *  reactive channel for in-place mutation of a custom value (distinct from a `let` cell). */
export type GenerationRef = unknown;
/** A re-runnable evaluation of an AST region: a zero-arg thunk the host re-invokes to recompute a
 *  reactive value. Backs both a {@link Region} prop thunk and a leaf effect. */
export type EffectRegion = () => unknown;

/** A REACTIVE prop value. An arg — or an object-literal ENTRY inside an arg — whose value
 *  expression reads a reactive `let` is emitted by the derive as a `Region` (a re-runnable thunk)
 *  instead of an eager value, so the runtime can register a per-attribute leaf effect over it. A
 *  static value stays a plain `HostValue`. `isRegion` lets a host tell them apart; test doubles
 *  ignore Regions (treat them as opaque values). */
export interface Region {
  /** Brand marking this object as a reactive `Region` (vs a plain {@link HostValue}). Always `true`. */
  readonly __region: true;
  /** The re-runnable thunk that recomputes the prop's value; the host re-invokes it inside a leaf
   *  effect so the attribute tracks the reactive `let`s it reads. */
  run: EffectRegion;
}
/** Wrap a re-runnable thunk as a {@link Region} — the tagged form the derive emits for a reactive prop
 *  value so a host can register a per-attribute leaf effect over it. */
export function region(run: EffectRegion): Region { return { __region: true, run }; }
/** Type guard: is `v` a {@link Region} (a reactive prop thunk), as opposed to a static {@link HostValue}?
 *  A host calls this to decide whether to register a leaf effect; the test doubles use it to resolve a
 *  Region to its current value once. */
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
  /** The declined-call intent: `'component'` for an in-DSL `component` instance the host didn't
   *  recognize as a registered head, `'unknown'` for an unregistered head (the extension-seam
   *  fallthrough). Also the brand {@link isWrapper} checks for. */
  readonly __mlWrap: 'component' | 'unknown';
  /** The call head the host declined to build. */
  readonly head: string;
  /** The identity key already minted by lowering for this call site. */
  readonly key: string;
  /** The resolved ordered arguments, keeping their parsed name/reactive roles ({@link Arg}). */
  readonly args: Arg[];
  /** The children already built for this call. */
  readonly children: HostValue[];
}
/** Build a {@link LangWrapper} — the tagged fallback the derive emits when {@link HostEnvironment.resolveCall}
 *  returns `{ handled: false }` and lang (being view-free) cannot build the node itself. */
export function wrapper(kind: 'component' | 'unknown', head: string, key: string, args: Arg[], children: HostValue[]): LangWrapper {
  return { __mlWrap: kind, head, key, args, children };
}
/** Type guard: is `v` a {@link LangWrapper} (a declined-call fallback the runtime derive materializes),
 *  as opposed to a host-built {@link HostValue}? */
export function isWrapper(v: unknown): v is LangWrapper {
  return typeof v === 'object' && v !== null && typeof (v as { __mlWrap?: unknown }).__mlWrap === 'string';
}

/** One ordered call argument. lang preserves the parsed name-vs-position info instead of discarding
 *  it into a flat value array: `name` is present iff the author wrote `name: value`, and `reactive`
 *  is true iff `value` is a `Region` (it reads a reactive `let`). lang does NOT classify args beyond
 *  this — the runtime interprets per head. */
export interface Arg {
  /** The argument's value: a static {@link HostValue}, or a {@link Region} if it reads reactive state. */
  value: HostValue;
  /** Present iff the author wrote `name: value` — the parsed argument name. Absent for a positional arg. */
  name?: string;
  /** True iff `value` is a {@link Region} (the argument reads a reactive `let`). */
  reactive?: boolean;
}

/** Builds a node from an already-minted key + the ordered args + children.
 *  `args` is raw source order (e.g. text → [{value:'hi'}, {value:{size:48}}]); the runtime interprets
 *  per head. lang does NOT classify args. An arg's `value` (or an object-entry within it) may be a
 *  `Region` (reactive); a host that wants live reactivity registers a leaf effect per Region, while
 *  `RecordingHostEnv` just reads `region.run()` once. */
export interface HostEnvironment {
  /** Resolve a call head to a host value. A resolved result defaults to a NODE (`kind` absent): an
   *  opaque domain value valid only in child position — a non-array object in expression position is
   *  rejected. A host may instead return `kind: 'value'` to declare a PURE scalar/array/record VALUE,
   *  which is deep-frozen and allowed in expression position (it MUST be pure + deterministic, never a
   *  live/mutable node). Returning `{ handled: false }` declines the call so the derive emits a
   *  {@link LangWrapper} fallback instead.
   *  @param head - the call head (the identifier at the call site).
   *  @param key - the identity key minted by lowering for this call site.
   *  @param args - the ordered arguments in source order, keeping name/reactive roles ({@link Arg}); an
   *                arg's `value` (or an object-entry within it) may be a {@link Region} (reactive).
   *  @param children - the already-built children.
   *  @param span - the call-site source span, for a diagnostic the host may raise. */
  resolveCall(
    head: string,
    key: string,
    args: Arg[],
    children: HostValue[],
    span: SourceSpan,
  ):
    | {
        /** Marks the call as built by this host. */
        handled: true;
        /** The built host value (a node by default). */
        value: HostValue;
        /** `'value'` to declare a pure, deep-frozen scalar/array/record allowed in expression position;
         *  omitted for a node (child-position only). */
        kind?: 'value';
      }
    | {
        /** Marks the call as declined so the derive emits a {@link LangWrapper} fallback. */
        handled: false;
      };
  /** The set of heads this host builds — lang uses it for a did-you-mean on an unknown head. OPTIONAL so
   *  test doubles + non-diagnosing hosts still satisfy the contract (absent ⇒ metael stays permissive). */
  knownHeads?: ReadonlySet<string>;
}

/** A {@link HostEnvironment} that also receives the reactive host out-of-band. The `resolveCall` port
 *  carries no host parameter by design (the runtime hands the host over exactly once, before the walk),
 *  so a stateful domain env exposes `bindHost` to capture it. Named here — the single home of the
 *  env-contract vocabulary — so a domain env, `composeEnvs`, and `mount`'s `envFactory` all reference one
 *  type instead of an inline `HostEnvironment & { bindHost(host): void }` intersection. */
export interface BindableHostEnv extends HostEnvironment {
  /** Receive the reactive host before the walk runs `resolveCall`. Called once by the runtime's derive
   *  (via its `onHost` callback). */
  bindHost(host: ReactiveHost): void;
}

/** A metael-run scope: disposing it tears down every cell + leaf effect allocated inside `run`.
 *  Backed by a native `DisposableStack`. The standard `Disposable` (`{ [Symbol.dispose](): void }`)
 *  IS the teardown type — no bespoke `() => void` alias. */
export interface Scope<T> extends Disposable {
  /** The value produced by the scoped `run` — returned alongside the teardown so a caller keeps the
   *  result and disposes the owner in one handle. */
  readonly value: T;
}

/** The reactivity + identity capability a domain injects: reactive `let` cells, leaf effects, owner
 *  scopes, per-value generation signals, and an optional clock. lang drives these; a domain supplies the
 *  concrete fine-grained reactive runtime behind them. The kernel imports nothing from a domain — every
 *  handle ({@link CellRef}/{@link GenerationRef}/{@link Scope}) is opaque. */
export interface ReactiveHost {
  /** Allocate one cell per component-scoped reactive `let`, seeded with `initial`. `cellKey` is a STABLE
   *  per-component-instance identity (component-instance key + let name + occurrence ordinal, minted by
   *  lowering); it lets a host LATCH the cell's settled value across a re-derive (a surviving instance
   *  keeps its state, a new instance resets). OPTIONAL so non-component/legacy callers + test doubles
   *  still work — when omitted the cell always uses `initial`. */
  allocateCell(initial: unknown, cellKey?: string): CellRef;
  /** Read a cell's current value, and register it as a dependency of the effect currently tracking (if a
   *  scope is active), so a later {@link ReactiveHost.writeCell} re-runs that effect. */
  readCell(cell: CellRef): unknown;
  /** Write a cell's value and schedule its dependents to re-run. */
  writeCell(cell: CellRef, value: unknown): void;
  /** Open a tracking scope, pipe the region's value to `sink` synchronously now + on each dependent
   *  write, and return a native `Disposable` that stops the effect (idempotent). A keyed-diff `remove`
   *  disposes this so the leaf effect can't leak. */
  runLeafEffect(region: EffectRegion, sink: (v: unknown) => void): Disposable;
  /** Run `run` inside an owner boundary and return its value plus teardown as a {@link Scope}. Every cell
   *  + leaf effect allocated inside `run` is registered on the scope, so disposing the returned Scope
   *  tears them all down at once. */
  scope<T>(run: () => T): Scope<T>;
  /** Allocate a fresh per-VALUE generation signal — a tracked reactive number, distinct from a component
   *  `let` cell: NOT keyed, NOT latched, NOT exported in state. A mutable custom value (a typed array)
   *  owns one so its in-place mutation can signal reactive dependents. */
  allocateGeneration(): GenerationRef;
  /** Read a generation's current number, registering it as a dependency if a scope is active — a reactive
   *  READ of the owning value subscribes here. */
  readGeneration(gen: GenerationRef): number;
  /** Bump a generation (a strictly-increasing number → always fires, unlike a same-reference
   *  {@link ReactiveHost.writeCell} no-op). An in-place WRITE of the owning value calls this; the re-run
   *  is coalesced to one at the `change()` flush. */
  touchGeneration(gen: GenerationRef): void;
  /** OPTIONAL time capability supplying the injected {@link Clock}. INJECTED (not read from an ambient
   *  global) so a run is replayable: a domain/test supplies a frozen or recorded clock and the same
   *  source + inputs + clock reproduce the same trace. ABSENT (a host that injects no clock) → the
   *  datetime builtins fail loud with `ML-LANG-NO-CLOCK` and return null, never a fabricated zero. */
  clock?(): Clock;
}

/** A replayable time source: `now()` = wall-clock ms since the epoch; `monotonic()` = a non-decreasing
 *  high-res reading. Both are injected so datetime is deterministic under replay (never ambient time). */
export interface Clock {
  /** Wall-clock time in milliseconds since the Unix epoch. */
  now(): number;
  /** A monotonically-non-decreasing high-resolution reading, for measuring elapsed durations. */
  monotonic(): number;
}

/** Mints identity keys for reconciliation. The runtime supplies an impl closing over the domain's
 *  own key/hash scheme so lang never imports the domain. `content` (for list items) is opaque. */
export interface KeyMinter {
  /** Mint the reconciliation key for a STRUCTURAL (statically-positioned) child, from its parent's key,
   *  its node kind, and its lexical ordinal among siblings.
   *  @param parentKey - the parent node's key.
   *  @param kind - the child's node kind (the call head).
   *  @param lexicalOrdinal - the child's 0-based position among its siblings in source order. */
  structural(parentKey: string, kind: string, lexicalOrdinal: number): string;
  /** Mint the reconciliation key for a LIST-ITEM child, preferring the author-supplied key when present
   *  and falling back to the ordinal otherwise.
   *  @param parentKey - the parent node's key.
   *  @param kind - the item's node kind.
   *  @param authorKey - the author-supplied identity for the item (opaque to lang), or nullish.
   *  @param ordinal - the item's 0-based position in the list, used when `authorKey` is absent.
   *  @param content - the item's fields, opaque to lang, available to a domain's own key/hash scheme. */
  listItem(parentKey: string, kind: string, authorKey: unknown, ordinal: number, content: Record<string, unknown>): string;
}

// --- Test doubles (plain-storage host + path minter make lang a plain evaluator) ---

/** A minimal reactive effect record for PlainStorageHost's dependency tracking. */
interface PlainEffect { disposed: boolean; deps: Set<Set<PlainEffect>>; runOnce(): void }

/** A frozen clock for deterministic/replay runs: `now()` and `monotonic()` both report the same `t` on
 *  every read. A test injects one (via `new PlainStorageHost(() => frozenClock(t))`) so `now()`/`monotonic()`
 *  become reproducible — time as a recorded input rather than ambient wall-clock. */
export function frozenClock(t: number): Clock {
  return { now: () => t, monotonic: () => t };
}

/** The default REAL clock a live host uses when none is injected: wall-clock ms + a high-res monotonic
 *  reading. Fine for a running app; a test/replay run overrides it with a `frozenClock`. */
function realClock(): Clock {
  return { now: () => Date.now(), monotonic: () => performance.now() };
}

/** A minimal in-memory {@link ReactiveHost} that makes lang a plain evaluator in isolation — the kernel's
 *  own test/default reactive backend, with no domain dependency. Cells are stored in a plain array (no
 *  latching — `cellKey` is ignored), effects track cell/generation reads and re-run on write, and
 *  `scope` tears down via a native `DisposableStack`. The clock factory defaults to the real clock;
 *  inject `() => frozenClock(t)` for a deterministic/replay run. */
export class PlainStorageHost implements ReactiveHost {
  private cells: unknown[] = [];
  // Per-cell subscriber sets: cellIndex → the effects that read it during their last run.
  private subscribers: Set<PlainEffect>[] = [];
  // The effect currently running (tracking). null when no effect is active (e.g. evaluator tests).
  private currentEffect: PlainEffect | null = null;
  // The DisposableStack of the innermost open scope(), or null when not inside a scope.
  private currentOwner: DisposableStack | null = null;
  // The injected clock factory. Defaults to the REAL clock so existing `new PlainStorageHost()` callers
  // are unaffected; a test supplies `() => frozenClock(t)` to make now()/monotonic() deterministic.
  private readonly clockFactory: () => Clock;

  /** Construct the store, optionally injecting a clock factory.
   *  @param clock - a factory for the {@link Clock} each `clock()` call returns; defaults to the real
   *                 clock so existing callers are unaffected. Pass `() => frozenClock(t)` for a
   *                 deterministic/replay run. */
  constructor(clock: () => Clock = realClock) {
    this.clockFactory = clock;
  }

  /** Return a {@link Clock} from the injected factory (the real clock unless one was supplied). */
  clock(): Clock { return this.clockFactory(); }

  /** Allocate a reactive cell holding `initial`, returning its {@link CellRef} handle.
   *  @remarks The plain store ignores `cellKey` (no latching) and always stores the initializer. */
  allocateCell(initial: unknown, _cellKey?: string): CellRef {
    this.cells.push(initial);
    const index = this.cells.length - 1;
    this.subscribers[index] = new Set();
    // The plain array store needs no per-cell teardown — the scope tears down effects, not cells.
    return index;
  }

  /** Read a cell's current value, tracking it as a dependency of any effect currently running.
   *  @remarks If an effect is tracking, subscribes it to this cell so a later {@link PlainStorageHost.writeCell}
   *           re-runs it. */
  readCell(cell: CellRef): unknown {
    // If an effect is tracking, subscribe it to this cell so a later writeCell re-runs it.
    if (this.currentEffect) {
      const subs = this.subscribers[cell as number]!;
      subs.add(this.currentEffect);
      this.currentEffect.deps.add(subs);
    }
    return this.cells[cell as number];
  }

  /** Write a cell's value and schedule its dependents to re-run.
   *  @remarks Re-runs every effect currently subscribed to this cell (over a snapshot, since a re-run
   *           mutates the subscriber set). */
  writeCell(cell: CellRef, value: unknown): void {
    this.cells[cell as number] = value;
    // Re-run every effect currently subscribed to this cell (snapshot — re-runs mutate the set).
    for (const effect of [...this.subscribers[cell as number]!]) {
      effect.runOnce();
    }
  }

  /** Open a tracking scope over `region`, pipe its value to `sink` now and on each dependent write, and
   *  return a native `Disposable` that stops the effect (idempotent).
   *  @remarks Each run clears the prior cell-deps before re-reading, and if the initial synchronous run
   *           throws, the partial dependency set is unsubscribed before the error propagates so a thrown
   *           effect never lingers in a subscriber set. */
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

  /** Run `run` inside an owner boundary and return its value plus teardown as a {@link Scope}, disposing
   *  every cell + leaf effect allocated inside when the Scope is disposed.
   *  @remarks Backed by a native `DisposableStack`; if `run` throws partway, whatever was already
   *           registered is torn down before the error propagates so no in-scope effect leaks. */
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

  // Per-value generation signals: index → {value, subscribers}. Separate from cells (not latched).
  private generations: { value: number; subs: Set<PlainEffect> }[] = [];

  /** Allocate a fresh per-value generation signal, returning its {@link GenerationRef} handle.
   *  @remarks Stored in a plain array separate from cells (a generation is not latched). */
  allocateGeneration(): GenerationRef {
    this.generations.push({ value: 0, subs: new Set() });
    return this.generations.length - 1;
  }
  /** Read a generation's current number, tracking it as a dependency of any effect currently running.
   *  @remarks Subscribes the tracking effect (if any) to this generation. */
  readGeneration(gen: GenerationRef): number {
    const g = this.generations[gen as number]!;
    if (this.currentEffect) { g.subs.add(this.currentEffect); this.currentEffect.deps.add(g.subs); }
    return g.value;
  }
  /** Bump a generation's number so its reactive dependents re-run.
   *  @remarks Increments the number and re-runs every subscribed effect (over a snapshot of the set). */
  touchGeneration(gen: GenerationRef): void {
    const g = this.generations[gen as number]!;
    g.value += 1;
    for (const effect of [...g.subs]) effect.runOnce();
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

/** A test-double {@link HostEnvironment} that answers (or, with an allowlist, selectively declines) every
 *  call and RECORDS what reached it — so a test can assert the {@link Arg} shape (name/reactive) and the
 *  built value. Having no reactivity, it resolves each {@link Region} to its current value once (deeply,
 *  through object-entries) so lang's own lower tests see plain values. */
export class RecordingHostEnv implements HostEnvironment {
  private readonly known?: string[];
  /** The allowlist surfaced as a Set for lang's did-you-mean, or `undefined` when permissive (no
   *  allowlist) so a permissive env exposes no {@link HostEnvironment.knownHeads} — metael stays
   *  permissive by default. */
  readonly knownHeads?: ReadonlySet<string>;
  /** The log of resolved calls, in order, so a test can assert what reached the host. */
  readonly calls: {
    /** The resolved call's head. */
    head: string;
    /** The identity key minted for the call site. */
    key: string;
    /** The ordered arguments as recorded, with each {@link Region} resolved to its value. */
    args: Arg[];
    /** The children passed to the call. */
    children: HostValue[];
  }[] = [];
  /** Construct the recorder, optionally with an allowlist of known heads.
   *  @param known - the heads to answer, as a positional `string[]` OR an `{ known }` options object;
   *                 omit to answer every head permissively. A head outside a supplied allowlist is
   *                 declined (`{ handled: false }`). */
  constructor(known?: string[] | { known?: string[] }) {
    this.known = Array.isArray(known) ? known : known?.known;
    if (this.known) this.knownHeads = new Set(this.known);
  }
  /** Resolve a call head to a host value, recording the head/key/args/children that reached it.
   *  @remarks Declines a head outside a supplied allowlist; otherwise resolves each {@link Region}
   *           argument (and any Region nested in an object-entry) to its current value once, records the
   *           call, and returns the recorded head/key/args/children as the built value. */
  resolveCall(head: string, key: string, args: Arg[], children: HostValue[], _span?: SourceSpan):
    {
      /** Marks the call as answered by this recorder. */
      handled: true;
      /** The recorded head/key/args/children returned as the built value. */
      value: HostValue;
    } | {
      /** Marks a head outside the allowlist as declined. */
      handled: false;
    } {
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
  /** Mint the reconciliation key for a structural (statically-positioned) child.
   *  @remarks Mints a path-style key `parentKey/kind#ordinal`. */
  structural(parentKey: string, kind: string, lexicalOrdinal: number): string {
    return `${parentKey}/${kind}#${lexicalOrdinal}`;
  }
  /** Mint the reconciliation key for a list-item child, preferring an author-supplied key over the ordinal.
   *  @remarks Mints `parentKey/kind[authorKey]` when an author key is present, else falls back to
   *           `parentKey/kind~ordinal`. */
  listItem(parentKey: string, kind: string, authorKey: unknown, ordinal: number): string {
    return authorKey != null ? `${parentKey}/${kind}[${String(authorKey)}]` : `${parentKey}/${kind}~${ordinal}`;
  }
}
