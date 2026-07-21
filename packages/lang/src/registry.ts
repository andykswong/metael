// packages/lang/src/registry.ts
// The builtin registry seam. A domain (a standard library) supplies a BuiltinModule; the evaluator
// dispatches an unbound call head to a registered Builtin, passing a capability context. This replaces
// the former hardcoded intrinsic cascade — the language kernel privileges no builtin by name.
import type { Expr } from './ast.ts';
import type { SourceSpan } from './diagnostics.ts';
import type { GenerationRef } from './ports.ts';

/** A builtin's capability profile (metadata for a classifier / a compile consumer). */
export type BuiltinProfile = 'core' | 'host';
/** How faithfully a builtin's result reproduces AWAY from the interpreter: `'exact'` (bit-for-bit
 *  identical on any target), `'gpu-tolerant'` (reproducible within a GPU's floating-point tolerance), or
 *  `'cpu-only'` (must run on the interpreter — not reproducible on a GPU target). Metadata for a
 *  classifier / a compile consumer; it does not affect interpretation. */
export type Portability = 'exact' | 'gpu-tolerant' | 'cpu-only';
/** The static self-description a {@link Builtin} declares: its call name, capability profile, and the
 *  metadata a classifier / a compile consumer reads to decide whether (and how) the call reproduces on
 *  another target. Pure data — it carries no behavior. */
export interface BuiltinSpec {
  /** The call head this builtin answers to — the identifier used at the call site (e.g. `'sqrt'`). */
  readonly name: string;
  /** Whether this is a domain-agnostic `'core'` builtin or a `'host'`-supplied capability
   *  ({@link BuiltinProfile}). */
  readonly profile: BuiltinProfile;
  /** How faithfully the result reproduces away from the interpreter ({@link Portability}). */
  readonly portability: Portability;
  /** True when the builtin accepts a closure argument (a mapping/filtering predicate), so a consumer
   *  knows a call site may carry an arrow/function value. */
  readonly takesClosure: boolean;
  /** The accepted argument-count range as `[min, max]`, inclusive. */
  readonly arity: readonly [number, number];
  /** True when the builtin is DECLARED in the catalog but NOT dispatched — a name whose classification is
   *  reserved for a future consumer, without adding a code path now. */
  readonly future?: boolean;
  /** The name this builtin lowers to on a compile target when it differs from {@link BuiltinSpec.name}
   *  (e.g. a source head that maps to a different intrinsic in generated code). Omitted when the target
   *  name matches the call name. */
  readonly lowerName?: string;
}

/** The capability context a builtin uses to interact with the interpreter — the ONLY channel by which a
 *  builtin reaches evaluation, budget, diagnostics, determinism, and reactive plumbing. It exposes NO
 *  concrete value constructor (vec/mat/buffer builders live in the library, not here). */
export interface BuiltinCtx {
  /** Charge `steps` against the run's fuel/deadline budget (default 1; a non-positive/non-finite value is
   *  clamped to 1). A builtin doing self-contained native work (e.g. splitting a string of length n) SHOULD
   *  charge it in one call — `tick(n)` — so the call fails closed with `ML-LANG-BUDGET` BEFORE the native
   *  work on a pathological input, without an O(n) accounting loop. A builtin whose loop invokes a user
   *  closure MUST instead tick once PER iteration, so the budget is checked between closure calls and an
   *  unbounded traversal cannot run unchecked.
   *  @param steps - the number of budget steps to charge; defaults to 1. */
  tick(steps?: number): void;
  /** Raise a diagnostic from within the builtin. Fails the call closed (the builtin returns its safe
   *  fallback) rather than throwing.
   *  @param code - the `ML-*` diagnostic code.
   *  @param message - the human-readable message.
   *  @param span - the source span to attribute; defaults to the call-site {@link BuiltinCtx.span}. */
  error(code: string, message: string, span?: SourceSpan): void;
  /** Draw the next value from the run's seeded PRNG (in `[0, 1)`), so a builtin's randomness stays part
   *  of the run's reproducible `rand()`/`range()` sequence. */
  rng(): number;
  /** The host clock capability, or undefined if the host injected none (then a datetime builtin fails
   *  loud + returns null — never a fake 0). */
  clock(): {
    /** Wall-clock time in milliseconds since the Unix epoch. */
    now(): number;
    /** A monotonic timestamp in milliseconds, suitable for measuring elapsed durations. */
    monotonic(): number;
  } | undefined;
  /** Evaluate argument i (0-based). NOTE: this RE-EVALUATES on each call (re-ticks + re-runs side
   *  effects) — it is not memoized. A builtin needing a value twice MUST bind it to a local. */
  evalArg(i: number): unknown;
  /** The number of argument expressions supplied at the call site. */
  argCount(): number;
  /** Invoke a value that is either an arrow closure or a user `function` as fn(...args). Returns null if v is neither. */
  callClosure(v: unknown, args: unknown[]): unknown;
  /** Allocate a fresh per-value generation reference ({@link GenerationRef}) a builtin can attach to a
   *  value it constructs, so a later in-place mutation of that value can signal reactive dependents. */
  allocateGeneration(): GenerationRef;
  /** Subscribe the current reactive scope to a value's generation, so reading it inside an effect
   *  re-runs when the value is mutated in place.
   *  @param g - the generation reference obtained from {@link BuiltinCtx.allocateGeneration}. */
  readGeneration(g: GenerationRef): void;
  /** Deep-freeze a result so it carries the same immutability guarantee as any interpreter value. */
  freeze<T>(v: T): T;
  /** The run's string-growth cap (characters). A builtin that BUILDS a string (e.g. join) fails closed
   *  with ML-LANG-BUDGET before crossing it, exactly as the `+` operator does. */
  readonly maxStringLength: number;
  /** The raw call-site span (for diagnostics a builtin raises itself). */
  readonly span: SourceSpan;
}

/** How the interpreter invokes a builtin. `argExprs` are UNEVALUATED — the builtin pulls values via
 *  ctx.evalArg(i) so closure-taking + short-circuit builtins stay expressible. */
export interface Builtin {
  /** The static self-description used for dispatch and classification ({@link BuiltinSpec}). */
  readonly spec: BuiltinSpec;
  /** Run the builtin. Called by the interpreter when a call head resolves to this builtin.
   *  @param ctx - the capability context ({@link BuiltinCtx}), the builtin's only channel back into the
   *               interpreter.
   *  @param argExprs - the UNEVALUATED argument expressions; the builtin pulls each value via
   *                    `ctx.evalArg(i)`, so closure-taking and short-circuit builtins stay expressible.
   *  @returns the call's result value. */
  invoke(ctx: BuiltinCtx, argExprs: readonly Expr[]): unknown;
}

/** A pluggable standard-library unit: a set of builtins a consumer injects at evaluateProgram. */
export interface BuiltinModule {
  /** The builtins this module contributes. Registered by name via {@link buildRegistry}. */
  readonly builtins: readonly Builtin[];
}

/** The resolved dispatch table: a read-only name→{@link Builtin} map the interpreter consults to
 *  resolve an unbound call head. Built from a consumer's modules by {@link buildRegistry}. */
export type BuiltinRegistry = ReadonlyMap<string, Builtin>;

/** Build a name→Builtin map. Later modules override earlier ones on a name clash (last wins). */
export function buildRegistry(modules: readonly BuiltinModule[]): BuiltinRegistry {
  const map = new Map<string, Builtin>();
  for (const mod of modules) for (const b of mod.builtins) map.set(b.spec.name, b);
  return map;
}

/** The shared empty {@link BuiltinRegistry} — used for a builtin-free run so no allocation or module
 *  wiring is needed when a consumer injects no standard-library modules. */
export const EMPTY_REGISTRY: BuiltinRegistry = new Map();
