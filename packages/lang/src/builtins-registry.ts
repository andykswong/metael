// The machine-readable catalog of metael's intrinsic builtins. Each entry declares the builtin's
// capability PROFILE and, for numeric builtins, its cross-target PORTABILITY class. This is the single
// source of truth for "which names are intrinsics" (a user `function` of the same name shadows one) and
// the input a static profile classifier + any future codegen consumes. Pure data — no evaluator import.

/** A builtin's capability profile.
 *  - 'core': no closure args, no heap types (scalar / fixed-shape). Expressible on a restricted
 *    (compile-to-shader / linear-memory) target as well as the interpreter.
 *  - 'host': takes a closure argument and/or touches heap-dynamic types (strings, objects, dynamic
 *    arrays). Only expressible on a full-data-model (interpreter / runtime-backed) target. */
export type BuiltinProfile = 'core' | 'host';

/** Cross-target reproducibility of a numeric builtin's result.
 *  - 'exact': IEEE-754 correctly-rounded; bit-identical across targets (given a fixed rounding mode).
 *  - 'gpu-tolerant': ULP/absolute-error bounded on a shader target; not bit-identical across vendors.
 *  - 'cpu-only': touches heap/string/closure; has no compile-to-shader lowering at all. */
export type Portability = 'exact' | 'gpu-tolerant' | 'cpu-only';

export interface BuiltinSpec {
  readonly name: string;
  readonly profile: BuiltinProfile;
  readonly portability: Portability;
  readonly takesClosure: boolean;
  /** Fixed positional arity, or a [min, max] range (max = Infinity for variadic). Callback/optional
   *  args are counted in the range. Advisory metadata — the evaluator still validates arg shapes. */
  readonly arity: readonly [number, number];
  /** True when the builtin is DECLARED in the model but NOT YET dispatched by the evaluator (a
   *  pre-tagged future-tier entry, e.g. transcendentals awaiting a shader/compute consumer). */
  readonly future?: boolean;
}

/** Every intrinsic metael knows about, keyed by name. Implemented entries (future !== true) are
 *  dispatched by the evaluator; future entries reserve the name's classification for the day a
 *  consumer needs it, without adding a code path now. */
export const BUILTINS: Readonly<Record<string, BuiltinSpec>> = Object.freeze({
  // --- seeded / already shipped ---
  rand:        { name: 'rand',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [0, 0] },
  range:       { name: 'range',       profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },
  map:         { name: 'map',         profile: 'host', portability: 'cpu-only',     takesClosure: true,  arity: [2, 2] },
  filter:      { name: 'filter',      profile: 'host', portability: 'cpu-only',     takesClosure: true,  arity: [2, 2] },
  reduce:      { name: 'reduce',      profile: 'host', portability: 'cpu-only',     takesClosure: true,  arity: [3, 3] },
  keys:        { name: 'keys',        profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },
  values:      { name: 'values',      profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },
  entries:     { name: 'entries',     profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },
  fromEntries: { name: 'fromEntries', profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },

  // --- query / predicate ---
  some:        { name: 'some',        profile: 'host', portability: 'cpu-only',     takesClosure: true,  arity: [2, 2] },
  every:       { name: 'every',       profile: 'host', portability: 'cpu-only',     takesClosure: true,  arity: [2, 2] },
  find:        { name: 'find',        profile: 'host', portability: 'cpu-only',     takesClosure: true,  arity: [2, 2] },
  findIndex:   { name: 'findIndex',   profile: 'host', portability: 'cpu-only',     takesClosure: true,  arity: [2, 2] },
  includes:    { name: 'includes',    profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [2, 2] },

  // --- ordering / slicing ---
  sort:        { name: 'sort',        profile: 'host', portability: 'cpu-only',     takesClosure: true,  arity: [1, 2] },
  slice:       { name: 'slice',       profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [2, 3] },
  reverse:     { name: 'reverse',     profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },

  // --- string bridge ---
  split:       { name: 'split',       profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [2, 2] },
  join:        { name: 'join',        profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [2, 2] },
  chars:       { name: 'chars',       profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },
  toUpperCase: { name: 'toUpperCase', profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },
  toLowerCase: { name: 'toLowerCase', profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },
  trim:        { name: 'trim',        profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },

  // --- numeric core ---
  min:         { name: 'min',         profile: 'core', portability: 'exact',        takesClosure: false, arity: [2, 2] },
  max:         { name: 'max',         profile: 'core', portability: 'exact',        takesClosure: false, arity: [2, 2] },
  abs:         { name: 'abs',         profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1] },
  sign:        { name: 'sign',        profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1] },
  floor:       { name: 'floor',       profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1] },
  ceil:        { name: 'ceil',        profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1] },
  round:       { name: 'round',       profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1] },
  clamp:       { name: 'clamp',       profile: 'core', portability: 'exact',        takesClosure: false, arity: [3, 3] },
  sqrt:        { name: 'sqrt',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1] },
  pow:         { name: 'pow',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2] },
  format:      { name: 'format',      profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [2, 2] },

  // --- future tier (declared, NOT dispatched — reserve the classification) ---
  sin:         { name: 'sin',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], future: true },
  cos:         { name: 'cos',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], future: true },
  exp:         { name: 'exp',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], future: true },
  log:         { name: 'log',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], future: true },
  fract:       { name: 'fract',       profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1], future: true },
  step:        { name: 'step',        profile: 'core', portability: 'exact',        takesClosure: false, arity: [2, 2], future: true },
  mix:         { name: 'mix',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3], future: true },
  smoothstep:  { name: 'smoothstep',  profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3], future: true },
});

/** True iff `name` is a metael intrinsic (implemented OR future-declared). Callers that need only
 *  the dispatched set should also check `!BUILTINS[name].future`. */
export function isBuiltin(name: string): boolean {
  return Object.prototype.hasOwnProperty.call(BUILTINS, name);
}

/** The set of builtin names the evaluator actually dispatches (implemented, not future-only). */
export const IMPLEMENTED_BUILTINS: ReadonlySet<string> = new Set(
  Object.values(BUILTINS).filter((b) => !b.future).map((b) => b.name),
);
