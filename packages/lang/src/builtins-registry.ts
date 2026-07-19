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
  /** For a numeric builtin that lowers to a native target function (dot/cross/sqrt/sin/…), the target
   *  builtin name. A compile consumer maps a call to this native name; a spec with no lowerName is
   *  CPU/interpreter-only in a shader kernel. */
  readonly lowerName?: string;
}

/** Every intrinsic metael knows about, keyed by name. Implemented entries (future !== true) are
 *  dispatched by the evaluator; future entries reserve the name's classification for the day a
 *  consumer needs it, without adding a code path now. */
export const BUILTINS: Readonly<Record<string, BuiltinSpec>> = Object.freeze({
  // --- seeded / already shipped ---
  rand:        { name: 'rand',        profile: 'core', portability: 'cpu-only',     takesClosure: false, arity: [0, 0] },
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
  sqrt:        { name: 'sqrt',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'sqrt' },
  pow:         { name: 'pow',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], lowerName: 'pow' },
  format:      { name: 'format',      profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [2, 2] },

  // --- typed-array constructors (custom-type protocol; buffers) ---
  f32:         { name: 'f32',         profile: 'core', portability: 'gpu-tolerant', takesClosure: true,  arity: [1, 2] },
  f64:         { name: 'f64',         profile: 'core', portability: 'exact',        takesClosure: true,  arity: [1, 2] },
  i32:         { name: 'i32',         profile: 'core', portability: 'exact',        takesClosure: true,  arity: [1, 2] },
  u32:         { name: 'u32',         profile: 'core', portability: 'exact',        takesClosure: true,  arity: [1, 2] },

  // --- vec/mat constructors + numeric builtins (custom-type protocol; value math) ---
  vec2:        { name: 'vec2',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 2] },
  vec3:        { name: 'vec3',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 3] },
  vec4:        { name: 'vec4',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 4] },
  mat2:        { name: 'mat2',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [0, 4] },
  mat3:        { name: 'mat3',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [0, 9] },
  mat4:        { name: 'mat4',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [0, 16] },
  mat2x3:      { name: 'mat2x3',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [6, 6] },
  mat2x4:      { name: 'mat2x4',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [8, 8] },
  mat3x2:      { name: 'mat3x2',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [6, 6] },
  mat3x4:      { name: 'mat3x4',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [12, 12] },
  mat4x2:      { name: 'mat4x2',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [8, 8] },
  mat4x3:      { name: 'mat4x3',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [12, 12] },
  dot:         { name: 'dot',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], lowerName: 'dot' },
  cross:       { name: 'cross',       profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], lowerName: 'cross' },
  normalize:   { name: 'normalize',   profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'normalize' },
  length:      { name: 'length',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'length' },
  transpose:   { name: 'transpose',   profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'transpose' },
  determinant: { name: 'determinant', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'determinant' },
  // No lowerName: WGSL has no inverse() at all (hand-emitted per matrix size); GLSL has inverse() natively
  // (emitted by an explicit name override). The gate accepts it on portability 'gpu-tolerant'.
  inverse:     { name: 'inverse',     profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1] },
  distance:    { name: 'distance',    profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], lowerName: 'distance' },
  reflect:     { name: 'reflect',     profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], lowerName: 'reflect' },
  refract:     { name: 'refract',     profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3], lowerName: 'refract' },
  faceforward: { name: 'faceforward', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3] },

  // --- quaternions (vec4 layout (x,y,z,w) = imaginary xyz + real w; no distinct quat value type) ---
  // No lowerName: every q* op is HAND-EMITTED inline (or via a small prelude helper) on all targets —
  // there is no native quaternion type or builtin in WGSL/GLSL. The gate accepts them on 'gpu-tolerant'.
  qmul:       { name: 'qmul',       profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2] },
  qconj:      { name: 'qconj',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1] },
  qinvert:    { name: 'qinvert',    profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1] },
  qaxisangle: { name: 'qaxisangle', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2] },
  qrotate:    { name: 'qrotate',    profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2] },
  qslerp:     { name: 'qslerp',     profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3] },
  qmat:       { name: 'qmat',       profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1] },

  // --- transcendentals (implemented; native-lowerable to a shader builtin) ---
  sin:         { name: 'sin',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'sin' },
  cos:         { name: 'cos',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'cos' },
  tan:         { name: 'tan',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'tan' },
  sinh:        { name: 'sinh',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'sinh' },
  cosh:        { name: 'cosh',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'cosh' },
  tanh:        { name: 'tanh',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'tanh' },
  asin:        { name: 'asin',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'asin' },
  acos:        { name: 'acos',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'acos' },
  atan:        { name: 'atan',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'atan' },
  atan2:       { name: 'atan2',       profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2] },
  exp:         { name: 'exp',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'exp' },
  exp2:        { name: 'exp2',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'exp2' },
  log:         { name: 'log',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'log' },
  log2:        { name: 'log2',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'log2' },
  inverseSqrt: { name: 'inverseSqrt', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1] },
  fract:       { name: 'fract',       profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1], lowerName: 'fract' },
  degrees:     { name: 'degrees',     profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1], lowerName: 'degrees' },
  radians:     { name: 'radians',     profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1], lowerName: 'radians' },
  trunc:       { name: 'trunc',       profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1], lowerName: 'trunc' },
  step:        { name: 'step',        profile: 'core', portability: 'exact',        takesClosure: false, arity: [2, 2], lowerName: 'step' },
  mix:         { name: 'mix',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3], lowerName: 'mix' },
  smoothstep:  { name: 'smoothstep',  profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3], lowerName: 'smoothstep' },
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
