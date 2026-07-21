// The machine-readable catalog of metael's builtins. Each entry declares the builtin's capability PROFILE
// and, for numeric builtins, its cross-target PORTABILITY class. This is the single source of truth for
// "which names are intrinsics" (a user `function` of the same name shadows one) and the input a static
// profile classifier + any future codegen consumes. Pure data — no evaluator import. The BuiltinSpec/
// BuiltinProfile/Portability TYPES live in @metael/lang (the registry seam); this module owns the DATA.
import type { BuiltinSpec, BuiltinProfile, Portability } from '@metael/lang';

export type { BuiltinSpec, BuiltinProfile, Portability };

/** Every intrinsic metael knows about, keyed by name. The numeric entries are dispatched by the numeric
 *  builtin module (MATH_BUILTINS); the collection/string/structural/random/datetime entries are dispatched
 *  by the standard-library module; `range` alone is the language kernel's own intrinsic (a bounded-loop
 *  primitive the compute-lowering gate + interpreter oracle depend on, so it cannot move to the standard
 *  library). The profile/portability metadata classifies cross-target reproducibility for a classifier or a
 *  codegen consumer. */
export const BUILTINS: Readonly<Record<string, BuiltinSpec>> = Object.freeze({
  // --- seeded / collection / string (rand dispatched by the standard library; range by the kernel) ---
  rand:        { name: 'rand',        profile: 'core', portability: 'cpu-only',     takesClosure: false, arity: [0, 0] },
  range:       { name: 'range',       profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },
  map:         { name: 'map',         profile: 'host', portability: 'cpu-only',     takesClosure: true,  arity: [2, 2] },
  filter:      { name: 'filter',      profile: 'host', portability: 'cpu-only',     takesClosure: true,  arity: [2, 2] },
  reduce:      { name: 'reduce',      profile: 'host', portability: 'cpu-only',     takesClosure: true,  arity: [3, 3] },
  keys:        { name: 'keys',        profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },
  values:      { name: 'values',      profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },
  entries:     { name: 'entries',     profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },
  object:      { name: 'object',      profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },
  has:         { name: 'has',         profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [2, 2] },

  // --- datetime (read the host's injected clock capability; time is a replayable input) ---
  now:         { name: 'now',         profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [0, 0] },
  monotonic:   { name: 'monotonic',   profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [0, 0] },

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
  format:      { name: 'format',      profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [2, 2] },
  codePointAt: { name: 'codePointAt', profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [2, 2] },

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
  // Componentwise (Hadamard) matrix product. GLSL has matrixCompMult; WGSL has no equivalent. No lowerName
  // (no hand-emit yet); classified core/gpu-tolerant for a future codegen consumer.
  matrixCompMult: { name: 'matrixCompMult', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2] },

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

  // --- affine transform composition + camera projections (return matrices / a decomposed object) ---
  // These build/consume matrices; no shader lowerName yet (may not be gate-lowerable). decompose returns a
  // heap object ({t,r,s}) so it is host/cpu-only; the rest return a mat and stay core/gpu-tolerant.
  transformation: { name: 'transformation', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3] },
  decompose:      { name: 'decompose',      profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1] },
  perspective:    { name: 'perspective',    profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [4, 4] },
  ortho:          { name: 'ortho',          profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [6, 6] },
  lookAt:         { name: 'lookAt',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3] },

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
  // Floored modulo — sign follows the DIVISOR (unlike the `%` operator, which is a truncated remainder that
  // takes the sign of the dividend). No lowerName: it needs a bespoke emitter case per target — GLSL `mod`
  // is ALREADY floored (native), but WGSL `%` is truncated, so WGSL emits the floored form `x - y*floor(x/y)`.
  mod:         { name: 'mod',         profile: 'core', portability: 'exact',        takesClosure: false, arity: [2, 2] },

  // --- transcendentals (native-lowerable to a shader builtin) ---
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
  // Inverse hyperbolics — a native shader builtin of the SAME name on both targets (WGSL asinh/acosh/atanh;
  // GLSL ES 3.00 asinh/acosh/atanh), so the emitter's generic `lowerName` path handles them with no bespoke
  // case. Out-of-domain (acosh x<1, atanh |x|>=1) returns NaN on both the interpreter and the native shaders,
  // so they carry NO fail-loud domain guard (unlike asin/acos/log) — that raw-NaN behavior is what keeps the
  // GPU output identical to the interpreter oracle (a CPU-only guard would break that cross-target parity).
  asinh:       { name: 'asinh',       profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'asinh' },
  acosh:       { name: 'acosh',       profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'acosh' },
  atanh:       { name: 'atanh',       profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'atanh' },
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

  // --- integer bit operations (32-bit unsigned) ---
  // BOTH are shader-lowered (the compute-lowering gate ACCEPTS a scalar bit input). The argument is coerced to a
  // 32-bit unsigned integer by TRUNCATION toward zero, matching the interpreter's ToUint32 / `x >>> 0` (a
  // fractional input like 3.9 counts bits of 3, not 4): WGSL has NATIVE countOneBits/reverseBits (on u32) reached
  // via `bitcast<u32>(i32(x))`, and GLSL ES 3.00 — which lacks the ES-3.10 bitCount/bitfieldReverse — hand-rolls
  // both via a `uint(int(x))`/bitwise-op prelude helper (a SWAR popcount, a 32-bit reverse). For the INTEGER inputs
  // these ops are meant for, the shader result matches the interpreter oracle exactly: the f32 output store is
  // lossless (a popcount is 0..32, always f32-exact; reversal preserves the ≤24-bit significant SPAN of an
  // f32-exact integer). `gpu-tolerant` (not `exact`) documents the u32-reinterpret-of-f32 boundary AND that the
  // input is coerced by truncation like `>>>0`, not any value divergence for the intended integer inputs.
  countOneBits: { name: 'countOneBits', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1] },
  reverseBits:  { name: 'reverseBits',  profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1] },
});

/** True iff `name` is a metael intrinsic in this catalog. */
export function isBuiltin(name: string): boolean {
  return Object.prototype.hasOwnProperty.call(BUILTINS, name);
}
