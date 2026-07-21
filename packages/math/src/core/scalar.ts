// packages/math/src/core/scalar.ts
// Plain scalar math. Pure number→number; domain guards are the LANGUAGE BINDING's concern (this returns
// the raw IEEE result, NaN for out-of-domain). round is half-to-even for cross-target bit-identity.
/** The numeric backing store a component-wise operation writes into: an `f32`/`f64` typed array or a
 *  plain `number[]`. The store's kind selects the per-element coercion (see {@link coerceFor}). */
export type Store = Float32Array | Float64Array | number[];
/** Per-element coercion for a store: f32 storage rounds via Math.fround; f64/number[] are exact. */
export function coerceFor(store: Store): (x: number) => number {
  return store instanceof Float32Array ? (x) => Math.fround(x) : (x) => x;
}
/** Smaller of two numbers (IEEE `Math.min`; propagates NaN). */
export const min = (a: number, b: number) => Math.min(a, b);
/** Larger of two numbers (IEEE `Math.max`; propagates NaN). */
export const max = (a: number, b: number) => Math.max(a, b);
/** Absolute value (`Math.abs`). */
export const abs = Math.abs;
/** Sign of `x`: -1, -0, 0, +1, or NaN (`Math.sign`). */
export const sign = Math.sign;
/** Largest integer ≤ `x` (`Math.floor`). */
export const floor = Math.floor;
/** Smallest integer ≥ `x` (`Math.ceil`). */
export const ceil = Math.ceil;
/** Round to the nearest integer, ties to even (banker's rounding) — chosen over `Math.round`'s
 *  round-half-up so results are bit-identical across compile targets. */
export function roundHalfEven(x: number): number { const r = Math.round(x); return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r; }
/** Round to the nearest integer, ties to even — the exported name for {@link roundHalfEven}. */
export const round = roundHalfEven;
/** Constrain `x` to the closed interval `[lo, hi]` (assumes `lo <= hi`). */
export const clamp = (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi);
/** `x` raised to the power `y` (`Math.pow`). */
export const pow = Math.pow;
/** Sine of `x`; `x` is in radians (`Math.sin`). */
export const sin = Math.sin;
/** Cosine of `x`; `x` is in radians (`Math.cos`). */
export const cos = Math.cos;
/** Tangent of `x`; `x` is in radians (`Math.tan`). */
export const tan = Math.tan;
/** Hyperbolic sine of `x` (`Math.sinh`). */
export const sinh = Math.sinh;
/** Hyperbolic cosine of `x` (`Math.cosh`). */
export const cosh = Math.cosh;
/** Hyperbolic tangent of `x` (`Math.tanh`). */
export const tanh = Math.tanh;
/** Arctangent of `x`, in radians, in `[-π/2, π/2]` (`Math.atan`). */
export const atan = Math.atan;
/** Angle in radians, in `(-π, π]`, of the point `(x, y)` from the positive x-axis (`Math.atan2`). */
export const atan2 = Math.atan2;
/** Inverse hyperbolic sine of `x` (`Math.asinh`). */
export const asinh = Math.asinh;
/** Inverse hyperbolic cosine of `x` (`Math.acosh`; NaN for `x < 1`). */
export const acosh = Math.acosh;
/** Inverse hyperbolic tangent of `x` (`Math.atanh`; NaN for `|x| > 1`). */
export const atanh = Math.atanh;
/** `e` raised to the power `x` (`Math.exp`). */
export const exp = Math.exp;
/** 2 raised to the power `x` (`2 ** x`). */
export const exp2 = (x: number) => 2 ** x;
/** Fractional part of `x`: `x - floor(x)`, always in `[0, 1)` (so `fract(-0.25) === 0.75`). */
export const fract = (x: number) => x - Math.floor(x);
/** Convert radians to degrees. */
export const degrees = (r: number) => (r * 180) / Math.PI;
/** Convert degrees to radians. */
export const radians = (d: number) => (d * Math.PI) / 180;
/** Integer part of `x` toward zero, discarding the fraction (`Math.trunc`). */
export const trunc = Math.trunc;
/** GLSL-style step: `0` when `x < edge`, else `1`. */
export const step = (edge: number, x: number) => (x < edge ? 0 : 1);
/** Linear interpolation: `a + (b - a) * t`. `t` is unclamped (extrapolates outside `[0, 1]`). */
export const mix = (a: number, b: number, t: number) => a + (b - a) * t;
// Domain-guarded funcs return NaN out-of-domain (not ±Inf) so component-wise vector results stay
// bit-identical. Scalar-position fail-loud diagnostics are a language-surface concern layered on top;
// the core here returns the raw NaN result.
/** Non-negative square root of `x`; NaN for `x < 0` (rather than throwing) so component-wise vector
 *  results stay bit-identical. */
export const sqrt = (x: number) => (x < 0 ? NaN : Math.sqrt(x));
/** Arcsine of `x`, in radians; NaN outside the domain `[-1, 1]`. */
export const asin = (x: number) => (x < -1 || x > 1 ? NaN : Math.asin(x));
/** Arccosine of `x`, in radians; NaN outside the domain `[-1, 1]`. */
export const acos = (x: number) => (x < -1 || x > 1 ? NaN : Math.acos(x));
/** Natural logarithm of `x`; NaN for `x <= 0`. */
export const log = (x: number) => (x <= 0 ? NaN : Math.log(x));
/** Base-2 logarithm of `x`; NaN for `x <= 0`. */
export const log2 = (x: number) => (x <= 0 ? NaN : Math.log2(x));
/** Reciprocal square root `1 / sqrt(x)`; NaN for `x <= 0`. */
export const inverseSqrt = (x: number) => (x <= 0 ? NaN : 1 / Math.sqrt(x));
/** GLSL-style smoothstep: 0 below `e0`, 1 above `e1`, with a smooth Hermite ramp between. The
 *  degenerate case `e0 === e1` returns `0` (not NaN). */
export function smoothstep(e0: number, e1: number, x: number): number { const t = e1 === e0 ? 0 : Math.min(Math.max((x - e0) / (e1 - e0), 0), 1); return t * t * (3 - 2 * t); }
/** Floored modulo: x - y*floor(x/y). Sign follows the DIVISOR (GLSL/WGSL `mod`), unlike JS `%`. */
export const mod = (x: number, y: number) => x - y * Math.floor(x / y);
