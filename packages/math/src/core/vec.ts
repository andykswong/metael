// packages/math/src/core/vec.ts
// Plain flat-store vector math. Componentwise / standard geometric ops over `number[]` (f64 precision).
//
// Out-param convention: `out = op(...args, out?)`. When `out` is omitted a fresh `number[]` is returned;
// when supplied it is written into and returned. Aliasing an input as the output is always legal —
// purely componentwise ops (add/sub/mul/div/scale) touch only index `i` to write `out[i]`, so an aliased
// output can never corrupt a still-pending read; ops that need a whole-vector scalar (normalize/reflect/
// refract/faceforward) compute that scalar before any write; and `cross` stages its three components in
// locals before writing, so an aliased `out` stays safe.
//
// Precision-agnostic by design: no f32 rounding here (that is the caller/binding's concern). Zero-length
// and shape-mismatch validation is a language-surface concern; these assume well-formed, equal-length
// inputs. length-0 normalize divides by 0 → NaN/±Inf components, matching shader `normalize`.

/** Componentwise sum a[i] + b[i]. */
export function add(a: readonly number[], b: readonly number[], out?: number[]): number[] {
  const o = out ?? new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) o[i] = (a[i] as number) + (b[i] as number);
  return o;
}

/** Componentwise difference a[i] - b[i]. */
export function sub(a: readonly number[], b: readonly number[], out?: number[]): number[] {
  const o = out ?? new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) o[i] = (a[i] as number) - (b[i] as number);
  return o;
}

/** Componentwise (Hadamard) product a[i] * b[i]. */
export function mul(a: readonly number[], b: readonly number[], out?: number[]): number[] {
  const o = out ?? new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) o[i] = (a[i] as number) * (b[i] as number);
  return o;
}

/** Componentwise quotient a[i] / b[i]. */
export function div(a: readonly number[], b: readonly number[], out?: number[]): number[] {
  const o = out ?? new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) o[i] = (a[i] as number) / (b[i] as number);
  return o;
}

/** Scale every component by a scalar: a[i] * s. */
export function scale(a: readonly number[], s: number, out?: number[]): number[] {
  const o = out ?? new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) o[i] = (a[i] as number) * s;
  return o;
}

/** Dot product Σ a[i]·b[i]. */
export function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] as number) * (b[i] as number);
  return s;
}

/** Euclidean length √(Σ a[i]²). */
export function length(a: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] as number) * (a[i] as number);
  return Math.sqrt(s);
}

/** Euclidean distance between a and b: √(Σ (a[i]−b[i])²). */
export function distance(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = (a[i] as number) - (b[i] as number); s += d * d; }
  return Math.sqrt(s);
}

/** Unit vector a / length(a). length 0 → NaN/±Inf components (no guard, matching native normalize). */
export function normalize(a: readonly number[], out?: number[]): number[] {
  const len = length(a);                 // whole-vector scalar computed before any write → aliasing-safe
  const o = out ?? new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) o[i] = (a[i] as number) / len;
  return o;
}

/** Cross product of two 3-vectors: [a1·b2−a2·b1, a2·b0−a0·b2, a0·b1−a1·b0]. */
export function cross(a: readonly number[], b: readonly number[], out?: number[]): number[] {
  const a0 = a[0] as number, a1 = a[1] as number, a2 = a[2] as number;
  const b0 = b[0] as number, b1 = b[1] as number, b2 = b[2] as number;
  // Stage all three components before writing so an aliased `out` (=== a or b) can't corrupt a read.
  const x = a1 * b2 - a2 * b1;
  const y = a2 * b0 - a0 * b2;
  const z = a0 * b1 - a1 * b0;
  const o = out ?? new Array<number>(3);
  o[0] = x; o[1] = y; o[2] = z;
  return o;
}

/** Reflect incident I about normal N (assumed normalized): I − 2·dot(I,N)·N. */
export function reflect(I: readonly number[], N: readonly number[], out?: number[]): number[] {
  const d = dot(I, N);                    // scalar first → aliasing-safe
  const o = out ?? new Array<number>(I.length);
  for (let i = 0; i < I.length; i++) o[i] = (I[i] as number) - 2 * d * (N[i] as number);
  return o;
}

/**
 * Refract incident I through a surface with normal N and ratio of indices `eta`.
 * k = 1 − eta²·(1 − dot(I,N)²); k < 0 → total internal reflection → all-zeros; else
 * eta·I − (eta·dot(I,N) + √k)·N.
 */
export function refract(I: readonly number[], N: readonly number[], eta: number, out?: number[]): number[] {
  const d = dot(I, N);
  const k = 1 - eta * eta * (1 - d * d);
  const o = out ?? new Array<number>(I.length);
  if (k < 0) { for (let i = 0; i < I.length; i++) o[i] = 0; return o; }
  const sq = Math.sqrt(k);
  for (let i = 0; i < I.length; i++) o[i] = eta * (I[i] as number) - (eta * d + sq) * (N[i] as number);
  return o;
}

/** Orient N to face away from I relative to reference Nref: dot(Nref,I) < 0 → N, else −N. */
export function faceforward(N: readonly number[], I: readonly number[], Nref: readonly number[], out?: number[]): number[] {
  const d = dot(Nref, I);
  const o = out ?? new Array<number>(N.length);
  for (let i = 0; i < N.length; i++) o[i] = d < 0 ? (N[i] as number) : -(N[i] as number);
  return o;
}
