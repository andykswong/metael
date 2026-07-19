// The custom-value-type dispatch protocol. A value may carry a non-forgeable, Symbol-keyed type
// DESCRIPTOR defining how the interpreter treats its operators, accessors, assignment, iteration,
// truthiness, display, and lowering. The grammar/AST are unchanged; only the evaluator's value model
// gains this seam. A mutable value (a typed array) additionally carries an opaque GENERATION handle
// (a reactive change-signal owned by the ReactiveHost) and a per-value FROZEN box (const-immutability).
import type { BinOp } from './ast.ts';
import type { GenerationRef } from './ports.ts';

/** Returned by an operator handler that does not define an op/operand-combination. The INTERPRETER
 *  (not the descriptor) then applies the uniform fallback: == / != → reference identity; an
 *  arithmetic/relational op → ML-LANG-OP-UNSUPPORTED. (A throw or `null` would be ambiguous — null is
 *  a legal metael value — so a dedicated sentinel is required.) */
export const NOT_HANDLED: unique symbol = Symbol('ml.notHandled');
export type NotHandled = typeof NOT_HANDLED;

export type LowerElement = 'f32' | 'f64' | 'i32' | 'u32';
export type LowerShape = 'scalar' | 'vecN' | 'matMxN';
export type LowerAccess = 'value' | 'linear-buffer';

/** A neutral, target-agnostic operator lowering — NOT shader text. A compile consumer renders it to
 *  WGSL/GLSL/CPU. Keyed on the language's BinOp (plus 'neg') via Lowering.ops. */
export type LowerOp =
  | { readonly kind: 'componentwise'; readonly op: 'add' | 'sub' | 'mul' | 'div' }
  | { readonly kind: 'unary'; readonly op: 'neg' }
  | { readonly kind: 'scale' }
  | { readonly kind: 'dot' }
  | { readonly kind: 'matmul' }
  | { readonly kind: 'builtin-call'; readonly name: string };

/** A domain-agnostic description of how a type is STORED/ACCESSED and how its operators/members map to
 *  target ops. Complete: a compile consumer never hardcodes a type — a new numeric type is a library
 *  (a descriptor with `ops` + `members`) needing zero emitter edits. */
export interface Lowering {
  readonly element: LowerElement;
  readonly shape: LowerShape;
  readonly n?: number;        // scalar-buffer element count (unchanged use)
  readonly rows?: number;     // vec/mat rows (a vec is rows×1)
  readonly cols?: number;     // vec/mat cols (a vec is 1)
  readonly gpuStorable: boolean;
  readonly access: LowerAccess;
  readonly ops?: Readonly<Partial<Record<string, LowerOp>>>;
  readonly members?: { readonly kind: 'swizzle' | 'component'; readonly of: string };
}

/** How the interpreter treats a value of a custom type. Handlers are built-in host code (typed arrays,
 *  vec/mat) trusted like a resolveCall head. A mutating handler is gated by `frozen?` BEFORE it runs
 *  (the interpreter checks and emits ML-LANG-IMMUTABLE for a const value). An operator handler returns
 *  NOT_HANDLED for an op/combination it does not define. */
export interface TypeDescriptor {
  readonly name: string;
  readonly frozen?: (v: unknown) => boolean;
  binary?(op: BinOp, left: unknown, right: unknown): unknown;
  equals?(a: unknown, b: unknown): boolean;
  neg?(v: unknown): unknown;
  getMember?(v: unknown, prop: string): unknown;
  getIndex?(v: unknown, key: number | string): unknown;
  setMember?(v: unknown, prop: string, val: unknown): void;
  setIndex?(v: unknown, key: number | string, val: unknown): void;
  iterate?(v: unknown): Iterable<unknown>;
  /** Zero-copy access to a linear-buffer value's backing store (only present on `access: 'linear-buffer'`
   *  descriptors). Returns the raw TypedArray + its element kind so a consumer (a compute backend) can read
   *  it WITHOUT the O(n) `iterate` → number[] copy. The returned data is the live store — treat it read-only. */
  bufferView?(v: unknown): { readonly data: ArrayLike<number>; readonly element: LowerElement };
  truthy?(v: unknown): boolean;
  display?(v: unknown): string;
  readonly lower?: Lowering;
}

// Non-enumerable, non-forgeable tags (the language surface cannot read or set a Symbol-keyed field).
// A frozen box is a mutable holder so a const declaration can freeze a value in place after construction.
const DESCRIPTOR: unique symbol = Symbol('ml.descriptor');
const GENERATION: unique symbol = Symbol('ml.generation');
const FROZEN: unique symbol = Symbol('ml.frozen');

interface FrozenBox { frozen: boolean }

/** Attach a descriptor to a value. A mutable descriptor (has setIndex/setMember) also gets a frozen box
 *  (starts writable) and, if provided, a generation handle for reactive in-place mutation. */
export function tagCustom<T extends object>(v: T, descriptor: TypeDescriptor, gen?: GenerationRef): T {
  Object.defineProperty(v, DESCRIPTOR, { value: descriptor, enumerable: false, configurable: false, writable: false });
  const mutable = descriptor.setIndex !== undefined || descriptor.setMember !== undefined;
  if (mutable) {
    Object.defineProperty(v, FROZEN, { value: { frozen: false } satisfies FrozenBox, enumerable: false, configurable: false, writable: false });
    if (gen !== undefined) Object.defineProperty(v, GENERATION, { value: gen, enumerable: false, configurable: false, writable: false });
  }
  return v;
}

export function descriptorOf(v: unknown): TypeDescriptor | undefined {
  return (typeof v === 'object' && v !== null) ? (v as { [DESCRIPTOR]?: TypeDescriptor })[DESCRIPTOR] : undefined;
}
export function isCustomType(v: unknown): boolean {
  return descriptorOf(v) !== undefined;
}
/** The value's opaque generation handle (a per-buffer reactive change-signal), or undefined for an
 *  immutable/non-buffer custom value. A consumer reads it via host.readGeneration(generationOf(v)). */
export function generationOf(v: unknown): GenerationRef | undefined {
  return (typeof v === 'object' && v !== null) ? (v as { [GENERATION]?: GenerationRef })[GENERATION] : undefined;
}
/** True iff the value is a tagged linear buffer (a typed array). Narrow uses only — classification for
 *  lowering goes through descriptorOf(v).lower.access, not this. */
export function isTypedArray(v: unknown): boolean {
  const d = descriptorOf(v);
  return d !== undefined && d.lower?.access === 'linear-buffer';
}
/** Mark a mutable custom value immutable (a const declaration / a deep-freeze of a container). A no-op
 *  for an immutable type (no frozen box). */
export function markFrozen(v: unknown): void {
  const box = (typeof v === 'object' && v !== null) ? (v as { [FROZEN]?: FrozenBox })[FROZEN] : undefined;
  if (box) box.frozen = true;
}
export function isFrozenCustom(v: unknown): boolean {
  const box = (typeof v === 'object' && v !== null) ? (v as { [FROZEN]?: FrozenBox })[FROZEN] : undefined;
  return box ? box.frozen : false;
}

// ─────────────────────────────────────────── typed arrays (first custom-type consumer) ───────────────────────────────────────────

/** Per-element coercion + WGSL storage element for each typed-array kind. i32/u32 truncate-then-wrap
 *  mod 2^32; f32 rounds via Math.fround; f64 is exact. NaN/±Inf/-0 follow the underlying TypedArray. */
interface BufferKind { readonly ctor: { new (n: number): { [i: number]: number; length: number } }; readonly element: LowerElement; readonly gpuStorable: boolean; coerce(x: number): number }
const F32: BufferKind = { ctor: Float32Array, element: 'f32', gpuStorable: true, coerce: (x) => Math.fround(x) };
const F64: BufferKind = { ctor: Float64Array, element: 'f64', gpuStorable: false, coerce: (x) => x };
const I32: BufferKind = { ctor: Int32Array,   element: 'i32', gpuStorable: true, coerce: (x) => x | 0 };
const U32: BufferKind = { ctor: Uint32Array,  element: 'u32', gpuStorable: true, coerce: (x) => x >>> 0 };
export const BUFFER_KINDS: Readonly<Record<'f32' | 'f64' | 'i32' | 'u32', BufferKind>> = { f32: F32, f64: F64, i32: I32, u32: U32 };

/** A marker error a descriptor handler throws to surface a value error (bounds/coercion) — the
 *  interpreter's read/write hooks catch it and map it to a diagnostic (descriptors are stateless/shared
 *  and can't reach the Runner). */
export class BufferError extends Error { constructor(readonly code: string, readonly detail: string) { super(code); } }

/** The shared typed-array descriptor for a given kind. One descriptor object per kind (stateless). The
 *  backing store is a real JS TypedArray held on a field the language can't read (Symbol-keyed). */
const STORE: unique symbol = Symbol('ml.buffer.store');
function makeTypedArrayDescriptor(kind: BufferKind): TypeDescriptor {
  const lower: Lowering = { element: kind.element, shape: 'scalar', gpuStorable: kind.gpuStorable, access: 'linear-buffer' };
  const storeOf = (v: unknown): { [i: number]: number; length: number } => (v as { [STORE]: { [i: number]: number; length: number } })[STORE];
  return {
    name: `${kind.element}buffer`,
    frozen: isFrozenCustom,
    getIndex: (v, key) => {
      if (typeof key !== 'number') return NOT_HANDLED;
      const store = storeOf(v);
      if (!Number.isInteger(key) || key < 0 || key >= store.length) throw new BufferError('ML-LANG-INDEX-RANGE', `index ${String(key)} is out of range (length ${store.length})`);
      return store[key];
    },
    getMember: (v, prop) => (prop === 'length' ? storeOf(v).length : NOT_HANDLED),
    setIndex: (v, key, val) => {
      const store = storeOf(v);
      if (typeof key !== 'number' || !Number.isInteger(key) || key < 0 || key >= store.length) throw new BufferError('ML-LANG-INDEX-RANGE', `index ${String(key)} is out of range (length ${store.length})`);
      if (typeof val !== 'number') throw new BufferError('ML-LANG-BUILTIN-ARG', 'a typed-array element must be a number');
      store[key] = kind.coerce(val);
    },
    iterate: (v) => { const store = storeOf(v); const out: number[] = []; for (let i = 0; i < store.length; i++) out.push(store[i] as number); return out; },
    bufferView: (v) => ({ data: storeOf(v), element: kind.element }),
    display: (v) => { const store = storeOf(v); const head: string[] = []; for (let i = 0; i < Math.min(store.length, 8); i++) head.push(String(store[i])); const abbreviated = store.length > 8; return `${kind.element}[${head.join(', ')}${abbreviated ? `, … (len ${store.length})` : ''}]`; },
    lower,
  };
}
/** Allocate + tag a typed-array custom value. The backing TypedArray is Symbol-hidden; `gen` is the
 *  reactive change-signal for in-place mutation. */
export function makeTypedArray(kind: keyof typeof BUFFER_KINDS, store: { [i: number]: number; length: number }, gen: GenerationRef): object {
  const box = {};
  Object.defineProperty(box, STORE, { value: store, enumerable: false, configurable: false, writable: false });
  return tagCustom(box, TYPED_ARRAY_DESCRIPTORS[kind], gen);
}
export const TYPED_ARRAY_DESCRIPTORS: Readonly<Record<'f32' | 'f64' | 'i32' | 'u32', TypeDescriptor>> = {
  f32: makeTypedArrayDescriptor(F32), f64: makeTypedArrayDescriptor(F64), i32: makeTypedArrayDescriptor(I32), u32: makeTypedArrayDescriptor(U32),
};

// ─────────────────────────────────────────── vec/mat (second custom-type consumer — immutable value math) ───────────────────────────────────────────

const VEC_STORE: unique symbol = Symbol('ml.vec.store');
const SWIZZLE = 'xyzw';
interface VecStore { c: ArrayLike<number>; rows: number; cols: number }
const storeOfVec = (v: unknown): VecStore => (v as { [VEC_STORE]: VecStore })[VEC_STORE];
/** A vec is a single-column matrix. */
const isVec = (s: VecStore): boolean => s.cols === 1;
/** Test-only reader for the (Symbol-hidden) store — no language-surface access. */
export function vecStoreOf(v: unknown): VecStore { return storeOfVec(v); }

function vecLower(rows: number, cols: number, element: LowerElement = 'f32'): Lowering {
  const mat = cols > 1;
  return {
    element, shape: mat ? 'matMxN' : 'vecN', rows, cols, gpuStorable: true, access: 'value',
    ops: mat
      ? { '*': { kind: 'matmul' } }
      : {
          '+': { kind: 'componentwise', op: 'add' }, '-': { kind: 'componentwise', op: 'sub' },
          '/': { kind: 'componentwise', op: 'div' }, '*': { kind: 'componentwise', op: 'mul' },
          'neg': { kind: 'unary', op: 'neg' },   // negation, NOT subtraction — a compile consumer emits `-v`
        },
    members: mat ? undefined : { kind: 'swizzle', of: SWIZZLE },
  };
}

const vecDescriptors = new Map<string, TypeDescriptor>();
function vecDescriptor(rows: number, cols: number): TypeDescriptor {
  const key = cols > 1 ? `mat${cols}x${rows}` : `vec${rows}`;
  const existing = vecDescriptors.get(key);
  if (existing) return existing;
  const componentwise = (op: 'add' | 'sub' | 'mul' | 'div', a: ArrayLike<number>, b: ArrayLike<number>): number[] => {
    const out: number[] = [];
    for (let i = 0; i < a.length; i++) { const x = a[i] as number; const y = b[i] as number; out.push(op === 'add' ? x + y : op === 'sub' ? x - y : op === 'mul' ? x * y : x / y); }
    return out;
  };
  const desc: TypeDescriptor = {
    name: key,
    binary: (o, l, r) => {
      const ls = descriptorOf(l) ? storeOfVec(l) : null;
      const rs = descriptorOf(r) ? storeOfVec(r) : null;
      const lMat = ls && !isVec(ls); const rMat = rs && !isVec(rs);
      // vec ∘ scalar / scalar ∘ vec — scale
      if (ls && isVec(ls) && typeof r === 'number' && (o === '*' || o === '/')) return makeVec(Array.from(ls.c, (x) => o === '*' ? x * r : x / r));
      if (rs && isVec(rs) && typeof l === 'number' && o === '*') return makeVec(Array.from(rs.c, (x) => x * l));
      // mat ∘ scalar / scalar ∘ mat — scale (componentwise, same shape)
      if (lMat && typeof r === 'number' && (o === '*' || o === '/')) return makeMat(Array.from(ls!.c, (x) => o === '*' ? x * r : x / r), ls!.rows, ls!.cols);
      if (rMat && typeof l === 'number' && o === '*') return makeMat(Array.from(rs!.c, (x) => x * l), rs!.rows, rs!.cols);
      // matmul: mat * (mat|vec) — a vec is the cols===1 case
      if (lMat && rs && o === '*') { const p = matmul(ls!, rs); return p ? (p.cols === 1 ? makeVec(p.c) : makeMat(p.c, p.rows, p.cols)) : NOT_HANDLED; }
      // vec componentwise + - * / (equal length)
      if (ls && rs && isVec(ls) && isVec(rs) && ls.rows === rs.rows && (o === '+' || o === '-' || o === '*' || o === '/'))
        return makeVec(componentwise(o === '+' ? 'add' : o === '-' ? 'sub' : o === '*' ? 'mul' : 'div', Array.from(ls.c), Array.from(rs.c)));
      return NOT_HANDLED;
    },
    equals: (l, r) => {
      const ls = descriptorOf(l) ? storeOfVec(l) : null; const rs = descriptorOf(r) ? storeOfVec(r) : null;
      if (!ls || !rs || ls.rows !== rs.rows || ls.cols !== rs.cols) return false;
      return Array.from(ls.c).every((x, i) => x === rs.c[i]);
    },
    neg: (v) => { const s = storeOfVec(v); const c = Array.from(s.c, (x) => -x); return isVec(s) ? makeVec(c) : makeMat(c, s.rows, s.cols); },
    getMember: (v, prop) => {
      const s = storeOfVec(v);
      if (!isVec(s)) return NOT_HANDLED;   // matrices have no swizzle
      if (prop.length === 1) { const i = SWIZZLE.indexOf(prop); return (i >= 0 && i < s.rows) ? s.c[i] : NOT_HANDLED; }
      const idxs = [...prop].map((ch) => SWIZZLE.indexOf(ch));
      if (idxs.some((i) => i < 0 || i >= s.rows) || idxs.length < 2 || idxs.length > 4) return NOT_HANDLED;
      return makeVec(idxs.map((i) => s.c[i] as number));
    },
    display: (v) => { const s = storeOfVec(v); return `${isVec(s) ? `vec${s.rows}` : `mat${s.cols}x${s.rows}`}(${Array.from(s.c).join(', ')})`; },
    lower: vecLower(rows, cols),
  };
  vecDescriptors.set(key, desc);
  return desc;
}
/** Column-major matrix product A(R×K) · B(K×C) → R×C. A vec is a K×1 / R×1 column. Null on inner-dim
 *  mismatch (A.cols !== B.rows). Column-major flat index (r,c) → c*rows + r. */
function matmul(a: VecStore, b: VecStore): { c: number[]; rows: number; cols: number } | null {
  if (a.cols !== b.rows) return null;
  const R = a.rows, K = a.cols, C = b.cols;
  const out = new Array<number>(R * C).fill(0);
  for (let c = 0; c < C; c++) for (let r = 0; r < R; r++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += (a.c[k * R + r] as number) * (b.c[c * K + k] as number);
    out[c * R + r] = s;
  }
  return { c: out, rows: R, cols: C };
}
export function makeVec(components: number[]): object {
  const box = {};
  Object.defineProperty(box, VEC_STORE, { value: { c: components.map((x) => Math.fround(x)), rows: components.length, cols: 1 } satisfies VecStore, enumerable: false, configurable: false, writable: false });
  return tagCustom(box, vecDescriptor(components.length, 1));
}
export function makeMat(components: number[], rows: number, cols: number): object {
  const box = {};
  Object.defineProperty(box, VEC_STORE, { value: { c: components.map((x) => Math.fround(x)), rows, cols } satisfies VecStore, enumerable: false, configurable: false, writable: false });
  return tagCustom(box, vecDescriptor(rows, cols));
}
export function identityMat(n: number): number[] { const out = new Array<number>(n * n).fill(0); for (let i = 0; i < n; i++) out[i * n + i] = 1; return out; }
