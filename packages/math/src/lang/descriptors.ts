// The boxed vec/mat values for the language surface — immutable value math over the custom-value
// descriptor protocol. A vec is a single-column matrix; a mat carries rows×cols. The descriptor machinery
// (tagCustom/descriptorOf/NOT_HANDLED/BufferError + the Lowering/TypeDescriptor types) lives in
// @metael/lang; the plain flat-store arithmetic lives in @metael/math core. This module wraps core:
// it boxes results, applies per-store precision (f32 rounds via Math.fround; f64 is exact), and threads
// the element kind through EVERY operator so a chain of f64 operands never silently downcasts to f32.
import type { TypeDescriptor, Lowering, LowerElement } from '@metael/lang';
import { tagCustom, descriptorOf, NOT_HANDLED, BufferError } from '@metael/lang';
import { matmul as coreMatmul, matColumn as coreMatColumn, add as coreAdd, sub as coreSub, mul as coreMul, div as coreDiv } from '@metael/math';

const VEC_STORE: unique symbol = Symbol('ml.vec.store');
const SWIZZLE = 'xyzw';

/** The Symbol-hidden backing store: the flat column-major components + the shape + the element precision. */
interface VecStore { c: ArrayLike<number>; rows: number; cols: number; element: LowerElement }
const storeOfVec = (v: unknown): VecStore => (v as { [VEC_STORE]: VecStore })[VEC_STORE];
/** A vec is a single-column matrix. */
const isVec = (s: VecStore): boolean => s.cols === 1;
/** Test-only reader for the (Symbol-hidden) store — no language-surface access. */
export function vecStoreOf(v: unknown): VecStore { return storeOfVec(v); }

/** The result precision of an operation: f64 if EITHER store is f64, else the shared element. */
function resultElement(a: VecStore, b?: VecStore | null): LowerElement {
  if (a.element === 'f64') return 'f64';
  if (b && b.element === 'f64') return 'f64';
  return a.element;
}

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
function vecDescriptor(rows: number, cols: number, element: LowerElement = 'f32'): TypeDescriptor {
  const shapeKey = cols > 1 ? `mat${cols}x${rows}` : `vec${rows}`;
  const key = `${shapeKey}:${element}`;   // an f32 and an f64 descriptor of the same shape must NOT collide
  const existing = vecDescriptors.get(key);
  if (existing) return existing;
  // The core add/sub/mul/div are purely componentwise and NEVER mutate their inputs (they write a fresh `out`),
  // so pass the stores straight through — no defensive `Array.from` copy. The call site likewise passes
  // `ls.c`/`rs.c` (already `number[]`) directly. coreAdd/etc. accept `readonly number[]`.
  const componentwise = (op: 'add' | 'sub' | 'mul' | 'div', a: readonly number[], b: readonly number[]): number[] =>
    op === 'add' ? coreAdd(a, b) : op === 'sub' ? coreSub(a, b) : op === 'mul' ? coreMul(a, b) : coreDiv(a, b);
  const mat = cols > 1;
  const desc: TypeDescriptor = {
    name: shapeKey,
    binary: (o, l, r) => {
      const ls = descriptorOf(l) ? storeOfVec(l) : null;
      const rs = descriptorOf(r) ? storeOfVec(r) : null;
      const lMat = ls && !isVec(ls); const rMat = rs && !isVec(rs);
      // vec ∘ scalar / scalar ∘ vec — scale
      if (ls && isVec(ls) && typeof r === 'number' && (o === '*' || o === '/')) return makeVec(Array.from(ls.c, (x) => o === '*' ? x * r : x / r), resultElement(ls));
      if (rs && isVec(rs) && typeof l === 'number' && o === '*') return makeVec(Array.from(rs.c, (x) => x * l), resultElement(rs));
      // mat ∘ scalar / scalar ∘ mat — scale (componentwise, same shape)
      if (lMat && typeof r === 'number' && (o === '*' || o === '/')) return makeMat(Array.from(ls!.c, (x) => o === '*' ? x * r : x / r), ls!.rows, ls!.cols, resultElement(ls!));
      if (rMat && typeof l === 'number' && o === '*') return makeMat(Array.from(rs!.c, (x) => x * l), rs!.rows, rs!.cols, resultElement(rs!));
      // matmul: mat * (mat|vec) — a vec is the cols===1 case
      if (lMat && rs && o === '*') { const p = matmul(ls!, rs); if (!p) return NOT_HANDLED; const e = resultElement(ls!, rs); return p.cols === 1 ? makeVec(p.c, e) : makeMat(p.c, p.rows, p.cols, e); }
      // vec componentwise + - * / (equal length). Pass the stores straight to `componentwise` — core does not
      // mutate them and writes a fresh result array, so the two prior `Array.from` copies were redundant.
      if (ls && rs && isVec(ls) && isVec(rs) && ls.rows === rs.rows && (o === '+' || o === '-' || o === '*' || o === '/'))
        return makeVec(componentwise(o === '+' ? 'add' : o === '-' ? 'sub' : o === '*' ? 'mul' : 'div', ls.c as number[], rs.c as number[]), resultElement(ls, rs));
      return NOT_HANDLED;
    },
    equals: (l, r) => {
      const ls = descriptorOf(l) ? storeOfVec(l) : null; const rs = descriptorOf(r) ? storeOfVec(r) : null;
      if (!ls || !rs || ls.rows !== rs.rows || ls.cols !== rs.cols) return false;
      return Array.from(ls.c).every((x, i) => x === rs.c[i]);
    },
    neg: (v) => { const s = storeOfVec(v); const c = Array.from(s.c, (x) => -x); return isVec(s) ? makeVec(c, s.element) : makeMat(c, s.rows, s.cols, s.element); },
    getMember: (v, prop) => {
      const s = storeOfVec(v);
      if (!isVec(s)) return NOT_HANDLED;   // matrices have no swizzle
      if (prop.length === 1) { const i = SWIZZLE.indexOf(prop); return (i >= 0 && i < s.rows) ? s.c[i] : NOT_HANDLED; }
      const idxs = [...prop].map((ch) => SWIZZLE.indexOf(ch));
      if (idxs.some((i) => i < 0 || i >= s.rows) || idxs.length < 2 || idxs.length > 4) return NOT_HANDLED;
      return makeVec(idxs.map((i) => s.c[i] as number), s.element);
    },
    // A matrix indexes to its i-th COLUMN as a fresh immutable vec (column-major); a vec has no getIndex
    // (swizzle-only). Out-of-range throws BufferError, which the interpreter's readMember maps to a diagnostic.
    getIndex: mat
      ? (v, key) => {
          if (typeof key !== 'number') return NOT_HANDLED;
          const s = storeOfVec(v);
          if (!Number.isInteger(key) || key < 0 || key >= s.cols) throw new BufferError('ML-LANG-INDEX-RANGE', `index ${String(key)} is out of range (length ${s.cols})`);
          return makeVec(coreMatColumn(s.c as number[], s.rows, s.cols, key), s.element);
        }
      : undefined,
    display: (v) => { const s = storeOfVec(v); return `${isVec(s) ? `vec${s.rows}` : `mat${s.cols}x${s.rows}`}(${Array.from(s.c).join(', ')})`; },
    lower: vecLower(rows, cols, element),
  };
  vecDescriptors.set(key, desc);
  return desc;
}

/** Column-major matrix product A(R×K) · B(K×C) → R×C. A vec is a K×1 / R×1 column. Null on inner-dim
 *  mismatch (A.cols !== B.rows). Delegates the arithmetic to the core column-major matmul. */
function matmul(a: VecStore, b: VecStore): { c: number[]; rows: number; cols: number } | null {
  if (a.cols !== b.rows) return null;
  const out = coreMatmul(a.c as number[], a.rows, a.cols, b.c as number[], b.rows, b.cols);
  return { c: out, rows: a.rows, cols: b.cols };
}

/** Build a boxed vec (an n×1 column). Components are coerced to `element` (f32 rounds via Math.fround;
 *  f64 is exact) and stored on a Symbol-hidden field the language cannot read. */
export function makeVec(components: number[], element: LowerElement = 'f32'): object {
  const coerce = element === 'f64' ? (x: number) => x : (x: number) => Math.fround(x);
  const box = {};
  Object.defineProperty(box, VEC_STORE, { value: { c: components.map(coerce), rows: components.length, cols: 1, element } satisfies VecStore, enumerable: false, configurable: false, writable: false });
  return tagCustom(box, vecDescriptor(components.length, 1, element));
}

/** Build a boxed matrix (rows×cols, column-major flat). Components are coerced to `element` as in makeVec. */
export function makeMat(components: number[], rows: number, cols: number, element: LowerElement = 'f32'): object {
  const coerce = element === 'f64' ? (x: number) => x : (x: number) => Math.fround(x);
  const box = {};
  Object.defineProperty(box, VEC_STORE, { value: { c: components.map(coerce), rows, cols, element } satisfies VecStore, enumerable: false, configurable: false, writable: false });
  return tagCustom(box, vecDescriptor(rows, cols, element));
}

/** The flat column-major identity of size n (element (i,i) = 1). */
export function identityMat(n: number): number[] { const out = new Array<number>(n * n).fill(0); for (let i = 0; i < n; i++) out[i * n + i] = 1; return out; }
