// The boxed typed-array (linear-buffer) values for the language surface. Each kind (f32/f64/i32/u32)
// wraps a real JS TypedArray on a Symbol-hidden field and exposes it to the interpreter through the
// custom-value descriptor protocol (index read/write, length, iteration, zero-copy bufferView, display).
// The descriptor machinery + BufferError live in @metael/lang; the INSTANCES live here.
import type { TypeDescriptor, Lowering, LowerElement, GenerationRef } from '@metael/lang';
import { tagCustom, NOT_HANDLED, BufferError, isFrozenCustom } from '@metael/lang';

/** Per-element coercion + storage element for each typed-array kind. i32/u32 truncate-then-wrap mod 2^32;
 *  f32 rounds via Math.fround; f64 is exact. NaN/±Inf/-0 follow the underlying TypedArray. */
export interface BufferKind {
  /** Constructor for the backing JS TypedArray of this kind (e.g. `Float32Array`), allocating `n` elements. */
  readonly ctor: {
    /** Allocate a zero-filled backing array of `n` elements. */
    new (n: number): { [i: number]: number; length: number };
  };
  /** The lowering element tag for this kind (`'f32'`/`'f64'`/`'i32'`/`'u32'`). */
  readonly element: LowerElement;
  /** Whether values of this kind can back a GPU storage buffer (`f64` cannot; the rest can). */
  readonly gpuStorable: boolean;
  /** Coerce an incoming JS number to this kind's stored representation: f32 rounds via `Math.fround`,
   *  i32/u32 truncate-then-wrap mod 2^32, f64 is exact. */
  coerce(x: number): number;
}
const F32: BufferKind = { ctor: Float32Array, element: 'f32', gpuStorable: true, coerce: (x) => Math.fround(x) };
const F64: BufferKind = { ctor: Float64Array, element: 'f64', gpuStorable: false, coerce: (x) => x };
const I32: BufferKind = { ctor: Int32Array,   element: 'i32', gpuStorable: true, coerce: (x) => x | 0 };
const U32: BufferKind = { ctor: Uint32Array,  element: 'u32', gpuStorable: true, coerce: (x) => x >>> 0 };
/** The {@link BufferKind} table keyed by element tag — the per-kind ctor/coercion/gpu-storability facts
 *  the typed-array descriptors are built from. */
export const BUFFER_KINDS: Readonly<Record<'f32' | 'f64' | 'i32' | 'u32', BufferKind>> = { f32: F32, f64: F64, i32: I32, u32: U32 };

/** The backing store lives on a Symbol-keyed field the language surface cannot read or set. */
const STORE: unique symbol = Symbol('ml.buffer.store');

/** The shared typed-array descriptor for a given kind. One descriptor object per kind (stateless). The
 *  backing store is a real JS TypedArray held on a Symbol-keyed field the language can't read. */
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

/** The shared custom-value type descriptor for each typed-array kind, keyed by element tag — one
 *  stateless descriptor per kind, wiring index read/write, `length`, iteration, zero-copy `bufferView`,
 *  and display into the custom-value protocol. {@link makeTypedArray} tags an allocation with one of these. */
export const TYPED_ARRAY_DESCRIPTORS: Readonly<Record<'f32' | 'f64' | 'i32' | 'u32', TypeDescriptor>> = {
  f32: makeTypedArrayDescriptor(F32), f64: makeTypedArrayDescriptor(F64), i32: makeTypedArrayDescriptor(I32), u32: makeTypedArrayDescriptor(U32),
};
