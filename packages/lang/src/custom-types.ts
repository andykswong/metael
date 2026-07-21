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
/** The type of the {@link NOT_HANDLED} sentinel — the return type of an operator/accessor handler that
 *  declines to handle a given op/operand combination, leaving the interpreter to apply its uniform
 *  fallback. */
export type NotHandled = typeof NOT_HANDLED;

/** The element kind of a lowered numeric type's backing store: 32-bit float, 64-bit float, 32-bit
 *  signed int, or 32-bit unsigned int. Determines the emitted numeric width. */
export type LowerElement = 'f32' | 'f64' | 'i32' | 'u32';
/** The structural shape of a lowered type: a `scalar`, an N-wide `vecN`, or an M×N `matMxN`. */
export type LowerShape = 'scalar' | 'vecN' | 'matMxN';
/** How a lowered value is accessed: a single `value` (a scalar/vec/mat register), or a
 *  `linear-buffer` (an indexable element store). */
export type LowerAccess = 'value' | 'linear-buffer';

/** A neutral, target-agnostic operator lowering — NOT shader text. A compile consumer renders it to a
 *  concrete backend (e.g. WGSL/GLSL/CPU). Keyed on the language's {@link BinOp} (plus `'neg'`) via
 *  {@link Lowering.ops}, so a descriptor declares how each of its operators maps to a target op without
 *  the emitter hardcoding the type. */
export type LowerOp =
  | {
      /** Discriminant: a per-element binary op applied component-by-component (e.g. `vec + vec`). */
      readonly kind: 'componentwise';
      /** Which componentwise arithmetic op to apply. */
      readonly op: 'add' | 'sub' | 'mul' | 'div';
    }
  | {
      /** Discriminant: unary negation of every component. */
      readonly kind: 'unary';
      /** The unary op — always negation. */
      readonly op: 'neg';
    }
  /** A scalar-times-value scale (e.g. `2 * vec`). */
  | {
      /** Discriminant: a scalar-times-value scale (e.g. `2 * vec`). */
      readonly kind: 'scale';
    }
  | {
      /** Discriminant: a vector dot product. */
      readonly kind: 'dot';
    }
  | {
      /** Discriminant: a matrix (or matrix–vector) multiply. */
      readonly kind: 'matmul';
    }
  | {
      /** Discriminant: a call to a named target-side builtin function. */
      readonly kind: 'builtin-call';
      /** The name of the target-side builtin to call. */
      readonly name: string;
    };

/** A domain-agnostic description of how a type is STORED/ACCESSED and how its operators/members map to
 *  target ops. Complete on its own: a compile consumer never hardcodes a type — a new numeric type is a
 *  library (a descriptor whose {@link Lowering} carries `ops` + `members`) that needs zero emitter edits.
 *
 *  @remarks Attached to a {@link TypeDescriptor} via {@link TypeDescriptor.lower}. */
export interface Lowering {
  /** The element kind of the backing store ({@link LowerElement}). */
  readonly element: LowerElement;
  /** The structural shape — scalar, vector, or matrix ({@link LowerShape}). */
  readonly shape: LowerShape;
  /** Scalar-buffer element count (the number of elements in a linear buffer). */
  readonly n?: number;
  /** Row count for a vector/matrix (a vector is `rows`×1). */
  readonly rows?: number;
  /** Column count for a matrix (a vector is 1 column). */
  readonly cols?: number;
  /** Whether a value of this type can be placed in GPU storage (drives whether a compile consumer may
   *  keep it resident on-device). */
  readonly gpuStorable: boolean;
  /** How the value is accessed — a single value or a linear buffer ({@link LowerAccess}). */
  readonly access: LowerAccess;
  /** The per-operator lowerings, keyed by {@link BinOp} (plus `'neg'`). A missing key means the
   *  operator has no target lowering for this type. */
  readonly ops?: Readonly<Partial<Record<string, LowerOp>>>;
  /** How member access lowers: a `swizzle` (multi-component, e.g. `v.xy`) or a single `component`
   *  read, plus `of` — the underlying element/type name the members project from. */
  readonly members?: {
    /** Whether member access is a multi-component `swizzle` or a single `component` read. */
    readonly kind: 'swizzle' | 'component';
    /** The underlying element/type name the members project from. */
    readonly of: string;
  };
}

/** How the interpreter treats a value of a custom type. Handlers are built-in host code (typed arrays,
 *  vec/mat) trusted like a resolveCall head. A mutating handler is gated by `frozen?` BEFORE it runs
 *  (the interpreter checks and emits ML-LANG-IMMUTABLE for a const value). An operator handler returns
 *  NOT_HANDLED for an op/combination it does not define. */
export interface TypeDescriptor {
  /** The type's display name, used in diagnostics (e.g. `type 'vec3' has no member 'w'`) and as the
   *  fallback `[name]` string when no {@link TypeDescriptor.display} handler is defined. */
  readonly name: string;
  /** An optional extra immutability predicate: returns `true` when a value should be treated as frozen
   *  independent of the interpreter's own const/frozen box (OR-ed with it). A mutating handler is gated
   *  by this before it runs. */
  readonly frozen?: (v: unknown) => boolean;
  /** Handle a binary operator between two values (at least one carrying this descriptor). Returns the
   *  result, or {@link NOT_HANDLED} for an op/operand combination this type does not define.
   *  @param op - the binary operator ({@link BinOp}).
   *  @param left - the left operand.
   *  @param right - the right operand.
   *  @returns the operator result, or {@link NOT_HANDLED}. */
  binary?(op: BinOp, left: unknown, right: unknown): unknown;
  /** Decide value equality for `==`/`!=`, preferred over {@link TypeDescriptor.binary} when present.
   *  @param a - the first operand.
   *  @param b - the second operand.
   *  @returns `true` iff the two values are equal by this type's semantics. */
  equals?(a: unknown, b: unknown): boolean;
  /** Handle unary negation (`-v`). Returns the negated value, or {@link NOT_HANDLED} if undefined for
   *  this type.
   *  @param v - the value to negate.
   *  @returns the negated value, or {@link NOT_HANDLED}. */
  neg?(v: unknown): unknown;
  /** Read a named member (`v.prop`). Returns the member value, or {@link NOT_HANDLED} to signal no such
   *  member (the interpreter then raises `ML-LANG-UNKNOWN-MEMBER`).
   *  @param v - the container value.
   *  @param prop - the member name.
   *  @returns the member value, or {@link NOT_HANDLED}. */
  getMember?(v: unknown, prop: string): unknown;
  /** Read an indexed element (`v[key]`). Returns the element, or {@link NOT_HANDLED} to signal no such
   *  index/key. Also serves string keys when {@link TypeDescriptor.getMember} is absent.
   *  @param v - the container value.
   *  @param key - the numeric index or string key.
   *  @returns the element value, or {@link NOT_HANDLED}. */
  getIndex?(v: unknown, key: number | string): unknown;
  /** Write a named member in place (`v.prop = val`). Its presence marks the type MUTABLE (the
   *  interpreter allocates a frozen box on tag). Throws a {@link BufferError} to surface a value error.
   *  @param v - the container value.
   *  @param prop - the member name.
   *  @param val - the value to store. */
  setMember?(v: unknown, prop: string, val: unknown): void;
  /** Write an indexed element in place (`v[key] = val`). Its presence marks the type MUTABLE. Throws a
   *  {@link BufferError} to surface a bounds/coercion error.
   *  @param v - the container value.
   *  @param key - the numeric index or string key.
   *  @param val - the value to store. */
  setIndex?(v: unknown, key: number | string, val: unknown): void;
  /** Produce the elements a `for … of` over this value iterates.
   *  @param v - the value to iterate.
   *  @returns an iterable of the value's elements. */
  iterate?(v: unknown): Iterable<unknown>;
  /** Zero-copy access to a linear-buffer value's backing store (only present on `access: 'linear-buffer'`
   *  descriptors). Returns the raw backing array + its element kind so a consumer (a compute backend) can
   *  read it WITHOUT the O(n) {@link TypeDescriptor.iterate} → `number[]` copy. The returned data is the
   *  live store — treat it read-only.
   *  @param v - the linear-buffer value.
   *  @returns the live backing array and its {@link LowerElement} kind. */
  bufferView?(v: unknown): {
    /** The live backing array — treat it as read-only. */
    readonly data: ArrayLike<number>;
    /** The element kind of the backing array ({@link LowerElement}). */
    readonly element: LowerElement;
  };
  /** Decide the value's truthiness for `if`/`&&`/`||`/`!`, overriding the language's default rule.
   *  @param v - the value to test.
   *  @returns the value's truthiness. */
  truthy?(v: unknown): boolean;
  /** Coerce the value to a string for string `+` and `join`. Should be bounded (not proportional to a
   *  large backing store). Absent → the language uses the `[name]` fallback.
   *  @param v - the value to stringify.
   *  @returns the value's display string. */
  display?(v: unknown): string;
  /** How this type is stored/accessed and how its operators/members lower to target ops
   *  ({@link Lowering}). Present on types a compile consumer can lower; absent for an interpreter-only
   *  type. */
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

/** Read the {@link TypeDescriptor} attached to a value by {@link tagCustom}, or `undefined` if the value
 *  carries none (a plain metael value). This is the single lookup every dispatch site uses to decide
 *  whether a value has custom-type behavior.
 *  @param v - the value to inspect.
 *  @returns the value's descriptor, or `undefined`. */
export function descriptorOf(v: unknown): TypeDescriptor | undefined {
  return (typeof v === 'object' && v !== null) ? (v as { [DESCRIPTOR]?: TypeDescriptor })[DESCRIPTOR] : undefined;
}
/** Whether a value carries a custom-type {@link TypeDescriptor} (i.e. {@link descriptorOf} is defined).
 *  @param v - the value to test.
 *  @returns `true` iff the value is a tagged custom type. */
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
/** Whether a mutable custom value has been frozen (by {@link markFrozen}). Always `false` for an
 *  immutable custom value (which has no frozen box) and for a non-custom value. The interpreter checks
 *  this before running a mutating handler, emitting `ML-LANG-IMMUTABLE` for a frozen value.
 *  @param v - the value to test.
 *  @returns `true` iff the value is a frozen mutable custom value. */
export function isFrozenCustom(v: unknown): boolean {
  const box = (typeof v === 'object' && v !== null) ? (v as { [FROZEN]?: FrozenBox })[FROZEN] : undefined;
  return box ? box.frozen : false;
}

// ─────────────────────────────────────────── linear-buffer error signal ───────────────────────────────────────────

/** A marker error a descriptor handler throws to surface a value error (bounds/coercion) — the
 *  interpreter's read/write hooks catch it and map it to a diagnostic (descriptors are stateless/shared
 *  and can't reach the Runner). The concrete buffer/vec/mat INSTANCES live in a standard-library module;
 *  this error type is part of the shared protocol so a library handler can raise a diagnostic. */
export class BufferError extends Error {
  /**
   * Construct a buffer error carrying a diagnostic code and detail message.
   *
   * @param code - the diagnostic code the interpreter surfaces (e.g. a bounds/coercion code).
   * @param detail - the human-readable diagnostic message.
   */
  constructor(
    /** The diagnostic code the interpreter maps this error to when it catches the throw. */
    readonly code: string,
    /** The human-readable diagnostic detail message. */
    readonly detail: string,
  ) { super(code); }
}
