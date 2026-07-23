/** A builtin's capability profile (metadata for a classifier / a compile consumer). */
export type BuiltinProfile = 'core' | 'host';
/** How faithfully a builtin's result reproduces AWAY from the interpreter: `'exact'` (bit-for-bit
 *  identical on any target), `'gpu-tolerant'` (reproducible within a GPU's floating-point tolerance), or
 *  `'cpu-only'` (must run on the interpreter — not reproducible on a GPU target). */
export type Portability = 'exact' | 'gpu-tolerant' | 'cpu-only';
/** The static self-description of a builtin: its call name, capability profile, and the metadata a
 *  classifier / a compile consumer reads to decide whether (and how) the call reproduces on another
 *  target. Pure data — it carries no behavior. */
export interface BuiltinSpec {
  /** The call head this builtin answers to — the identifier used at the call site (e.g. `'sqrt'`). */
  readonly name: string;
  /** Whether this is a domain-agnostic `'core'` builtin or a `'host'`-supplied capability. */
  readonly profile: BuiltinProfile;
  /** How faithfully the result reproduces away from the interpreter. */
  readonly portability: Portability;
  /** True when the builtin accepts a closure argument (a mapping/filtering predicate). */
  readonly takesClosure: boolean;
  /** The accepted argument-count range as `[min, max]`, inclusive. */
  readonly arity: readonly [number, number];
  /** True when the builtin is DECLARED in a profile but NOT dispatched — a name whose classification is
   *  reserved for a future consumer, without adding a code path now. */
  readonly future?: boolean;
  /** The name this builtin lowers to on a compile target when it differs from `name`. */
  readonly lowerName?: string;
  /** A one-line human description shown in a hover card / completion detail. */
  readonly doc?: string;
  /** A one-line description of what a call returns, rendered as `Returns <returnDoc>.` after the
   *  description + per-arg list in a hover card. When absent, the hover omits the returns line. */
  readonly returnDoc?: string;
  /** The named parameters, in call-position order — powering signature help + a param-named hover
   *  signature (the same role {@link HeadSpec.params} plays for a head). */
  readonly params?: readonly HeadParam[];
}

/** One parameter of a head, for signature help + param completion. */
export interface HeadParam {
  /** The parameter's display name. */
  readonly name: string;
  /** Optional human-readable description. */
  readonly doc?: string;
  /** True when the parameter may be omitted. */
  readonly optional?: boolean;
  /** True when the parameter is variadic (absorbs the remaining args). */
  readonly rest?: boolean;
}

/** The static self-description of a host CALL HEAD (a domain-built node/value), analogous to
 *  {@link BuiltinSpec} for builtins. Pure tooling metadata — never read by the interpreter. */
export interface HeadSpec {
  /** The head name (the call-site identifier, e.g. `'div'` or `'gpu'`). */
  readonly name: string;
  /** The head's parameters, in order. */
  readonly params: readonly HeadParam[];
  /** The accepted argument-count range as `[min, max]`, inclusive. */
  readonly arity: readonly [number, number];
  /** Whether the head builds a `'node'` (child-position) or a `'value'` (expression-position). */
  readonly returns: 'node' | 'value';
  /** Optional human-readable description. */
  readonly doc?: string;
  /** A one-line description of what the head returns, rendered as `Returns <returnDoc>.` after the
   *  description + per-arg list in a hover card. Preferred over the coarse {@link HeadSpec.returns}
   *  kind for reader-facing text; when absent, the hover omits the returns line. */
  readonly returnDoc?: string;
  /** True when the `head { ... }` wrap-block shorthand applies (the head takes children). */
  readonly takesChildren?: boolean;
}

/** One accessible member of a custom value type, for member completion + hover. */
export interface MemberSpec {
  /** The member name (e.g. `'x'`, `'xyz'`). */
  readonly name: string;
  /** Optional human-readable description. */
  readonly doc?: string;
  /** Whether it is a single `component` read, a multi-component `swizzle`, or a `method`. */
  readonly kind: 'swizzle' | 'component' | 'method';
}

/** A static projection of a custom value type — its accessible members + the builtin names that
 *  construct it — with no runtime handlers. */
export interface TypeDescriptorMeta {
  /** The type name (e.g. `'vec3'`). */
  readonly name: string;
  /** The members completion offers on a value of this type. */
  readonly members: readonly MemberSpec[];
  /** Builtin names whose result is a value of this type (e.g. `['vec3']`). */
  readonly constructors: readonly string[];
  /** Optional human-readable description. */
  readonly doc?: string;
}
