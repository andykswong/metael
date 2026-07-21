// The reactive-component AST. Discriminated unions, extensible via a fail-closed default in the
// evaluator.
import type { SourceSpan } from './diagnostics.ts';

/**
 * An expression AST node: the parsed form of every value-producing construct in the language.
 *
 * A discriminated union keyed by the `kind` string tag; each variant also carries a {@link SourceSpan}
 * (`span`) locating it in the source so diagnostics can point at it. Produced by the parser and walked
 * by the eval-free evaluator, which dispatches on `kind` and fails closed on any variant it does not
 * recognize.
 *
 * @remarks
 * Structural children reference {@link Expr} recursively (and {@link Stmt} for block bodies), so an
 * `Expr` is the root of an expression subtree. Operator variants carry a {@link BinOp} tag; the
 * `arrow` parameters use {@link Pattern}.
 */
export type Expr =
  /** A numeric literal (e.g. `42`, `3.14`). */
  | { kind: 'number'; value: number; span: SourceSpan }
  /** A string literal. */
  | { kind: 'string'; value: string; span: SourceSpan }
  /** A boolean literal (`true` / `false`). */
  | { kind: 'bool'; value: boolean; span: SourceSpan }
  /** The `null` literal. */
  | { kind: 'null'; span: SourceSpan }
  /** An identifier reference — a lookup of the binding named `name`. */
  | { kind: 'ident'; name: string; span: SourceSpan }
  /** A static property access, `object.property`. */
  | { kind: 'member'; object: Expr; property: string; span: SourceSpan }
  /** A computed index access, `object[index]`. */
  | { kind: 'index'; object: Expr; index: Expr; span: SourceSpan }
  /** An object literal, `{ … }`, built from its {@link ObjectEntry} list. */
  | { kind: 'object'; entries: ObjectEntry[]; span: SourceSpan }
  /** An array literal, `[ … ]`, built from its {@link ArrayElement} list. */
  | { kind: 'array'; elements: ArrayElement[]; span: SourceSpan }
  /** An arrow function, `(params) => body`. `body` is a single {@link Expr} for an expression body, or a
   *  `Stmt[]` block for a `=> { … }` body. Parameters are {@link Pattern}s. */
  | { kind: 'arrow'; params: Pattern[]; body: Expr | Stmt[]; span: SourceSpan }
  /** A call, `callee(args)`. A trailing `{ … }` block (`block` present) marks a wrapping element — a call
   *  whose child statements are the wrapped content. */
  | { kind: 'call'; callee: Expr; args: Expr[]; block?: Stmt[]; span: SourceSpan }
  /** A prefix unary operation: numeric negation (`-`) or logical not (`!`). */
  | { kind: 'unary'; op: '-' | '!'; operand: Expr; span: SourceSpan }
  /** A binary operation combining `left` and `right` under a {@link BinOp} operator. */
  | { kind: 'binary'; op: BinOp; left: Expr; right: Expr; span: SourceSpan }
  /** A ternary conditional, `test ? then : else`. */
  | { kind: 'cond'; test: Expr; then: Expr; else: Expr; span: SourceSpan };

/** An array literal element: a normal value, or `...expr` splicing an array's elements in. */
export interface ArrayElement {
  /** The element expression — either the value itself, or (when {@link ArrayElement.spread} is `true`)
   *  the array whose elements are spliced in. */
  readonly value: Expr;
  /** `true` when this element is a `...expr` spread that splices another array's elements in place. */
  readonly spread: boolean;
}
/** An object literal entry: a `key: value` pair, or `...expr` spreading an object's own keys in
 *  (spread entries carry no key; `key` is '' and ignored). */
export interface ObjectEntry {
  /** The property name for a `key: value` pair. Empty and ignored when {@link ObjectEntry.spread} is
   *  `true`. */
  readonly key: string;
  /** The value expression — either the entry's value, or (when {@link ObjectEntry.spread} is `true`) the
   *  object whose own keys are spread in. */
  readonly value: Expr;
  /** `true` when this entry is a `...expr` spread that copies another object's own keys in place. */
  readonly spread: boolean;
}

/**
 * The set of binary operators, used as the `op` tag of the `binary` {@link Expr} variant.
 *
 * @remarks
 * Covers arithmetic (`+ - * / %`), equality/relational comparison (`== != < <= > >=`), and short-circuit
 * logical operators (`&&` / `||`).
 */
export type BinOp = '+' | '-' | '*' | '/' | '%' | '==' | '!=' | '<' | '<=' | '>' | '>=' | '&&' | '||';

/**
 * A parameter / destructuring pattern (e.g. `component KPI({ label, value })`).
 *
 * A discriminated union keyed by `kind`: a plain binding name, or a shallow object/array destructuring
 * that binds the listed field/element names.
 */
export type Pattern =
  /** A plain binding name (e.g. `x`). */
  | { kind: 'name'; name: string }
  /** An object-destructuring pattern, `{ a, b }`, binding each named field. */
  | { kind: 'objectPattern'; fields: string[] }
  /** An array-destructuring pattern, `[a, b]`, binding each positional element name. */
  | { kind: 'arrayPattern'; elements: string[] };

/**
 * A statement AST node: the parsed form of every declaration, control-flow, and effect construct.
 *
 * A discriminated union keyed by `kind`; each variant carries a {@link SourceSpan} (`span`) for
 * diagnostics. Statement bodies are `Stmt[]` blocks and their sub-expressions are {@link Expr}s.
 * Produced by the parser and walked by the eval-free evaluator, which fails closed on any unrecognized
 * variant.
 */
export type Stmt =
  /** A `const name = init` binding (immutable). */
  | { kind: 'const'; name: string; init: Expr; span: SourceSpan }
  /** A `let name = init` binding (reactive when declared inside a `component` body). */
  | { kind: 'let'; name: string; init: Expr; span: SourceSpan }
  /** An assignment, `target = value`, to an existing binding or member/index target. */
  | { kind: 'assign'; target: Expr; value: Expr; span: SourceSpan }
  /** A `function` declaration — a pure, non-reactive callable with parameter {@link Pattern}s. */
  | { kind: 'function'; name: string; params: Pattern[]; body: Stmt[]; span: SourceSpan }
  /** A `component` declaration — a stateful callable whose `let`s are reactive. */
  | { kind: 'component'; name: string; params: Pattern[]; body: Stmt[]; span: SourceSpan }
  /** An `if (test) { then } [else { else }]` conditional; `else` is absent when there is no else branch. */
  | { kind: 'if'; test: Expr; then: Stmt[]; else?: Stmt[]; span: SourceSpan }
  /** A `for (binding of iterable) { body }` loop that binds each element in turn. */
  | { kind: 'for'; binding: string; iterable: Expr; body: Stmt[]; span: SourceSpan }
  /** A `while (test) { body }` loop. */
  | { kind: 'while'; test: Expr; body: Stmt[]; span: SourceSpan }
  /** A `return value?` statement; `value` is absent for a bare `return`. */
  | { kind: 'return'; value?: Expr; span: SourceSpan }
  /** An expression evaluated for its value/effect (e.g. a call), `expr`. */
  | { kind: 'expr'; expr: Expr; span: SourceSpan };

/** A parsed program: the ordered top-level statements the evaluator runs, in source order. */
export interface Program {
  /** The top-level statements, in source order. */
  readonly stmts: Stmt[];
}

/**
 * Property names that must never be read or written through member/index access.
 *
 * @remarks
 * Guards against prototype-pollution: accessing `__proto__`, `constructor`, or `prototype` on a value
 * is refused so program source cannot reach the JS prototype chain. The evaluator consults this set on
 * every member/index operation and fails closed when a key is present.
 */
export const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);
