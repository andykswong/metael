// The reactive-component AST. Discriminated unions, extensible via a fail-closed default in the
// evaluator.
import type { SourceSpan } from './diagnostics.ts';

export type Expr =
  | { kind: 'number'; value: number; span: SourceSpan }
  | { kind: 'string'; value: string; span: SourceSpan }
  | { kind: 'bool'; value: boolean; span: SourceSpan }
  | { kind: 'null'; span: SourceSpan }
  | { kind: 'ident'; name: string; span: SourceSpan }
  | { kind: 'member'; object: Expr; property: string; span: SourceSpan }
  | { kind: 'index'; object: Expr; index: Expr; span: SourceSpan }
  | { kind: 'object'; entries: ObjectEntry[]; span: SourceSpan }
  | { kind: 'array'; elements: ArrayElement[]; span: SourceSpan }
  | { kind: 'arrow'; params: Pattern[]; body: Expr | Stmt[]; span: SourceSpan }   // Stmt[] = block body `=> { … }`
  | { kind: 'call'; callee: Expr; args: Expr[]; block?: Stmt[]; span: SourceSpan }  // block present ⇒ wrapping element
  | { kind: 'unary'; op: '-' | '!'; operand: Expr; span: SourceSpan }
  | { kind: 'binary'; op: BinOp; left: Expr; right: Expr; span: SourceSpan }
  | { kind: 'cond'; test: Expr; then: Expr; else: Expr; span: SourceSpan };   // ternary `?:`

/** An array literal element: a normal value, or `...expr` splicing an array's elements in. */
export interface ArrayElement { readonly value: Expr; readonly spread: boolean }
/** An object literal entry: a `key: value` pair, or `...expr` spreading an object's own keys in
 *  (spread entries carry no key; `key` is '' and ignored). */
export interface ObjectEntry { readonly key: string; readonly value: Expr; readonly spread: boolean }

export type BinOp = '+' | '-' | '*' | '/' | '%' | '==' | '!=' | '<' | '<=' | '>' | '>=' | '&&' | '||';

/** Parameter / destructuring pattern (e.g. `component KPI({ label, value })`). */
export type Pattern =
  | { kind: 'name'; name: string }
  | { kind: 'objectPattern'; fields: string[] }
  | { kind: 'arrayPattern'; elements: string[] };

export type Stmt =
  | { kind: 'const'; name: string; init: Expr; span: SourceSpan }
  | { kind: 'let'; name: string; init: Expr; span: SourceSpan }
  | { kind: 'assign'; target: Expr; value: Expr; span: SourceSpan }
  | { kind: 'function'; name: string; params: Pattern[]; body: Stmt[]; span: SourceSpan }
  | { kind: 'component'; name: string; params: Pattern[]; body: Stmt[]; span: SourceSpan }
  | { kind: 'if'; test: Expr; then: Stmt[]; else?: Stmt[]; span: SourceSpan }
  | { kind: 'for'; binding: string; iterable: Expr; body: Stmt[]; span: SourceSpan }
  | { kind: 'while'; test: Expr; body: Stmt[]; span: SourceSpan }
  | { kind: 'return'; value?: Expr; span: SourceSpan }
  | { kind: 'expr'; expr: Expr; span: SourceSpan };

export interface Program { readonly stmts: Stmt[] }

export const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);
