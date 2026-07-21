// A canonical AST → DSL-text printer. PURE + additive: it imports only the AST types and produces the
// surface syntax the grammar defines. The correctness proof is a conservation law (print.test.ts):
// stripSpans(parseProgram(printProgram(ast)).program) deep-equals stripSpans(ast) — reprinting changes
// spans (byte offsets shift), so the comparison is span-stripped. Extra defensive parens are legal
// (they reparse away). No new AST kinds; the bare-ident wrap shorthand prints as a zero-arg call + block.
import type { Expr, Stmt, Program, Pattern, ArrayElement, ObjectEntry } from './ast.ts';

const INDENT = '  ';

/**
 * The upper bound on the printer's own recursion depth. The printer recurses structurally over the AST,
 * but the parser accepts UNBOUNDED left-spine chains (member/index/call/binary) because its depth guard
 * counts only nested-expression recursion, not its iterative postfix/binary loops — so a valid parser AST
 * can be arbitrarily deep. {@link printExpr} and {@link stripSpans} bound their recursion at this cap and
 * fail CLOSED with a typed, catchable {@link PrintDepthError} rather than letting a raw `RangeError` (stack
 * overflow) escape into the host, mirroring the parser's own fail-closed depth guard.
 *
 * @remarks The cap sits far above any realistic nesting (the parser bounds nested expressions at 512; only
 *          degenerate chains exceed it) and comfortably below the native stack limit, so a pathological AST
 *          fails loud, not ugly.
 */
export const MAX_PRINT_DEPTH = 1500;
let printDepth = 0;

/** Thrown (and catchable) when an AST nests past {@link MAX_PRINT_DEPTH} — a controlled, documented
 *  failure in place of an uncontrolled stack-overflow `RangeError`, so the printer stays a total function. */
export class PrintDepthError extends Error {
  /** Construct the error with the fixed message `'AST too deeply nested to print'` and a `name` of
   *  `'PrintDepthError'`, so a caller can distinguish it from a generic `Error` when catching. */
  constructor() { super('AST too deeply nested to print'); this.name = 'PrintDepthError'; }
}

/** Quote + escape a string literal, inverting the lexer's escapes (backslash, the quote, newline, tab).
 *  Other characters pass through verbatim. Uses double quotes (the lexer accepts either; we normalize). */
export function printString(value: string): string {
  let out = '"';
  for (const ch of value) {
    if (ch === '\\') out += '\\\\';
    else if (ch === '"') out += '\\"';
    else if (ch === '\n') out += '\\n';
    else if (ch === '\t') out += '\\t';
    else out += ch;
  }
  return out + '"';
}

/** Recursively drop every `span` field so two ASTs can be compared structurally (reprinting shifts
 *  byte offsets). Returns a deep copy with spans removed; arrays and plain objects are recursed. Bounds
 *  its own recursion (a deeply-nested AST would otherwise overflow the stack) and fails closed with
 *  {@link PrintDepthError} — so, like the printer proper, it is a total function. */
export function stripSpans(node: unknown): unknown {
  if (++printDepth > MAX_PRINT_DEPTH) { printDepth--; throw new PrintDepthError(); }
  try {
    if (Array.isArray(node)) return node.map(stripSpans);
    if (node !== null && typeof node === 'object') {
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
        if (k === 'span') continue;
        out[k] = stripSpans(v);
      }
      return out;
    }
    return node;
  } finally { printDepth--; }
}

/** A lexer-round-trippable numeric literal. `String(n)` prints large/small magnitudes in exponent form
 *  (`1e21`→`"1e+21"`, `0.0000001`→`"1e-7"`) which the lexer splits into several tokens, and NaN/Infinity
 *  have no literal form at all. Degrade the (degenerate) non-finite values to `0` so printing never
 *  produces un-lexable text; otherwise expand the exponent form to a plain decimal by SHIFTING the
 *  decimal point of `String(n)`'s shortest round-trip digits — no re-rounding, so every finite double
 *  round-trips exactly. Only non-negative magnitudes reach here (a negative is a `unary '-'` over its
 *  magnitude), but the leading-sign group is kept for defensiveness. */
function printNumber(n: number): string {
  if (!Number.isFinite(n)) return '0';       // NaN/Infinity are not lexable literals; degrade safely
  const s = String(n);
  if (!/[eE]/.test(s)) return s;             // plain decimal already — round-trips as-is
  const m = /^(-?)(\d+)(?:\.(\d+))?[eE]([+-]?\d+)$/.exec(s);
  if (!m) return s;                          // unreachable for a finite value with an exponent; defensive
  const sign = m[1] ?? '';
  const intPart = m[2] ?? '';
  const fracPart = m[3] ?? '';
  const expStr = m[4] ?? '0';
  const digits = intPart + fracPart;
  const point = intPart.length + Number(expStr);   // decimal-point index within `digits` after the shift
  let out: string;
  if (point <= 0) out = `0.${'0'.repeat(-point)}${digits}`;           // 0.000…d  (small fraction)
  else if (point >= digits.length) out = digits + '0'.repeat(point - digits.length);   // d000…  (large int)
  else out = `${digits.slice(0, point)}.${digits.slice(point)}`;      // d.d      (point lands mid-digits)
  return sign + out;
}

/** An object key is printed bare only when it is a valid identifier; anything else (hyphenated
 *  `aria-label`, digit-leading `123abc`, punctuation) is quoted, which reparses exactly since the parser
 *  reads the key from a token's `.value` and a string token carries the unescaped key. */
function printKey(key: string): string {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(key) ? key : printString(key);
}

/**
 * Print a single expression node to its canonical DSL surface syntax.
 *
 * @param expr - the expression AST node to render.
 * @param depth - the current indentation depth, applied when the expression contains a `{ … }` block (an
 *                arrow/function body or a call's trailing block). Defaults to `0` (top level).
 * @returns the source text for `expr`. Defensive parentheses may be added around operator/ternary
 *          children; they reparse away, so the text still round-trips through the parser.
 * @throws {@link PrintDepthError} when the expression nests past {@link MAX_PRINT_DEPTH}, so a degenerate
 *         chain fails closed with a catchable error instead of overflowing the native stack.
 */
export function printExpr(expr: Expr, depth = 0): string {
  if (++printDepth > MAX_PRINT_DEPTH) { printDepth--; throw new PrintDepthError(); }
  try {
    return printExprInner(expr, depth);
  } finally { printDepth--; }
}

function printExprInner(expr: Expr, depth: number): string {
  switch (expr.kind) {
    case 'number': return printNumber(expr.value);
    case 'string': return printString(expr.value);
    case 'bool': return expr.value ? 'true' : 'false';
    case 'null': return 'null';
    case 'ident': return expr.name;
    // The object/callee of an access is routed through parenChild: a non-atomic head (binary/unary/
    // cond/arrow) must be wrapped or the postfix `.`/`[`/`(` would bind tighter than the head's operator
    // (e.g. `(a + b).c` must not flatten to `a + b.c`, `(() => 1)()` must not become `() => 1()`).
    // Atomic heads (member/index/call/string/bool/null/ident/object/array) stay bare, so `f().x` and
    // `a.b.c` are unparenthesized. A NUMBER head is NOT atomic here: `(1).foo` must keep its parens, or
    // the lexer would greedily consume the `.` into the number token (`1.foo` → `1.` + `foo`).
    case 'member': return `${parenMemberObject(expr.object, depth)}.${printMemberProperty(expr.property)}`;
    case 'index': return `${parenChild(expr.object, depth)}[${printExpr(expr.index, depth)}]`;
    case 'call': {
      const args = expr.args.map((a) => printExpr(a, depth)).join(', ');
      const call = `${parenChild(expr.callee, depth)}(${args})`;
      return expr.block ? `${call} ${printBlock(expr.block, depth)}` : call;
    }
    case 'unary': return `${expr.op}${parenChild(expr.operand, depth)}`;
    case 'binary': return `${parenChild(expr.left, depth)} ${expr.op} ${parenChild(expr.right, depth)}`;
    case 'cond': return `${parenChild(expr.test, depth)} ? ${printExpr(expr.then, depth)} : ${printExpr(expr.else, depth)}`;
    case 'object': return printObject(expr.entries, depth);
    case 'array': return printArray(expr.elements, depth);
    case 'arrow': {
      // An arrow node may carry destructuring params (`{ a, b }` / `[x, y]`) — the parser produces one
      // from a `function` EXPRESSION, whose params allow patterns. But arrow SYNTAX (`(…) => …`) only
      // reparses bare-ident params, so an arrow with any non-name param must print as a function
      // expression to round-trip; a name-only arrow keeps the concise arrow form.
      const nameOnly = expr.params.every((p) => p.kind === 'name');
      const params = `(${expr.params.map(printPattern).join(', ')})`;
      if (Array.isArray(expr.body)) {
        const block = printBlock(expr.body, depth);
        return nameOnly ? `${params} => ${block}` : `function${params} ${block}`;
      }
      // (A function expression has a statement-block body only; an expression-bodied arrow always has
      // name-only params in practice, but guard anyway by wrapping a non-name arrow's expr body in a block.)
      const body = printExpr(expr.body, depth);
      if (!nameOnly) return `function${params} {\n${INDENT.repeat(depth + 1)}${body}\n${INDENT.repeat(depth)}}`;
      // A body that begins with `{` (an object literal) would reparse as a statement block after `=>`,
      // so parenthesize it — the grammar's `() => ({ … })` rule.
      return `${params} => ${body.startsWith('{') ? `(${body})` : body}`;
    }
  }
}

/** The token printed after a member `.`: a bare identifier when valid, else a QUOTED string. The parser's
 *  dot-access reads the next token's `.value` as the property, so `a."x-y"` parses to a `member` node with
 *  property `x-y` — exactly like the source. A bare `.x-y` would instead re-lex as `a.x` minus `y`, and a
 *  computed `a["x-y"]` would parse to a different node kind (`index`), so the quoted-dot form is the one
 *  that round-trips a non-identifier member property. Escaping matches `printString`. */
function printMemberProperty(property: string): string {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(property) ? property : printString(property);
}

/** The object of a `.member` access. Same as parenChild, but ALSO wraps a bare number literal head:
 *  `(1).foo` must keep its parens because the lexer greedily eats the `.` into the number token, so an
 *  unparenthesized `1.foo` re-lexes as the number `1.` followed by `foo`. (Only the `.`-access position
 *  has this hazard; `1[0]` / `1(2)` / `1 + x` lex fine, so parenChild leaves a number bare there.) */
function parenMemberObject(e: Expr, depth: number): string {
  return e.kind === 'number' ? `(${printExpr(e, depth)})` : parenChild(e, depth);
}

/** Wrap a non-atomic child in parens. Conservative: extra parens are always legal (they reparse away),
 *  so we parenthesize any operator/ternary child rather than track exact precedence. Atomic nodes
 *  (literals, idents, member/index/call access, bracketed literals) never need wrapping. */
function parenChild(e: Expr, depth: number): string {
  const atomic = e.kind === 'number' || e.kind === 'string' || e.kind === 'bool' || e.kind === 'null'
    || e.kind === 'ident' || e.kind === 'member' || e.kind === 'index' || e.kind === 'call'
    || e.kind === 'object' || e.kind === 'array';
  return atomic ? printExpr(e, depth) : `(${printExpr(e, depth)})`;
}

function printObject(entries: readonly ObjectEntry[], depth: number): string {
  if (entries.length === 0) return '{}';
  const parts = entries.map((e) => e.spread ? `...${printExpr(e.value, depth)}` : `${printKey(e.key)}: ${printExpr(e.value, depth)}`);
  return `{ ${parts.join(', ')} }`;
}

function printArray(elements: readonly ArrayElement[], depth: number): string {
  const parts = elements.map((e) => e.spread ? `...${printExpr(e.value, depth)}` : printExpr(e.value, depth));
  return `[${parts.join(', ')}]`;
}

function printPattern(p: Pattern): string {
  switch (p.kind) {
    case 'name': return p.name;
    case 'objectPattern': return p.fields.length === 0 ? '{}' : `{ ${p.fields.join(', ')} }`;
    case 'arrayPattern': return `[${p.elements.join(', ')}]`;
  }
}

/** Print a `{ … }` block of statements at the given indent depth. An empty block is `{}`. */
export function printBlock(body: readonly Stmt[], depth: number): string {
  if (body.length === 0) return '{}';
  const inner = body.map((st) => INDENT.repeat(depth + 1) + printStmt(st, depth + 1)).join('\n');
  return `{\n${inner}\n${INDENT.repeat(depth)}}`;
}

/**
 * Print a single statement node to its canonical DSL surface syntax.
 *
 * @param stmt - the statement AST node to render (a declaration, assignment, control-flow form, or a
 *               bare expression statement).
 * @param depth - the current indentation depth; nested `{ … }` blocks (via {@link printBlock}) indent
 *                one level deeper than this.
 * @returns the source text for `stmt`, without a trailing newline.
 */
export function printStmt(stmt: Stmt, depth: number): string {
  switch (stmt.kind) {
    case 'const': return `const ${stmt.name} = ${printExpr(stmt.init, depth)}`;
    case 'let': return `let ${stmt.name} = ${printExpr(stmt.init, depth)}`;
    case 'assign': return `${printExpr(stmt.target, depth)} = ${printExpr(stmt.value, depth)}`;
    case 'function': return `function ${stmt.name}(${stmt.params.map(printPattern).join(', ')}) ${printBlock(stmt.body, depth)}`;
    case 'component': return `component ${stmt.name}(${stmt.params.map(printPattern).join(', ')}) ${printBlock(stmt.body, depth)}`;
    case 'if': {
      const head = `if (${printExpr(stmt.test, depth)}) ${printBlock(stmt.then, depth)}`;
      return stmt.else ? `${head} else ${printBlock(stmt.else, depth)}` : head;
    }
    case 'for': return `for (const ${stmt.binding} of ${printExpr(stmt.iterable, depth)}) ${printBlock(stmt.body, depth)}`;
    case 'while': return `while (${printExpr(stmt.test, depth)}) ${printBlock(stmt.body, depth)}`;
    case 'return': return stmt.value === undefined ? 'return' : `return ${printExpr(stmt.value, depth)}`;
    case 'expr': return printExpr(stmt.expr, depth);
  }
}

/**
 * Print a whole program back to canonical DSL surface syntax — the inverse of parsing.
 *
 * @param program - the parsed program whose top-level statements are rendered.
 * @returns the program source: each top-level statement printed at depth `0`, joined by newlines.
 * @remarks Round-trips through the parser up to spans: `stripSpans(parseProgram(printProgram(ast)).program)`
 *          deep-equals `stripSpans(ast)`. Reprinting shifts byte offsets, so the equivalence is checked
 *          with {@link stripSpans}; defensive parentheses the printer adds reparse away.
 * @throws {@link PrintDepthError} when the AST nests past {@link MAX_PRINT_DEPTH}.
 */
export function printProgram(program: Program): string {
  return program.stmts.map((st) => printStmt(st, 0)).join('\n');
}
