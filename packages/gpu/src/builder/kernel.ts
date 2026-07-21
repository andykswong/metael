// A TSL-style JS builder for whole metael kernels — statements and control flow, not just expressions.
// `kernel(fn)` runs `fn` ONCE with fresh param KNodes and CAPTURES the statements the helpers
// (`letVar`/`set`/`forRange`/`ifThen`/`ret`) record, assembling the SAME `component` AST the parser
// emits. Control flow that JS would execute is instead traced into a statement list — the standard
// builder-trace technique — so a builder-authored kernel inherits the existing gate/emit/oracle/dispatch
// path unchanged. Imports ONLY @metael/lang (AST types + `Environment`) and its `./node.ts` siblings —
// no runtime, no evaluator, no host.
import { Environment } from '@metael/lang';
import type { Stmt, Pattern, UserFn } from '@metael/lang';
import { KNode, param, toExpr } from './node.ts';

/** The zero span every builder-authored node carries. The builder has no source text, so there is
 *  nothing to point diagnostics at; equivalence with parser output is compared via `stripSpans`, which
 *  drops the `span` field, so the exact value never affects a compare. */
const ZERO_SPAN = { start: 0, end: 0 };

/** The fixed decl name for a builder-authored kernel. JS gives the builder only the callback's arity,
 *  not a name, so every builder kernel is named `K` (semantically irrelevant — a kernel head is the
 *  binding the engine dispatches, and its param names are just thread-coordinate bindings). */
const KERNEL_NAME = 'K';

/**
 * The ambient statement-capture stack. Each open scope (the kernel root, a `forRange` body, an `ifThen`
 * branch) pushes a fresh `Stmt[]` frame; a helper records into the top frame. A module-level stack (not
 * an argument threaded through every helper) is what lets the plain callbacks — `() => set(acc, …)` —
 * record without carrying a context object, exactly as the builder-trace technique intends.
 */
const frames: Stmt[][] = [];

/**
 * The current `forRange` nesting depth — 0 outside any loop, 1 inside the outermost, 2 inside a loop
 * nested in that, and so on. Used ONLY to mint a distinct loop-variable name per nesting level so a nested
 * `forRange` never shadows its parent: depth 0 → `i`, depth 1 → `i1`, depth 2 → `i2`, … A single
 * (non-nested) loop still gets `i` — the natural source name equivalence tests compare against. Mirrors the
 * `frames` stack: incremented on entering a loop body, decremented on leaving (via `try/finally`), so a
 * throwing body cannot leave the counter skewed for a later `forRange`.
 */
let forDepth = 0;

/** Append `stmt` to the currently open capture frame. Throws a clear error when no frame is open — i.e.
 *  a statement helper was called outside a `kernel(...)`/`kernelAst(...)` trace. */
function push(stmt: Stmt): void {
  const top = frames[frames.length - 1];
  if (top === undefined) {
    throw new Error('metael kernel builder: a statement helper (letVar/set/forRange/ifThen/ret) was called outside kernel(...) — open a kernel scope first.');
  }
  top.push(stmt);
}

/** Open a fresh capture frame, run `capture` (which records into it via the helpers), then close and
 *  return the frame. `try/finally` pops even if `capture` throws, so a failed trace never corrupts the
 *  stack for a later `kernel(...)` call. */
function captureFrame(capture: () => void): Stmt[] {
  const frame: Stmt[] = [];
  frames.push(frame);
  try {
    capture();
  } finally {
    frames.pop();
  }
  return frame;
}

/**
 * Declare a mutable `let` binding, e.g. `const acc = letVar('acc', lit(0))`.
 *
 * Records a `let` statement into the current frame and returns a {@link KNode} that references the new
 * binding, so the caller can read and reassign it (`set(acc, acc.add(i))`).
 * @param name The binding name.
 * @param init The initial value.
 * @returns A KNode referencing the binding.
 */
export function letVar(name: string, init: KNode): KNode {
  push({ kind: 'let', name, init: init.expr, span: ZERO_SPAN });
  return param(name);
}

/**
 * Assign to an existing binding or index/member target: `set(acc, acc.add(i))` → `acc = acc + i`.
 *
 * Records an `assign` statement into the current frame.
 * @param target The assignment target (a binding or index/member KNode).
 * @param value The value to assign.
 */
export function set(target: KNode, value: KNode): void {
  push({ kind: 'assign', target: target.expr, value: value.expr, span: ZERO_SPAN });
}

/**
 * Capture a `for i of range(n) { … }` loop: `forRange(4, (i) => set(acc, acc.add(i)))`.
 *
 * Opens a nested capture frame, invokes `body(param(loopVar))` ONCE to trace the loop body into it, then
 * records a `for` statement whose iterable is a `range(n)` call — structurally identical to the parser's
 * `for (const … of range(…))`. The loop variable is minted PER NESTING DEPTH so a nested loop never
 * shadows its parent: the outermost loop is `i`, one nested inside it `i1`, the next `i2`, … A single
 * (non-nested) loop is `i`, matching the natural source the equivalence tests compare against. Distinct
 * names matter because the CPU emitter runs the body over one flat scope with no per-loop child scope, so a
 * shadowed `i` would clobber the outer loop's value — the interpreter oracle (distinct names) would not.
 * @param n The iteration count — a KNode or a bare number (coerced to a numeric literal).
 * @param body Invoked once with a KNode for the loop variable; its recorded statements become the body.
 */
export function forRange(n: KNode | number, body: (i: KNode) => void): void {
  const loopVar = forDepth === 0 ? 'i' : `i${forDepth}`;
  const captured = captureFrame(() => {
    forDepth++;
    try {
      body(param(loopVar));
    } finally {
      forDepth--;
    }
  });
  push({
    kind: 'for',
    binding: loopVar,
    iterable: { kind: 'call', callee: { kind: 'ident', name: 'range', span: ZERO_SPAN }, args: [toExpr(n)], span: ZERO_SPAN },
    body: captured,
    span: ZERO_SPAN,
  });
}

/**
 * Capture an `if (cond) { then } [else { els }]` conditional.
 *
 * Opens a nested frame for `then` (and, if given, another for `els`), tracing each branch's statements.
 * When no `els` is passed the `else` key is OMITTED entirely (the parser omits it — an `else: undefined`
 * would break `stripSpans`-based equivalence with parsed output).
 * @param cond The test expression.
 * @param then Invoked once; its recorded statements become the `then` block.
 * @param els Optional; invoked once when present, its statements become the `else` block.
 */
export function ifThen(cond: KNode, then: () => void, els?: () => void): void {
  const thenBlock = captureFrame(then);
  if (els === undefined) {
    push({ kind: 'if', test: cond.expr, then: thenBlock, span: ZERO_SPAN });
  } else {
    const elseBlock = captureFrame(els);
    push({ kind: 'if', test: cond.expr, then: thenBlock, else: elseBlock, span: ZERO_SPAN });
  }
}

/**
 * Record an explicit `return value` statement, e.g. `ret(acc)`.
 *
 * Equivalent to the idiomatic arrow-return form (`kernel((row, col) => row.add(col))`): both emit a
 * `return` statement, which is the ONLY form the emitters lower to the kernel's output write. `ret(...)` is
 * for kernels whose body is a statement block (`kernel((p0) => { … ret(acc); })`) where the return is not
 * the callback's own trailing value.
 * @param value The value to return.
 */
export function ret(value: KNode): void {
  push({ kind: 'return', value: value.expr, span: ZERO_SPAN });
}

/**
 * Assemble a kernel's params + captured body from a trace callback. Shared by {@link kernel} and
 * {@link kernelAst} so both agree on param synthesis and return handling.
 *
 * Arity comes from `fn.length`; params are synthesized as positional names `p0`,`p1`,… (JS drops the
 * callback's real param names at runtime). The root frame is opened, `fn` is invoked once with the param
 * KNodes, and — if it returns a KNode — a `return` statement carrying that expression is appended. A value
 * returned from the JS callback IS a return, and `return` is the ONLY statement the emitters lower to the
 * kernel's output write, so this is the dispatchable mapping: `kernel((row, col) => row.add(col))` MEANS
 * "return row + col". (A trailing bare-expression statement — the parser's `{ … acc }` form — is discarded
 * by the emitters, so mapping the returned KNode to an `expr` stmt would silently dispatch all zeros.)
 */
function assemble(fn: (...params: KNode[]) => KNode | void): { params: Pattern[]; body: Stmt[] } {
  const arity = fn.length;
  const names = Array.from({ length: arity }, (_, i) => `p${i}`);
  const params: Pattern[] = names.map((name) => ({ kind: 'name', name }));
  const body = captureFrame(() => {
    const returned = fn(...names.map((name) => param(name)));
    if (returned instanceof KNode) {
      // A returned KNode is a return: emit a `return` stmt (the dispatchable form the emitters write),
      // NOT an `expr` stmt (a trailing bare expression, which the emitters discard → all-zeros output).
      push({ kind: 'return', value: returned.expr, span: ZERO_SPAN });
    }
  });
  return { params, body };
}

/**
 * Assemble the `component` DECL AST for a builder-authored kernel — the representation compared against
 * the parser's decl node for equivalence.
 *
 * @param fn The trace callback; its arity determines the params (`p0`,`p1`,…), and the statements its
 *   helpers record — plus a `return` stmt carrying the returned KNode, if it returns one — become the body.
 * @returns A `component` `Stmt` node named `K`.
 */
export function kernelAst(fn: (...params: KNode[]) => KNode | void): Stmt {
  const { params, body } = assemble(fn);
  return { kind: 'component', name: KERNEL_NAME, params, body, span: ZERO_SPAN };
}

/**
 * Build a kernel as a {@link UserFn} closure value — what the engine's `gpu(kernel, cfg)` consumes.
 *
 * Shares {@link assemble}'s param/body capture with {@link kernelAst}, then constructs the `UserFn` object
 * DIRECTLY (`evaluateProgram` takes a source string, not an assembled AST, so it cannot be routed
 * through — and the direct build keeps `/builder` on `@metael/lang` alone, no interpreter run). The
 * closure is a fresh empty {@link Environment}: a param-only kernel resolves its builtins via the
 * registry the engine injects at dispatch, not via a captured closure.
 * @param fn The trace callback; its arity determines the kernel's params.
 * @returns A `component` `UserFn` named `K`.
 */
export function kernel(fn: (...params: KNode[]) => KNode | void): UserFn {
  const { params, body } = assemble(fn);
  return { __mlFn: true, name: KERNEL_NAME, params, body, closure: new Environment(), isComponent: true };
}
