// The eval-free tree-walking evaluator. Asserts the fuel/deadline/depth budgets + the 4-layer
// FORBIDDEN_KEYS guard + the MAX_STRING_LENGTH cap over the JS/ES grammar.
// Total & non-throwing: author errors, budget/recursion limits,
// unknown calls, and forbidden-key access all become Diagnostics + a safe `null` — nothing
// escapes into the host. Reactive `let` read/write route through the ReactiveHost cells (NOT
// Environment.assign); unbound calls dispatch to the HostEnvironment builtin.
import type { Diagnostic } from './diagnostics.ts';
import { makeDiagnostic } from './diagnostics.ts';
import type { Expr, Stmt, Pattern } from './ast.ts';
import { FORBIDDEN_KEYS } from './ast.ts';
import { parseProgram } from './parser.ts';
import { Environment } from './environment.ts';
import type { HostEnvironment, ReactiveHost, CellRef } from './ports.ts';

export const DEFAULT_MAX_STEPS = 100_000;
export const DEFAULT_MAX_TIME_MS = 1000;
export const DEFAULT_MAX_DEPTH = 64;
export const MAX_STRING_LENGTH = 10_000_000;
const TIME_CHECK_INTERVAL = 1024;

export interface EvalOptions {
  data?: unknown;
  seed?: number;
  host: ReactiveHost;
  env: HostEnvironment;
  maxSteps?: number;
  maxTimeMs?: number;
  maxDepth?: number;
  maxStringLength?: number;    // string-growth cap on `+` (default MAX_STRING_LENGTH) — testable small (F3)
  insideComponent?: boolean;   // gates reactive `let`
}
export interface EvalResult { value: unknown; diagnostics: Diagnostic[] }

/** Thrown INTERNALLY only, never escapes evaluateProgram; carries a `return` value up the stack.
 *  EXPORTED for the runtime derive so its top-level catch can recognize a bubbled `return`. */
export class ReturnSignal { constructor(readonly value: unknown) {} }
/** Thrown INTERNALLY only; signals the step/time/depth budget was exhausted — caught at the top.
 *  EXPORTED for the runtime derive so the derive walk fails closed with ML-LANG-BUDGET, never hangs. */
export class BudgetSignal extends Error { constructor(readonly reason: string) { super(reason); } }

/** A user-defined `function`/`component` DECLARATION. A structured object (not a JS closure) so the
 *  runtime derive can introspect a `component` to collect its children. `isComponent` runs the body
 *  with insideComponent=true so its `let`s are reactive; a `function` runs pure. Invoked via
 *  callUserFn (implicit-last-expr return; depth cap). Arrows are real JS closures instead.
 *  EXPORTED for the runtime derive — it introspects a `component` UserFn to child-collect its body. */
export interface UserFn {
  readonly __mlFn: true;
  readonly name: string;         // the declared name — lowering uses it as the node head (e.g. 'KPI')
  readonly params: Pattern[];
  readonly body: Stmt[];
  readonly closure: Environment;
  readonly isComponent: boolean;
}
export function isUserFn(v: unknown): v is UserFn {
  return typeof v === 'object' && v !== null && (v as { __mlFn?: unknown }).__mlFn === true;
}

/** An arrow closure carries its AST (params + body + captured env) so the runtime derive can RE-LOWER
 *  it as a node-producer in child position (a `renderItem: (r) => KPI(r)` render-prop): plainly
 *  evaluating it would hit the node-in-expression-position rejection, so the derive re-walks the body. */
export interface ArrowInfo { readonly params: Pattern[]; readonly body: Expr | Stmt[]; readonly env: Environment }
export function arrowInfo(v: unknown): ArrowInfo | undefined {
  return typeof v === 'function' ? (v as { __mlArrow?: ArrowInfo }).__mlArrow : undefined;
}

/** EXPORTED for the runtime derive: the shared budget/diagnostics context threaded through both the
 *  evaluator and the derive walk. `resolveOptions` builds the defaulted option bag `Runner` needs. */
export function resolveOptions(options: EvalOptions): Required<Pick<EvalOptions, 'maxSteps' | 'maxTimeMs' | 'maxDepth' | 'maxStringLength'>> & EvalOptions {
  return { maxSteps: DEFAULT_MAX_STEPS, maxTimeMs: DEFAULT_MAX_TIME_MS, maxDepth: DEFAULT_MAX_DEPTH, maxStringLength: MAX_STRING_LENGTH, ...options };
}

export class Runner {
  steps = 0;
  depth = 0;
  readonly deadline: number;
  readonly diagnostics: Diagnostic[] = [];
  /** The key of the component INSTANCE currently being child-collected (set by the runtime derive's
   *  instantiateComponent, save/restored for nesting). Empty at the top level / outside a component.
   *  Combined with a `let`'s name + occurrence ordinal it forms the STABLE cell identity that latches
   *  reactive state S across a re-derive (a surviving instance keeps its state; a new one resets). */
  currentComponentKey = '';
  /** Per-current-instance occurrence counter for reactive-`let` NAMES, so `let n` allocated more than
   *  once within ONE instance (e.g. a loop body) gets distinct suffixes (#0, #1, …). Reset per instance
   *  by instantiateComponent. Canonical top-of-component `let`s each hit #0 and latch perfectly. */
  letOccurrences = new Map<string, number>();
  constructor(readonly opt: Required<Pick<EvalOptions, 'maxSteps' | 'maxTimeMs' | 'maxDepth' | 'maxStringLength'>> & EvalOptions) {
    this.deadline = Date.now() + opt.maxTimeMs;
  }
  tick(): void {
    if (++this.steps > this.opt.maxSteps) throw new BudgetSignal('steps');
    if (this.steps % TIME_CHECK_INTERVAL === 0 && Date.now() > this.deadline) throw new BudgetSignal('time');
  }
  error(code: string, message: string, span?: Expr['span']): void {
    this.diagnostics.push(makeDiagnostic(code, message, span));
  }
}

// ─────────────────────────────────────────── expressions ───────────────────────────────────────────

export function evalExpr(expr: Expr, env: Environment, r: Runner): unknown {
  r.tick();
  switch (expr.kind) {
    case 'number': return expr.value;
    case 'string': return expr.value;
    case 'bool': return expr.value;
    case 'null': return null;
    case 'ident': return evalIdent(expr, env, r);
    case 'member': {
      const object = evalExpr(expr.object, env, r);
      return readMember(object, expr.property, expr, r);
    }
    case 'index': {
      const object = evalExpr(expr.object, env, r);
      const key = evalExpr(expr.index, env, r);
      if (typeof key !== 'string' && typeof key !== 'number') {
        if (key === null || key === undefined) { r.error('ML-LANG-BAD-KEY', 'index key is null/undefined', expr.span); return null; }
        return readMember(object, String(key), expr, r);
      }
      return readMember(object, key, expr, r);
    }
    case 'object': {
      const out: Record<string, unknown> = {};
      for (const { key, value } of expr.entries) {
        if (FORBIDDEN_KEYS.has(key)) { r.error('ML-LANG-FORBIDDEN', `forbidden key '${key}'`, expr.span); continue; }
        out[key] = evalExpr(value, env, r);
      }
      return out;
    }
    case 'array': return expr.elements.map((e) => evalExpr(e, env, r));
    case 'unary': return evalUnary(expr, env, r);
    case 'binary': return evalBinary(expr, env, r);
    case 'cond': return truthy(evalExpr(expr.test, env, r)) ? evalExpr(expr.then, env, r) : evalExpr(expr.else, env, r);
    case 'arrow': return makeArrowClosure(expr, env, r);
    case 'call': return evalCall(expr, env, r);
    default: {
      const u = expr as { kind?: unknown; span?: Expr['span'] };
      r.error('ML-LANG-UNKNOWN-KIND', `unknown expression kind '${String(u.kind)}'`, u.span);
      return null;
    }
  }
}

/** Read an identifier. A reactive `let` binding reads through host.readCell (registers a dep when
 *  a leaf-effect scope is active); a `const`/param/`function` binding reads its stored value. An
 *  unbound identifier fails closed to null with a diagnostic. */
function evalIdent(expr: Extract<Expr, { kind: 'ident' }>, env: Environment, r: Runner): unknown {
  const name = expr.name;
  if (FORBIDDEN_KEYS.has(name)) { r.error('ML-LANG-FORBIDDEN', `forbidden identifier '${name}'`, expr.span); return null; }
  if (!env.has(name)) { r.error('ML-LANG-UNKNOWN-VAR', `unknown variable '${name}'`, expr.span); return null; }
  const meta = env.meta(name);
  if (meta?.kind === 'let' && meta.cell !== undefined) return r.opt.host.readCell(meta.cell as CellRef) ?? null;
  return env.get(name) ?? null;
}

function evalUnary(expr: Extract<Expr, { kind: 'unary' }>, env: Environment, r: Runner): unknown {
  const operand = evalExpr(expr.operand, env, r);
  switch (expr.op) {
    case '-': return -toNum(operand);
    case '!': return !truthy(operand);
    default: r.error('ML-LANG-UNKNOWN-OP', `unknown unary operator`, expr.span); return null;
  }
}

function evalBinary(expr: Extract<Expr, { kind: 'binary' }>, env: Environment, r: Runner): unknown {
  const op = expr.op;
  // short-circuit boolean operators — the right operand is NOT evaluated when the result is decided.
  if (op === '&&') { const l = evalExpr(expr.left, env, r); return truthy(l) ? evalExpr(expr.right, env, r) : l; }
  if (op === '||') { const l = evalExpr(expr.left, env, r); return truthy(l) ? l : evalExpr(expr.right, env, r); }

  const left = evalExpr(expr.left, env, r);
  const right = evalExpr(expr.right, env, r);
  switch (op) {
    case '+': {
      if (typeof left === 'string' || typeof right === 'string') {
        const ls = strOf(left), rs = strOf(right);
        // Fail CLOSED before allocating: a doubling loop grows a string exponentially while ticking
        // only per node, so the step budget can't bound it. String-cap is treated as a BUDGET case.
        if (ls.length + rs.length > r.opt.maxStringLength) {
          r.error('ML-LANG-BUDGET', `string result would exceed the ${r.opt.maxStringLength}-character limit`, expr.span);
          return null;
        }
        return ls + rs;
      }
      return toNum(left) + toNum(right);
    }
    case '-': return toNum(left) - toNum(right);
    case '*': return toNum(left) * toNum(right);
    case '/': { const b = toNum(right); if (b === 0) { r.error('ML-LANG-DIV-ZERO', 'division by zero', expr.span); return null; } return toNum(left) / b; }
    case '%': { const b = toNum(right); if (b === 0) { r.error('ML-LANG-DIV-ZERO', 'modulo by zero', expr.span); return null; } return toNum(left) % b; }
    case '==': return looseEquals(left, right);
    case '!=': return !looseEquals(left, right);
    case '<': return compare(left, right) < 0;
    case '<=': return compare(left, right) <= 0;
    case '>': return compare(left, right) > 0;
    case '>=': return compare(left, right) >= 0;
    default: r.error('ML-LANG-UNKNOWN-OP', `unknown binary operator`, expr.span); return null;
  }
}

/** An arrow becomes a REAL JS closure capturing the CURRENT Environment (net-new vs expr): invoking
 *  it binds args in a fresh child scope and runs the body over the interpreter's own budget machinery.
 *  An EXPRESSION body yields its value; a BLOCK body runs its statements for effect (a `hover = h`
 *  assignment routes through host.writeCell, driving reactivity) and yields its implicit last-expr
 *  value. Arrows run pure (insideComponent=false) and are opaque HostValues to callers (handler-prop
 *  arrows are stored, not auto-invoked). The depth cap applies on invocation. */
function makeArrowClosure(expr: Extract<Expr, { kind: 'arrow' }>, closure: Environment, r: Runner): (...args: unknown[]) => unknown {
  const body = expr.body;
  const fn = (...args: unknown[]): unknown => {
    if (++r.depth > r.opt.maxDepth) { r.depth--; throw new BudgetSignal('depth'); }
    const frame = new Environment(closure);
    bindParams(expr.params, args, frame);
    try {
      return Array.isArray(body) ? execBlockValue(body, frame, r, false) : evalExpr(body, frame, r);
    } catch (sig) {
      if (sig instanceof ReturnSignal) return sig.value;
      throw sig;
    } finally {
      r.depth--;
    }
  };
  // Tag the closure with its AST so the runtime derive can re-lower it as a node-producer (render-prop).
  (fn as { __mlArrow?: ArrowInfo }).__mlArrow = { params: expr.params, body, env: closure };
  return fn;
}

/** CALL RESOLUTION ORDER:
 *   1. callee resolves to an Environment-bound callable → invoke it (implicit-last-expr; depth cap).
 *   2. else an unbound head → dispatch to the HostEnvironment builtin (env.resolveCall).
 *   3. else → fail closed with ML-LANG-UNKNOWN-CALL + null. */
function evalCall(expr: Extract<Expr, { kind: 'call' }>, env: Environment, r: Runner): unknown {
  const callee = expr.callee;
  // (1) an ident bound to a user callable, OR any callee expression that evaluates to one.
  if (callee.kind === 'ident') {
    const name = callee.name;
    if (env.has(name)) {
      const bound = env.get(name);
      if (isUserFn(bound)) return callUserFn(bound, expr.args.map((a) => evalExpr(a, env, r)), r);
      if (typeof bound === 'function') return invokeClosure(bound as (...a: unknown[]) => unknown, expr, env, r);
      // A bound-but-not-callable name in call position is not a valid call → fail closed.
    } else {
      // (2) unbound head → HostEnvironment builtin dispatch (rand/range/node heads in expr position).
      return dispatchBuiltin(name, expr, env, r);
    }
  } else {
    const target = evalExpr(callee, env, r);
    if (isUserFn(target)) return callUserFn(target, expr.args.map((a) => evalExpr(a, env, r)), r);
    if (typeof target === 'function') return invokeClosure(target as (...a: unknown[]) => unknown, expr, env, r);
  }
  // (3) unresolved call in expression position → fail closed, never throw.
  const head = callee.kind === 'ident' ? callee.name : '<expr>';
  r.error('ML-LANG-UNKNOWN-CALL', `unknown call '${head}'`, expr.span);
  return null;
}

/** Invoke a JS-closure callable (an arrow). Its own depth/budget guarding lives in the closure. */
function invokeClosure(fn: (...a: unknown[]) => unknown, expr: Extract<Expr, { kind: 'call' }>, env: Environment, r: Runner): unknown {
  return fn(...expr.args.map((a) => evalExpr(a, env, r)));
}

function dispatchBuiltin(head: string, expr: Extract<Expr, { kind: 'call' }>, env: Environment, r: Runner): unknown {
  const args = expr.args.map((a) => evalExpr(a, env, r));
  // Wrap each evaluated value into the Arg shape. lang's evaluator does NOT populate name/reactive
  // here — that is a @metael/runtime (derive) responsibility; it only carries the value.
  const resolved = r.opt.env.resolveCall(head, '', args.map((value) => ({ value })), [], expr.span);
  if (resolved.handled) {
    // A SCALAR builtin (rand/range/double) resolves to a scalar — a primitive or an array. A
    // non-array OBJECT means the host built a NODE (it is a node head, e.g. text/layout, or a
    // permissive node-builder env answered an unknown head): a node is only valid in CHILD
    // position (collected by the runtime derive), NOT in scalar expression position — fail closed here.
    const v = resolved.value ?? null;
    if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
      r.error('ML-LANG-UNKNOWN-CALL', `unknown call '${head}' (node head is not valid in expression position)`, expr.span);
      return null;
    }
    return v;
  }
  r.error('ML-LANG-UNKNOWN-CALL', `unknown call '${head}'`, expr.span);
  return null;
}

function callUserFn(fn: UserFn, args: unknown[], r: Runner): unknown {
  if (++r.depth > r.opt.maxDepth) { r.depth--; throw new BudgetSignal('depth'); }
  const frame = new Environment(fn.closure);
  bindParams(fn.params, args, frame);
  try {
    return execBlockValue(fn.body, frame, r, fn.isComponent);
  } catch (sig) {
    if (sig instanceof ReturnSignal) return sig.value;
    throw sig;   // BudgetSignal propagates to the top
  } finally {
    r.depth--;
  }
}

/** Bind positional args into `frame` per each param pattern (name / object-destructure / array-destructure).
 *  EXPORTED for the runtime derive: it binds a component's params before child-collecting its body. */
export function bindParams(params: Pattern[], args: unknown[], frame: Environment): void {
  params.forEach((p, i) => {
    const arg = i < args.length ? args[i] : null;
    if (p.kind === 'name') { frame.define(p.name, arg, { kind: 'const' }); return; }
    if (p.kind === 'objectPattern') {
      const obj = (arg !== null && typeof arg === 'object') ? arg as Record<string, unknown> : {};
      for (const f of p.fields) frame.define(f, obj[f] ?? null, { kind: 'const' });
      return;
    }
    // arrayPattern
    const arr = Array.isArray(arg) ? arg : [];
    p.elements.forEach((name, idx) => frame.define(name, arr[idx] ?? null, { kind: 'const' }));
  });
}

// ─────────────────────────────────────────── statements ───────────────────────────────────────────

/** Run a block for its VALUE: the implicit last-expression return. Declarations/effects yield
 *  null; the last `expr` statement's value is the block's value. A `return` inside throws ReturnSignal.
 *  A fresh child scope is used so block-local bindings don't leak. `insideComponent` gates reactive `let`. */
export function execBlockValue(body: Stmt[], parent: Environment, r: Runner, insideComponent: boolean): unknown {
  const env = new Environment(parent);
  let last: unknown = null;
  for (const stmt of body) last = execStmt(stmt, env, r, insideComponent);
  return last;
}

/** Execute one statement; returns its VALUE (for implicit-last-expr). Non-expr statements return null.
 *  EXPORTED for the runtime derive: it runs a component body's non-node statements (const/let/assign/
 *  function decls) for effect while child-collecting. */
export function execStmt(stmt: Stmt, env: Environment, r: Runner, insideComponent: boolean): unknown {
  r.tick();
  switch (stmt.kind) {
    case 'expr': return evalExpr(stmt.expr, env, r);
    case 'const': {
      if (env.hasOwn(stmt.name)) { r.error('ML-LANG-REDECL', `'${stmt.name}' already declared`, stmt.span); return null; }
      env.define(stmt.name, evalExpr(stmt.init, env, r), { kind: 'const' });
      return null;
    }
    case 'let': {
      // Reactive state is component-scope-only. Outside a component → ML-LANG-LET-SCOPE.
      if (!insideComponent) { r.error('ML-LANG-LET-SCOPE', `reactive 'let' is only allowed inside a component`, stmt.span); return null; }
      // Redeclaration guard (parser only dedups the top level; block bodies must check here too, else
      // a `const x; let x` silently converts the const cell to a reactive let — a const-immutability bypass).
      if (env.hasOwn(stmt.name)) { r.error('ML-LANG-REDECL', `'${stmt.name}' already declared`, stmt.span); return null; }
      // Compute the STABLE cell key = component-instance key + let name + per-instance occurrence
      // ordinal, so a host can latch this cell's settled value across a re-derive (Task-6). No instance
      // key (top-level / non-component context) → undefined → the host always uses the initializer.
      // EDGE: if the SAME name is `let`-declared more than once within one instance (e.g. inside a loop
      // body), the occurrence ordinal keeps them distinct within a pass but may not carry perfectly
      // across a data-driven control-flow change (row count shifts the ordinals). Canonical usage is a
      // top-of-component `let`, which always hits #0 and latches correctly.
      let cellKey: string | undefined;
      if (r.currentComponentKey) {
        const occ = r.letOccurrences.get(stmt.name) ?? 0;
        r.letOccurrences.set(stmt.name, occ + 1);
        cellKey = `${r.currentComponentKey}::${stmt.name}#${occ}`;
      }
      const cell = r.opt.host.allocateCell(evalExpr(stmt.init, env, r), cellKey);
      env.define(stmt.name, undefined, { kind: 'let', cell });
      return null;
    }
    case 'assign': execAssign(stmt, env, r); return null;
    case 'function': {
      if (env.hasOwn(stmt.name)) { r.error('ML-LANG-REDECL', `'${stmt.name}' already declared`, stmt.span); return null; }
      env.define(stmt.name, { __mlFn: true, name: stmt.name, params: stmt.params, body: stmt.body, closure: env, isComponent: false } satisfies UserFn, { kind: 'const' });
      return null;
    }
    case 'component': {
      if (env.hasOwn(stmt.name)) { r.error('ML-LANG-REDECL', `'${stmt.name}' already declared`, stmt.span); return null; }
      env.define(stmt.name, { __mlFn: true, name: stmt.name, params: stmt.params, body: stmt.body, closure: env, isComponent: true } satisfies UserFn, { kind: 'const' });
      return null;
    }
    case 'if': {
      if (truthy(evalExpr(stmt.test, env, r))) execBlockValue(stmt.then, env, r, insideComponent);
      else if (stmt.else) execBlockValue(stmt.else, env, r, insideComponent);
      return null;
    }
    case 'for': {
      const iter = evalExpr(stmt.iterable, env, r);
      if (!Array.isArray(iter)) { r.error('ML-LANG-FOR-ITER', 'for-of expects an array to iterate', stmt.span); return null; }
      for (const item of iter) {
        r.tick();
        const loopEnv = new Environment(env);
        loopEnv.define(stmt.binding, item, { kind: 'const' });
        execBlockValue(stmt.body, loopEnv, r, insideComponent);
      }
      return null;
    }
    case 'while': {
      // Each iteration ticks the budget so an unbounded loop fails closed (ML-LANG-BUDGET), never hangs.
      while (truthy(evalExpr(stmt.test, env, r))) {
        r.tick();
        execBlockValue(stmt.body, env, r, insideComponent);
      }
      return null;
    }
    case 'return': throw new ReturnSignal(stmt.value ? evalExpr(stmt.value, env, r) : null);
    default: {
      const u = stmt as { kind?: unknown; span?: Expr['span'] };
      r.error('ML-LANG-UNIMPL', `statement '${String(u.kind)}' not supported`, u.span);
      return null;
    }
  }
}

/** Assignment. A simple ident target: to a reactive `let` → host.writeCell (NEVER Environment.assign);
 *  to a `const` → ML-LANG-CONST. A member/index target: guarded in-place write on the container, with
 *  the computed-key FORBIDDEN guard. Fails closed on an unbound / non-object target. */
function execAssign(stmt: Extract<Stmt, { kind: 'assign' }>, env: Environment, r: Runner): void {
  const target = stmt.target;
  const value = evalExpr(stmt.value, env, r);

  if (target.kind === 'ident') {
    const name = target.name;
    if (!env.has(name)) { r.error('ML-LANG-UNBOUND-ASSIGN', `cannot assign to undeclared variable '${name}'`, stmt.span); return; }
    const meta = env.meta(name);
    if (meta?.kind === 'let') {
      // Reactive-let write routes through the ReactiveHost cell so dependents are scheduled.
      if (meta.cell !== undefined) r.opt.host.writeCell(meta.cell as CellRef, value);
      else env.assign(name, value);   // defensive: a let without a cell (shouldn't happen)
      return;
    }
    // const (or a param, which is const-bound) → immutable.
    r.error('ML-LANG-CONST', `cannot reassign const '${name}'`, stmt.span);
    return;
  }

  if (target.kind === 'member') {
    if (FORBIDDEN_KEYS.has(target.property)) { r.error('ML-LANG-FORBIDDEN', `forbidden key '${target.property}'`, stmt.span); return; }
    const container = evalExpr(target.object, env, r);
    writeMember(container, target.property, value, stmt.span, r);
    return;
  }

  if (target.kind === 'index') {
    const container = evalExpr(target.object, env, r);
    const key = evalExpr(target.index, env, r);
    // COMPUTED-key guard: the parse-time guard only covers literal `.k`/`["lit"]`; guard the case
    // where a computed key EVALUATES to a forbidden string here.
    if (typeof key === 'string' && FORBIDDEN_KEYS.has(key)) { r.error('ML-LANG-FORBIDDEN', `forbidden key '${key}'`, stmt.span); return; }
    if (typeof key !== 'string' && typeof key !== 'number') { writeMember(container, String(key), value, stmt.span, r); return; }
    writeMember(container, key, value, stmt.span, r);
    return;
  }

  r.error('ML-LANG-BAD-LVALUE', 'unsupported assignment target', stmt.span);
}

/** Guarded in-place write. On an array, only a canonical in-bounds index is written (no `.length`
 *  mutation, no sparse holes). On a plain object any non-forbidden key is set. Else fails closed. */
function writeMember(container: unknown, key: string | number, value: unknown, span: Expr['span'], r: Runner): void {
  if (typeof key === 'string' && FORBIDDEN_KEYS.has(key)) { r.error('ML-LANG-FORBIDDEN', `forbidden key '${key}'`, span); return; }
  if (Array.isArray(container)) {
    let idx: number | null = null;
    if (typeof key === 'number') idx = key;
    else { const n = Number(key); if (String(n) === key) idx = n; }
    if (idx !== null && Number.isInteger(idx) && idx >= 0 && idx < container.length) container[idx] = value;
    else r.error('ML-LANG-INDEX-RANGE', `index ${String(key)} out of range`, span);
    return;
  }
  if (container !== null && typeof container === 'object') { (container as Record<string, unknown>)[String(key)] = value; return; }
  r.error('ML-LANG-ASSIGN-INDEX', 'cannot assign into a non-object/array', span);
}

// ─────────────────────────────────────────── member read + coercions ───────────────────────────────────────────

function readMember(container: unknown, key: string | number, expr: Expr, r: Runner): unknown {
  if (typeof key === 'string' && FORBIDDEN_KEYS.has(key)) { r.error('ML-LANG-FORBIDDEN', `forbidden path segment '${key}'`, expr.span); return null; }
  if (container === null || container === undefined) return null;
  if (Array.isArray(container)) {
    if (typeof key === 'number') {
      if (!Number.isInteger(key) || key < 0 || key >= container.length) { r.error('ML-LANG-INDEX-RANGE', `index ${key} is out of range (length ${container.length})`, expr.span); return null; }
      return container[key] ?? null;
    }
    if (key === 'length') return container.length;
  }
  if (typeof container === 'object') {
    const k = String(key);
    if (Object.prototype.hasOwnProperty.call(container, k)) return (container as Record<string, unknown>)[k] ?? null;
    return null;
  }
  if (typeof container === 'string') {
    if (typeof key === 'number') { if (key < 0 || key >= container.length) { r.error('ML-LANG-INDEX-RANGE', `index ${key} is out of range (length ${container.length})`, expr.span); return null; } return container[key]; }
    if (key === 'length') return container.length;
  }
  return null;
}

export function truthy(v: unknown): boolean {
  return !(v === false || v === null || v === undefined || v === 0 || v === '' || (typeof v === 'number' && Number.isNaN(v)));
}
function toNum(v: unknown): number {
  if (typeof v === 'number') return v;
  if (typeof v === 'boolean') return v ? 1 : 0;
  if (typeof v === 'string') { const t = v.trim(); return t === '' ? NaN : Number(t); }
  return NaN;
}
function strOf(v: unknown): string {
  if (v === null || v === undefined) return '';
  if (typeof v === 'string') return v;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  try { return JSON.stringify(v) ?? ''; } catch { return ''; }
}
function looseEquals(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a === null || a === undefined) return b === null || b === undefined;
  if (b === null || b === undefined) return false;
  if (typeof a === 'number' && typeof b === 'string') return a === toNum(b) && b.trim() !== '';
  if (typeof a === 'string' && typeof b === 'number') return toNum(a) === b && a.trim() !== '';
  return false;
}
function compare(a: unknown, b: unknown): number {
  if (typeof a === 'string' && typeof b === 'string') return a < b ? -1 : a > b ? 1 : 0;
  const na = toNum(a), nb = toNum(b);
  if (Number.isNaN(na) || Number.isNaN(nb)) return NaN;
  return na < nb ? -1 : na > nb ? 1 : 0;
}

// ─────────────────────────────────────────── program entry ───────────────────────────────────────────

/** Execute the top-level program. The top level is NOT a component (so a bare `let` → ML-LANG-LET-SCOPE);
 *  a `component` body executes with insideComponent=true, a `function` body with insideComponent=false.
 *  Returns the implicit last-expression value of the top-level block. */
function execProgram(stmts: Stmt[], root: Environment, r: Runner, insideComponent: boolean): unknown {
  let last: unknown = null;
  for (const stmt of stmts) last = execStmt(stmt, root, r, insideComponent);
  return last;
}

export function evaluateProgram(source: string, options: EvalOptions): EvalResult {
  const opt = {
    maxSteps: DEFAULT_MAX_STEPS, maxTimeMs: DEFAULT_MAX_TIME_MS, maxDepth: DEFAULT_MAX_DEPTH,
    maxStringLength: MAX_STRING_LENGTH, ...options,
  };
  const runner = new Runner(opt);
  const root = new Environment();
  if ('data' in options) root.define('data', options.data, { kind: 'const' });
  let value: unknown = null;
  try {
    // Parse INSIDE the try: a deeply-nested source can overflow the recursive-descent parser
    // (RangeError); nothing may escape into the host (module never-throw contract).
    const { program, diagnostics } = parseProgram(source);
    runner.diagnostics.push(...diagnostics);
    value = execProgram(program.stmts, root, runner, options.insideComponent ?? false); // top level is NOT a component
  } catch (e) {
    if (e instanceof BudgetSignal) runner.diagnostics.push(makeDiagnostic('ML-LANG-BUDGET', `evaluation budget exceeded: ${e.reason}`));
    else if (e instanceof ReturnSignal) value = e.value;   // a top-level `return` yields its value
    else runner.diagnostics.push(makeDiagnostic('ML-LANG-INTERNAL', String(e)));
  }
  return { value, diagnostics: runner.diagnostics };
}
