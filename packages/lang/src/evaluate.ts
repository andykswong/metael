// The eval-free tree-walking evaluator. Asserts the fuel/deadline/depth budgets + the 4-layer
// FORBIDDEN_KEYS guard + the MAX_STRING_LENGTH cap over the JS/ES grammar.
// Total & non-throwing: author errors, budget/recursion limits,
// unknown calls, and forbidden-key access all become Diagnostics + a safe `null` — nothing
// escapes into the host. Reactive `let` read/write route through the ReactiveHost cells (NOT
// Environment.assign); unbound calls dispatch to the HostEnvironment builtin.
import type { Diagnostic } from './diagnostics.ts';
import { makeDiagnostic } from './diagnostics.ts';
import type { Expr, Stmt, Pattern, BinOp } from './ast.ts';
import { FORBIDDEN_KEYS } from './ast.ts';
import { parseProgram } from './parser.ts';
import { Environment } from './environment.ts';
import { makeSeededRng, range as seededRange } from './determinism.ts';
import { IMPLEMENTED_BUILTINS } from './builtins-registry.ts';
import { defaultCompare, stableSort } from './sort.ts';
import type { HostEnvironment, ReactiveHost, CellRef } from './ports.ts';
import { descriptorOf, isCustomType, generationOf, isFrozenCustom, markFrozen, NOT_HANDLED, makeTypedArray, BUFFER_KINDS, BufferError, makeVec, makeMat, identityMat, vecStoreOf } from './custom-types.ts';

export const DEFAULT_MAX_STEPS = 100_000;
export const DEFAULT_MAX_TIME_MS = 1000;
export const DEFAULT_MAX_DEPTH = 64;
export const MAX_STRING_LENGTH = 10_000_000;
export const MAX_BUFFER_LENGTH = 16_777_216;   // 2^24 elements — a typed-array construction cap (ML-LANG-BUDGET over)
const TIME_CHECK_INTERVAL = 1024;

export interface EvalOptions {
  data?: unknown;
  seed?: number;
  host: ReactiveHost;
  env: HostEnvironment;
  maxSteps?: number;
  maxTimeMs?: number;
  maxDepth?: number;
  maxStringLength?: number;    // string-growth cap on `+` (default MAX_STRING_LENGTH) — testable small
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

export interface MakeCallableOpts {
  host: ReactiveHost; env: HostEnvironment;
  maxSteps?: number; maxTimeMs?: number; maxDepth?: number; maxStringLength?: number; seed?: number;
}
/** Build a plain JS callable that invokes a UserFn under a FRESH Runner (its own budget). Used to run a
 *  kernel per sampled coordinate as a top-level call — distinct from a reentrant in-interpretation call,
 *  which shares the ambient Runner. A raised maxSteps sizes the Runner for a large loop bound. */
export function makeCallable(fn: UserFn, opts: MakeCallableOpts): (...args: unknown[]) => unknown {
  const runner = new Runner(resolveOptions(opts));
  return (...args: unknown[]) => {
    try { return callUserFn(fn, args, runner); }
    catch (e) { if (e instanceof BudgetSignal || e instanceof ReturnSignal) return null; throw e; }
  };
}
/** Read a free identifier's resolved value from a UserFn's captured closure. A reactive `let` reads
 *  through host.readCell (registers a dep if a scope is active). Returns undefined if unbound. */
export function readClosureValue(fn: UserFn, name: string, host: ReactiveHost): unknown {
  const env = fn.closure;
  if (!env.has(name)) return undefined;
  const meta = env.meta(name);
  if (meta?.kind === 'let' && meta.cell !== undefined) return host.readCell(meta.cell as CellRef) ?? null;
  return env.get(name) ?? null;
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
  /** The seeded PRNG for the intrinsic rand()/range builtins. Seeded from EvalOptions.seed so
   *  `result = f(source, data, seed, state)` — same source + same seed → identical rand() sequence.
   *  A fresh Runner per run re-seeds identically, so a re-run is byte-stable. */
  readonly rng: () => number;
  constructor(readonly opt: Required<Pick<EvalOptions, 'maxSteps' | 'maxTimeMs' | 'maxDepth' | 'maxStringLength'>> & EvalOptions) {
    this.deadline = Date.now() + opt.maxTimeMs;
    this.rng = makeSeededRng(opt.seed ?? 0);
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
      for (const entry of expr.entries) {
        if (entry.spread) {
          const src = evalExpr(entry.value, env, r);
          if (src !== null && typeof src === 'object' && !Array.isArray(src)) {
            for (const [k, v] of Object.entries(src as Record<string, unknown>)) {
              if (!FORBIDDEN_KEYS.has(k)) out[k] = v;
            }
          } else r.error('ML-LANG-SPREAD', 'spread of a non-object in an object literal', expr.span);
          continue;
        }
        if (FORBIDDEN_KEYS.has(entry.key)) { r.error('ML-LANG-FORBIDDEN', `forbidden key '${entry.key}'`, expr.span); continue; }
        out[entry.key] = evalExpr(entry.value, env, r);
      }
      return deepFreeze(out);
    }
    case 'array': {
      const out: unknown[] = [];
      for (const el of expr.elements) {
        const val = evalExpr(el.value, env, r);
        if (el.spread) {
          if (Array.isArray(val)) out.push(...val);
          else r.error('ML-LANG-SPREAD', 'spread of a non-array in an array literal', expr.span);
        } else out.push(val);
      }
      return deepFreeze(out);
    }
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
    case '-': return isCustomType(operand) ? evalCustomNeg(operand, expr, r) : -toNum(operand);
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
  // Custom-type dispatch fires when an operand carries a descriptor — EXCEPT string `+`, which is a
  // primitive fast path: concatenation coerces a custom operand through its bounded `display` (strOf),
  // matching scalar string-coercion. So `"" + custom` yields the display string, not a dispatched op.
  const stringConcat = op === '+' && (typeof left === 'string' || typeof right === 'string');
  if (!stringConcat && (isCustomType(left) || isCustomType(right))) return evalCustomBinary(op, left, right, expr, r);
  switch (op) {
    case '+': {
      if (typeof left === 'string' || typeof right === 'string') {
        // Subscribe a reactive read to a custom operand's in-place mutation when it is stringified: strOf
        // does not receive the Runner (it is also called from the `join` builtin), so we register the
        // generation dep HERE — the only string-concat site where `r` is in scope — so `"" + buf` in a UI
        // re-renders on an in-place write, matching how readMember subscribes on element access.
        for (const operand of [left, right]) { const g = generationOf(operand); if (g !== undefined) r.opt.host.readGeneration(g); }
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

// The six non-square matrix constructors, name → [rows, cols]. A matCxR builds C columns of R rows, so
// its stored shape is R rows × C cols. Column-major: the flat args fill column 0 top-to-bottom, then
// column 1, and so on.
const NON_SQUARE_MAT: Readonly<Record<string, readonly [number, number]>> = {
  mat2x3: [3, 2], mat2x4: [4, 2], mat3x2: [2, 3], mat3x4: [4, 3], mat4x2: [2, 4], mat4x3: [3, 4],
};

// The determinant of an n×n column-major matrix (n = 2, 3, or 4). Flat element (row, col) lives at
// index col*n + row. Hardcoded cofactor expansions keep the result stable and match the shader targets'
// native determinant().
function matDeterminant(c: number[], n: number): number {
  const e = (row: number, col: number): number => c[col * n + row] as number;
  if (n === 2) return e(0, 0) * e(1, 1) - e(0, 1) * e(1, 0);
  if (n === 3) {
    return e(0, 0) * (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1))
         - e(0, 1) * (e(1, 0) * e(2, 2) - e(1, 2) * e(2, 0))
         + e(0, 2) * (e(1, 0) * e(2, 1) - e(1, 1) * e(2, 0));
  }
  // n === 4 — cofactor expansion along the first row, each 3×3 minor expanded inline.
  const m3 = (r0: number, r1: number, r2: number, c0: number, c1: number, c2: number): number =>
      e(r0, c0) * (e(r1, c1) * e(r2, c2) - e(r1, c2) * e(r2, c1))
    - e(r0, c1) * (e(r1, c0) * e(r2, c2) - e(r1, c2) * e(r2, c0))
    + e(r0, c2) * (e(r1, c0) * e(r2, c1) - e(r1, c1) * e(r2, c0));
  return e(0, 0) * m3(1, 2, 3, 1, 2, 3)
       - e(0, 1) * m3(1, 2, 3, 0, 2, 3)
       + e(0, 2) * m3(1, 2, 3, 0, 1, 3)
       - e(0, 3) * m3(1, 2, 3, 0, 1, 2);
}

// The determinant of a k×k column-major matrix (flat, element (row, col) at index col*k+row), by cofactor
// expansion along the first column. k=1 is the scalar; k=2 is the direct 2×2 formula; larger k recurses on
// the (k-1)×(k-1) minors. General over k so it serves BOTH the top-level inverse determinant AND the smaller
// minors the adjugate needs (n−1 = 1/2/3 for n = 2/3/4).
function detFlat(m: number[], k: number): number {
  if (k === 1) return m[0] as number;
  if (k === 2) return (m[0] as number) * (m[3] as number) - (m[2] as number) * (m[1] as number);   // col-major [a,b,c,d] → a*d − c*b
  let sum = 0;
  for (let row = 0; row < k; row++) {
    const sign = row % 2 === 0 ? 1 : -1;                    // cofactor sign along column 0
    sum += sign * (m[row] as number) * detFlat(subMat(m, k, row, 0), k - 1);   // element (row, 0) lives at flat index row
  }
  return sum;
}
// The (k−1)×(k−1) column-major submatrix of a k×k column-major matrix, formed by deleting row `dr` and column
// `dc`. Iterating remaining columns (outer) then remaining rows (inner) pushes elements in column-major order.
function subMat(m: number[], k: number, dr: number, dc: number): number[] {
  const out: number[] = [];
  for (let col = 0; col < k; col++) {
    if (col === dc) continue;
    for (let row = 0; row < k; row++) {
      if (row === dr) continue;
      out.push(m[col * k + row] as number);
    }
  }
  return out;
}
// The inverse of an n×n column-major matrix (n = 2, 3, or 4) via the closed-form adjugate/determinant:
// inv[i][j] = cofactor(j, i) / det = ((−1)^(i+j) · minor(j, i)) / det, where minor(j, i) is the determinant of
// the submatrix with row j and column i deleted. The result is stored column-major (element (row=i, col=j) at
// flat index j*n+i). If det ≈ 0 (a singular matrix) the elements are ±Inf/NaN — undefined, matching the shader
// targets' native behavior (GLSL/WGSL neither guard nor define a singular inverse); no special guard is added.
function matInverse(c: number[], n: number): number[] {
  const det = detFlat(c, n);
  const out = new Array<number>(n * n);
  for (let i = 0; i < n; i++) {          // i = row of the inverse
    for (let j = 0; j < n; j++) {        // j = column of the inverse
      const sign = (i + j) % 2 === 0 ? 1 : -1;
      const minor = detFlat(subMat(c, n, j, i), n - 1);
      out[j * n + i] = (sign * minor) / det;
    }
  }
  return out;
}

function dispatchBuiltin(head: string, expr: Extract<Expr, { kind: 'call' }>, env: Environment, r: Runner): unknown {
  // Intrinsic seeded builtins — resolved BEFORE the host so determinism-with-randomness is a language
  // guarantee, not a per-domain re-implementation. Only fires for the UNBOUND-head path (a user
  // `function rand`/`range` shadows via evalCall's callee-resolution step, which never reaches here).
  if (head === 'f32' || head === 'f64' || head === 'i32' || head === 'u32') {
    r.tick();
    const kind = head as keyof typeof BUFFER_KINDS;
    const spec = BUFFER_KINDS[kind];
    const a0 = evalExpr(expr.args[0] ?? { kind: 'null', span: expr.span }, env, r);
    let store: { [i: number]: number; length: number };
    if (typeof a0 === 'number') {
      const n = Math.floor(a0);
      if (!Number.isFinite(n) || n < 0) { r.error('ML-LANG-BUILTIN-ARG', `${head}(n) — n must be a non-negative number`, expr.span); return deepFreeze([]); }
      if (n > MAX_BUFFER_LENGTH) { r.error('ML-LANG-BUDGET', `${head}(${n}) exceeds the ${MAX_BUFFER_LENGTH}-element cap`, expr.span); return deepFreeze([]); }
      store = new spec.ctor(n);
      const genFn = expr.args[1] ? evalExpr(expr.args[1], env, r) : undefined;
      if (genFn !== undefined) {
        const call = typeof genFn === 'function' ? (i: number) => (genFn as (...xs: unknown[]) => unknown)(i) : isUserFn(genFn) ? (i: number) => callUserFn(genFn, [i], r) : null;
        if (!call) { r.error('ML-LANG-BUILTIN-ARG', `${head}(n, fn) — the second argument must be a function`, expr.span); return deepFreeze([]); }
        for (let i = 0; i < n; i++) { r.tick(); const val = call(i); store[i] = spec.coerce(typeof val === 'number' ? val : NaN); }
      }
    } else if (Array.isArray(a0)) {
      if (a0.length > MAX_BUFFER_LENGTH) { r.error('ML-LANG-BUDGET', `${head}([…]) exceeds the ${MAX_BUFFER_LENGTH}-element cap`, expr.span); return deepFreeze([]); }
      store = new spec.ctor(a0.length);
      for (let i = 0; i < a0.length; i++) { r.tick(); const val = a0[i]; store[i] = spec.coerce(typeof val === 'number' ? val : NaN); }
    } else {
      r.error('ML-LANG-BUILTIN-ARG', `${head}(n | […] | (n, fn)) — bad argument`, expr.span);
      return deepFreeze([]);
    }
    const gen = r.opt.host.allocateGeneration();
    return makeTypedArray(kind, store, gen);
  }
  if (head === 'vec2' || head === 'vec3' || head === 'vec4') {
    r.tick();
    const n = head === 'vec2' ? 2 : head === 'vec3' ? 3 : 4;
    const a = expr.args.map((x) => evalExpr(x, env, r));
    if (a.some((x) => typeof x !== 'number')) { r.error('ML-LANG-BUILTIN-ARG', `${head}(numbers) — all components must be numbers`, expr.span); return deepFreeze([]); }
    const nums = a as number[];
    if (nums.length === 1) return makeVec(new Array<number>(n).fill(nums[0] as number));   // splat
    if (nums.length !== n) { r.error('ML-LANG-BUILTIN-ARG', `${head} needs 1 (splat) or ${n} components`, expr.span); return deepFreeze([]); }
    return makeVec(nums);
  }
  if (Object.hasOwn(NON_SQUARE_MAT, head)) {   // own-property check: a prototype-inherited head (constructor/toString/…) is NOT a ctor
    r.tick();
    const [rows, cols] = NON_SQUARE_MAT[head]!;
    const a = expr.args.map((x) => evalExpr(x, env, r));
    if (a.some((x) => typeof x !== 'number') || a.length !== rows * cols) { r.error('ML-LANG-BUILTIN-ARG', `${head}(${rows * cols} numbers, column-major)`, expr.span); return deepFreeze([]); }
    return makeMat(a as number[], rows, cols);
  }
  if (head === 'mat2' || head === 'mat3' || head === 'mat4') {
    r.tick();
    const n = head === 'mat2' ? 2 : head === 'mat3' ? 3 : 4;
    const a = expr.args.map((x) => evalExpr(x, env, r));
    if (a.length === 0) return makeMat(identityMat(n), n, n);
    if (a.some((x) => typeof x !== 'number') || a.length !== n * n) { r.error('ML-LANG-BUILTIN-ARG', `${head}() (identity) or ${head}(${n * n} numbers, column-major)`, expr.span); return deepFreeze([]); }
    return makeMat(a as number[], n, n);
  }
  if (head === 'transpose') {
    r.tick();
    const m = evalExpr(expr.args[0] ?? { kind: 'null', span: expr.span }, env, r);
    const d = descriptorOf(m);
    if (!d || d.lower?.shape !== 'matMxN') { r.error('ML-LANG-BUILTIN-ARG', 'transpose(mat)', expr.span); return deepFreeze([]); }
    // Column-major transpose. Input is R rows × C cols; element (r,c) lives at flat index c*R+r. The
    // transpose is C rows × R cols and maps (r,c) → (c,r); output element at row=c, col=r lives at flat
    // index r*C+c. So out[r*C+c] = in[c*R+r] for r∈[0,R), c∈[0,C).
    const s = vecStoreOf(m); const R = s.rows, C = s.cols; const out = new Array<number>(R * C);
    for (let c = 0; c < C; c++) for (let rr = 0; rr < R; rr++) out[rr * C + c] = s.c[c * R + rr] as number;
    return makeMat(out, C, R);
  }
  if (head === 'determinant') {
    r.tick();
    const m = evalExpr(expr.args[0] ?? { kind: 'null', span: expr.span }, env, r);
    const d = descriptorOf(m);
    const s = d?.lower?.shape === 'matMxN' ? vecStoreOf(m) : null;
    if (!s || s.rows !== s.cols) { r.error('ML-LANG-BUILTIN-ARG', 'determinant(square mat)', expr.span); return deepFreeze([]); }
    return matDeterminant(Array.from(s.c), s.rows);
  }
  if (head === 'inverse') {
    r.tick();
    const m = evalExpr(expr.args[0] ?? { kind: 'null', span: expr.span }, env, r);
    const d = descriptorOf(m);
    const s = d?.lower?.shape === 'matMxN' ? vecStoreOf(m) : null;
    if (!s || s.rows !== s.cols) { r.error('ML-LANG-BUILTIN-ARG', 'inverse(square mat)', expr.span); return deepFreeze([]); }
    return makeMat(matInverse(Array.from(s.c), s.rows), s.rows, s.rows);
  }
  if (head === 'qmat') {
    r.tick();
    // qmat(q:vec4) → the 3×3 rotation matrix of the quaternion, COLUMN-MAJOR (element (r,c) at flat index c*3+r).
    const q = evalExpr(expr.args[0] ?? { kind: 'null', span: expr.span }, env, r);
    const d = descriptorOf(q);
    if (!d || d.lower?.shape !== 'vecN' || (d.lower.rows ?? 0) !== 4) { r.error('ML-LANG-BUILTIN-ARG', 'qmat(vec4 quaternion)', expr.span); return deepFreeze([]); }
    const s = vecStoreOf(q); const x = s.c[0] as number, y = s.c[1] as number, z = s.c[2] as number, w = s.c[3] as number;
    const xx=x*x, yy=y*y, zz=z*z, xy=x*y, xz=x*z, yz=y*z, wx=w*x, wy=w*y, wz=w*z;
    // Columns of the rotation matrix. col0 = R·x̂, col1 = R·ŷ, col2 = R·ẑ; laid out column-major so
    // out[c*3+r] = column c's row r. This is the transpose-free companion to qrotate: qmat(q)·v ≡ qrotate(q,v).
    return makeMat([
      1-2*(yy+zz), 2*(xy+wz),   2*(xz-wy),     // col 0
      2*(xy-wz),   1-2*(xx+zz), 2*(yz+wx),     // col 1
      2*(xz+wy),   2*(yz-wx),   1-2*(xx+yy),   // col 2
    ], 3, 3);
  }
  if (head === 'dot' || head === 'cross' || head === 'normalize' || head === 'length' ||
      head === 'distance' || head === 'reflect' || head === 'refract' || head === 'faceforward' ||
      head === 'qmul' || head === 'qconj' || head === 'qinvert' || head === 'qaxisangle' || head === 'qrotate' ||
      head === 'qslerp') {
    r.tick();
    const a = expr.args.map((x) => evalExpr(x, env, r));
    const comps = (v: unknown): number[] | null => {
      const d = descriptorOf(v);
      if (!d || d.lower?.shape !== 'vecN') return null;
      const n = d.lower.rows ?? 0; const out: number[] = [];
      for (let i = 0; i < n; i++) { const c = d.getMember!(v, 'xyzw'[i] as string); if (typeof c !== 'number') return null; out.push(c); }
      return out;
    };
    const num = (v: unknown): number | null => (typeof v === 'number' && !Number.isNaN(v)) ? v : null;
    const badVec = (): unknown => { r.error('ML-LANG-BUILTIN-ARG', `${head}(vec…) — argument must be a vector`, expr.span); return deepFreeze([]); };
    if (head === 'dot') { const x = comps(a[0]); const y = comps(a[1]); if (!x || !y || x.length !== y.length) return badVec(); return x.reduce((s, xi, i) => s + xi * (y[i] as number), 0); }
    if (head === 'cross') { const x = comps(a[0]); const y = comps(a[1]); if (!x || !y || x.length !== 3 || y.length !== 3) return badVec(); return makeVec([x[1]!*y[2]!-x[2]!*y[1]!, x[2]!*y[0]!-x[0]!*y[2]!, x[0]!*y[1]!-x[1]!*y[0]!]); }
    if (head === 'length') { const x = comps(a[0]); if (!x) return badVec(); return Math.sqrt(x.reduce((s, xi) => s + xi * xi, 0)); }
    // distance(a, b) = length(a - b) = sqrt(Σ (aᵢ - bᵢ)²).
    if (head === 'distance') { const x = comps(a[0]); const y = comps(a[1]); if (!x || !y || x.length !== y.length) return badVec(); return Math.sqrt(x.reduce((s, xi, i) => s + (xi - (y[i] as number)) ** 2, 0)); }
    // reflect(I, N) = I - 2·dot(N, I)·N (GLSL/WGSL semantics; N is assumed normalized).
    if (head === 'reflect') { const I = comps(a[0]); const N = comps(a[1]); if (!I || !N || I.length !== N.length) return badVec(); const d = I.reduce((s, ii, i) => s + ii * (N[i] as number), 0); return makeVec(I.map((ii, i) => ii - 2 * d * (N[i] as number))); }
    // refract(I, N, eta): k = 1 - eta²·(1 - dot(N,I)²); k < 0 → total internal reflection → zero vector;
    // else eta·I - (eta·dot(N,I) + √k)·N.
    if (head === 'refract') { const I = comps(a[0]); const N = comps(a[1]); const eta = num(a[2]); if (!I || !N || eta === null || I.length !== N.length) return badVec(); const d = I.reduce((s, ii, i) => s + ii * (N[i] as number), 0); const k = 1 - eta * eta * (1 - d * d); if (k < 0) return makeVec(I.map(() => 0)); const sq = Math.sqrt(k); return makeVec(I.map((ii, i) => eta * ii - (eta * d + sq) * (N[i] as number))); }
    // faceforward(N, I, Nref): dot(Nref, I) < 0 → N, else -N (orient N to face away from I).
    if (head === 'faceforward') { const N = comps(a[0]); const I = comps(a[1]); const Nref = comps(a[2]); if (!N || !I || !Nref || N.length !== I.length || N.length !== Nref.length) return badVec(); const d = Nref.reduce((s, ni, i) => s + ni * (I[i] as number), 0); return makeVec(N.map((ni) => d < 0 ? ni : -ni)); }
    // ─── quaternions (vec4 layout (x,y,z,w) = imaginary xyz + real w) ───
    // A quat IS a vec4; each op requires its argument to be exactly a vec4 (`comps` returns any width, so
    // the length===4 guard rejects a vec2/vec3). qconj negates the imaginary part; qinvert = qconj / |q|²
    // (the multiplicative inverse — for a unit quat this equals the conjugate); qmul is the Hamilton product.
    if (head === 'qconj') { const q = comps(a[0]); if (!q || q.length !== 4) return badVec(); return makeVec([-q[0]!, -q[1]!, -q[2]!, q[3]!]); }
    if (head === 'qinvert') { const q = comps(a[0]); if (!q || q.length !== 4) return badVec(); const d = q[0]!**2 + q[1]!**2 + q[2]!**2 + q[3]!**2; return makeVec([-q[0]!/d, -q[1]!/d, -q[2]!/d, q[3]!/d]); }
    if (head === 'qmul') {
      const A = comps(a[0]); const B = comps(a[1]); if (!A || A.length !== 4 || !B || B.length !== 4) return badVec();
      const [ax, ay, az, aw] = A as [number, number, number, number]; const [bx, by, bz, bw] = B as [number, number, number, number];
      return makeVec([aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz]);
    }
    // qaxisangle(axis:vec3, angle) → the unit quat (axis·sin(θ/2), cos(θ/2)). The axis must be a vec3.
    if (head === 'qaxisangle') { const ax = comps(a[0]); const ang = num(a[1]); if (!ax || ax.length !== 3 || ang === null) return badVec(); const s = Math.sin(ang/2); return makeVec([ax[0]!*s, ax[1]!*s, ax[2]!*s, Math.cos(ang/2)]); }
    // qrotate(q:vec4, v:vec3) → the rotated vec3 via the optimized `v + 2·cross(q.xyz, cross(q.xyz, v) + q.w·v)`:
    // t = 2·cross(q.xyz, v); c2 = cross(q.xyz, t); result = v + q.w·t + c2. Cheaper than building qmat and equal to it.
    if (head === 'qrotate') { const q = comps(a[0]); const v = comps(a[1]); if (!q || q.length !== 4 || !v || v.length !== 3) return badVec();
      const t = [2*(q[1]!*v[2]!-q[2]!*v[1]!), 2*(q[2]!*v[0]!-q[0]!*v[2]!), 2*(q[0]!*v[1]!-q[1]!*v[0]!)];   // 2·cross(q.xyz, v)
      const c2 = [q[1]!*t[2]!-q[2]!*t[1]!, q[2]!*t[0]!-q[0]!*t[2]!, q[0]!*t[1]!-q[1]!*t[0]!];               // cross(q.xyz, t)
      return makeVec([v[0]!+q[3]!*t[0]!+c2[0]!, v[1]!+q[3]!*t[1]!+c2[1]!, v[2]!+q[3]!*t[2]!+c2[2]!]);
    }
    // qslerp(a:vec4, b:vec4, t) → spherical linear interpolation of the two quats. The antipodal fix (dot<0 →
    // negate b, so the shorter arc is taken); a small-angle NORMALIZED-lerp fallback (dot>0.9995 → sin θ ≈ 0 would
    // divide-by-near-zero, so lerp+normalize instead); else the sin-weighted great-circle blend.
    if (head === 'qslerp') { const A = comps(a[0]); const B = comps(a[1]); const t = num(a[2]); if (!A || A.length !== 4 || !B || B.length !== 4 || t === null) return badVec();
      let dot = A[0]!*B[0]! + A[1]!*B[1]! + A[2]!*B[2]! + A[3]!*B[3]!; let b = B.slice(); if (dot < 0) { b = b.map((x) => -x); dot = -dot; }
      if (dot > 0.9995) { const out = A.map((ai, i) => ai + t * (b[i]! - ai)); const len = Math.hypot(...out); return makeVec(out.map((x) => x / len)); }
      const th = Math.acos(dot); const s = Math.sin(th); const wa = Math.sin((1 - t) * th) / s; const wb = Math.sin(t * th) / s;
      return makeVec(A.map((ai, i) => wa * ai + wb * b[i]!));
    }
    // normalize
    const x = comps(a[0]); if (!x) return badVec(); const len = Math.sqrt(x.reduce((s, xi) => s + xi * xi, 0));
    return makeVec(x.map((xi) => xi / len));   // len===0 → NaN components, matching native normalize
  }
  if (head === 'rand') { r.tick(); return r.rng(); }
  if (head === 'range') {
    r.tick();
    const n = Number(evalExpr(expr.args[0] ?? { kind: 'number', value: 0, span: expr.span }, env, r));
    return seededRange(Number.isFinite(n) ? n : 0);
  }
  // Pure collection builtins — intrinsic free functions that RETURN NEW frozen collections (never
  // mutate). Unbound-head-only (a user function of the same name shadows via evalCall's callee
  // resolution, which never reaches here). Each ticks the budget per call + per element so a large
  // collection fails closed with ML-LANG-BUDGET. Callbacks are ordinary closures invoked as fn(x, i).
  if (IMPLEMENTED_BUILTINS.has(head) && head !== 'rand' && head !== 'range') {
    r.tick();
    const a = expr.args.map((x) => evalExpr(x, env, r));
    // A callback may be EITHER an arrow (a real JS closure — typeof 'function') OR a user-declared
    // `function` (a structured callable object, invoked via callUserFn). Normalize both to a uniform
    // `(...xs) => unknown` invoker so a named function works as a callback, not just an arrow.
    const asFn = (v: unknown): ((...xs: unknown[]) => unknown) | null => {
      if (typeof v === 'function') return v as (...xs: unknown[]) => unknown;   // an arrow closure
      if (isUserFn(v)) return (...xs: unknown[]) => callUserFn(v, xs, r);        // a user `function`
      return null;
    };
    const badArg = (msg: string): unknown => { r.error('ML-LANG-BUILTIN-ARG', msg, expr.span); return deepFreeze([]); };
    // Accept any value the language can iterate: a plain array as-is, or a custom value whose descriptor
    // exposes `iterate` (a typed array — the same seam for-of uses). Returns null for a non-iterable, so the
    // caller's existing bad-argument diagnostic fires. Materializes to a plain array (results stay plain
    // arrays). A whole-buffer read must register the value's generation dependency — like for-of, index, and
    // concat — so a reactive context re-runs on an in-place write; iterate alone would not subscribe.
    const asArray = (xs: unknown): unknown[] | null => {
      if (Array.isArray(xs)) return xs;
      const desc = descriptorOf(xs);
      const iter = desc?.iterate?.(xs);
      if (!iter) return null;
      const gen = generationOf(xs);
      if (gen !== undefined) r.opt.host.readGeneration(gen);
      return Array.from(iter);
    };

    switch (head) {
      case 'map': {
        const xs = asArray(a[0]); const fn = asFn(a[1]);
        if (!xs || !fn) return badArg(`map(array, fn) — bad arguments`);
        const out: unknown[] = []; xs.forEach((x, i) => { r.tick(); out.push(fn(x, i)); }); return deepFreeze(out);
      }
      case 'filter': {
        const xs = asArray(a[0]); const fn = asFn(a[1]);
        if (!xs || !fn) return badArg(`filter(array, fn) — bad arguments`);
        const out: unknown[] = []; xs.forEach((x, i) => { r.tick(); if (truthy(fn(x, i))) out.push(x); }); return deepFreeze(out);
      }
      case 'reduce': {
        const xs = asArray(a[0]); const fn = asFn(a[1]); const init = a[2];
        if (!xs || !fn) return badArg(`reduce(array, fn, init) — bad arguments`);
        let acc = init; xs.forEach((x, i) => { r.tick(); acc = fn(acc, x, i); }); return acc;   // acc may be a scalar; a collection acc is already frozen by its own eval
      }
      case 'keys': {
        const o = a[0];
        if (o === null || typeof o !== 'object' || Array.isArray(o)) return badArg(`keys(object) — bad argument`);
        return deepFreeze(Object.keys(o as object));
      }
      case 'values': {
        const o = a[0];
        if (o === null || typeof o !== 'object' || Array.isArray(o)) return badArg(`values(object) — bad argument`);
        return deepFreeze(Object.values(o as Record<string, unknown>));
      }
      case 'entries': {
        const o = a[0];
        if (o === null || typeof o !== 'object' || Array.isArray(o)) return badArg(`entries(object) — bad argument`);
        return deepFreeze(Object.entries(o as Record<string, unknown>).map(([k, v]) => [k, v]));
      }
      case 'fromEntries': {
        const pairs = a[0];
        if (!Array.isArray(pairs)) return badArg(`fromEntries(array of [key, value]) — bad argument`);
        const out: Record<string, unknown> = {};
        for (const p of pairs) { r.tick(); if (Array.isArray(p) && typeof p[0] === 'string' && !FORBIDDEN_KEYS.has(p[0])) out[p[0]] = p[1]; }
        return deepFreeze(out);
      }
      case 'some': {
        const xs = asArray(a[0]); const fn = asFn(a[1]);
        if (!xs || !fn) return badArg(`some(array, fn) — bad arguments`);
        for (let i = 0; i < xs.length; i++) { r.tick(); if (truthy(fn(xs[i], i))) return true; }
        return false;
      }
      case 'every': {
        const xs = asArray(a[0]); const fn = asFn(a[1]);
        if (!xs || !fn) return badArg(`every(array, fn) — bad arguments`);
        for (let i = 0; i < xs.length; i++) { r.tick(); if (!truthy(fn(xs[i], i))) return false; }
        return true;
      }
      case 'find': {
        const xs = asArray(a[0]); const fn = asFn(a[1]);
        if (!xs || !fn) return badArg(`find(array, fn) — bad arguments`);
        for (let i = 0; i < xs.length; i++) { r.tick(); if (truthy(fn(xs[i], i))) return xs[i] ?? null; }
        return null;
      }
      case 'findIndex': {
        const xs = asArray(a[0]); const fn = asFn(a[1]);
        if (!xs || !fn) return badArg(`findIndex(array, fn) — bad arguments`);
        for (let i = 0; i < xs.length; i++) { r.tick(); if (truthy(fn(xs[i], i))) return i; }
        return -1;
      }
      case 'includes': {
        const xs = asArray(a[0]); const v = a[1];
        if (!xs) return badArg(`includes(array, value) — bad argument`);
        for (const x of xs) { r.tick(); if (looseEquals(x, v)) return true; }
        return false;
      }
      case 'sort': {
        const xs = asArray(a[0]); const cmpArg = a[1];
        if (!xs) return badArg(`sort(array, comparator?) — bad argument`);
        // Tick PER COMPARISON on the default path too — an O(n log n) sort of a large array must be
        // budget-charged or it bypasses the step + time guards (tick() is the only deadline check point).
        if (cmpArg === undefined) return deepFreeze(stableSort(xs, (x, y) => { r.tick(); return defaultCompare(x, y); }));
        const cmpFn = asFn(cmpArg);
        if (!cmpFn) return badArg(`sort(array, comparator) — comparator is not callable`);
        let flagged = false;
        const cmp = (x: unknown, y: unknown): number => {
          r.tick();
          const res = cmpFn(x, y);
          if (typeof res !== 'number' || Number.isNaN(res)) {
            if (!flagged) { flagged = true; r.error('ML-LANG-BUILTIN-ARG', 'sort comparator must return a number', expr.span); }
            return 0;   // keep relative order on a bad return (stable)
          }
          return res;
        };
        return deepFreeze(stableSort(xs, cmp));
      }
      case 'slice': {
        const xs = asArray(a[0]);
        if (!xs) return badArg(`slice(array, start, end?) — bad argument`);
        const len = xs.length;
        const norm = (v: unknown, dflt: number): number => {
          if (v === undefined) return dflt;
          const n = Math.trunc(toNum(v));
          if (Number.isNaN(n)) return dflt;
          return n < 0 ? Math.max(len + n, 0) : Math.min(n, len);
        };
        const start = norm(a[1], 0);
        const end = norm(a[2], len);
        const out: unknown[] = [];
        for (let i = start; i < end; i++) { r.tick(); out.push(xs[i]); }
        return deepFreeze(out);
      }
      case 'reverse': {
        const xs = asArray(a[0]);
        if (!xs) return badArg(`reverse(array) — bad argument`);
        const out: unknown[] = [];
        for (let i = xs.length - 1; i >= 0; i--) { r.tick(); out.push(xs[i]); }
        return deepFreeze(out);
      }
      case 'split': {
        const s = a[0]; const sep = a[1];
        if (typeof s !== 'string' || typeof sep !== 'string') return badArg(`split(string, separator) — bad arguments`);
        const parts = sep === '' ? Array.from(s) : s.split(sep);
        parts.forEach(() => r.tick());
        return deepFreeze(parts);
      }
      case 'join': {
        const xs = asArray(a[0]); const sep = a[1];
        if (!xs || typeof sep !== 'string') return badArg(`join(array, separator) — bad arguments`);
        const parts: string[] = [];
        // Fail CLOSED before the result crosses the string cap (mirrors the `+` operator): a large
        // collection of large strings could otherwise build a string past the engine limit and throw
        // a raw RangeError. Treat the cap as a budget trip, before allocating the joined string.
        let total = 0;
        for (let i = 0; i < xs.length; i++) {
          r.tick();
          const s = strOf(xs[i]);
          total += s.length + (i > 0 ? sep.length : 0);
          if (total > r.opt.maxStringLength) {
            r.error('ML-LANG-BUDGET', `string result would exceed the ${r.opt.maxStringLength}-character limit`, expr.span);
            return null;
          }
          parts.push(s);
        }
        return parts.join(sep);
      }
      case 'chars': {
        const s = a[0];
        if (typeof s !== 'string') return badArg(`chars(string) — bad argument`);
        const cs = Array.from(s);
        cs.forEach(() => r.tick());
        return deepFreeze(cs);
      }
      case 'toUpperCase': {
        const s = a[0];
        if (typeof s !== 'string') return badArg(`toUpperCase(string) — bad argument`);
        return s.toUpperCase();
      }
      case 'toLowerCase': {
        const s = a[0];
        if (typeof s !== 'string') return badArg(`toLowerCase(string) — bad argument`);
        return s.toLowerCase();
      }
      case 'trim': {
        const s = a[0];
        if (typeof s !== 'string') return badArg(`trim(string) — bad argument`);
        return s.trim();
      }
      case 'min': case 'max': case 'abs': case 'sign': case 'floor':
      case 'ceil': case 'round': case 'clamp': case 'sqrt': case 'pow':
      case 'sin': case 'cos': case 'exp': case 'log': case 'fract':
      case 'step': case 'mix': case 'smoothstep':
      case 'tan': case 'sinh': case 'cosh': case 'tanh':
      case 'asin': case 'acos': case 'atan': case 'atan2':
      case 'exp2': case 'log2': case 'inverseSqrt':
      case 'degrees': case 'radians': case 'trunc': {
        // Coerce a numeric arg or fail: returns null when the value is not a finite/coercible number.
        const num = (v: unknown): number | null => {
          if (typeof v === 'number') return Number.isNaN(v) ? null : v;
          const n = toNum(v);
          return Number.isNaN(n) ? null : n;
        };
        const bad = (): unknown => badArg(`${head}(number, …) — non-numeric argument`);
        const arity = NUMERIC_ARITY[head] ?? a.length;
        // Componentwise vec application (GLSL semantics): if any of the head's arity-relevant args carries
        // a vecN descriptor, map the scalar op over its components, broadcasting a plain scalar arg to every
        // component, and return a fresh vec of the common width. Detection + reads are scoped to the first
        // `arity` args, so a vec in a beyond-arity slot never promotes an otherwise-scalar call. All vec
        // args in that prefix must share ONE width — a mismatch is a fail-loud arg error (GLSL rejects
        // `min(vec2, vec3)` at compile time, so the interpreter oracle must too; it never truncates). A NaN
        // component (e.g. sqrt(negative), log(<=0), or a non-numeric scalar broadcast) is KEPT — native
        // shaders compute componentwise and never abort the whole vector.
        const vecComps = (v: unknown): number[] | null => {
          const d = descriptorOf(v);
          if (!d || d.lower?.shape !== 'vecN') return null;
          const n = d.lower.rows ?? 0; const out: number[] = [];
          for (let i = 0; i < n; i++) out.push(Number(d.getMember!(v, 'xyzw'[i] as string)));
          return out;
        };
        const relevant = a.slice(0, arity);
        const vecArgs = relevant.map(vecComps).filter((c): c is number[] => c !== null);
        if (vecArgs.length > 0) {
          const width = vecArgs[0]!.length;
          if (vecArgs.some((c) => c.length !== width)) {
            return badArg(`${head}(vec…) — vector arguments must be the same width`);
          }
          const scalarAt = (arg: unknown, i: number): number => { const c = vecComps(arg); return c ? (c[i] ?? NaN) : (num(arg) ?? NaN); };
          const out: number[] = [];
          for (let i = 0; i < width; i++) {
            const xs: number[] = [];
            for (let k = 0; k < arity; k++) xs.push(scalarAt(relevant[k], i));
            out.push(scalarMath(head, xs));
          }
          return makeVec(out);
        }
        // Scalar path — byte-identical to the pre-vec dispatch: coerce each needed arg (null → fail loud),
        // intercept sqrt/log domains for their specific diagnostics, then compute via scalarMath.
        const xs: number[] = [];
        for (let k = 0; k < arity; k++) { const v = num(a[k]); if (v === null) return bad(); xs.push(v); }
        if (head === 'sqrt' && xs[0]! < 0) return badArg(`sqrt(x) — x must be >= 0`);
        if (head === 'log' && xs[0]! <= 0) return badArg(`log(x) — x must be > 0`);
        if ((head === 'asin' || head === 'acos') && (xs[0]! < -1 || xs[0]! > 1)) return badArg(`${head}(x) — x must be in [-1, 1]`);
        if ((head === 'log2' || head === 'inverseSqrt') && xs[0]! <= 0) return badArg(`${head}(x) — x must be > 0`);
        return scalarMath(head, xs);
      }
      case 'format': {
        const x = a[0]; const digits = a[1];
        if (typeof x !== 'number' || Number.isNaN(x)) return badArg(`format(number, digits) — first argument must be a number`);
        if (typeof digits !== 'number' || !Number.isInteger(digits) || digits < 0 || digits > 100) return badArg(`format(number, digits) — digits must be an integer in [0, 100]`);
        return x.toFixed(digits);
      }
      default: {
        // A registry-admitted name with no case yet fails loud here, with args evaluated exactly once —
        // never silently falling through to the host resolveCall path (which would re-evaluate args).
        r.error('ML-LANG-UNKNOWN-CALL', `unknown call '${head}'`, expr.span);
        return null;
      }
    }
  }
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
    // A host may return a PURE value (kind:'value') — allowed in expression position and deep-frozen so
    // it carries the same immutability guarantee as an intrinsic result. Without the tag, a non-array
    // object is a domain NODE (only valid in child position) → fail closed here as before.
    if (resolved.kind === 'value') return deepFreeze(v);
    if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
      r.error('ML-LANG-UNKNOWN-CALL', `unknown call '${head}' (node head is not valid in expression position)`, expr.span);
      return null;
    }
    return v;
  }
  r.error('ML-LANG-UNKNOWN-CALL', `unknown call '${head}'`, expr.span);
  return null;
}

export function callUserFn(fn: UserFn, args: unknown[], r: Runner): unknown {
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
      const cval = evalExpr(stmt.init, env, r);
      if (isCustomType(cval)) markFrozen(cval);
      env.define(stmt.name, cval, { kind: 'const' });
      return null;
    }
    case 'let': {
      // Reactive state is component-scope-only. Outside a component → ML-LANG-LET-SCOPE.
      if (!insideComponent) { r.error('ML-LANG-LET-SCOPE', `reactive 'let' is only allowed inside a component`, stmt.span); return null; }
      // Redeclaration guard (parser only dedups the top level; block bodies must check here too, else
      // a `const x; let x` silently converts the const cell to a reactive let — a const-immutability bypass).
      if (env.hasOwn(stmt.name)) { r.error('ML-LANG-REDECL', `'${stmt.name}' already declared`, stmt.span); return null; }
      // Compute the STABLE cell key = component-instance key + let name + per-instance occurrence
      // ordinal, so a host can latch this cell's settled value across a re-derive. No instance
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
      const iterValue = evalExpr(stmt.iterable, env, r);
      // Arrays iterate elements; strings iterate Unicode code points (Array.from — matching chars()/
      // split("") and JS for..of). NOTE: this deliberately differs from string indexing s[i] and
      // .length, which are UTF-16 code-unit based — for astral characters (emoji, CJK Ext-B) the item
      // count differs from .length and s[i] may return a lone surrogate. Any other value → fail-loud.
      let iter: unknown[];
      if (Array.isArray(iterValue)) iter = iterValue;
      else if (typeof iterValue === 'string') iter = Array.from(iterValue);
      else {
        const desc = descriptorOf(iterValue);
        // Subscribe a reactive read to the iterated value's in-place mutation: a for-of over a buffer is
        // a whole-value read that bypasses readMember, so it must register the generation dep itself, or a
        // UI iterating the buffer never re-renders on an in-place write.
        const gen = generationOf(iterValue);
        if (gen !== undefined) r.opt.host.readGeneration(gen);
        if (desc?.iterate) iter = Array.from(desc.iterate(iterValue));
        else { r.error('ML-LANG-FOR-ITER', 'for-of expects an array or string to iterate', stmt.span); return null; }
      }
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
    // A forbidden-key write is a prototype-pollution attempt — surface the specific security diagnostic
    // (it is blocked as immutable too, but the more-specific code is the useful one).
    if (FORBIDDEN_KEYS.has(target.property)) { r.error('ML-LANG-FORBIDDEN', `forbidden key '${target.property}'`, stmt.span); return; }
    const base = evalExpr(target.object, env, r);
    const desc = descriptorOf(base);
    if (desc?.setMember) {
      // Immutability is the interpreter's own FROZEN box (set by a `const` binding / deepFreeze), OR-ed
      // with any extra descriptor-defined frozen condition — so a `const`-bound value (and any alias
      // sharing the same box by reference) is immutable even if the descriptor omits a `frozen` handler.
      if (isFrozenCustom(base) || desc.frozen?.(base)) { r.error('ML-LANG-IMMUTABLE', `cannot assign to a member of a frozen value`, stmt.span); return; }
      try { desc.setMember(base, target.property, value); } catch (e) { if (e instanceof BufferError) { r.error(e.code, e.detail, stmt.span); return; } throw e; }
      const gen = generationOf(base);
      if (gen !== undefined) r.opt.host.touchGeneration(gen);
      return;
    }
    r.error('ML-LANG-IMMUTABLE', `cannot assign to a member of an immutable value; rebuild with spread instead (e.g. o = { ...o, ${target.property}: … })`, stmt.span);
    return;
  }

  if (target.kind === 'index') {
    // Resolve the key only to catch a computed forbidden key (prototype pollution); the write itself is
    // rejected as immutable unless the base is a mutable custom value with a setIndex handler.
    const key = evalExpr(target.index, env, r);
    if (typeof key === 'string' && FORBIDDEN_KEYS.has(key)) { r.error('ML-LANG-FORBIDDEN', `forbidden key '${key}'`, stmt.span); return; }
    const base = evalExpr(target.object, env, r);
    const desc = descriptorOf(base);
    if (desc?.setIndex) {
      // Interpreter-owned FROZEN box OR-ed with a descriptor frozen condition — see the member branch.
      if (isFrozenCustom(base) || desc.frozen?.(base)) { r.error('ML-LANG-IMMUTABLE', `cannot assign to an index of a frozen value`, stmt.span); return; }
      if (typeof key !== 'number' && typeof key !== 'string') { r.error('ML-LANG-BAD-KEY', 'index key is null/undefined', stmt.span); return; }
      try { desc.setIndex(base, key, value); } catch (e) { if (e instanceof BufferError) { r.error(e.code, e.detail, stmt.span); return; } throw e; }
      const gen = generationOf(base);
      if (gen !== undefined) r.opt.host.touchGeneration(gen);
      return;
    }
    r.error('ML-LANG-IMMUTABLE', 'cannot assign to an index of an immutable value; rebuild with spread or a builtin (map/filter/…) instead', stmt.span);
    return;
  }

  r.error('ML-LANG-BAD-LVALUE', 'unsupported assignment target', stmt.span);
}

// ─────────────────────────────────────────── member read + coercions ───────────────────────────────────────────

function readMember(container: unknown, key: string | number, expr: Expr, r: Runner): unknown {
  if (typeof key === 'string' && FORBIDDEN_KEYS.has(key)) { r.error('ML-LANG-FORBIDDEN', `forbidden path segment '${key}'`, expr.span); return null; }
  if (container === null || container === undefined) return null;
  const desc = descriptorOf(container);
  if (desc) {
    const gen = generationOf(container);
    if (gen !== undefined) r.opt.host.readGeneration(gen);
    try {
      if (typeof key === 'number') {
        if (desc.getIndex) { const res = desc.getIndex(container, key); if (res !== NOT_HANDLED) return res; }
      } else {
        if (desc.getMember) { const res = desc.getMember(container, key); if (res !== NOT_HANDLED) return res; }
        else if (desc.getIndex) { const res = desc.getIndex(container, key); if (res !== NOT_HANDLED) return res; }
      }
    } catch (e) {
      if (e instanceof BufferError) { r.error(e.code, e.detail, expr.span); return null; }
      throw e;
    }
    r.error('ML-LANG-UNKNOWN-MEMBER', `type '${desc.name}' has no member '${String(key)}'`, expr.span);
    return null;
  }
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
  const d = descriptorOf(v);
  if (d?.truthy) return d.truthy(v);
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
  const d = descriptorOf(v);
  if (d) return d.display ? d.display(v) : `[${d.name}]`;
  try { return JSON.stringify(v) ?? ''; } catch { return ''; }
}
/** Round half-to-even ("banker's rounding") — matches the shader `round` builtins so a `core` numeric
 *  result is bit-identical across targets (JS Math.round is half-up, which would diverge at x.5). */
function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff < 0.5) return f;
  if (diff > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;   // exactly .5 → nearest even
}
/** The number of leading args each scalar numeric builtin reads. The scalar caller coerces exactly
 *  this many (fail-loud on the first non-numeric one), matching the pre-vec per-head arg-count guards. */
const NUMERIC_ARITY: Readonly<Record<string, number>> = {
  min: 2, max: 2, abs: 1, sign: 1, floor: 1, ceil: 1, round: 1, clamp: 3,
  sqrt: 1, pow: 2, sin: 1, cos: 1, exp: 1, log: 1, fract: 1, step: 2, mix: 3, smoothstep: 3,
  tan: 1, sinh: 1, cosh: 1, tanh: 1, asin: 1, acos: 1, atan: 1, atan2: 2,
  exp2: 1, log2: 1, inverseSqrt: 1, degrees: 1, radians: 1, trunc: 1,
};
/** The pure scalar math for the numeric builtins, over already-coerced numbers (xs[0..2]). Returns a
 *  raw result — which may itself be NaN (e.g. pow(-2, 0.5), or sin(Infinity)), exactly as the direct
 *  Math.* call did before. `sqrt(x<0)` and `log(x<=0)` return NaN here; the SCALAR caller intercepts
 *  those domains first for their specific diagnostics, so it never observes this NaN — but the vec
 *  path uses it directly, keeping a NaN component (native shaders compute componentwise and never
 *  abort the whole vector). `round` is half-to-even for cross-target exactness. */
function scalarMath(head: string, xs: readonly number[]): number {
  const x = xs[0] ?? NaN; const y = xs[1] ?? NaN; const z = xs[2] ?? NaN;
  switch (head) {
    case 'min': return Math.min(x, y);
    case 'max': return Math.max(x, y);
    case 'abs': return Math.abs(x);
    case 'sign': return Math.sign(x);
    case 'floor': return Math.floor(x);
    case 'ceil': return Math.ceil(x);
    case 'round': return roundHalfEven(x);
    case 'clamp': return Math.min(Math.max(x, y), z);
    case 'sqrt': return x < 0 ? NaN : Math.sqrt(x);
    case 'pow': return Math.pow(x, y);
    case 'sin': return Math.sin(x);
    case 'cos': return Math.cos(x);
    case 'tan': return Math.tan(x);
    case 'sinh': return Math.sinh(x);
    case 'cosh': return Math.cosh(x);
    case 'tanh': return Math.tanh(x);
    case 'asin': return (x < -1 || x > 1) ? NaN : Math.asin(x);
    case 'acos': return (x < -1 || x > 1) ? NaN : Math.acos(x);
    case 'atan': return Math.atan(x);
    case 'atan2': return Math.atan2(x, y);   // atan2(y, x): xs[0]=y (numerator), xs[1]=x (denominator)
    case 'exp': return Math.exp(x);
    case 'exp2': return Math.pow(2, x);
    case 'log': return x <= 0 ? NaN : Math.log(x);
    case 'log2': return x <= 0 ? NaN : Math.log2(x);
    case 'inverseSqrt': return x <= 0 ? NaN : 1 / Math.sqrt(x);
    case 'degrees': return x * 180 / Math.PI;
    case 'radians': return x * Math.PI / 180;
    case 'trunc': return Math.trunc(x);
    case 'fract': return x - Math.floor(x);
    case 'step': return y < x ? 0 : 1;          // step(edge, x) → x < edge ? 0 : 1
    case 'mix': return x + (y - x) * z;
    case 'smoothstep': { const t = y === x ? 0 : Math.min(Math.max((z - x) / (y - x), 0), 1); return t * t * (3 - 2 * t); }
    default: return NaN;
  }
}
/** Deep-freeze a DSL-created collection so it is immutable by construction. Recurses into arrays +
 *  plain objects; a function short-circuits at the first guard (typeof !== 'object'), so a handler
 *  value stays callable. Short-circuits already-frozen values (idempotent, cycle-safe for the acyclic
 *  values the evaluator builds). */
function deepFreeze<T>(v: T): T {
  if (v === null || typeof v !== 'object' || Object.isFrozen(v)) return v;
  // A custom value is an opaque leaf: never Object.freeze it (a typed array's integer-indexed store
  // would throw). But it IS made immutable here via its own frozen box, so a value reached through a
  // frozen container (a const-bound object/array literal, injected data, a kind:'value' host return)
  // cannot be mutated in place. markFrozen is a no-op for an immutable custom value (no frozen box).
  if (isCustomType(v)) { markFrozen(v); return v; }
  if (Array.isArray(v)) { for (const x of v) deepFreeze(x); return Object.freeze(v) as T; }
  for (const val of Object.values(v as Record<string, unknown>)) deepFreeze(val);
  return Object.freeze(v) as T;
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

/** Dispatch a binary op when at least one operand carries a descriptor. Rule: try the LEFT operand's
 *  descriptor first, then the RIGHT's (so `2 * vec` reaches vec's binary as (op, 2, vec)). Equality
 *  prefers a dedicated `equals?` handler on either operand, else `binary('==')`, else reference identity
 *  (never fail-loud). An arithmetic/relational op returning NOT_HANDLED on both → ML-LANG-OP-UNSUPPORTED. */
function evalCustomBinary(op: BinOp, left: unknown, right: unknown, expr: Extract<Expr, { kind: 'binary' }>, r: Runner): unknown {
  const dl = descriptorOf(left);
  const dr = descriptorOf(right);
  if (op === '==' || op === '!=') {
    // A custom value compared with null/undefined follows reference identity — a custom value is a
    // non-null object, so it is never == null. Guard here so a descriptor handler is never called with a
    // null/undefined operand it may not tolerate (the idiomatic `x == null` must never fail-loud).
    if (left === null || left === undefined || right === null || right === undefined) {
      const eq = left === right;
      return op === '==' ? eq : !eq;
    }
    let eq: boolean;
    if (dl?.equals) eq = dl.equals(left, right);
    else if (dr?.equals) eq = dr.equals(left, right);
    else {
      // Equality FALLBACK always asks binary('==') (never binary('!=')) so `==` and `!=` stay symmetric;
      // the outer `op === '==' ? eq : !eq` negates for `!=`. A NOT_HANDLED result → reference identity.
      let res: unknown = dl?.binary ? dl.binary('==', left, right) : NOT_HANDLED;
      if (res === NOT_HANDLED) res = dr?.binary ? dr.binary('==', left, right) : NOT_HANDLED;
      eq = res === NOT_HANDLED ? left === right : Boolean(res);
    }
    return op === '==' ? eq : !eq;
  }
  let res: unknown = dl?.binary ? dl.binary(op, left, right) : NOT_HANDLED;
  if (res === NOT_HANDLED) res = dr?.binary ? dr.binary(op, left, right) : NOT_HANDLED;
  if (res === NOT_HANDLED) { r.error('ML-LANG-OP-UNSUPPORTED', `operator '${op}' is not defined for this type`, expr.span); return null; }
  return res;
}
function evalCustomNeg(x: unknown, expr: Extract<Expr, { kind: 'unary' }>, r: Runner): unknown {
  const d = descriptorOf(x);
  const res = d?.neg ? d.neg(x) : NOT_HANDLED;
  if (res === NOT_HANDLED) { r.error('ML-LANG-OP-UNSUPPORTED', `unary '-' is not defined for this type`, expr.span); return null; }
  return res;
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
  // Deep-freeze injected data at the boundary so it is immutable-by-construction like every DSL value.
  // Without this, a builtin that returns deepFreeze(out) over an array aliasing data's own element
  // objects (sort/slice/reverse/map/values) would freeze the host's LIVE objects in place — a silent
  // cross-boundary mutation. Freezing here makes those re-freezes no-ops (deepFreeze short-circuits
  // already-frozen values) and realizes "the DSL cannot mutate anything it sees" structurally.
  if ('data' in options) root.define('data', deepFreeze(options.data), { kind: 'const' });
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
