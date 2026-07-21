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
import type { HostEnvironment, ReactiveHost, CellRef } from './ports.ts';
import { descriptorOf, isCustomType, generationOf, isFrozenCustom, markFrozen, NOT_HANDLED, BufferError } from './custom-types.ts';
import type { BuiltinCtx, BuiltinModule, BuiltinRegistry } from './registry.ts';
import { buildRegistry, EMPTY_REGISTRY } from './registry.ts';

/** Default fuel budget: the maximum number of evaluation steps before an {@link EvalOptions.maxSteps}
 *  override, after which evaluation fails closed with an `ML-LANG-BUDGET` diagnostic. */
export const DEFAULT_MAX_STEPS = 100_000;
/** Default deadline budget in milliseconds (checked periodically), overridable via
 *  {@link EvalOptions.maxTimeMs}. Exceeding it fails closed with an `ML-LANG-BUDGET` diagnostic. */
export const DEFAULT_MAX_TIME_MS = 1000;
/** Default recursion-depth budget for nested calls/blocks, overridable via {@link EvalOptions.maxDepth}.
 *  Exceeding it fails closed with an `ML-LANG-BUDGET` diagnostic rather than overflowing the JS stack. */
export const DEFAULT_MAX_DEPTH = 64;
/** Default cap on the length a string may grow to via `+` concatenation, overridable via
 *  {@link EvalOptions.maxStringLength}. A result that would exceed it fails closed with `ML-LANG-BUDGET`. */
export const MAX_STRING_LENGTH = 10_000_000;
export const MAX_BUFFER_LENGTH = 16_777_216;   // 2^24 elements — a typed-array construction cap (ML-LANG-BUDGET over)
const TIME_CHECK_INTERVAL = 1024;

/**
 * The inputs to {@link evaluateProgram}: the host capabilities a run needs plus the optional
 * determinism seed, injected data, and budget overrides.
 *
 * The run is a pure function of these: `result = f(source, data, seed, host-state)` — the same source
 * with the same `data`/`seed` and an equivalently-seeded host reproduces the same result and the same
 * `rand()`/`range()` sequence.
 */
export interface EvalOptions {
  /** Data made available to the program as the `data` binding. Deep-frozen at the boundary, so the
   *  program cannot mutate anything the host passes in. */
  data?: unknown;
  /** Seed for the intrinsic `rand()`/`range()` PRNG. The same seed yields the same sequence; omitted
   *  defaults to `0`. */
  seed?: number;
  /** The reactive host: cells/effects, per-value generation signals, and the optional clock capability
   *  ({@link ReactiveHost}). */
  host: ReactiveHost;
  /** The host environment that resolves an unbound call head to a host value ({@link HostEnvironment}). */
  env: HostEnvironment;
  /** Fuel budget override — max evaluation steps before failing closed with `ML-LANG-BUDGET`. Defaults
   *  to {@link DEFAULT_MAX_STEPS}. */
  maxSteps?: number;
  /** Deadline budget override in milliseconds. Defaults to {@link DEFAULT_MAX_TIME_MS}. */
  maxTimeMs?: number;
  /** Recursion-depth budget override. Defaults to {@link DEFAULT_MAX_DEPTH}. */
  maxDepth?: number;
  /** Cap on the length a string may grow to via `+`. Defaults to {@link MAX_STRING_LENGTH} (a small
   *  value is useful in tests). */
  maxStringLength?: number;
  /** Whether the top level is evaluated as a component body — gates reactive `let`. Defaults to `false`
   *  (the program root is not a component). */
  insideComponent?: boolean;
  /** Standard-library modules whose builtins the evaluator resolves an unbound call head against (e.g.
   *  a numeric library so `vec3(...)`/`sqrt(...)` dispatch). The language kernel registers none itself;
   *  omit for a builtin-free run. */
  builtins?: readonly BuiltinModule[];
}

/** The outcome of {@link evaluateProgram}: the produced value and the diagnostics collected during the
 *  run. `evaluateProgram` never throws — an author error, a budget/recursion limit, an unknown call, or
 *  a forbidden-key access all surface here as a diagnostic with `value` set to a safe `null`. */
export interface EvalResult {
  /** The value the program evaluated to (its last top-level expression or a top-level `return`), or
   *  `null` if the run failed closed. */
  value: unknown;
  /** Every diagnostic collected during parsing + evaluation, in order. Empty on a fully-successful run. */
  diagnostics: Diagnostic[];
}

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
  /** Brand marking this object as a metael user function (vs a plain JS closure). Always `true`. */
  readonly __mlFn: true;
  /** The declared name — lowering uses it as the node head (e.g. `'KPI'`). */
  readonly name: string;
  /** The declared parameter patterns. */
  readonly params: Pattern[];
  /** The function body statements. */
  readonly body: Stmt[];
  /** The lexical environment captured at declaration. */
  readonly closure: Environment;
  /** `true` for a `component` (stateful; runs with reactive `let`), `false` for a pure `function`. */
  readonly isComponent: boolean;
}
/** Type guard: is `v` a metael {@link UserFn} (a declared `function`/`component`), as opposed to a plain
 *  JS value or an arrow closure? */
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

/** Options for {@link makeCallable}: the host capabilities + optional seed and budget overrides the
 *  fresh evaluation context needs. */
export interface MakeCallableOpts {
  /** The reactive host ({@link ReactiveHost}). */
  host: ReactiveHost;
  /** The host environment that resolves an unbound call head ({@link HostEnvironment}). */
  env: HostEnvironment;
  /** Fuel budget override. A raised value sizes the context for a large loop bound. Defaults to
   *  {@link DEFAULT_MAX_STEPS}. */
  maxSteps?: number;
  /** Deadline budget override in milliseconds. Defaults to {@link DEFAULT_MAX_TIME_MS}. */
  maxTimeMs?: number;
  /** Recursion-depth budget override. Defaults to {@link DEFAULT_MAX_DEPTH}. */
  maxDepth?: number;
  /** String-growth cap override. Defaults to {@link MAX_STRING_LENGTH}. */
  maxStringLength?: number;
  /** PRNG seed for the callable's `rand()`/`range()`. Defaults to `0`. */
  seed?: number;
  /** Builtin modules the callable resolves an unbound call head against (e.g. the numeric
   *  library so a kernel body's `vec3(...)`/`sqrt(...)` dispatch). Omit for a builtin-free callable. */
  builtins?: readonly BuiltinModule[];
}
/**
 * Build a plain JS callable that invokes a {@link UserFn} under a FRESH evaluation context (its own
 * budget).
 *
 * @param fn - the user function/component to invoke.
 * @param opts - host capabilities + budget/seed overrides ({@link MakeCallableOpts}).
 * @returns a function `(...args) => value` that runs `fn` with the given arguments; it returns `null`
 *          rather than throwing if the budget is exhausted mid-call.
 * @remarks Used to run a kernel per sampled coordinate as a top-level call — distinct from a reentrant
 *          in-interpretation call, which shares the ambient context.
 */
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
  /** The injected builtin registry (name→Builtin). An unbound call head resolves here FIRST; an
   *  unregistered head falls through to the intrinsic cascade and then fails closed. EMPTY_REGISTRY
   *  when the consumer injected no modules. */
  readonly registry: BuiltinRegistry;
  constructor(readonly opt: Required<Pick<EvalOptions, 'maxSteps' | 'maxTimeMs' | 'maxDepth' | 'maxStringLength'>> & EvalOptions) {
    this.deadline = Date.now() + opt.maxTimeMs;
    this.rng = makeSeededRng(opt.seed ?? 0);
    this.registry = opt.builtins ? buildRegistry(opt.builtins) : EMPTY_REGISTRY;
  }
  tick(steps = 1): void {
    // Charge `steps` fuel at once (default 1). A builtin doing self-contained native work (e.g. splitting
    // a string of length n) charges `tick(n)` up front — O(1) instead of an n-iteration accounting loop —
    // so it fails closed BEFORE the native work on a pathological input. Loops that invoke a user closure
    // must still tick once PER iteration (the tick is the loop's budget beat, interleaved with each call).
    // Guard the increment: a non-finite (NaN/±Infinity) or non-positive `n` from a buggy builtin must not
    // corrupt the count or skip the step cap — clamp such a value to a single step.
    const n = Number.isFinite(steps) && steps >= 1 ? Math.floor(steps) : 1;
    const before = this.steps;
    this.steps += n;
    if (this.steps > this.opt.maxSteps) throw new BudgetSignal('steps');
    // Deadline is checked once per TIME_CHECK_INTERVAL window. A multi-step tick can jump PAST a window
    // boundary without landing on a multiple, so test whether the increment CROSSED a boundary — not
    // `steps % INTERVAL === 0`, which a large `n` would silently skip.
    if (Math.floor(before / TIME_CHECK_INTERVAL) !== Math.floor(this.steps / TIME_CHECK_INTERVAL) && Date.now() > this.deadline) {
      throw new BudgetSignal('time');
    }
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
        // A non-string/non-number key can never be a valid member/index key. Fail CLOSED to a localized
        // diagnostic + null — the same fail-closed contract every readMember path honors — rather than
        // coercing via String(). A record is null-prototype (no inherited toString), so String(record) THROWS
        // ('Cannot convert object to primitive value'); coercing here would let that throw escape to the
        // top-level catch → ML-LANG-INTERNAL, aborting the WHOLE program and losing every sibling's output.
        // Booleans coerce cleanly to a string key (String(true)='true'), preserving the prior behavior for them.
        if (typeof key === 'object' || typeof key === 'function') { r.error('ML-LANG-BAD-KEY', 'index key is not a string or number', expr.span); return null; }
        if (key === null || key === undefined) { r.error('ML-LANG-BAD-KEY', 'index key is null/undefined', expr.span); return null; }
        return readMember(object, String(key), expr, r);
      }
      return readMember(object, key, expr, r);
    }
    case 'object': {
      // A null-prototype record: a metael object exposes NO inherited `__proto__`/`constructor`/`toString`,
      // closing a class of prototype-based sandbox escapes (defense in depth over the FORBIDDEN_KEYS guard).
      const out: Record<string, unknown> = Object.create(null);
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

function makeBuiltinCtx(expr: Extract<Expr, { kind: 'call' }>, env: Environment, r: Runner): BuiltinCtx {
  return {
    tick: (steps) => r.tick(steps),
    error: (code, message, span) => r.error(code, message, span ?? expr.span),
    rng: () => r.rng(),
    // Returns undefined when the host injected no clock — the datetime builtin then raises
    // ML-LANG-NO-CLOCK and returns null (fail-closed; NEVER a fake 0).
    clock: () => {
      const c = (r.opt.host as { clock?: () => { now(): number; monotonic(): number } }).clock;
      return c ? c.call(r.opt.host) : undefined;
    },
    evalArg: (i) => evalExpr(expr.args[i] ?? { kind: 'null', span: expr.span }, env, r),
    argCount: () => expr.args.length,
    callClosure: (v, args) => (typeof v === 'function' ? (v as (...a: unknown[]) => unknown)(...args) : isUserFn(v) ? callUserFn(v, args, r) : null),
    allocateGeneration: () => r.opt.host.allocateGeneration(),
    readGeneration: (g) => r.opt.host.readGeneration(g),
    freeze: (v) => deepFreeze(v),
    maxStringLength: r.opt.maxStringLength,
    span: expr.span,
  };
}

function dispatchBuiltin(head: string, expr: Extract<Expr, { kind: 'call' }>, env: Environment, r: Runner): unknown {
  const _b = r.registry.get(head);
  if (_b) return _b.invoke(makeBuiltinCtx(expr, env, r), expr.args);
  // `range` stays a language-kernel intrinsic (resolved before the host): it is the bounded-loop primitive
  // the compute-lowering gate hardcodes as the only lowerable loop form (`for … of range(n)`) and the
  // interpreter oracle relies on, so it cannot move to the injectable standard library. Only fires for the
  // UNBOUND-head path (a user `function range` shadows via evalCall's callee-resolution step, which never
  // reaches here). `rand` moved to the standard-library `random` module (registry-dispatched above).
  if (head === 'range') {
    r.tick();
    const n = Number(evalExpr(expr.args[0] ?? { kind: 'number', value: 0, span: expr.span }, env, r));
    return seededRange(Number.isFinite(n) ? n : 0);
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

/**
 * Invoke a {@link UserFn} with the given arguments under an existing evaluation context, returning its
 * value (its last expression, or a `return` value). Enforces the recursion-depth budget.
 *
 * @param fn - the user function/component to call.
 * @param args - positional arguments bound to the function's parameters.
 * @param r - the ambient evaluation context (shared with the caller).
 * @returns the value produced by the call.
 */
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

/** The language's canonical truthiness test (used by `if`/`&&`/`||`/`!` and filtering builtins).
 *  @internal Exported so a standard-library builtin matches the language's own truthiness exactly
 *  rather than re-deriving it; not part of the end-user API. */
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
/** The language's canonical value→string coercion (used by string `+` and `join`): custom values via
 *  their bounded `display`, primitives via String, collections via JSON.
 *  @internal Exported so a standard-library string builtin stringifies exactly as the language does
 *  rather than re-deriving it; not part of the end-user API. */
export function strOf(v: unknown): string {
  if (v === null || v === undefined) return '';
  if (typeof v === 'string') return v;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  const d = descriptorOf(v);
  if (d) return d.display ? d.display(v) : `[${d.name}]`;
  try { return JSON.stringify(v) ?? ''; } catch { return ''; }
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
/** The language's loose-equality relation (the `==` operator's semantics).
 *  @internal Exported so a standard-library builtin (`includes`) matches `==` exactly rather than
 *  re-implementing (and risking drift from) the language's own semantics; not part of the end-user API. */
export function looseEquals(a: unknown, b: unknown): boolean {
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

/**
 * Parse and evaluate a metael program, returning its value and diagnostics.
 *
 * The entry point of the language kernel. Total and non-throwing: a parse error, an author error, a
 * budget/recursion limit, an unknown call, or a forbidden-key access all become diagnostics with a safe
 * `null` value — nothing escapes into the host. Injected `data` is deep-frozen at the boundary, so the
 * program cannot mutate anything the host passes in. The kernel privileges no builtin; supply the
 * vocabulary a program needs via {@link EvalOptions.builtins}.
 *
 * @param source - the program source text.
 * @param options - host capabilities, the determinism seed, injected data, and budget overrides
 *                  ({@link EvalOptions}).
 * @returns the produced value + the diagnostics collected during the run ({@link EvalResult}).
 * @example
 * ```ts
 * const { value, diagnostics } = evaluateProgram('1 + 2', { host, env });
 * // value === 3, diagnostics === []
 * ```
 */
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
