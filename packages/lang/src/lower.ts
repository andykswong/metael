// The generic child-collection walk — the AST→host-value transform.
//
// The RETURN-MECHANISM SPLIT:
//   • function body            → implicit last-expression VALUE (execBlockValue — the evaluator).
//   • component body / wrap {}  → ordered CHILD LIST (collectChildren — this module).
//
// The walk runs ON TOP of the evaluator: it reuses evalExpr for argument/iterable evaluation and the
// shared Runner (budgets + diagnostics), so a runaway walk fails closed with ML-LANG-BUDGET exactly
// like evaluation. lang stays PURE: identity keys are minted through the injected KeyMinter port;
// nodes are built through the HostEnvironment port. lang imports nothing from any domain — the domain
// output type is opaque HostValue here.
//
// The root is a designated ENTRY component (default "Story"). A missing entry is ML-LANG-NO-ENTRY.

import type { Diagnostic } from './diagnostics.ts';
import { makeDiagnostic } from './diagnostics.ts';
import type { Expr, Stmt } from './ast.ts';
import { parseProgram } from './parser.ts';
import { Environment } from './environment.ts';
import type { HostValue, KeyMinter, Arg } from './ports.ts';
import { region, wrapper, isRegion } from './ports.ts';
import type { EvalOptions } from './evaluate.ts';
import {
  Runner, ReturnSignal, BudgetSignal, resolveOptions, evalExpr, bindParams, execStmt, truthy,
  type UserFn, isUserFn, arrowInfo,
} from './evaluate.ts';

export interface LowerOptions extends EvalOptions {
  minter: KeyMinter;
  entry?: string;
  /** Opt `data` into reactivity so a `data.x` read lowers to a trackable Region (Proxy-FREE). */
  reactiveData?: boolean;
}
export interface LowerResult { value: unknown; diagnostics: Diagnostic[] }

/** Lift a lowered value into the ordered Arg shape the host contract wants. lang's core grammar has
 *  no named-arg syntax at the call level (a call arg is positional), so `name` stays undefined; the
 *  load-bearing part is `reactive`, set when the value is a Region (it reads reactive state). An
 *  object-literal arg keeps its entries RAW inside `value` — only the top-level arg position is lifted. */
function toArg(value: unknown): Arg { return { value, reactive: isRegion(value) }; }

/** Per-wrapper-scope key context: `parentKey` prefixes minted child keys; `kindOrdinals` gives a
 *  fresh per-parent-per-kind LEXICAL ordinal so sibling `text`s become text#0, text#1, … A fresh
 *  `kindOrdinals` is created for every child block so ordinals restart under each parent. */
interface KeyContext {
  readonly parentKey: string;
  readonly kindOrdinals: Map<string, number>;
}
function childContext(parentKey: string): KeyContext {
  return { parentKey, kindOrdinals: new Map() };
}
function nextOrdinal(ctx: KeyContext, kind: string): number {
  const n = ctx.kindOrdinals.get(kind) ?? 0;
  ctx.kindOrdinals.set(kind, n + 1);
  return n;
}

/** A for-of iteration hint: the node produced by this iteration is a LIST ITEM, so it is keyed via
 *  minter.listItem (author `key` prop → stable key; else content-hash + this ordinal tiebreak) rather
 *  than a structural ordinal — so keyed items stay stable and duplicate-content siblings differ. */
interface ListHint { readonly ordinal: number }

export function lowerEntry(source: string, opts: LowerOptions): LowerResult {
  const runner = new Runner(resolveOptions(opts));
  const root = new Environment();
  if ('data' in opts) {
    // Opting `data` into reactivity binds it with `let` metadata, so readsReactiveLet treats a
    // `data.x` read as a reactive Region (Proxy-FREE, interpreter-mediated — the read is intercepted at
    // the eval-site, never via a reactive()/Proxy over the data object). Default stays `const` (eager).
    root.define('data', opts.data, { kind: opts.reactiveData ? 'let' : 'const' });
  }

  let value: unknown = null;
  try {
    const { program, diagnostics } = parseProgram(source);
    runner.diagnostics.push(...diagnostics);

    // Run the top-level DECLARATIONS for effect (const/function/component). The top level is not a
    // component, so a bare `let` is ML-LANG-LET-SCOPE (via execStmt). Non-declaration top-level
    // statements execute too (harmless); the root is ONLY the entry component.
    for (const stmt of program.stmts) execStmt(stmt, root, runner, false);

    const entryName = opts.entry ?? 'Story';
    const entry = root.get(entryName);
    if (!isUserFn(entry) || !entry.isComponent) {
      runner.diagnostics.push(makeDiagnostic('ML-LANG-NO-ENTRY', `entry component '${entryName}' is not defined`));
      return { value: null, diagnostics: runner.diagnostics };
    }
    // The entry is the PARENTLESS root — its key is just `${entryName}#0` (no leading-slash prefix; the
    // minter's structural() prepends `${parent}/`, which would give `/Story#0` for an empty parent).
    // Children below it mint through minter.structural('Story#0', kind, ord) → 'Story#0/box#0' etc.
    const entryKey = `${entryName}#0`;
    value = instantiateComponent(entry, entryName, entryKey, [], runner, opts);
  } catch (e) {
    if (e instanceof BudgetSignal) runner.diagnostics.push(makeDiagnostic('ML-LANG-BUDGET', `lowering budget exceeded: ${e.reason}`));
    else if (e instanceof ReturnSignal) value = e.value;
    else runner.diagnostics.push(makeDiagnostic('ML-LANG-INTERNAL', String(e)));
  }
  return { value, diagnostics: runner.diagnostics };
}

/** Instantiate an in-DSL `component`: bind its (possibly-destructured) params in a child scope of the
 *  component's own closure, child-collect its body (insideComponent=true so its `let`s are reactive),
 *  and produce the wrapping node. The host gets FIRST refusal via resolveCall(componentName, …) so a
 *  runtime host can build a proper group node for a user component instance; if the host declines
 *  (handled:false — e.g. an in-DSL component the host doesn't know), lang falls back to a uniform
 *  opaque wrapper `{ head, key, args, children }` so the derived tree stays well-formed.
 *  `args` is the RAW lowered arg list: it feeds bindParams (raw values) AND is lifted to Arg[] for
 *  resolveCall/wrapper (carrying the reactive flag). */
function instantiateComponent(
  fn: UserFn, name: string, key: string, args: unknown[], r: Runner, opts: LowerOptions,
): HostValue {
  if (++r.depth > r.opt.maxDepth) { r.depth--; throw new BudgetSignal('depth'); }
  // Enter this component INSTANCE's reactive-state scope: its `let`s key their cells by THIS instance's
  // key (stable across re-derives → latches state). Save the caller's scope + occurrence counter and
  // restore them in `finally` so a nested child component (and the parent's tail) resume correctly.
  const savedKey = r.currentComponentKey;
  const savedOccurrences = r.letOccurrences;
  r.currentComponentKey = key;
  r.letOccurrences = new Map();
  try {
    const frame = new Environment(fn.closure);
    bindParams(fn.params, args, frame);
    // A `return` inside a component body (the guard-clause idiom `if (…) return`) throws ReturnSignal
    // during child-collection. Catch it HERE so it stops this component's collection without unwinding
    // to lowerEntry (which would clobber the whole tree with the returned scalar). Children collected
    // before the return are kept (the guard ran; the tail did not).
    const children = collectChildren(fn.body, frame, childContext(key), r, opts);
    const liftedArgs = args.map(toArg);
    const resolved = r.opt.env.resolveCall(name, key, liftedArgs, children, { start: 0, end: 0 });
    // Host built the instance node (a runtime host could special-case a component head) → use it.
    // Otherwise emit a 'component' wrapper: the runtime derive materializes it as a structural group.
    return resolved.handled ? resolved.value : wrapper('component', name, key, liftedArgs, children);
  } finally {
    r.depth--;
    r.currentComponentKey = savedKey;
    r.letOccurrences = savedOccurrences;
  }
}

/** Walk a component body / wrap block in SOURCE ORDER, producing the ordered child node list.
 *  Node-producing statements append; declarations/effects (const/let/assign/function/component) run
 *  via the evaluator and append nothing; control flow (for/if/while) flattens its produced nodes. */
function collectChildren(body: Stmt[], env: Environment, ctx: KeyContext, r: Runner, opts: LowerOptions): HostValue[] {
  const out: HostValue[] = [];
  const scope = new Environment(env);   // block scope: local decls don't leak to siblings
  try {
    for (const stmt of body) collectStmt(stmt, scope, ctx, r, opts, out);
  } catch (e) {
    // A `return` (guard-clause idiom) stops collection at this block boundary, keeping the children
    // gathered before it. It must NOT unwind to lowerEntry (which would clobber the whole tree).
    if (!(e instanceof ReturnSignal)) throw e;
  }
  return out;
}

function collectStmt(stmt: Stmt, env: Environment, ctx: KeyContext, r: Runner, opts: LowerOptions, out: HostValue[]): void {
  r.tick();
  switch (stmt.kind) {
    case 'expr':
      collectValue(lowerExprToChildren(stmt.expr, env, ctx, r, opts, undefined), out);
      return;
    case 'for': {
      const iter = evalExpr(stmt.iterable, env, r);
      if (!Array.isArray(iter)) { r.error('ML-LANG-FOR-ITER', 'for-of expects an array to iterate', stmt.span); return; }
      iter.forEach((item, ordinal) => {
        r.tick();
        const loopEnv = new Environment(env);
        loopEnv.define(stmt.binding, item, { kind: 'const' });
        // A for-of body produces list item(s) for this ordinal — nodes are keyed via listItem.
        collectForBody(stmt.body, loopEnv, ctx, r, opts, { ordinal }, out);
      });
      return;
    }
    case 'if': {
      const branch = truthy(evalExpr(stmt.test, env, r)) ? stmt.then : stmt.else;
      // Fresh block scope so branch-local const/let don't leak to siblings (mirrors execBlockValue).
      if (branch) { const branchScope = new Environment(env); for (const s of branch) collectStmt(s, branchScope, ctx, r, opts, out); }
      return;
    }
    case 'while': {
      while (truthy(evalExpr(stmt.test, env, r))) {
        r.tick();
        // Fresh scope PER ITERATION so a body-local `const` is re-defined each pass (else it leaks
        // from iteration 1 and every row renders the stale first value). The loop counter lives in
        // the outer `env` and reassigns through the parent chain.
        const iterScope = new Environment(env);
        for (const s of stmt.body) collectStmt(s, iterScope, ctx, r, opts, out);
      }
      return;
    }
    // Declarations / effects run for effect; they append NO children.
    case 'const': case 'let': case 'assign': case 'function': case 'component': case 'return':
      execStmt(stmt, env, r, true);
      return;
    default: {
      const u = stmt as { kind?: unknown; span?: Expr['span'] };
      r.error('ML-LANG-UNIMPL', `statement '${String(u.kind)}' not supported in child position`, u.span);
    }
  }
}

/** Lower a for-of body for one iteration: its direct node-producing statements are LIST ITEMS keyed
 *  via listItem (the `hint` carries the iteration ordinal). A block-scope is used so loop-body decls
 *  don't leak between iterations. */
function collectForBody(body: Stmt[], env: Environment, ctx: KeyContext, r: Runner, opts: LowerOptions, hint: ListHint, out: HostValue[]): void {
  const scope = new Environment(env);
  for (const stmt of body) {
    if (stmt.kind === 'expr') collectValue(lowerExprToChildren(stmt.expr, scope, ctx, r, opts, hint), out);
    else collectStmt(stmt, scope, ctx, r, opts, out);   // nested control-flow / decls handled normally
  }
}

/** Append a produced value as a child. Flattens arrays (a producer yielding multiple nodes). Only
 *  NODE-SHAPED values (objects — a built node or a slot) are placed; a bare primitive statement value
 *  (a `"Welcome";` / `42;` / `data.title;` run for effect) is dropped, as is null/undefined. HostValue
 *  is opaque to lang, but every node the host/lang builds is an object, and no node is a primitive —
 *  so "typeof object" cleanly separates node-producing statements from plain-value ones. */
function collectValue(v: unknown, out: HostValue[]): void {
  if (v === null || v === undefined) return;
  if (Array.isArray(v)) { for (const x of v) collectValue(x, out); return; }
  if (typeof v !== 'object') return;   // a bare primitive value statement runs for effect, not placed
  out.push(v);
}

/** Lower a child-position expression to node(s). The common case is a `call`; a bare producer or an
 *  already-instantiated node (a slot expression like `p.body`) is also valid. Returns a HostValue, an
 *  array of them, or null (nothing to append). */
function lowerExprToChildren(expr: Expr, env: Environment, ctx: KeyContext, r: Runner, opts: LowerOptions, hint: ListHint | undefined): unknown {
  if (expr.kind === 'call') return lowerCall(expr, env, ctx, r, opts, hint);
  // A non-call child expression: evaluate it. If it is already a node (a slot, e.g. `p.body`), place
  // it directly (it keeps its caller-site key). collectValue drops a bare non-node primitive; a slot
  // node is an object so it survives.
  return evalExpr(expr, env, r);
}

/** Lower a call in CHILD position (resolution order):
 *   1. Resolve the callee: a bound in-DSL component/function/arrow producer, or an unbound HOST head.
 *   2. Mint this node's key — listItem (author key / content-hash + ordinal) when in a for-of, else a
 *      structural ordinal — always from the AUTHORED head, so shorthands stay legible.
 *   3. A component producer → instantiate (child-collect its body). A function/arrow producer → invoke
 *      and place its returned node(s). A HOST head → eval args IN ORDER, collect the wrap block, and
 *      build the node via resolveCall (lifting the raw args to Arg[]). */
function lowerCall(expr: Extract<Expr, { kind: 'call' }>, env: Environment, ctx: KeyContext, r: Runner, opts: LowerOptions, hint: ListHint | undefined): unknown {
  const callee = expr.callee;

  // Resolve the callee value. A bare ident may be a bound producer OR an unbound HOST head (box/text/
  // …). A member/other expression (e.g. `p.renderItem`) evaluates to a producer value.
  let calleeValue: unknown;
  let identHead: string | undefined;
  if (callee.kind === 'ident') {
    identHead = callee.name;
    calleeValue = env.has(identHead) ? env.get(identHead) : undefined;
  } else {
    calleeValue = evalExpr(callee, env, r);
  }

  // (2a) In-DSL component producer → instantiate + child-collect its body. The head/key use the
  //      component's DECLARED name (so a `renderItem: Card` render-prop keys as Card, not the prop name).
  if (isUserFn(calleeValue) && calleeValue.isComponent) {
    const rawArgs = expr.args.map((a) => lowerArg(a, env, ctx, r, opts));
    const key = mintKey(ctx, calleeValue.name, rawArgs, hint, opts);
    return instantiateComponent(calleeValue, calleeValue.name, key, rawArgs, r, opts);
  }
  // (2b) An arrow producer (e.g. `renderItem: (r) => Card(r)`) → RE-LOWER its body in child position
  //      (evaluating it would hit the evaluator's node-in-expression rejection). Bind the args in a
  //      child of the arrow's captured env, then lower the body expression as this placement's node.
  const arrow = arrowInfo(calleeValue);
  if (arrow) {
    if (++r.depth > r.opt.maxDepth) { r.depth--; throw new BudgetSignal('depth'); }
    try {
      const frame = new Environment(arrow.env);
      bindParams(arrow.params, expr.args.map((a) => evalExpr(a, env, r)), frame);
      if (Array.isArray(arrow.body)) {                 // block-bodied arrow: last statement is the node
        return collectChildren(arrow.body, frame, ctx, r, opts);
      }
      return lowerExprToChildren(arrow.body, frame, ctx, r, opts, hint);
    } finally {
      r.depth--;
    }
  }
  // (2c) A pure `function` producer invoked in child position → its implicit-last-expression node.
  if (isUserFn(calleeValue) && !calleeValue.isComponent) {
    const key = mintKey(ctx, calleeValue.name, [], hint, opts);
    return instantiateFunctionProducer(calleeValue, expr, env, key, r, opts);
  }

  // (3) A HOST head (core vocabulary / registered component / unknown). The head is the authored ident.
  //     A non-ident callee that resolved to a non-producer is an error (fail closed).
  if (identHead === undefined) { r.error('ML-LANG-UNKNOWN-CALL', 'child-position call does not resolve to a node producer', expr.span); return null; }
  const rawArgs = expr.args.map((a) => lowerArg(a, env, ctx, r, opts));
  const key = mintKey(ctx, identHead, rawArgs, hint, opts);
  const children = expr.block ? collectChildren(expr.block, env, childContext(key), r, opts) : [];
  const liftedArgs = rawArgs.map(toArg);
  const resolved = r.opt.env.resolveCall(identHead, key, liftedArgs, children, expr.span);
  // handled:false (an unregistered head, not core vocabulary / registered component / in-DSL component)
  // → an 'unknown' wrapper; the runtime derive materializes a fallback node + a diagnostic (the
  //   extension-seam fallthrough). A test double that answers every head never returns false.
  return resolved.handled ? resolved.value : wrapper('unknown', identHead, key, liftedArgs, children);
}

/** A pure `function` producer invoked in child position yields its implicit-last-expression node. The
 *  function body may itself contain a node-producing call (e.g. `function tile(k){ text(k) }`), which
 *  the evaluator would reject in expression position — so re-lower the body's final expression. */
function instantiateFunctionProducer(fn: UserFn, expr: Extract<Expr, { kind: 'call' }>, env: Environment, key: string, r: Runner, opts: LowerOptions): unknown {
  if (++r.depth > r.opt.maxDepth) { r.depth--; throw new BudgetSignal('depth'); }
  try {
    const frame = new Environment(fn.closure);
    bindParams(fn.params, expr.args.map((a) => evalExpr(a, env, r)), frame);
    // Child-collect the body: its node-producing statements become the produced node(s); the function's
    // declarations run for effect. (A function producer typically has a single trailing node.)
    return collectChildren(fn.body, frame, childContext(key), r, opts);
  } finally {
    r.depth--;
  }
}

/** Mint a node key: listItem when this node is a for-of list item (author `key` prop → stable key;
 *  else content-hash of the args + the iteration ordinal tiebreak), else a structural lexical ordinal.
 *  Keys are minted from the RAW lowered args (before the Arg lift). */
function mintKey(ctx: KeyContext, head: string, args: unknown[], hint: ListHint | undefined, opts: LowerOptions): string {
  if (hint) {
    const content = argContent(args);
    // The reserved `key` prop is RECONCILIATION IDENTITY — it must be its resolved VALUE at derive time,
    // never a reactive Region. Under `reactiveData`, a data-derived key (e.g. `{ key: data.prefix + r.id }`)
    // lowers to a Region like every other data read; resolving it here yields the current key string (it
    // re-resolves to the new value on each re-derive). Left unresolved, String(region) → "[object Object]"
    // and every such row would collapse onto ONE identity key.
    const authorKey = isRegion(content.key) ? content.key.run() ?? null : content.key ?? null;
    return opts.minter.listItem(ctx.parentKey, head, authorKey, hint.ordinal, content);
  }
  return opts.minter.structural(ctx.parentKey, head, nextOrdinal(ctx, head));
}

/** Merge the object-typed args into one record for author-key extraction + content hashing (list
 *  items). Non-object args (a bare `text(k)` content string) contribute nothing to the key content. */
function argContent(args: unknown[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const a of args) {
    if (a !== null && typeof a === 'object' && !Array.isArray(a)) Object.assign(out, a as Record<string, unknown>);
  }
  return out;
}

/** Lower a call ARGUMENT. Two rules layer here:
 *   • REGION rule: if the arg expression — or an object-literal ENTRY within it — statically reads a
 *     reactive `let` binding, emit that position as a `region(() => evalExpr(...))` (a re-runnable
 *     thunk) so the runtime can register a per-attribute leaf effect. Else eager.
 *   • SLOT rule: an arg that is a component/producer CALL (e.g. `body: Card("x")`) is LOWERED to a node
 *     (the evaluator would reject a node in expression position), so it can be placed as a slot.
 *  A bare component/function reference passed as a value stays the producer itself (a render-prop).
 *  Returns the RAW lowered value/object/Region — the top-level call site lifts it into an Arg. */
function lowerArg(arg: Expr, env: Environment, ctx: KeyContext, r: Runner, opts: LowerOptions): unknown {
  // An object literal: lower each entry independently (each entry is itself an arg position).
  if (arg.kind === 'object') {
    const out: Record<string, unknown> = {};
    for (const { key, value } of arg.entries) out[key] = lowerArg(value, env, ctx, r, opts);
    return out;
  }
  // A SLOT FIRST — before the reactive-Region check: an arg that is a call to an in-DSL COMPONENT
  // lowers to an (already-built) node so it can be placed (e.g. `Panel({ body: Card(hover) })`). The
  // slot's OWN reactive args become Regions inside the recursive lowerCall→lowerArg, so a reactive
  // slot is still fully reactive — but the slot itself must not be wrapped as a Region (that would leak
  // a raw Region object into the child list). A pure `function` call in arg position is a normal VALUE
  // call (implicit-last-expr), evaluated — so only a COMPONENT callee triggers slot lowering.
  if (arg.kind === 'call' && calleeIsComponent(arg.callee, env)) {
    return lowerCall(arg, env, ctx, r, opts, undefined);
  }
  // A reactive arg (reads a reactive `let`) → a Region thunk for a runtime leaf effect.
  if (readsReactiveLet(arg, env)) return region(() => evalExpr(arg, env, r));
  return evalExpr(arg, env, r);
}

/** True when a call's callee is a bound in-DSL `component` (so an arg-position call is a SLOT node to
 *  be lowered, not an evaluated value). A pure `function`/arrow callee is NOT a slot — its call is a
 *  value. Only a bare ident is checked (a component is referenced by name); a computed callee that
 *  yields a component is out of scope for slots. */
function calleeIsComponent(callee: Expr, env: Environment): boolean {
  if (callee.kind !== 'ident') return false;
  const v = env.has(callee.name) ? env.get(callee.name) : undefined;
  return isUserFn(v) && v.isComponent;
}

/** Static free-variable scan: does this expression READ a reactive `let` binding (kind:'let') visible
 *  in `env`? Used to decide whether a prop/entry becomes a reactive Region. Conservative and read-only
 *  — it walks the expression tree without evaluating. Arrow bodies are NOT scanned (a handler like
 *  `(h) => { hover = h }` WRITES state; its body runs on invocation, not at derive). */
function readsReactiveLet(expr: Expr, env: Environment): boolean {
  switch (expr.kind) {
    case 'ident':
      return env.meta(expr.name)?.kind === 'let';
    case 'member':
      return readsReactiveLet(expr.object, env);
    case 'index':
      return readsReactiveLet(expr.object, env) || readsReactiveLet(expr.index, env);
    case 'unary':
      return readsReactiveLet(expr.operand, env);
    case 'binary':
      return readsReactiveLet(expr.left, env) || readsReactiveLet(expr.right, env);
    case 'cond':
      return readsReactiveLet(expr.test, env) || readsReactiveLet(expr.then, env) || readsReactiveLet(expr.else, env);
    case 'object':
      return expr.entries.some((e) => readsReactiveLet(e.value, env));
    case 'array':
      return expr.elements.some((e) => readsReactiveLet(e, env));
    case 'call':
      return (expr.callee.kind !== 'ident' && readsReactiveLet(expr.callee, env)) || expr.args.some((a) => readsReactiveLet(a, env));
    // Literals and arrows do not (statically) read a reactive let for the purpose of prop reactivity.
    default:
      return false;
  }
}
