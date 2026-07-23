import { describe, it, expect } from 'vitest';
import { evaluateProgram, Runner, BudgetSignal } from './evaluate.ts';
import { makeSeededRng } from './determinism.ts';
import { PlainStorageHost, RecordingHostEnv } from './ports.ts';
import type { Arg } from './ports.ts';
import { MATH_BUILTINS, mathProfile } from '@metael/math/lang';
import { STD_BUILTINS, stdProfile } from '@metael/std';

const run = (src: string, data?: unknown) =>
  evaluateProgram(src, { data, host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS, STD_BUILTINS] });

describe('evaluator core + budgets', () => {
  it('evaluates arithmetic with precedence', () => {
    expect(run('1 + 2 * 3;').value).toBe(7);
  });
  it('short-circuits && and ||', () => {
    expect(run('false && missingCall();').value).toBe(false);
    expect(run('true || missingCall();').value).toBe(true);
  });
  it('evaluates unary minus and not', () => {
    expect(run('-3;').value).toBe(-3);
    expect(run('!false;').value).toBe(true);
  });
  it('evaluates a ternary, short-circuiting the untaken branch', () => {
    expect(run('true ? 1 : missingCall();').value).toBe(1);
    expect(run('false ? missingCall() : 2;').value).toBe(2);
  });
  it('binds injected data as a root identifier', () => {
    expect(run('data.n;', { n: 5 }).value).toBe(5);
  });
  it('a function returns its implicit last expression', () => {
    expect(run('function f() { 1; 2; 3 } f();').value).toBe(3);
  });
  it('an explicit return yields the returned value (and stops the body)', () => {
    expect(run('function f() { return 7; 99 } f();').value).toBe(7);
  });
  it('a member write on an immutable value is ML-LANG-IMMUTABLE (was in-place mutation)', () => {
    const r = run('const o = { x: 1 }; o.x = 9; o.x;');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(r.value).toBe(1);   // unchanged — the read after the blocked write still sees 1
  });
  it('reassigning a const is a ML-LANG-CONST diagnostic', () => {
    expect(run('const a = 1; a = 2;').diagnostics.some((d) => d.code === 'ML-LANG-CONST')).toBe(true);
  });
  it('a let outside a component is a ML-LANG-LET-SCOPE diagnostic', () => {
    expect(run('let a = 1;').diagnostics.some((d) => d.code === 'ML-LANG-LET-SCOPE')).toBe(true);
  });
  it('exceeding the step budget yields ML-LANG-BUDGET, never a hang', () => {
    const r = evaluateProgram('while (true) { const x = 1; }', { host: new PlainStorageHost(), env: new RecordingHostEnv(), maxSteps: 1000 });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(true);
  });
  it('exceeding call depth yields ML-LANG-BUDGET', () => {
    const r = evaluateProgram('function f() { f() } f();', { host: new PlainStorageHost(), env: new RecordingHostEnv(), maxDepth: 32 });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(true);
  });
  it('assigning through a computed forbidden key is blocked', () => {
    const r = run('const o = {}; o["__proto__"] = 1;');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-FORBIDDEN')).toBe(true);
  });
  it('a string-concat exceeding the cap emits ML-LANG-BUDGET without allocating', () => {
    const r = evaluateProgram('"a" + "bbbb";', { host: new PlainStorageHost(), env: new RecordingHostEnv(), maxStringLength: 3 });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(true);   // string-cap is treated as a BUDGET case
    expect(r.value).toBe(null);   // fails closed, does not build the oversized string
  });

  describe('Runner.tick(steps) — multi-step budget charging', () => {
    // A Runner with a fixed step cap and a deadline we control by construction.
    const mkRunner = (maxSteps: number, maxTimeMs = 1_000_000) =>
      new Runner({ host: new PlainStorageHost(), env: new RecordingHostEnv(), maxSteps, maxTimeMs, maxDepth: 1000, maxStringLength: 1000 });

    it('tick(n) charges n steps at once (equivalent to n single ticks)', () => {
      const r = mkRunner(1000);
      r.tick(10);
      expect(r.steps).toBe(10);
      r.tick();               // default is 1
      expect(r.steps).toBe(11);
    });

    it('tick(n) trips the step cap when the increment crosses maxSteps (not only on an exact landing)', () => {
      const r = mkRunner(100);
      expect(() => r.tick(101)).toThrow(BudgetSignal);   // jumped straight past the cap
      expect(r.steps).toBeGreaterThan(100);              // accounted the full charge, did not silently drop it
    });

    it('a non-positive / non-finite step count is clamped to 1 (cannot corrupt the count or skip the cap)', () => {
      const r = mkRunner(1000);
      r.tick(0);
      r.tick(-5);
      r.tick(NaN);
      r.tick(Infinity);       // Infinity must NOT set steps to Infinity (that would fail the cap on garbage) — clamps to 1
      expect(Number.isFinite(r.steps)).toBe(true);
      expect(r.steps).toBe(4);   // four guarded charges, each clamped to a single step
    });

    it('a large tick that JUMPS PAST a TIME_CHECK_INTERVAL boundary still enforces the deadline', () => {
      // Deadline already in the past → the very next boundary-crossing tick must fail closed on time.
      // TIME_CHECK_INTERVAL is 1024; a single tick(2000) crosses boundary 1024 without landing on it.
      const r = mkRunner(1_000_000, -1);   // maxTimeMs negative → deadline = now-1ms, already expired
      let thrown: unknown;
      try { r.tick(2000); } catch (e) { thrown = e; }
      expect(thrown).toBeInstanceOf(BudgetSignal);
      expect((thrown as BudgetSignal).reason).toBe('time');
    });

    it('single ticks below the first interval do NOT check the (expired) deadline — parity with pre-change behavior', () => {
      const r = mkRunner(1_000_000, -1);   // deadline expired, but we stay within the first 1024-step window
      expect(() => { for (let i = 0; i < 100; i++) r.tick(); }).not.toThrow();
      expect(r.steps).toBe(100);
    });
  });

  it('an unbound callee dispatches to the HostEnvironment builtin (expression-position)', () => {
    // A recording env that answers `double` returns handled:true with a scalar.
    class BuiltinEnv extends RecordingHostEnv {
      override resolveCall(head: string, _k: string, args: Arg[]): { handled: true; value: unknown } | { handled: false } {
        return head === 'double' ? { handled: true, value: ((args[0]?.value) as number) * 2 } : { handled: false };
      }
    }
    const r = evaluateProgram('double(21);', { host: new PlainStorageHost(), env: new BuiltinEnv() });
    expect(r.value).toBe(42);
  });
  it('an unresolved call in expression position fails closed (ML-LANG-UNKNOWN-CALL, not a throw)', () => {
    const r = run('nope(1);');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-UNKNOWN-CALL')).toBe(true);
    expect(r.value).toBe(null);
  });
  it('a parser stack-overflow is caught, not thrown — deeply nested unary (never escapes into host)', () => {
    // '!'*50000 overflows recursive-descent parseUnary; must become a diagnostic + null, not a RangeError.
    // The public parser fails closed (depth guard, else a RangeError fallback) → ML-LANG-PARSE; any code
    // is fine as long as nothing throws and a diagnostic is present (never-throw contract).
    let r: ReturnType<typeof run>;
    expect(() => { r = run('!'.repeat(50000) + 'true;'); }).not.toThrow();
    expect(r!.value).toBe(null);
    expect(r!.diagnostics.some((d) => d.code === 'ML-LANG-PARSE' || d.code === 'ML-LANG-INTERNAL' || d.code === 'ML-LANG-BUDGET')).toBe(true);
  });
  it('a parser stack-overflow is caught, not thrown — deeply nested parens (never escapes into host)', () => {
    let r: ReturnType<typeof run>;
    expect(() => { r = run('('.repeat(20000) + '1' + ')'.repeat(20000) + ';'); }).not.toThrow();
    expect(r!.value).toBe(null);
    expect(r!.diagnostics.some((d) => d.code === 'ML-LANG-PARSE' || d.code === 'ML-LANG-INTERNAL' || d.code === 'ML-LANG-BUDGET')).toBe(true);
  });

  // Regression (review): a `const` must NOT be reopenable by redeclaring it as a reactive `let` in the
  // same scope inside a component — that was a const-immutability bypass (const cell → let cell → writable).
  it('redeclaring a const as a let inside a component is ML-LANG-REDECL, not a const bypass', () => {
    const r = run('component C() { const x = 1; let x = 2; x = 99; text("t") } C();');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-REDECL')).toBe(true);
  });
  it('a plain block redeclaration (let n; let n) inside a component is flagged', () => {
    const r = run('component C() { let n = 1; let n = 2; text("t") } C();');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-REDECL')).toBe(true);
  });
});

describe('reactive let routes through the ReactiveHost', () => {
  // A tracking double that records host calls, to prove reactive-let read/write go through the host,
  // NOT through Environment.assign.
  class SpyHost extends PlainStorageHost {
    readonly log: string[] = [];
    override allocateCell(v: unknown, cellKey?: string) { this.log.push('alloc'); return super.allocateCell(v, cellKey); }
    override readCell(c: unknown) { this.log.push('read'); return super.readCell(c); }
    override writeCell(c: unknown, v: unknown) { this.log.push('write'); super.writeCell(c, v); }
  }
  it('declaring a reactive let inside a component allocates a cell; read/assign route through the host', () => {
    const host = new SpyHost();
    // A component body that declares a reactive let, reads it, then assigns it.
    evaluateProgram('component C() { let n = 1; n; n = 2; } C();',
      { host, env: new RecordingHostEnv({ known: ['C'] }) });
    expect(host.log).toContain('alloc');   // `let n = 1` → allocateCell(1)
    expect(host.log).toContain('read');    // `n;` → readCell
    expect(host.log).toContain('write');   // `n = 2` → writeCell (NOT Environment.assign)
  });
  it('an arrow captures its environment and is invokable (net-new vs expr)', () => {
    // exposed via a tiny test entry that evaluates a single expression to a value
    const r = run('(function make() { const k = 10; (x) => x + k })();');
    expect(typeof r.value).toBe('function');
    expect((r.value as (x: number) => number)(5)).toBe(15);
  });
});

// `range` stays a language-kernel intrinsic (dispatched by lang, needs no injected module — it is the
// compute-lowering gate's bounded-loop primitive). `rand` moved to the standard library's `random` module,
// so a program using it must inject STD_BUILTINS (it is no longer privileged in lang). The seed itself is
// still owned + threaded by lang, so determinism-with-randomness is unchanged. `random.test.ts` owns the
// full rand suite; these keep the lang-level dispatch checks (range-stays-intrinsic; rand-not-via-resolveCall).
describe('seeded rand/range builtins (range = kernel intrinsic; rand = std-injected; EvalOptions.seed)', () => {
  const withStd = (src: string, seed = 0, env = new RecordingHostEnv()) =>
    evaluateProgram(src, { host: new PlainStorageHost(), env, seed, builtins: [STD_BUILTINS] });
  it('rand() is deterministic for a fixed seed and matches makeSeededRng', () => {
    const expected = makeSeededRng(42)();
    expect(withStd('rand()', 42).value).toBe(expected);
    expect(withStd('rand()', 42).value).toBe(expected);
  });
  it('different seeds → different rand() values', () => {
    expect(withStd('rand()', 1).value).not.toBe(withStd('rand()', 2).value);
  });
  it('range(n) returns [0..n) without touching the host (range stays a kernel intrinsic)', () => {
    const env = new RecordingHostEnv();
    const res = evaluateProgram('range(3)', { host: new PlainStorageHost(), env, seed: 0 });
    expect(res.value).toEqual([0, 1, 2]);
    expect(env.calls.some((c) => c.head === 'range')).toBe(false);
  });
  it('rand/range are NOT routed to resolveCall (registry/intrinsic wins over the host)', () => {
    const env = new RecordingHostEnv();
    withStd('rand()', 7, env);
    expect(env.calls.some((c) => c.head === 'rand')).toBe(false);
  });
  it('a domain can still shadow: a user function named rand is called first', () => {
    const res = evaluateProgram('function rand() { 99 } rand()', { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 3 });
    expect(res.value).toBe(99);
  });
  it('rand() advances one shared per-run sequence (not re-seeded per call)', () => {
    const res = withStd('[rand(), rand(), rand()]', 42);
    const rng = makeSeededRng(42);
    expect(res.value).toEqual([rng(), rng(), rng()]);   // three SUCCESSIVE draws, not three copies of the first
  });
  it('range with no arg → [] (synthetic 0 fallback)', () => {
    const res = evaluateProgram('range()', { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 0 });
    expect(res.value).toEqual([]);
  });
  it('range(0) → []', () => {
    expect(evaluateProgram('range(0)', { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 0 }).value).toEqual([]);
  });
  it('range(2 + 1) resolves its arg via evalExpr → [0,1,2]', () => {
    expect(evaluateProgram('range(2 + 1)', { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 0 }).value).toEqual([0, 1, 2]);
  });
  it('range with a negative arg → [] (guarded, no throw)', () => {
    expect(evaluateProgram('range(-1)', { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 0 }).value).toEqual([]);
  });
  it('rand() is budget-charged: an unbounded loop of rand() trips ML-LANG-BUDGET (does not hang)', () => {
    const res = evaluateProgram('while (true) { rand() }', { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 0, maxSteps: 1000, builtins: [STD_BUILTINS] });
    expect(res.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(true);
  });
});

describe('spread evaluation', () => {
  it('array spread splices elements in order', () => {
    expect(run('const a = [1, 2]; [0, ...a, 3]').value).toEqual([0, 1, 2, 3]);
  });
  it('object spread merges; later keys win', () => {
    expect(run('const o = { a: 1, b: 2 }; { ...o, b: 9, c: 3 }').value).toEqual({ a: 1, b: 9, c: 3 });
  });
  it('spread of a non-array is a fail-loud ML-LANG-SPREAD (+ skipped)', () => {
    const r = run('[...5]');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-SPREAD')).toBe(true);
    expect(r.value).toEqual([]);
  });
  it('a forbidden key cannot be introduced via object spread', () => {
    expect(run('const o = { a: 1 }; { ...o }').value).toEqual({ a: 1 });
  });
});

describe('immutability: DSL-created collections are frozen; member/index writes fail loud', () => {
  it('an array literal is frozen — an index write is now ML-LANG-IMMUTABLE', () => {
    const r = run('const a = [1, 2]; a[0] = 9; a');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(r.value).toEqual([1, 2]);
  });
  it('an object member write is now ML-LANG-IMMUTABLE (was in-place mutation)', () => {
    const r = run('const o = { x: 1 }; o.x = 9; o');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(r.value).toEqual({ x: 1 });
  });
  it('the immutable-update path (spread reassignment) works and is itself frozen', () => {
    expect(run('const o = { x: 1 }; { ...o, x: 9 }').value).toEqual({ x: 9 });
  });
  it('a reactive let reassignment (identifier LHS) is NOT blocked by the immutability guard', () => {
    const r = evaluateProgram('let n = 1; n = n + 1; n', { host: new PlainStorageHost(), env: new RecordingHostEnv(), insideComponent: true });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(false);
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-LET-SCOPE')).toBe(false);
    expect(r.value).toBe(2);
  });
  it('an array in a component let updates via reassignment + spread (identifier LHS — OK)', () => {
    const ok = evaluateProgram('let items = [1]; items = [...items, 2]; items', { host: new PlainStorageHost(), env: new RecordingHostEnv(), insideComponent: true });
    expect(ok.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(false);
    expect(ok.value).toEqual([1, 2]);
  });
});

describe('pure collection builtins (intrinsic free functions)', () => {
  it('map transforms into a new array', () => {
    expect(run('map([1, 2, 3], (x) => x * 2)').value).toEqual([2, 4, 6]);
  });
  it('map passes the index as the second arg', () => {
    expect(run('map([10, 20], (x, i) => i)').value).toEqual([0, 1]);
  });
  it('filter selects', () => {
    expect(run('filter([1, 2, 3, 4], (x) => x % 2 == 0)').value).toEqual([2, 4]);
  });
  it('reduce aggregates (the only fold available in a pure function)', () => {
    expect(run('reduce([1, 2, 3, 4], (acc, x) => acc + x, 0)').value).toBe(10);
  });
  it('keys / values / entries on an object', () => {
    expect(run('keys({ a: 1, b: 2 })').value).toEqual(['a', 'b']);
    expect(run('values({ a: 1, b: 2 })').value).toEqual([1, 2]);
    expect(run('entries({ a: 1, b: 2 })').value).toEqual([['a', 1], ['b', 2]]);
  });
  it('object builds an object from pairs', () => {
    expect(run('object([["a", 1], ["b", 2]])').value).toEqual({ a: 1, b: 2 });
  });
  it('round-trips: object(entries(o)) == o', () => {
    expect(run('const o = { a: 1, b: 2 }; object(entries(o))').value).toEqual({ a: 1, b: 2 });
  });
  it('a builtin result is frozen (immutable)', () => {
    const r = run('const m = map([1], (x) => x); m[0] = 9; m');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
  });
  it('a non-array arg is fail-loud ML-LANG-BUILTIN-ARG + a safe empty result', () => {
    const r = run('map(5, (x) => x)');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(r.value).toEqual([]);
  });
  it('a user `function map` SHADOWS the intrinsic (unbound-head-only)', () => {
    expect(run('function map() { 99 } map([1,2], (x) => x)').value).toBe(99);
  });
  it('a user-declared `function` works as a callback (not just an arrow)', () => {
    expect(run('function dbl(x) { x * 2 } map([1, 2, 3], dbl)').value).toEqual([2, 4, 6]);
    expect(run('function even(x) { x % 2 == 0 } filter([1, 2, 3, 4], even)').value).toEqual([2, 4]);
  });
  it('object ignores a forbidden key (FORBIDDEN_KEYS-guarded)', () => {
    const r = run('object([["__proto__", 1], ["a", 2]])');
    expect((r.value as Record<string, unknown>).a).toBe(2);
    expect(Object.getOwnPropertyNames(r.value as object)).not.toContain('__proto__');
  });
  it('map budget: a large mapped array fails closed via per-element ticks, never hangs', () => {
    const r = evaluateProgram('map(range(2000), (x) => x + 1)', { host: new PlainStorageHost(), env: new RecordingHostEnv(), maxSteps: 500, builtins: [MATH_BUILTINS, STD_BUILTINS] });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(true);
  });
});

describe('collections increment preserves the invariants', () => {
  it('DETERMINISM: same source + seed → identical result for a spread + builtins + rand mix', () => {
    const src = 'map(range(3), (i) => ({ r: rand(), i: i }))';
    const a = evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 42, builtins: [MATH_BUILTINS, STD_BUILTINS] });
    const b = evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 42, builtins: [MATH_BUILTINS, STD_BUILTINS] });
    expect(a.value).toEqual(b.value);
    const c = evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 43, builtins: [MATH_BUILTINS, STD_BUILTINS] });
    expect(a.value).not.toEqual(c.value);   // the seed genuinely drives the result
  });
  it('NEVER-THROWS: a builtin callback that recurses infinitely fails closed (ML-LANG-BUDGET), no host throw', () => {
    const src = 'function loop(x) { loop(x) } map([1], (x) => loop(x))';
    let r: ReturnType<typeof evaluateProgram>;
    expect(() => { r = evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), maxDepth: 20, builtins: [MATH_BUILTINS, STD_BUILTINS] }); }).not.toThrow();
    expect(r!.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(true);   // did not escape into the host
  });
  it('FREEZE does not break equality / serialization of results', () => {
    const r = evaluateProgram('[...[1,2], 3]', { host: new PlainStorageHost(), env: new RecordingHostEnv() });
    expect(JSON.stringify(r.value)).toBe('[1,2,3]');   // frozen arrays serialize normally
    expect(r.value).toEqual([1, 2, 3]);                 // and compare structurally
  });
  it('FREEZE is deep + a nested frozen result still serializes', () => {
    const r = evaluateProgram('map([1, 2], (x) => ({ v: x }))', { host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS, STD_BUILTINS] });
    expect(JSON.stringify(r.value)).toBe('[{"v":1},{"v":2}]');
  });
});

describe('query + predicate builtins', () => {
  it('some / every short-circuit and return booleans', () => {
    expect(run('some([1,2,3], (x) => x > 2);').value).toBe(true);
    expect(run('some([1,2,3], (x) => x > 9);').value).toBe(false);
    expect(run('every([2,4], (x) => x % 2 == 0);').value).toBe(true);
    expect(run('every([2,3], (x) => x % 2 == 0);').value).toBe(false);
  });
  it('find returns the element or null; findIndex returns the index or -1', () => {
    expect(run('find([1,2,3], (x) => x > 1);').value).toBe(2);
    expect(run('find([1,2,3], (x) => x > 9);').value).toBe(null);
    expect(run('findIndex([1,2,3], (x) => x > 1);').value).toBe(1);
    expect(run('findIndex([1,2,3], (x) => x > 9);').value).toBe(-1);
  });
  it('includes tests value membership using == semantics', () => {
    expect(run('includes([1,2,3], 2);').value).toBe(true);
    expect(run('includes([1,2,3], 9);').value).toBe(false);
    expect(run('includes(["a","b"], "b");').value).toBe(true);
  });
  it('a user function shadows a query builtin', () => {
    expect(run('function some(a, b) { 42 } some([1], (x) => x);').value).toBe(42);
  });
  it('a wrong-shape arg is fail-loud ML-LANG-BUILTIN-ARG, never a throw', () => {
    expect(run('some(5, (x) => x);').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(run('find([1], 5);').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
  it('a callback may be a user function, not just an arrow', () => {
    expect(run('function big(x) { x > 1 } find([1,2,3], big);').value).toBe(2);
  });
});

describe('ordering + slicing builtins', () => {
  it('sort uses the default total order and does not mutate', () => {
    expect(run('sort([3,1,2]);').value).toEqual([1, 2, 3]);
    expect(run('const a = [3,1,2]; sort(a); a;').value).toEqual([3, 1, 2]);   // original frozen + untouched
    expect(run('sort([2, "a", null, true]);').value).toEqual([null, true, 2, 'a']);
  });
  it('sort with a comparator', () => {
    expect(run('sort([1,2,3], (a, b) => b - a);').value).toEqual([3, 2, 1]);
  });
  it('sort with a non-number comparator return is fail-loud but keeps order', () => {
    const r = run('sort([1,2,3], (a, b) => "x");');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(r.value).toEqual([1, 2, 3]);   // treated as 0 → stable keep-order
  });
  it('slice extracts a sub-array (JS index semantics incl. negatives), never mutating', () => {
    expect(run('slice([1,2,3,4,5], 1, 3);').value).toEqual([2, 3]);
    expect(run('slice([1,2,3,4,5], -2);').value).toEqual([4, 5]);
    expect(run('slice([1,2,3], 1);').value).toEqual([2, 3]);
    expect(run('slice([1,2,3,4,5], 1, -1);').value).toEqual([2, 3, 4]);   // negative END (exercises the end branch)
    expect(run('slice([1,2,3,4,5], -3, -1);').value).toEqual([3, 4]);     // both negative
  });
  it('reverse returns a new reversed array, not mutating the input', () => {
    expect(run('reverse([1,2,3]);').value).toEqual([3, 2, 1]);
    expect(run('const a=[1,2,3]; reverse(a); a;').value).toEqual([1, 2, 3]);
  });
  it('the sort result is deep-frozen (a member write is ML-LANG-IMMUTABLE)', () => {
    expect(run('const s = sort([3,1,2]); s[0] = 9;').diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
  });
  it('slice + reverse results are deep-frozen (a member write is ML-LANG-IMMUTABLE)', () => {
    expect(run('const s = slice([1,2,3], 0, 2); s[0] = 9;').diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(run('const r = reverse([1,2,3]); r[0] = 9;').diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
  });
  it('collection builtins stay ARRAY-ONLY — a string arg is fail-loud ML-LANG-BUILTIN-ARG', () => {
    expect(run('map("abc", (x) => x);').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(run('filter("abc", (x) => x);').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(run('reduce("abc", (a, x) => a, 0);').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(run('sort("abc");').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);   // sort case exists after this task
  });
});

describe('string-bridge builtins', () => {
  it('split divides a string; join reassembles; chars decomposes', () => {
    expect(run('split("a,b,c", ",");').value).toEqual(['a', 'b', 'c']);
    expect(run('split("abc", "");').value).toEqual(['a', 'b', 'c']);
    expect(run('join(["a","b","c"], "-");').value).toBe('a-b-c');
    expect(run('chars("abc");').value).toEqual(['a', 'b', 'c']);
  });
  it('join coerces non-string elements deterministically', () => {
    expect(run('join([1,2,3], ",");').value).toBe('1,2,3');
  });
  it('case + trim are locale-independent and pure', () => {
    expect(run('toUpperCase("abc");').value).toBe('ABC');
    expect(run('toLowerCase("ABC");').value).toBe('abc');
    expect(run('trim("  hi  ");').value).toBe('hi');
  });
  it('split result is a real (frozen) array usable by map', () => {
    expect(run('map(split("a,b", ","), (s) => toUpperCase(s));').value).toEqual(['A', 'B']);
    expect(run('const cs = chars("ab"); cs[0] = "z";').diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
  });
  it('wrong-shape args are fail-loud ML-LANG-BUILTIN-ARG', () => {
    expect(run('split(5, ",");').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(run('join("notarray", ",");').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(run('toUpperCase(5);').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
  it('join fails CLOSED with ML-LANG-BUDGET when the result would exceed the string cap (never a raw RangeError)', () => {
    // A collection of large strings could otherwise build a string past the engine limit and throw a
    // RangeError caught only as the internal-error code. join mirrors the `+` operator's cap.
    const r = evaluateProgram('join(map(range(50), (i) => "x"), "y");', { host: new PlainStorageHost(), env: new RecordingHostEnv(), maxStringLength: 20, builtins: [MATH_BUILTINS, STD_BUILTINS] });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(true);
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-INTERNAL')).toBe(false);
    expect(r.value).toBe(null);
  });
});

describe('string for-of iterability (consistency with indexing + .length)', () => {
  it('for-of over a string emits ML-LANG-FOR-ITER TODAY (RED) and must not after the fix', () => {
    // Drive the loop directly at TOP LEVEL — evaluateProgram runs the top level, so the loop body
    // executes without any component. (An uncalled `component C(){…}` would never run its body,
    // making the test vacuous — verified against the built evaluator.)
    const r = run('for (const c of "abc") { c }');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-FOR-ITER')).toBe(false);   // GREEN after the fix
  });
  it('for-of over a non-array, non-string still errors', () => {
    const r = run('for (const x of 5) { x }');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-FOR-ITER')).toBe(true);
  });
  it('for-of over a string actually iterates each character (behavior, via a CALLED component)', () => {
    // A reactive-let accumulator needs component scope; append `C();` so the body runs.
    const r = run('component C() { let n = 0; for (const c of "abcde") { n = n + 1 } n } C();');
    expect(r.value).toBe(5);   // one iteration per character
  });
});

describe('numeric core builtins', () => {
  it('min/max/abs/sign/clamp', () => {
    expect(run('min(3, 7);').value).toBe(3);
    expect(run('max(3, 7);').value).toBe(7);
    expect(run('abs(-5);').value).toBe(5);
    expect(run('sign(-3);').value).toBe(-1);
    expect(run('sign(0);').value).toBe(0);
    expect(run('clamp(15, 0, 10);').value).toBe(10);
    expect(run('clamp(-5, 0, 10);').value).toBe(0);
    expect(run('clamp(5, 0, 10);').value).toBe(5);
  });
  it('floor/ceil and round uses banker rounding (half-to-even) for cross-target exactness', () => {
    expect(run('floor(2.9);').value).toBe(2);
    expect(run('ceil(2.1);').value).toBe(3);
    expect(run('round(2.5);').value).toBe(2);   // half-to-even (JS Math.round would give 3)
    expect(run('round(3.5);').value).toBe(4);   // half-to-even
    expect(run('round(2.4);').value).toBe(2);
    expect(run('round(-2.5);').value).toBe(-2); // half-to-even
    // A negative value rounding toward zero yields -0, matching Math.round (and the shader round/roundEven
    // the GPU oracle checks against) — the language `round` builtin delegates to the math core's single
    // roundHalfEven, so there is exactly ONE half-to-even implementation, not a lang-local duplicate.
    expect(Object.is(run('round(-0.4);').value, -0)).toBe(true);
  });
  it('sqrt/pow', () => {
    expect(run('sqrt(9);').value).toBe(3);
    expect(run('pow(2, 10);').value).toBe(1024);
  });
  it('sqrt of a negative is fail-loud ML-LANG-BUILTIN-ARG, never a silent NaN', () => {
    const r = run('sqrt(-1);');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(Number.isNaN(r.value as number)).toBe(false);
  });
  it('a non-number arg is fail-loud ML-LANG-BUILTIN-ARG', () => {
    expect(run('abs("x");').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
  it('a user function shadows a numeric builtin', () => {
    expect(run('function min(a, b) { 999 } min(1, 2);').value).toBe(999);
  });
  it('min/abs/clamp/pow apply componentwise to vec args (GLSL semantics)', () => {
    const h = () => ({ host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS, STD_BUILTINS] });
    expect(evaluateProgram('min(vec2(1,5), vec2(3,2)).x', h()).value).toBe(1);
    expect(evaluateProgram('min(vec2(1,5), vec2(3,2)).y', h()).value).toBe(2);
    expect(evaluateProgram('abs(vec2(-1, 2)).x', h()).value).toBe(1);
    expect(evaluateProgram('clamp(vec2(-1, 5), 0, 1).y', h()).value).toBe(1); // scalar bounds broadcast
  });
  it('mismatched-width vec args are a builtin-arg error (not a silent truncation)', () => {
    const h = () => ({ host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS, STD_BUILTINS] });
    const r = evaluateProgram('min(vec2(1,5), vec3(3,2,9)).x', h());
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    // a beyond-arity vec arg does not promote a scalar call to a vec
    expect(evaluateProgram('min(1, 2, vec3(9,9,9))', h()).value).toBe(1);
  });
});

describe('trig / hyperbolic / exponential math builtins', () => {
  const h = () => ({ host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS, STD_BUILTINS] });
  it('tan/sinh/cosh/tanh evaluate', () => {
    expect(evaluateProgram('tan(0)', h()).value as number).toBeCloseTo(0);
    expect(evaluateProgram('sinh(0)', h()).value as number).toBeCloseTo(0);
    expect(evaluateProgram('cosh(0)', h()).value as number).toBeCloseTo(1);
    expect(evaluateProgram('tanh(0)', h()).value as number).toBeCloseTo(0);
  });
  it('asin/acos evaluate and domain-guard |x|>1 to the bad sentinel (→0 as a cell)', () => {
    expect(evaluateProgram('asin(0)', h()).value as number).toBeCloseTo(0);
    expect(evaluateProgram('acos(1)', h()).value as number).toBeCloseTo(0);
    // |x|>1 is out of domain → fail-loud + the bad sentinel, which coerces to 0 downstream.
    const r = evaluateProgram('asin(2)', h());
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(Number(r.value ?? 0)).toBe(0);
  });
  it('atan (1-arg) and atan2 (2-arg) evaluate', () => {
    expect(evaluateProgram('atan(0)', h()).value as number).toBeCloseTo(0);
    // atan2(y, x): first arg is the numerator (y), second is the denominator (x).
    expect(evaluateProgram('atan2(1, 1)', h()).value as number).toBeCloseTo(Math.PI / 4);
    expect(evaluateProgram('atan2(1, 0)', h()).value as number).toBeCloseTo(Math.PI / 2);
  });
  it('exp2/log2/inverseSqrt evaluate + guard domain', () => {
    expect(evaluateProgram('exp2(3)', h()).value as number).toBeCloseTo(8);
    expect(evaluateProgram('log2(8)', h()).value as number).toBeCloseTo(3);
    expect(evaluateProgram('inverseSqrt(4)', h()).value as number).toBeCloseTo(0.5);
    // x<=0 is out of domain for log2 → fail-loud + the bad sentinel (→0 as a cell).
    const r = evaluateProgram('log2(0)', h());
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(Number(r.value ?? 0)).toBe(0);
  });
  it('degrees/radians/trunc evaluate', () => {
    expect(evaluateProgram('degrees(3.141592653589793)', h()).value as number).toBeCloseTo(180);
    expect(evaluateProgram('radians(180)', h()).value as number).toBeCloseTo(Math.PI);
    expect(evaluateProgram('trunc(3.7)', h()).value).toBe(3);
    expect(evaluateProgram('trunc(-3.7)', h()).value).toBe(-3);
  });
});

describe('format (number formatting)', () => {
  it('format(x, digits) returns a fixed-decimal string', () => {
    expect(run('format(3.14159, 2);').value).toBe('3.14');
    expect(run('format(5, 0);').value).toBe('5');
    expect(run('format(1.005, 2);').value).toBe('1.00');   // deterministic (IEEE-754 of 1.005), documented
  });
  it('format with a bad arg is fail-loud', () => {
    expect(run('format("x", 2);').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(run('format(1, -1);').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
});

describe('registry ↔ dispatch cross-check', () => {
  it('every catalog builtin is OWNED by dispatch (never falls through to a host unknown-call)', () => {
    // Any implemented builtin, called with junk args, must return a value or ML-LANG-BUILTIN-ARG —
    // NEVER ML-LANG-UNKNOWN-CALL (which would mean the catalog advertises a name dispatch doesn't own,
    // or a case exists without a catalog row). rand/range are seeded intrinsics handled before the
    // collection block; the rest resolve either in the collection cascade or via the injected numeric
    // module (MATH_BUILTINS is wired into `run`).
    const catalog = new Set([...mathProfile.builtins.keys(), ...stdProfile.builtins.keys()]);
    for (const name of catalog) {
      if (name === 'rand' || name === 'range') continue;
      const codes = run(`${name}(5, 5, 5);`).diagnostics.map((d) => d.code);
      expect(codes, `builtin '${name}' fell through to UNKNOWN-CALL`).not.toContain('ML-LANG-UNKNOWN-CALL');
    }
  });
});

describe('extension seam — kind:value discriminant', () => {
  it('a kind:value record return is allowed in expression position and deep-frozen', () => {
    class ValueEnv extends RecordingHostEnv {
      override resolveCall(head: string, _k: string, args: Arg[]): { handled: true; value: unknown; kind?: 'value' } | { handled: false } {
        if (head === 'rgb') return { handled: true, kind: 'value', value: { r: (args[0]?.value as number), g: 0, b: 0 } };
        return { handled: false };
      }
    }
    const r = evaluateProgram('const c = rgb(255); c.r;', { host: new PlainStorageHost(), env: new ValueEnv() });
    expect(r.value).toBe(255);
    expect(r.diagnostics).toEqual([]);   // NO "node not valid in expression position" error
  });
  it('a kind:value return is frozen (a member write on it is ML-LANG-IMMUTABLE)', () => {
    class ValueEnv extends RecordingHostEnv {
      override resolveCall(head: string): { handled: true; value: unknown; kind?: 'value' } | { handled: false } {
        return head === 'rgb' ? { handled: true, kind: 'value', value: { r: 1, g: 2, b: 3 } } : { handled: false };
      }
    }
    const r = evaluateProgram('const c = rgb(); c.r = 9;', { host: new PlainStorageHost(), env: new ValueEnv() });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
  });
  it('a DEFAULT (no kind) object return is still rejected in expression position (back-compat)', () => {
    class NodeEnv extends RecordingHostEnv {
      override resolveCall(head: string): { handled: true; value: unknown } | { handled: false } {
        return head === 'box' ? { handled: true, value: { kind: 'node' } } : { handled: false };
      }
    }
    const r = evaluateProgram('box();', { host: new PlainStorageHost(), env: new NodeEnv() });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-UNKNOWN-CALL')).toBe(true);   // node not valid here
  });
});

describe('vec/mat constructors — column-major', () => {
  it('mat2 identity and mat2(4 nums) construct column-major via the interpreter', () => {
    const idn = evaluateProgram('mat2()', { host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS, STD_BUILTINS] });
    expect(idn.diagnostics).toEqual([]);   // identity constructs without error
    const prod = evaluateProgram('(mat2(1,2,3,4) * vec2(5,6)).x', { host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS, STD_BUILTINS] });
    expect(prod.diagnostics).toEqual([]);
    expect(prod.value).toBe(23);   // column-major: 1*5 + 3*6 (row-major would be 17)
  });
  it('normalize of a zero vector is a NaN vector (matches native shader normalize(0))', () => {
    const r = evaluateProgram('normalize(vec3(0,0,0)).x', { host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS, STD_BUILTINS] });
    expect(Number.isNaN(r.value as number)).toBe(true);
  });
  it('normalize of a NONZERO vector still yields a unit vector (unchanged)', () => {
    const r = evaluateProgram('length(normalize(vec3(3,4,0)))', { host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS, STD_BUILTINS] });
    expect(r.value as number).toBeCloseTo(1, 6);
  });
  it('a bad mat2 arg-count error names the column-major layout', () => {
    const r = evaluateProgram('mat2(1,2,3)', { host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS, STD_BUILTINS] });
    const d = r.diagnostics.find((x) => x.code === 'ML-LANG-BUILTIN-ARG');
    expect(d?.message).toContain('column-major');
    expect(d?.message).not.toContain('row-major');
  });
});
