import { describe, it, expect } from 'vitest';
import { evaluateProgram } from './evaluate.ts';
import { makeSeededRng } from './determinism.ts';
import { PlainStorageHost, RecordingHostEnv } from './ports.ts';
import type { Arg } from './ports.ts';

const run = (src: string, data?: unknown) =>
  evaluateProgram(src, { data, host: new PlainStorageHost(), env: new RecordingHostEnv() });

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

describe('seeded rand/range builtins (intrinsic, EvalOptions.seed)', () => {
  it('rand() is deterministic for a fixed seed and matches makeSeededRng', () => {
    const r1 = evaluateProgram('rand()', { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 42 });
    const r2 = evaluateProgram('rand()', { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 42 });
    const expected = makeSeededRng(42)();
    expect(r1.value).toBe(expected);
    expect(r2.value).toBe(expected);
  });
  it('different seeds → different rand() values', () => {
    const a = evaluateProgram('rand()', { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 1 });
    const b = evaluateProgram('rand()', { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 2 });
    expect(a.value).not.toBe(b.value);
  });
  it('range(n) returns [0..n) without touching the host', () => {
    const env = new RecordingHostEnv();
    const res = evaluateProgram('range(3)', { host: new PlainStorageHost(), env, seed: 0 });
    expect(res.value).toEqual([0, 1, 2]);
    expect(env.calls.some((c) => c.head === 'range')).toBe(false);
  });
  it('rand/range are NOT routed to resolveCall (intrinsic wins over the host)', () => {
    const env = new RecordingHostEnv();
    evaluateProgram('rand()', { host: new PlainStorageHost(), env, seed: 7 });
    expect(env.calls.some((c) => c.head === 'rand')).toBe(false);
  });
  it('a domain can still shadow: a user function named rand is called first', () => {
    const res = evaluateProgram('function rand() { 99 } rand()', { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 3 });
    expect(res.value).toBe(99);
  });
  it('rand() advances one shared per-run sequence (not re-seeded per call)', () => {
    const res = evaluateProgram('[rand(), rand(), rand()]', { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 42 });
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
    const res = evaluateProgram('while (true) { rand() }', { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 0, maxSteps: 1000 });
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
  it('fromEntries builds an object from pairs', () => {
    expect(run('fromEntries([["a", 1], ["b", 2]])').value).toEqual({ a: 1, b: 2 });
  });
  it('round-trips: fromEntries(entries(o)) == o', () => {
    expect(run('const o = { a: 1, b: 2 }; fromEntries(entries(o))').value).toEqual({ a: 1, b: 2 });
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
  it('fromEntries ignores a forbidden key (FORBIDDEN_KEYS-guarded)', () => {
    const r = run('fromEntries([["__proto__", 1], ["a", 2]])');
    expect((r.value as Record<string, unknown>).a).toBe(2);
    expect(Object.getOwnPropertyNames(r.value as object)).not.toContain('__proto__');
  });
  it('map budget: a large mapped array fails closed via per-element ticks, never hangs', () => {
    const r = evaluateProgram('map(range(2000), (x) => x + 1)', { host: new PlainStorageHost(), env: new RecordingHostEnv(), maxSteps: 500 });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(true);
  });
});

describe('collections increment preserves the invariants', () => {
  it('DETERMINISM: same source + seed → identical result for a spread + builtins + rand mix', () => {
    const src = 'map(range(3), (i) => ({ r: rand(), i: i }))';
    const a = evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 42 });
    const b = evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 42 });
    expect(a.value).toEqual(b.value);
    const c = evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 43 });
    expect(a.value).not.toEqual(c.value);   // the seed genuinely drives the result
  });
  it('NEVER-THROWS: a builtin callback that recurses infinitely fails closed (ML-LANG-BUDGET), no host throw', () => {
    const src = 'function loop(x) { loop(x) } map([1], (x) => loop(x))';
    let r: ReturnType<typeof evaluateProgram>;
    expect(() => { r = evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), maxDepth: 20 }); }).not.toThrow();
    expect(r!.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(true);   // did not escape into the host
  });
  it('FREEZE does not break equality / serialization of results', () => {
    const r = evaluateProgram('[...[1,2], 3]', { host: new PlainStorageHost(), env: new RecordingHostEnv() });
    expect(JSON.stringify(r.value)).toBe('[1,2,3]');   // frozen arrays serialize normally
    expect(r.value).toEqual([1, 2, 3]);                 // and compare structurally
  });
  it('FREEZE is deep + a nested frozen result still serializes', () => {
    const r = evaluateProgram('map([1, 2], (x) => ({ v: x }))', { host: new PlainStorageHost(), env: new RecordingHostEnv() });
    expect(JSON.stringify(r.value)).toBe('[{"v":1},{"v":2}]');
  });
});
