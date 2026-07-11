import { describe, it, expect } from 'vitest';
import { evaluateProgram } from './evaluate.ts';
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
  it('assigns through a member LValue (o.x = v)', () => {
    expect(run('const o = { x: 1 }; o.x = 9; o.x;').value).toBe(9);
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

describe('reactive let routes through the ReactiveHost (F5)', () => {
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
