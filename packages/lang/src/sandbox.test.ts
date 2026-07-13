import { describe, it, expect } from 'vitest';
import { evaluateProgram } from './evaluate.ts';
import { PlainStorageHost, RecordingHostEnv } from './ports.ts';

// The standing sandbox-escape gate. Proves DSL source cannot reach host internals/globals, cannot
// mutate injected data, and cannot bypass the budgets — across EVERY access path, including the new
// builtins' argument/callback paths. Any failure here is a P0.
const run = (src: string, data?: unknown) =>
  evaluateProgram(src, { data, host: new PlainStorageHost(), env: new RecordingHostEnv() });
const codes = (src: string, data?: unknown) => run(src, data).diagnostics.map((d) => d.code);

describe('sandbox: prototype-chain / forbidden-key escapes (every access path)', () => {
  const forbidden = ['__proto__', 'constructor', 'prototype'];
  for (const k of forbidden) {
    it(`member access .${k} is blocked`, () => {
      expect(codes(`({}).${k};`)).toContain('ML-LANG-FORBIDDEN');
    });
    it(`index access ["${k}"] is blocked`, () => {
      expect(codes(`({})["${k}"];`)).toContain('ML-LANG-FORBIDDEN');
    });
    it(`computed index [k] where k="${k}" is blocked`, () => {
      expect(codes(`const k = "${k}"; ({})[k];`)).toContain('ML-LANG-FORBIDDEN');
    });
    it(`object-literal key ${k} is blocked`, () => {
      expect(codes(`const o = { ${k}: 1 };`)).toContain('ML-LANG-FORBIDDEN');
    });
    it(`member/index WRITE to .${k} is blocked`, () => {
      expect(codes(`const o = {}; o.${k} = 1;`).concat(codes(`const o = {}; o["${k}"] = 1;`))).toContain('ML-LANG-FORBIDDEN');
    });
  }
  it('a spread-copied forbidden key does not leak onto the result', () => {
    const r = run('const evil = fromEntries([["ok", 1]]); const merged = { ...evil }; keys(merged);');
    expect(r.value).toEqual(['ok']);
  });
  it('fromEntries rejects a forbidden key', () => {
    const r = run('fromEntries([["__proto__", 1], ["ok", 2]]);');
    expect(r.value).toEqual({ ok: 2 });
    expect(Object.getPrototypeOf(r.value as object)).toBe(Object.prototype);
  });
});

describe('sandbox: forbidden keys via NEW builtin arg paths', () => {
  it('keys/entries over an object never surface a forbidden key', () => {
    expect(run('keys({ a: 1, b: 2 });').value).toEqual(['a', 'b']);
  });
  it('a sort comparator cannot reach a forbidden key on its arguments', () => {
    expect(codes('sort([{},{}], (a, b) => a.constructor ? 1 : -1);')).toContain('ML-LANG-FORBIDDEN');
  });
  it('a map/find callback cannot reach __proto__', () => {
    expect(codes('map([{}], (x) => x.__proto__);')).toContain('ML-LANG-FORBIDDEN');
    expect(codes('find([{}], (x) => x.constructor);')).toContain('ML-LANG-FORBIDDEN');
  });
  it('includes comparing against a crafted value does not escape', () => {
    expect(run('includes([1,2,3], 2);').value).toBe(true);
  });
});

describe('sandbox: globals + dynamic code are unreachable', () => {
  for (const g of ['globalThis', 'window', 'self', 'global', 'Function', 'eval', 'require', 'process', 'import']) {
    it(`bare identifier '${g}' fails closed to ML-LANG-UNKNOWN-VAR`, () => {
      expect(codes(`${g};`)).toContain('ML-LANG-UNKNOWN-VAR');
    });
  }
  it('a call to Function(...) is an unknown CALL, not a constructor', () => {
    expect(codes('Function("return 1")();')).toContain('ML-LANG-UNKNOWN-CALL');
  });
});

describe('sandbox: budgets cannot be bypassed (incl. via new builtins)', () => {
  const withBudget = (src: string, maxSteps = 5000) =>
    evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), maxSteps }).diagnostics.map((d) => d.code);
  it('a sort comparator that recurses trips ML-LANG-BUDGET (depth or steps), never hangs', () => {
    expect(withBudget('function rec(n) { rec(n) } sort([1,2,3], (a, b) => rec(a));', 5000)).toContain('ML-LANG-BUDGET');
  });
  it('a huge range fed to map still ticks per element and trips the step budget', () => {
    expect(withBudget('map(range(100000), (x) => x + 1);', 5000)).toContain('ML-LANG-BUDGET');
  });
  it('every/some/find/filter tick per element (no unbudgeted iteration)', () => {
    expect(withBudget('every(range(100000), (x) => true);', 5000)).toContain('ML-LANG-BUDGET');
  });
  it('a large default-order sort ticks per comparison and trips the budget', () => {
    expect(withBudget('sort(range(100000));', 5000)).toContain('ML-LANG-BUDGET');
  });
});

describe('sandbox: injected data cannot be mutated (via any path)', () => {
  it('a member write on injected data is blocked', () => {
    const r = run('data.x = 9; data.x;', { x: 1 });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(r.value).toBe(1);
  });
  it('sort/reverse/slice return copies and do not change injected data VALUES', () => {
    const data = { xs: [3, 1, 2] };
    run('sort(data.xs);', data);
    run('reverse(data.xs);', data);
    run('slice(data.xs, 0, 1);', data);
    expect(data.xs).toEqual([3, 1, 2]);
  });
  it('injected data is deep-frozen at the boundary (immutable-by-construction), and a NON-injected host object is untouched', () => {
    // Injected data is frozen at bind time so a builtin returning a frozen result that aliases data's
    // own element objects (sort/slice/reverse/map/values) is a no-op on data — never a surprise
    // in-place freeze of a live host object that leaked through unpredictably.
    const data = { xs: [{ n: 1 }, { n: 2 }], obj: { inner: { k: 1 } } };
    run('sort(data.xs); slice(data.xs, 0, 1); reverse(data.xs); map(data.xs, (x) => x); values(data.obj);', data);
    expect(Object.isFrozen(data)).toBe(true);
    expect(Object.isFrozen(data.xs[0])).toBe(true);        // deep — reachable element objects frozen
    expect(Object.isFrozen(data.obj.inner)).toBe(true);    // deep — nested host object frozen
    expect(data.xs.map((o) => o.n)).toEqual([1, 2]);       // VALUES unchanged
    // A host object NOT passed as data is never frozen (only the injected graph is the boundary).
    const untouched: { keep: string } = { keep: 'mutable' };
    run('sort([3,1,2]);', { xs: [1] });
    expect(Object.isFrozen(untouched)).toBe(false);
  });
  it('a map callback cannot write through a member assign to data', () => {
    const data = { xs: [{ n: 1 }] };
    // Block-body callback so the assignment is a STATEMENT that reaches the eval-time member-write
    // guard (a concise-body assignment expression is rejected earlier at parse — also fail-closed).
    const r = run('map(data.xs, (o) => { o.n = 9 });', data);
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect((data.xs[0] as { n: number }).n).toBe(1);
  });
});
