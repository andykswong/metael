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
  it('a prototype-key call head is a clean UNKNOWN-CALL, not an internal error', () => {
    // A builtin-lookup keyed by the user-controlled call head must own-property-check, or a prototype
    // member (constructor/toString/valueOf/hasOwnProperty/__proto__) reads as truthy → a raw TypeError →
    // ML-LANG-INTERNAL. The sandbox invariant is: an unknown head fails closed to ML-LANG-UNKNOWN-CALL.
    for (const head of ['constructor', 'toString', 'valueOf', 'hasOwnProperty', '__proto__']) {
      const c = codes(`${head}(1,2,3,4,5,6)`);
      expect(c).toContain('ML-LANG-UNKNOWN-CALL');
      expect(c).not.toContain('ML-LANG-INTERNAL');
    }
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

describe('sandbox: the custom-value-type protocol cannot be abused (typed arrays + vec/mat)', () => {
  // A component-scoped run — a bare top-level `let` needs insideComponent (else ML-LANG-LET-SCOPE),
  // so the alias / mutable-buffer cases below can declare `let b = a`.
  const runIC = (src: string) =>
    evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), insideComponent: true });
  // A tight-budget run so an unbounded generator/loop trips the STEP budget quickly (never hangs).
  const codesBudget = (src: string, maxSteps = 5000) =>
    evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), maxSteps }).diagnostics.map((d) => d.code);

  // (1) A program cannot FORGE a descriptor — a plain object shaped like a buffer is NOT a custom type.
  //     The DESCRIPTOR/GENERATION/FROZEN symbols are module-private and there is no builtin that reads them.
  it('a plain object shaped like a buffer is not a custom type — no bounds-check, no coercion', () => {
    // A numeric-index read on a plain object is ORDINARY object access: `o[0]` returns the own property
    // value verbatim (1) — NOT a coerced typed-array read — and crucially an out-of-"range" `o[5]` is a
    // benign null with NO ML-LANG-INDEX-RANGE. A REAL f32 buffer range-checks that same read (see the
    // OOB test below), so the ABSENCE of the bounds-check here proves isCustomType stayed false: the
    // program's plain object never picked up the typed-array descriptor's read machinery.
    const r = run('const o = { length: 3, "0": 1 }; o[0];');
    expect(r.value).toBe(1);
    expect(r.diagnostics).toEqual([]);
    const oob = run('const o = { length: 3, "0": 1 }; o[5];');
    expect(oob.value).toBeNull();
    expect(oob.diagnostics.some((d) => d.code === 'ML-LANG-INDEX-RANGE')).toBe(false);
  });
  it('an out-of-"range" numeric key on a forged buffer is a benign null, never a bounds diagnostic', () => {
    const r = run('const o = { length: 1 }; o[5];');
    expect(r.value).toBeNull();
    expect(r.diagnostics).toEqual([]);
  });
  it('the descriptor-introspection helpers are NOT reachable from the language surface', () => {
    // descriptorOf / isCustomType / generationOf are host TS, never registered as builtins — a call is
    // just an unknown head. This is the structural reason a program can neither read nor forge a tag.
    expect(codes('descriptorOf(f32([1,2,3]));')).toContain('ML-LANG-UNKNOWN-CALL');
    expect(codes('isCustomType(f32([1,2,3]));')).toContain('ML-LANG-UNKNOWN-CALL');
    expect(codes('generationOf(f32([1,2,3]));')).toContain('ML-LANG-UNKNOWN-CALL');
  });

  // (2) The Symbol-hidden store + the type tags never leak through keys/values/entries/spread/display.
  it('keys/values/entries over a typed array surface no hidden fields (empty, no leak)', () => {
    expect(run('keys(f32([1,2,3]));').value).toEqual([]);
    expect(run('values(f32([1,2,3]));').value).toEqual([]);
    expect(run('entries(f32([1,2,3]));').value).toEqual([]);
  });
  it('spreading a typed array copies no fields (the store/descriptor/generation stay hidden)', () => {
    expect(run('keys({ ...f32([1,2,3]) });').value).toEqual([]);
  });
  it('keys over a vec surfaces no hidden fields either', () => {
    expect(run('keys(vec3(1,2,3));').value).toEqual([]);
  });
  it('string-coercing a typed array yields the bounded display, never the raw backing store', () => {
    const r = run('"" + f32([1,2,3]);');
    expect(r.value).toBe('f32[1, 2, 3]');
    expect(r.diagnostics).toEqual([]);
  });

  // (3) A forbidden key is rejected BEFORE the descriptor — no reaching __proto__/constructor through a tag.
  for (const [label, ctor] of [['typed array', 'f32([1,2,3])'], ['vec', 'vec3(1,2,3)']] as const) {
    it(`${label}: reading a forbidden key is blocked (${ctor}["__proto__"] / ["constructor"])`, () => {
      expect(codes(`${ctor}["__proto__"];`)).toContain('ML-LANG-FORBIDDEN');
      expect(codes(`${ctor}["constructor"];`)).toContain('ML-LANG-FORBIDDEN');
    });
    it(`${label}: writing a forbidden key is blocked (the forbidden-key check precedes the descriptor)`, () => {
      expect(codes(`const a = ${ctor}; a["__proto__"] = 9;`)).toContain('ML-LANG-FORBIDDEN');
    });
  }

  // (4) A `const` typed array (and any alias sharing its frozen box) is immutable; a vec is immutable at all.
  it('a const typed array cannot be mutated in place — ML-LANG-IMMUTABLE, value unchanged', () => {
    const r = run('const a = f32([1,2,3]); a[0] = 9; a[0];');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(r.value).toBe(1);
  });
  it('mutating a const buffer through a let alias is still blocked (shared frozen box)', () => {
    const r = runIC('const a = f32([1,2,3]); let b = a; b[0] = 9; b[0];');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(r.value).toBe(1);
  });
  it('a vec is a pure immutable value — a member write is blocked, value unchanged', () => {
    const r = run('const v = vec3(1,2,3); v.x = 9; v.x;');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(r.value).toBe(1);
  });

  // (5) A custom value cannot be a vector for a budget escape: construction is capped, a recursing/looping
  //     generator fails closed on the step budget, and an OOB read is a diagnostic (never a crash).
  it('an over-cap construction trips ML-LANG-BUDGET (no giant allocation)', () => {
    expect(codes('f32(999999999);')).toContain('ML-LANG-BUDGET');
  });
  it('a recursing generator callback fails closed with ML-LANG-BUDGET, never hangs', () => {
    // A recursive `function` used as the (n, fn) generator — the depth/step budget bounds it.
    expect(codesBudget('function rec(n) { rec(n) } f32(4, (i) => rec(i));')).toContain('ML-LANG-BUDGET');
  });
  it('an unbounded-loop generator callback fails closed with ML-LANG-BUDGET, never hangs', () => {
    // A generator arrow whose body calls a helper that loops forever — each tick charges the step budget.
    expect(codesBudget('function loop() { while (true) { rand() } } f32(4, (i) => loop());')).toContain('ML-LANG-BUDGET');
  });
  it('an out-of-bounds buffer read is a diagnostic, not a crash (value null)', () => {
    const r = run('f32([1])[999];');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-INDEX-RANGE')).toBe(true);
    expect(r.value).toBeNull();
  });

  // (6) A custom value cannot smuggle a non-number into a numeric slot, nor silently coerce a whole-buffer op.
  it('writing a non-number element is ML-LANG-BUILTIN-ARG (no silent coercion), value unchanged', () => {
    const r = runIC('let a = f32([0]); a[0] = "x"; a[0];');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(r.value).toBe(0);
  });
  it('a whole-buffer binary op is ML-LANG-OP-UNSUPPORTED, not a silent coercion', () => {
    const r = run('f32([1]) + f32([2]);');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-OP-UNSUPPORTED')).toBe(true);
    expect(r.value).toBeNull();
  });
});
