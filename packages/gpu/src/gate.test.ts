import { describe, it, expect } from 'vitest';
import { MATH_BUILTINS } from '@metael/math/lang';
import { evaluateProgram, isUserFn } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { gateKernel } from './gate.ts';

function kernelOf(src: string): { fn: UserFn; host: PlainStorageHost } {
  const host = new PlainStorageHost();
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
  if (!isUserFn(res.value)) throw new Error('expected the last expression to be the kernel function');
  return { fn: res.value, host };
}

describe('compute-lowerability gate — accepts real kernels', () => {
  it('accepts a scalar matmul kernel (buffers + a const + a for-of range + arithmetic)', () => {
    const { fn, host } = kernelOf(`
      const N = 4
      const a = f32(N * N, (i) => i)
      const b = f32(N * N, (i) => i)
      function product(row, col) {
        let sum = 0
        for (const k of range(N)) { sum = sum + a[row * N + k] * b[k * N + col] }
        return sum
      }
      product`);
    const v = gateKernel(fn, host);
    expect(v.core).toBe(true);
    expect(v.reasons).toEqual([]);
    expect(v.bindings.byName.get('a')?.role).toBe('buffer');
    expect(v.bindings.byName.get('N')?.role).toBe('scalar');
    expect(v.bindings.params).toEqual(['row', 'col']);
  });
  it('accepts a matmul kernel authored as a COMPONENT (a `let` accumulator is a valid Stmt in the walk)', () => {
    // Kernels are authored/run as `component` so a reactive `let sum` accumulator works (a plain function
    // runs insideComponent=false → ML-LANG-LET-SCOPE in the interpreter/oracle). The gate walks a component
    // body identically to a function body (a component IS a UserFn), so it must still gate this as core.
    const { fn, host } = kernelOf(`
      const N = 4
      const a = f32(N * N, (i) => i)
      const b = f32(N * N, (i) => i)
      component product(row, col) {
        let sum = 0
        for (const k of range(N)) { sum = sum + a[row * N + k] * b[k * N + col] }
        return sum
      }
      product`);
    expect(fn.isComponent).toBe(true);
    const v = gateKernel(fn, host);
    expect(v.core).toBe(true);
    expect(v.reasons).toEqual([]);
  });
  it('accepts a transcendental kernel (sin/cos)', () => {
    const { fn, host } = kernelOf(`
      const a = f32(16, (i) => i)
      function k(i) { return sin(a[i]) + cos(a[i]) }
      k`);
    expect(gateKernel(fn, host).core).toBe(true);
  });
  it('accepts a vec-math kernel (vec3 intermediates, scalar output)', () => {
    const { fn, host } = kernelOf(`
      const a = f32(48, (i) => i)
      function k(i) { const v = vec3(a[i*3], a[i*3+1], a[i*3+2]); return length(v) }
      k`);
    expect(gateKernel(fn, host).core).toBe(true);
  });
});

describe('compute-lowerability gate — rejects with a span-anchored MLGPU reason', () => {
  const reject = (src: string) => { const { fn, host } = kernelOf(src); return gateKernel(fn, host); };
  it('rejects a host builtin (format)', () => {
    const v = reject(`const a = f32(4, (i) => i)\nfunction k(i) { return format(a[i], 2) }\nk`);
    expect(v.core).toBe(false);
    expect(v.reasons.some((r) => r.code === 'MLGPU-NOT-LOWERABLE')).toBe(true);
  });
  it('rejects a string operation', () => {
    const v = reject(`function k(i) { return "x" + i }\nk`);
    expect(v.core).toBe(false);
  });
  it('accepts a plain all-number array input as a buffer (coerced to f32 at dispatch)', () => {
    // A plain metael array of all numbers is a valid buffer input — it is classified role:'buffer' and
    // coerced ONCE to an f32 store at dispatch. (The rejection now applies only to a MIXED/empty array.)
    const { fn, host } = kernelOf(`const a = [1, 2, 3]\nfunction k(i) { return a[i] }\nk`);
    const v = gateKernel(fn, host);
    expect(v.core).toBe(true);
    expect(v.reasons).toEqual([]);
    expect(v.bindings.byName.get('a')?.role).toBe('buffer');
  });
  it('rejects a MIXED (non-all-number) array input', () => {
    const v = reject(`const a = [1, "two", 3]\nfunction k(i) { return a[i] }\nk`);
    expect(v.core).toBe(false);
    expect(v.reasons.some((r) => r.code === 'MLGPU-BAD-INPUT')).toBe(true);
  });
  it('rejects a data-dependent unbounded while loop', () => {
    const v = reject(`const a = f32(4, (i) => i)\nfunction k(i) { let x = 0; while (x < a[i]) { x = x + 1 } return x }\nk`);
    expect(v.core).toBe(false);
  });
  it('a kernel indexing a member of a non-input (rA.value[i]) gets a targeted bind-a-local-first hint', () => {
    // `rA` is a resource wrapper (undeclared inside the kernel closure) indexed as `rA.value[i]`. The gate
    // flags `rA` as not-a-kernel-input; the message should point at the fix (bind `rA.value` to a local
    // OUTSIDE the kernel, then index that local), not just list valid input kinds.
    const v = reject(`function k(i) { return rA.value[i] * 2 }\nk`);
    expect(v.core).toBe(false);
    const badInput = v.reasons.find((r) => r.code === 'MLGPU-BAD-INPUT' && r.message.includes("'rA'"));
    expect(badInput).toBeDefined();
    expect(badInput!.message).toMatch(/const \w+ = rA\.value/);   // the targeted "bind a local first" hint
  });
  it('rejects buf + buf (a whole-buffer op the type does not define)', () => {
    const v = reject(`const a = f32(4, (i) => i)\nconst b = f32(4, (i) => i)\nfunction k(i) { const c = a + b; return c[i] }\nk`);
    expect(v.core).toBe(false);
  });
});

describe('gate — helper (callee) bodies are gated recursively', () => {
  it('rejects a kernel calling a helper with an un-lowerable while loop', () => {
    const { fn, host } = kernelOf(`
      const a = f32(4, (i) => i)
      function bad(x) { let s = 0; while (s < x) { s = s + 1 } return s }
      function k(i) { return bad(a[i]) }
      k`);
    const v = gateKernel(fn, host);
    expect(v.core).toBe(false);
    expect(v.reasons.some((r) => r.code === 'MLGPU-NOT-LOWERABLE')).toBe(true);
  });
  it('rejects a kernel whose helper uses a host builtin (format)', () => {
    const { fn, host } = kernelOf(`
      const a = f32(4, (i) => i)
      function bad(x) { return format(x, 2) }
      function k(i) { return bad(a[i]) }
      k`);
    expect(gateKernel(fn, host).core).toBe(false);
  });
  it('rejects a user function that SHADOWS a builtin name (abs) — the interpreter runs the user body, so the gate must not lower it as the intrinsic', () => {
    // A top-level `function abs(x){…}` binds `abs` as a callee, SHADOWING the abs intrinsic (the interpreter
    // resolves the closure binding first). Lowering it as native abs() would drop the user body → a silent
    // wrong answer on the GPU that the default (verify-off) path never catches. The gate must reject the
    // helper CALL (v1 can't inline helpers) so gate ↔ emitter agree.
    const { fn, host } = kernelOf(`
      const a = f32(4, (i) => i)
      function abs(x) { return x + 1000 }
      component k(i) { return abs(a[i]) }
      k`);
    const v = gateKernel(fn, host);
    expect(v.core).toBe(false);
    expect(v.reasons.some((r) => r.code === 'MLGPU-NOT-LOWERABLE')).toBe(true);
  });
  it('rejects a kernel calling even a clean scalar helper in v1 (the emitter cannot emit helper fns yet — inline it)', () => {
    // v1 defers helper-fn emission (the WGSL emitter inlines only the top-level body), so the gate rejects a
    // helper CALL even when the helper body itself is clean — gate ↔ emitter must agree. The later
    // helper-fn-emission increment will re-accept this by emitting `fn dbl(){}`.
    const { fn, host } = kernelOf(`
      const a = f32(4, (i) => i)
      function dbl(x) { return x * 2 }
      function k(i) { return dbl(a[i]) }
      k`);
    const v = gateKernel(fn, host);
    expect(v.core).toBe(false);
    expect(v.reasons.some((r) => r.code === 'MLGPU-NOT-LOWERABLE')).toBe(true);
  });
});

describe("gate — v1 rejects vec/mat uniform inputs + helper calls (deferred / emitter can't lower yet)", () => {
  it('rejects a vec/mat input used as a kernel uniform', () => {
    const { fn, host } = kernelOf(`const v = vec3(1, 2, 3)\nconst a = f32(4, (i) => i)\ncomponent k(i) { return a[i] + v.x }\nk`);
    const g = gateKernel(fn, host);
    expect(g.core).toBe(false);
    expect(g.reasons.some((r) => r.code === 'MLGPU-NOT-LOWERABLE')).toBe(true);
  });
  it('rejects calling a helper function (inline it instead)', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\nfunction dbl(x) { return x * 2 }\ncomponent k(i) { return dbl(a[i]) }\nk`);
    expect(gateKernel(fn, host).core).toBe(false);
  });
  it('STILL accepts a vec/mat used only as an INTERMEDIATE (not an input uniform)', () => {
    const { fn, host } = kernelOf(`const a = f32(48, (i) => i)\ncomponent k(i) { const u = vec3(a[i*3], a[i*3+1], a[i*3+2]); return length(u) }\nk`);
    expect(gateKernel(fn, host).core).toBe(true);
  });
  it('accepts a cross/normalize vec-intermediate kernel', () => {
    const { fn, host } = kernelOf(`const a = f32(48, (i) => i)\ncomponent k(i) { const u = vec3(a[i*3], a[i*3+1], a[i*3+2]); return length(cross(u, vec3(1,0,0))) }\nk`);
    expect(gateKernel(fn, host).core).toBe(true);
  });
});

describe('gate — a buffer has no whole-value form (only a[i] / a.length)', () => {
  const reject = (src: string) => { const { fn, host } = kernelOf(src); return gateKernel(fn, host); };
  it('rejects sin(a) — a whole buffer as a builtin arg', () => {
    expect(reject(`const a = f32(4,(i)=>i)\nfunction k(i){ return sin(a) }\nk`).core).toBe(false);
  });
  it('rejects returning a whole buffer', () => {
    expect(reject(`const a = f32(4,(i)=>i)\nfunction k(i){ return a }\nk`).core).toBe(false);
  });
  it('rejects a buffer aliased through a const then added (const c = a; c + d)', () => {
    expect(reject(`const a = f32(4,(i)=>i)\nconst b = f32(4,(i)=>i)\nfunction k(i){ const c = a; const d = b; const e = c + d; return e[i] }\nk`).core).toBe(false);
  });
  it('rejects a whole buffer as a ternary test', () => {
    expect(reject(`const a = f32(4,(i)=>i)\nfunction k(i){ return a ? 1 : 0 }\nk`).core).toBe(false);
  });
  it('STILL ACCEPTS indexed reads a[i] + b[i] and a.length', () => {
    const v = reject(`const a = f32(4,(i)=>i)\nconst b = f32(4,(i)=>i)\nfunction k(i){ let n = a.length; return a[i] + b[i] + n }\nk`);
    expect(v.core).toBe(true);
  });
});

describe('gate — an input buffer is read-only (a write to it has no lowering)', () => {
  const reject = (src: string) => { const { fn, host } = kernelOf(src); return gateKernel(fn, host); };
  it('rejects a kernel that writes to an input buffer (inputs are read-only; gate ↔ emitter agree)', () => {
    // The emitters bind inputs read-only (WGSL `var<storage, read>`; the CPU/GLSL paths never store to an
    // input), so an index-write `buf[i] = …` has no lowering. If the gate let it through, the interpreter
    // oracle would honor the write (mutating the caller's live buffer + bumping its generation) while the
    // dispatched shader silently drops it — the two references diverge. The gate must reject it.
    const v = reject(`const buf = f32([0, 1, 2, 3])
component k(i) { buf[i] = 777
return buf[i] * 2 }
k`);
    expect(v.core).toBe(false);
    expect(v.reasons.some((r) => r.code === 'MLGPU-INPUT-WRITE')).toBe(true);
  });
  it('STILL accepts a kernel that writes only a LOCAL accumulator (a `let`, not an input buffer)', () => {
    // `sum = sum + …` writes a local `let` (an ident target, not an input-buffer index) — perfectly
    // lowerable. The read-only rule is scoped to input buffers; it must not touch local writes.
    const v = reject(`const a = f32(4, (i) => i)
component k(i) { let sum = 0
sum = sum + a[i]
return sum }
k`);
    expect(v.core).toBe(true);
    expect(v.reasons).toEqual([]);
  });
});

describe('gate — robustness + wrapping block', () => {
  it('does NOT throw on range() with no arg — flags it', () => {
    const { fn, host } = kernelOf(`const a = f32(4,(i)=>i)\nfunction k(i){ let s = 0; for (const x of range()) { s = s + a[i] } return s }\nk`);
    let v!: ReturnType<typeof gateKernel>;
    expect(() => { v = gateKernel(fn, host); }).not.toThrow();
    expect(v.core).toBe(false);
  });
});

describe('gate — a value-position range() call is rejected at the gate (not at emit)', () => {
  const reject = (src: string) => { const { fn, host } = kernelOf(src); return gateKernel(fn, host); };
  it('rejects range() used as a VALUE (const init / return) with MLGPU-NOT-LOWERABLE — before any emitter', () => {
    // `range` is a `core`/`exact` intrinsic in the gate catalog, so the generic host/lowerName reject branch
    // never fires for it. But `range(n)` returns a COLLECTION, which has no scalar/vec lowering — the only
    // lowerable use is as the bound of a `for (… of range(n))` loop (intercepted in the `for` case). A value-
    // position `range()` call must be rejected here at the gate so it never reaches the emitter (which would
    // otherwise throw "no WGSL/GLSL lowering for builtin 'range'" → an MLGPU-EMIT). gate ↔ emitter agree.
    const v = reject(`component k(i) { const r = range(i)\n return r } k`);
    expect(v.core).toBe(false);
    const notLowerable = v.reasons.find((r) => r.code === 'MLGPU-NOT-LOWERABLE' && /range/.test(r.message));
    expect(notLowerable).toBeDefined();
    // The gate rejects it — it must NOT slip through to the emitter (no MLGPU-EMIT).
    expect(v.reasons.some((r) => r.code === 'MLGPU-EMIT')).toBe(false);
  });
  it('REGRESSION: a SUPPORTED `for (… of range(n))` loop is unaffected (still gates core=true)', () => {
    // The `for` case walks ONLY the bound `range(n)` arg, never the `range(...)` call node through walkExpr,
    // so the new value-position reject must NOT fire for the supported loop form.
    const v = reject(`component k(i) { let acc = 0\n for (const j of range(4)) { acc = acc + j }\n return acc } k`);
    expect(v.core).toBe(true);
    expect(v.reasons).toEqual([]);
  });
});

describe('gate — a matrix return has no output-cell form', () => {
  it('a matrix returned to a scalar output is rejected', () => {
    const { fn, host } = kernelOf('component k(i) { return mat2(1,2,3,4) } k');
    const v = gateKernel(fn, host, 1);
    expect(v.core).toBe(false);
    expect(v.reasons.some((r) => r.code === 'MLGPU-OUTPUT-SHAPE')).toBe(true);
  });
});

describe('gate — vec ± mat and mismatched matmul are rejected (inconsistent shape)', () => {
  it('vec + mat is a gate reject (inconsistent shape)', () => {
    const { fn, host } = kernelOf('component k(i) { return (vec2(i,i) + mat2(1,2,3,4)).x } k');
    const v = gateKernel(fn, host, 1);
    expect(v.core).toBe(false);
  });
  it('mat * mat with a mismatched inner dimension is a gate reject', () => {
    // mat2 (cols=2) * mat3 (rows=3): inner dims 2 ≠ 3 → undefined product. The reject must come from the
    // inner-mismatch -1 path (the binary signals -1, the `.x` swizzle propagates it) → an MLGPU-OUTPUT-SHAPE
    // reason — NOT from the matrix-output guard (a `member` node is not a matShapeOf-recognized ctor call).
    const { fn, host } = kernelOf('component k(i) { return (mat2(1,2,3,4) * mat3(1,2,3,4,5,6,7,8,9)).x } k');
    const v = gateKernel(fn, host, 1);
    expect(v.core).toBe(false);
    expect(v.reasons.some((r) => r.code === 'MLGPU-OUTPUT-SHAPE')).toBe(true);
  });
});

describe('gate — accepts EXACTLY the vec/mat op·shapes the interpreter accepts (gate ⇒ parity)', () => {
  // The interpreter's vec/mat descriptor `binary` handler (the correctness oracle) evaluates only a fixed
  // set of (op, left-shape, right-shape) triples; everything else returns NOT_HANDLED → ML-LANG-OP-
  // UNSUPPORTED → the cell is 0. A shader emits the native op and computes a real value, so a gate that
  // accepts a NOT_HANDLED combo silently diverges (verify is off by default; GLSL runs on WebGL2 in CI).
  // The gate must accept a combo iff the interpreter does.
  const core = (src: string): boolean => { const { fn, host } = kernelOf(`component k(i) { return ${src} } k`); return gateKernel(fn, host, 1).core; };
  it('rejects the op·shapes the interpreter NOT_HANDLEs (were silently GPU-divergent)', () => {
    expect(core('(vec2(i,i) * mat2(1,2,3,4)).x')).toBe(false);   // vec*mat (vec on the LEFT of a matrix) — not matmul
    expect(core('(2 / vec2(i,i)).x')).toBe(false);               // scalar / vec — the interpreter scale rule is `*` only
    expect(core('(vec2(i,i) + 5).x')).toBe(false);               // vec + scalar — scale is `*`|`/` only
    expect(core('(vec2(i,i) - 5).x')).toBe(false);               // vec - scalar
    expect(core('(5 + vec2(i,i)).x')).toBe(false);               // scalar + vec
    expect(core('(5 - vec2(i,i)).x')).toBe(false);               // scalar - vec
    expect(core('(mat2(1,2,3,4) + mat2(5,6,7,8)).x')).toBe(false); // mat + mat (both matrices, same shape)
    expect(core('(mat2(1,2,3,4) - mat2(5,6,7,8)).x')).toBe(false); // mat - mat
    expect(core('(2 / mat2(1,2,3,4)).x')).toBe(false);           // scalar / mat — scale is `*` only for a scalar-left
    expect(core('(vec3(1,2,3) + vec2(i,i)).x')).toBe(false);     // vec + vec, differing width
  });
  it('still accepts the op·shapes the interpreter evaluates', () => {
    expect(core('(mat2(1,2,3,4) * vec2(1,1)).x')).toBe(true);    // mat * vec — matmul (mat on the left)
    expect(core('(vec2(1,2) * 3).x')).toBe(true);                // vec * scalar
    expect(core('(3 * vec2(1,2)).x')).toBe(true);                // scalar * vec
    expect(core('(vec2(1,2) / 2).x')).toBe(true);                // vec / scalar
    expect(core('(vec2(1,2) + vec2(3,4)).x')).toBe(true);        // vec + vec, equal width
    expect(core('(vec2(1,2) - vec2(3,4)).x')).toBe(true);        // vec - vec, equal width
    expect(core('(vec2(1,2) * vec2(3,4)).x')).toBe(true);        // vec * vec, equal width
    expect(core('(vec2(1,2) / vec2(3,4)).x')).toBe(true);        // vec / vec, equal width
    expect(core('((mat2(1,2,3,4) * 2) * vec2(1,1)).x')).toBe(true); // mat*scalar (a mat) then mat*vec — matmul
    expect(core('i * 2 + 1')).toBe(true);                        // plain scalar arithmetic (the carve-out)
    expect(core('i + i')).toBe(true);                            // plain scalar
  });
});

describe('gate — rand() cannot be lowered (no deterministic shader match)', () => {
  it('a kernel using rand() is non-core with the PRECISE deterministic-oracle rejection (not a mislabeled BAD-INPUT)', () => {
    const { fn, host } = kernelOf('component k(i) { return rand() } k');
    const v = gateKernel(fn, host, 1);
    expect(v.core).toBe(false);
    // The dedicated rand branch must fire: the message names the deterministic-oracle mismatch, not the
    // generic "not a lowerable builtin" fallback. (`rand` is recognized in the gate catalog for this reason.)
    expect(v.reasons.some((r) => r.code === 'MLGPU-NOT-LOWERABLE' && /cannot match the deterministic interpreter oracle/.test(r.message))).toBe(true);
    // `rand` must NOT be mislabeled as a missing kernel input (the free-name path) — it is a recognized builtin.
    expect(v.reasons.some((r) => r.code === 'MLGPU-BAD-INPUT')).toBe(false);
  });
});

describe('gate — inverse requires a statically-sized square matrix argument (no silent bare-arg WGSL)', () => {
  const ok = (src: string): boolean => { const { fn, host } = kernelOf(src); return gateKernel(fn, host, 1).core; };
  const reasons = (src: string) => { const { fn, host } = kernelOf(src); return gateKernel(fn, host, 1).reasons; };

  it('inverse of a matrix constructor is lowerable', () => {
    // A direct `matN(...)` ctor arg → matSizeOf resolves the size → the emitter can hand-emit `_invN`.
    expect(ok('component k(i) { return (inverse(mat2(4,2,7,6)) * vec2(1,0)).x } k')).toBe(true);
  });
  it('inverse of a ctor-typed local is lowerable (const M = mat3(...); inverse(M))', () => {
    // The locals-aware rule: a `const M = mat3(...)` records M as a 3×3 local, so `inverse(M)` resolves.
    expect(ok('component k(i) { const M = mat3(2,0,1, 1,3,0, 0,2,1) return (inverse(M) * vec3(1,0,0)).x } k')).toBe(true);
  });
  it('inverse of transpose of a local is lowerable (the normal-matrix idiom)', () => {
    // matSizeOf recurses through transpose over a square local → resolvable. This is the canonical idiom the
    // old (pre-fix) emitter silently mis-lowered (dropping the inverse, emitting the bare transpose(M)).
    expect(ok('component k(i) { const M = mat3(2,0,1, 1,3,0, 0,2,1) return (inverse(transpose(M)) * vec3(1,0,0)).x } k')).toBe(true);
  });
  it('inverse of a matrix PRODUCT (a computed matrix) is gate-rejected — not silently mis-lowered', () => {
    const r = reasons('component k(i) { return (inverse(mat2(1,2,3,4) * mat2(5,6,7,8)) * vec2(1,0)).x } k');
    expect(r.some((d) => d.code === 'MLGPU-NOT-LOWERABLE' && /inverse/.test(d.message))).toBe(true);
  });
  it('inverse of a computed-matrix LOCAL (const P = A * B; inverse(P)) is gate-rejected', () => {
    // P's init is a mat*mat product → matSizeOf null → P is NOT recorded as a matrix local → inverse(P) rejects.
    const r = reasons('component k(i) { const A = mat2(1,2,3,4) const B = mat2(5,6,7,8) const P = A * B return (inverse(P) * vec2(1,0)).x } k');
    expect(r.some((d) => d.code === 'MLGPU-NOT-LOWERABLE' && /inverse/.test(d.message))).toBe(true);
  });
  it('inverse of a mat*scalar and of a ternary are gate-rejected (computed matrices)', () => {
    const rScale = reasons('component k(i) { const A = mat2(1,2,3,4) return (inverse(A * 2.0) * vec2(1,0)).x } k');
    expect(rScale.some((d) => d.code === 'MLGPU-NOT-LOWERABLE' && /inverse/.test(d.message))).toBe(true);
    const rCond = reasons('component k(i) { const A = mat2(1,2,3,4) const B = mat2(5,6,7,8) return (inverse(i > 0 ? A : B) * vec2(1,0)).x } k');
    expect(rCond.some((d) => d.code === 'MLGPU-NOT-LOWERABLE' && /inverse/.test(d.message))).toBe(true);
  });
});

describe('gate — output-shape inference for the P4 vec/mat ops', () => {
  it('reflect returns arg0 width; distance returns scalar', () => {
    const { fn, host } = kernelOf('component k(i) { return reflect(vec2(1,-1), vec2(0,1)) } k');
    expect(gateKernel(fn, host, 2).core).toBe(true); // vec2 output OK (was WRONGLY rejected before)
    const { fn: fn2, host: h2 } = kernelOf('component k(i) { return distance(vec2(0,0), vec2(3,4)) } k');
    expect(gateKernel(fn2, h2, 1).core).toBe(true); // scalar output OK
  });
  it('refract/faceforward take arg0 width; determinant folds to a scalar', () => {
    const rr = kernelOf('component k(i) { return refract(vec3(1,0,0), vec3(0,1,0), 0.5) } k');
    expect(gateKernel(rr.fn, rr.host, 3).core).toBe(true); // vec3 output OK
    const ff = kernelOf('component k(i) { return faceforward(vec2(1,0), vec2(0,1), vec2(0,-1)) } k');
    expect(gateKernel(ff.fn, ff.host, 2).core).toBe(true); // vec2 output OK
    const det = kernelOf('component k(i) { return determinant(mat2(1,2,3,4)) } k');
    expect(gateKernel(det.fn, det.host, 1).core).toBe(true); // scalar output OK
  });
  it('determinant of a non-square matrix is a gate reject', () => {
    const { fn, host } = kernelOf('component k(i) { return determinant(mat2x3(1,2,3,4,5,6)) } k');
    expect(gateKernel(fn, host, 1).core).toBe(false);
    expect(gateKernel(fn, host, 1).reasons.some((r) => r.code === 'MLGPU-NOT-LOWERABLE')).toBe(true);
  });
  it('determinant of a non-square LOCAL is a gate reject (locals-aware)', () => {
    // The locals-aware shape resolver catches `const m = mat2x3(...); determinant(m)` — the non-square shape
    // is recorded for the local, so the determinant square check rejects it (it did NOT before this fix).
    const { fn, host } = kernelOf('component k(i) { const m = mat2x3(1,2,3,4,5,6) return determinant(m) } k');
    const v = gateKernel(fn, host, 1);
    expect(v.core).toBe(false);
    expect(v.reasons.some((r) => r.code === 'MLGPU-NOT-LOWERABLE')).toBe(true);
  });
  it('determinant of a square LOCAL stays lowerable', () => {
    const { fn, host } = kernelOf('component k(i) { const m = mat2(1,2,3,4) return determinant(m) } k');
    expect(gateKernel(fn, host, 1).core).toBe(true);
  });
});

describe('gate — the structured constructors + column access (vecN composition, matMxN-from-columns, m[i])', () => {
  const coreN = (src: string, comps: number): boolean => { const { fn, host } = kernelOf(`component k(i) { return ${src} } k`); return gateKernel(fn, host, comps).core; };
  const reasons = (src: string, comps: number) => { const { fn, host } = kernelOf(`component k(i) { return ${src} } k`); return gateKernel(fn, host, comps).reasons; };

  it('accepts vecN composition (a vecM arg to a vecN ctor when the flattened width totals N)', () => {
    expect(coreN('vec3(vec2(i, i), i)', 3)).toBe(true);         // vec2 + scalar → vec3
    expect(coreN('vec4(vec2(i, i), vec2(i, i))', 4)).toBe(true); // vec2 + vec2 → vec4
    expect(coreN('vec4(vec3(i, i, i), i)', 4)).toBe(true);       // vec3 + scalar → vec4
    expect(coreN('vec4(i, vec2(i, i), i)', 4)).toBe(true);       // scalar + vec2 + scalar → vec4
  });
  it('accepts matMxN from column vecs (vecR args to a matMxN ctor)', () => {
    expect(coreN('(mat2(vec2(1, 2), vec2(3, 4)) * vec2(i, i)).x', 1)).toBe(true);   // 2 column vec2s
    expect(coreN('(mat3(vec3(1, 2, 3), vec3(4, 5, 6), vec3(7, 8, 9)) * vec3(i, i, i)).x', 1)).toBe(true);  // 3 column vec3s
  });
  it('accepts an index node whose object is a matMxN (m[i] column read) and reports it as a vecR output', () => {
    expect(coreN('mat3(1, 2, 3, 4, 5, 6, 7, 8, 9)[1].xyz', 3)).toBe(true);   // m[i] → vec3, .xyz swizzle → vec3 output
    expect(coreN('mat2(1, 2, 3, 4)[0].xy', 2)).toBe(true);                    // m[i] → vec2, .xy → vec2 output
    // A mat column bound to a LOCAL then swizzled resolves via the locals-aware shape map.
    const { fn, host } = kernelOf('component k(i) { const m = mat3(1, 2, 3, 4, 5, 6, 7, 8, 9) const c = m[1] return c.xyz } k');
    expect(gateKernel(fn, host, 3).core).toBe(true);
  });
  it('a mat-column vecR is a WIDTH-typed operand: a wrong-width output over m[i] is rejected', () => {
    // `mat3(...)[1]` is a vec3; requesting a vec2 output over it must be a shape reject (not a silent pass).
    const rs = reasons('mat3(1, 2, 3, 4, 5, 6, 7, 8, 9)[1].xyz', 2);
    expect(rs.some((r) => r.code === 'MLGPU-OUTPUT-SHAPE')).toBe(true);
  });
  it('rejects an over-range swizzle on a mat column (m[i] of a mat2 is a vec2 — .z is out of range)', () => {
    // `mat2(...)[0]` is a vec2; `.z` reads past its width — the interpreter NOT_HANDLEs it, so the gate must
    // reject (the mat-index shape now feeds the over-range swizzle check).
    const rs = reasons('mat2(1, 2, 3, 4)[0].z', 1);
    expect(rs.some((r) => r.code === 'MLGPU-NOT-LOWERABLE' && /swizzle/.test(r.message))).toBe(true);
  });
});

describe('gate — the newly-lowered scalar builtins accept, including the 32-bit bit ops', () => {
  const accepts = (src: string): boolean => { const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return ${src} }\nk`); return gateKernel(fn, host, 1).core; };

  it('accepts mod / asinh / acosh / atanh (all have a faithful shader lowering)', () => {
    expect(accepts('mod(a[i], 3)')).toBe(true);
    expect(accepts('asinh(a[i])')).toBe(true);
    expect(accepts('acosh(a[i])')).toBe(true);
    expect(accepts('atanh(a[i])')).toBe(true);
  });
  it('accepts countOneBits / reverseBits — WGSL native + a GLSL ES 3.00 prelude helper; the f32 store is lossless (a popcount is 0..32; reversal preserves the f32-exact bit-span)', () => {
    expect(accepts('countOneBits(a[i])')).toBe(true);
    expect(accepts('reverseBits(a[i])')).toBe(true);
  });
});
