import { describe, it, expect } from 'vitest';
import { evaluateProgram, isUserFn } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { gateKernel } from './gate.ts';

function kernelOf(src: string): { fn: UserFn; host: PlainStorageHost } {
  const host = new PlainStorageHost();
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv() });
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
  it('rejects a normal-array input (not a typed array)', () => {
    const v = reject(`const a = [1, 2, 3]\nfunction k(i) { return a[i] }\nk`);
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
