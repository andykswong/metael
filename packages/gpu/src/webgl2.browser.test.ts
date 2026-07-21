import { describe, it, expect } from 'vitest';
import { MATH_BUILTINS } from '@metael/math/lang';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, descriptorOf } from '@metael/lang';
import type { UserFn, HostEnvironment, Arg, HostValue, SourceSpan } from '@metael/lang';
import { RecordingHostEnv } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import { tryWebGl2Backend, mlgpuResidentBinds, mlgpuResetResidentBinds } from './device/webgl2.ts';
import { gateKernel } from './gate.ts';
import { emitGlsl } from './emit-glsl.ts';
import { emitWgsl } from './emit-wgsl.ts';
import { emitCpu } from './emit-cpu.ts';
import type { DispatchInput } from './device/index.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn { const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] }); if (!isUserFn(res.value)) throw new Error('kernel'); return res.value; }
function reducerOf(src: string, host: RuntimeReactiveHost): UserFn { const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] }); if (!isUserFn(res.value)) throw new Error('reducer'); return res.value; }

// Assemble a DispatchInput by hand, mirroring the engine's `resource.ts` assembly (gate → tri-target emit →
// zero-copy input resolution → scalar uniforms). Used to drive the backend contract DIRECTLY (the engine
// wiring for resident handles lands later); `extra` layers on `retainOutput` / `residentInputs`.
function assembleDispatch(kernel: UserFn, host: RuntimeReactiveHost, dims: readonly number[], extra: Partial<DispatchInput> = {}): DispatchInput {
  const { bindings } = gateKernel(kernel, host);
  const inputs: { name: string; data: Float32Array }[] = [];
  const scalars: { name: string; value: number }[] = [];
  for (const b of bindings.byName.values()) {
    if (b.role === 'buffer') {
      const desc = descriptorOf(b.value);
      const view = desc?.bufferView?.(b.value);
      const data = view && view.element === 'f32' && view.data instanceof Float32Array
        ? view.data
        : Float32Array.from((view?.data ?? desc?.iterate?.(b.value) ?? []) as ArrayLike<number>);
      inputs.push({ name: b.name, data });
    } else if (b.role === 'scalar') scalars.push({ name: b.name, value: b.value });
  }
  return { kernel, bindings, dims, precision: 'f32', wgsl: emitWgsl(kernel, bindings, 'f32'), glsl: emitGlsl(kernel, bindings, 'f32'), cpuRun: emitCpu(kernel, bindings, host), inputs, scalars, ...extra };
}
// Whether THIS runner can actually dispatch on WebGL2 (a real context + the float-color extension). When
// true, the tests below REQUIRE the webgl2 backend actually ran (not a silent CPU fallback) — the point of
// the task is proving the GLSL compute-via-fragment path works on a real adapter. When false (a runner
// without WebGL2 float), the ladder must fall to CPU and still match the oracle.
const webgl2Live = !!tryWebGl2Backend();
// Force the WebGL2 leg (no WebGPU): the ladder tries WebGL2, else falls to CPU. Either way the result must
// match the interpreter oracle — the point of the test is that the GLSL emitter + the texture-packing backend
// produce the SAME numbers as the reference, on whichever backend actually ran.
const deps = { tryWebGpu: async () => null, tryWebGl2: tryWebGl2Backend, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('@metael/gpu — real WebGL2 dispatch (Chromium)', () => {
  it('saxpy on WebGL2 matches the interpreter oracle + reports the actual backend', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`
      const N = 256
      const x = f32(N, (i) => i)
      const y = f32(N, (i) => 2 * i)
      component k(i) { return 3 * x[i] + y[i] }
      k`, host);
    const engine = new GpuEngine(host, deps);
    const cfg = { output: [256], backend: 'webgl2' as const, verify: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    // If this runner has live WebGL2 float support, the dispatch MUST have run on webgl2 (a real GPU
    // compute-via-fragment), not fallen to CPU — else the GLSL path would go unproven. Otherwise → cpu floor.
    expect(settled.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    expect(settled.match?.ok).toBe(true);                  // within tolerance of the interpreter
    expect((settled.value as number[] | null)?.[3]).toBeCloseTo(3 * 3 + 6, 3);   // output[3] = 3*3 + 2*3 = 15
  });

  it('a 2-D matmul kernel on WebGL2 matches the oracle (the cols×rows fragment map is correct)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`
      const N = 8
      const a = f32(N * N, (i) => i)
      const b = f32(N * N, (i) => i)
      component product(row, col) { let sum = 0; for (const k of range(N)) { sum = sum + a[row * N + k] * b[k * N + col] } return sum }
      product`, host);
    const engine = new GpuEngine(host, deps);
    const cfg = { output: [8, 8], backend: 'webgl2' as const, verify: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    expect(settled.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    expect(settled.match?.ok).toBe(true);
  });

  it('a negative-operand % kernel matches the oracle on WebGL2 (sign-of-dividend, not mod())', async () => {
    const host = new RuntimeReactiveHost();
    // x[i] ranges over negatives; the interpreter/CPU `%` is sign-of-dividend. GLSL mod() would diverge here,
    // so a match.ok proves the truncated-remainder lowering is oracle-faithful on a real adapter.
    const kernel = kernelOf(`
      const N = 32
      const x = f32(N, (i) => i - 16)
      component k(i) { return x[i] % 3 }
      k`, host);
    const engine = new GpuEngine(host, deps);
    const cfg = { output: [32], backend: 'webgl2' as const, verify: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    expect(settled.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    expect(settled.match?.ok).toBe(true);        // GLSL % ≡ the interpreter on negatives
    expect((settled.value as number[] | null)?.[0]).toBeCloseTo(-16 % 3, 3);   // -16 % 3 = -1 (sign of dividend)
  });

  it('a value-returning && kernel matches the oracle on WebGL2 (operand value, not a 0/1 bool)', async () => {
    const host = new RuntimeReactiveHost();
    // `a[i] && b[i]` returns b[i] when a[i] is truthy (non-zero), else a[i] (0). A 0/1 coercion would diverge.
    const kernel = kernelOf(`
      const N = 16
      const a = f32(N, (i) => i % 2)
      const b = f32(N, (i) => i * 10)
      component k(i) { return a[i] && b[i] }
      k`, host);
    const engine = new GpuEngine(host, deps);
    const cfg = { output: [16], backend: 'webgl2' as const, verify: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    expect(settled.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    expect(settled.match?.ok).toBe(true);         // GLSL && ≡ the interpreter's value-returning short-circuit
    expect((settled.value as number[] | null)?.[0]).toBeCloseTo(0, 3);    // a[0]=0 (falsy) → returns a[0]=0
    expect((settled.value as number[] | null)?.[3]).toBeCloseTo(30, 3);   // a[3]=1 (truthy) → returns b[3]=30
  });

  it('a scalar-uniform kernel matches the oracle on WebGL2 (the _u_ namespacing round-trips through the backend)', async () => {
    const host = new RuntimeReactiveHost();
    // `bias` is a scalar uniform (emitted `_u_bias`, set by the backend under that name). A wrong name →
    // location null → the uniform reads 0 → a mismatch. match.ok proves the round-trip.
    const kernel = kernelOf(`
      const N = 64
      const x = f32(N, (i) => i)
      const bias = 100
      component k(i) { return x[i] + bias }
      k`, host);
    const engine = new GpuEngine(host, deps);
    const cfg = { output: [64], backend: 'webgl2' as const, verify: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    expect(settled.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    expect(settled.match?.ok).toBe(true);
    expect((settled.value as number[] | null)?.[5]).toBeCloseTo(105, 3);   // x[5] + bias = 5 + 100
  });

  it('a divide-by-zero kernel matches the oracle on WebGL2 (guarded to 0, not native +inf)', async () => {
    const host = new RuntimeReactiveHost();
    // x[0]=0 → 1/0. The interpreter maps /0 to 0; a raw GLSL division would write +inf. match.ok + value[0]=0
    // proves the divisor guard produces oracle-faithful cells on a real adapter.
    const kernel = kernelOf(`
      const N = 16
      const x = f32(N, (i) => i)
      component k(i) { return 1 / x[i] }
      k`, host);
    const engine = new GpuEngine(host, deps);
    const cfg = { output: [16], backend: 'webgl2' as const, verify: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    expect(settled.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    expect(settled.match?.ok).toBe(true);          // GLSL /0 ≡ the interpreter's 0
    expect((settled.value as number[] | null)?.[0]).toBe(0);            // 1/0 → 0 (not Infinity)
    expect(Number.isFinite((settled.value as number[] | null)?.[1] ?? NaN)).toBe(true);   // 1/1 = 1 still correct
  });

  it('a sqrt(negative) kernel matches the oracle on WebGL2 (guarded to 0, not native NaN)', async () => {
    const host = new RuntimeReactiveHost();
    // x[i]-8 goes negative for i<8 → sqrt of a negative. The interpreter maps it to 0; raw GLSL sqrt → NaN.
    const kernel = kernelOf(`
      const N = 16
      const x = f32(N, (i) => i)
      component k(i) { return sqrt(x[i] - 8) }
      k`, host);
    const engine = new GpuEngine(host, deps);
    const cfg = { output: [16], backend: 'webgl2' as const, verify: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    expect(settled.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    expect(settled.match?.ok).toBe(true);          // GLSL sqrt(neg) ≡ the interpreter's 0
    expect((settled.value as number[] | null)?.[0]).toBe(0);            // sqrt(-8) → 0 (not NaN)
    expect((settled.value as number[] | null)?.[9]).toBeCloseTo(1, 3);  // sqrt(9-8)=1 still correct
  });

  it('a rank-3 kernel matches the oracle on WebGL2 (the flat-index → (x,y,z) decomposition round-trips)', async () => {
    const host = new RuntimeReactiveHost();
    // A distinct value per (x,y,z) cell (x*100 + y*10 + z). WebGL2 has no 3-D render target: the shader reads
    // the FLAT texel index and decomposes it back to (x,y,z) via z=_flat%D, y=(_flat/D)%H, x=_flat/(H*D). If
    // that decomposition (or the backend's _cols=H/_deps=D uniforms) were wrong the cells would scramble →
    // match.ok fails. The dims [W=2, H=3, D=4] are all distinct so a swapped axis can't accidentally agree.
    const kernel = kernelOf(`component k(x, y, z) { return x * 100 + y * 10 + z } k`, host);
    const engine = new GpuEngine(host, deps);
    const cfg = { output: [2, 3, 4], backend: 'webgl2' as const, verify: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    expect(settled.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    expect(settled.match?.ok).toBe(true);   // the decomposed (x,y,z) coords ≡ the interpreter oracle
    const out = settled.value as number[] | null;
    // Flat index = (x*H + y)*D + z with H=3, D=4. Cell (1,2,3) → (1*3+2)*4+3 = 23 → value 1*100+2*10+3 = 123.
    expect(out?.[23]).toBeCloseTo(123, 3);
    expect(out?.[0]).toBeCloseTo(0, 3);     // cell (0,0,0)
    // Cell (0,1,0) → (0*3+1)*4+0 = 4 → value 0*100+1*10+0 = 10 (proves y is decoded, not folded into x/z).
    expect(out?.[4]).toBeCloseTo(10, 3);
  });

  it('the emitted GLSL actually COMPILES + links on a real WebGL2 context (catches type errors a substring test misses)', async (ctx) => {
    const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(1, 1) : document.createElement('canvas');
    const gl = (canvas as HTMLCanvasElement).getContext('webgl2') as WebGL2RenderingContext | null;
    if (!gl || !gl.getExtension('EXT_color_buffer_float')) return ctx.skip('no WebGL2 context, or no EXT_color_buffer_float');
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`
      const N = 8
      const a = f32(N * N, (i) => i)
      const b = f32(N * N, (i) => i)
      component product(row, col) { let sum = 0; for (const k of range(N)) { sum = sum + a[row * N + k] * b[k * N + col] } return sum }
      product`, host);
    const { bindings } = gateKernel(kernel, host);
    const glsl = emitGlsl(kernel, bindings, 'f32');
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, glsl); gl.compileShader(fs);
    const ok = gl.getShaderParameter(fs, gl.COMPILE_STATUS) as boolean;
    const log = gl.getShaderInfoLog(fs) ?? '';
    expect(ok, `GLSL compile errors:\n${log}`).toBe(true);   // ZERO fragment-shader compile errors — the real gate
    gl.deleteShader(fs);
  });

  it('a resident output binds as the next kernel input on WebGL2 (same instance, no readback mismatch)', async () => {
    const back = tryWebGl2Backend();
    if (!back) { expect(webgl2Live).toBe(false); return; }   // no WebGL2 float here → skip (the cpu floor is covered elsewhere)
    const host = new RuntimeReactiveHost();
    const N = 256;
    // Stage A: y[i] = x[i] + 1. Retain its output on-device.
    const kA = kernelOf(`
      const N = ${N}
      const x = f32(N, (i) => i)
      component a(i) { return x[i] + 1 }
      a`, host);
    const resA = await back.dispatch(assembleDispatch(kA, host, [N], { retainOutput: true }));
    expect(resA.resident).toBeDefined();
    expect(resA.output[3]).toBeCloseTo(4, 3);   // x[3] + 1 = 4 (the retained buffer holds the right cells)

    // Stage B: z[i] = y[i] * 2, binding A's resident output as `y`. The stage-B kernel closes over a DUMMY
    // `y` (all zeros) so the CPU-fallback `inputs['y']` data is deliberately WRONG — if the backend uploaded
    // the fallback instead of binding A's resident texture, z would be 0. z === 2*(x+1) therefore proves the
    // resident-bind path (same instance, matching token) actually fed A's on-device output into stage B.
    const kB = kernelOf(`
      const N = ${N}
      const y = f32(N, (i) => 0)
      component b(i) { return y[i] * 2 }
      b`, host);
    const diB = assembleDispatch(kB, host, [N], { residentInputs: new Map([['y', resA.resident!.gpuBuffer]]) });
    // Sanity: the fallback `inputs['y']` really is the wrong (zero) data, so a pass can only come from residency.
    expect(diB.inputs.find((i) => i.name === 'y')!.data.every((v) => v === 0)).toBe(true);
    const resB = await back.dispatch(diB);
    for (const k of [0, 1, 7, 42, 128, 255]) expect(resB.output[k]).toBeCloseTo(2 * (k + 1), 3);   // z[k] = 2*(x[k]+1)

    resA.resident!.dispose();
    back[Symbol.dispose]();
  });

  it('an N-step gpu-buffer pipeline is correct on the real backend: each stage feeds the prior resident output', async () => {
    const host = new RuntimeReactiveHost();
    const engine = new GpuEngine(host, deps);
    const N = 128;
    // A HostEnvironment whose `resident()` head yields a specific prior-stage handle — the way a consumer
    // kernel closes over an upstream gpu-buffer as its own input.
    const residentEnv = (handle: unknown): HostEnvironment => ({
      resolveCall(head: string, _k: string, _a: Arg[], _c: HostValue[], _s: SourceSpan) {
        return head === 'resident' ? { handled: true as const, value: handle as HostValue, kind: 'value' as const } : { handled: false as const };
      },
    });
    // Settle one stage: dispatch, wait for the async task, then re-read the memo hit for the settled resource.
    const settle = async (kernel: UserFn, cfg: Parameters<GpuEngine['gpu']>[1]): Promise<ReturnType<GpuEngine['gpu']>> => {
      change(() => { engine.gpu(kernel, cfg); });
      await new Promise((r) => setTimeout(r, 300));
      let s!: ReturnType<GpuEngine['gpu']>;
      change(() => { s = engine.gpu(kernel, cfg); });
      return s;
    };
    // Stage A: a[i] = i, retained as a resident gpu-buffer.
    const kA = kernelOf(`const N = ${N}\nconst x = f32(N, (i) => i)\ncomponent a(i) { return x[i] }\na`, host);
    const sA = await settle(kA, { output: [N], backend: 'webgl2', outputType: 'gpu-buffer' });
    // Stage B: b[i] = hA[i] + 1, closing over A's resident handle; retained again as a gpu-buffer.
    const resB = evaluateProgram(`const hA = resident()\ncomponent b(i) { return hA[i] + 1 }\nb`, { host, env: residentEnv(sA.value) });
    const kB = resB.value as UserFn;
    const sB = await settle(kB, { output: [N], backend: 'webgl2', outputType: 'gpu-buffer' });
    // Stage C: c[i] = hB[i] * 2, closing over B's resident handle → C[k] = (k + 1) * 2.
    const resC = evaluateProgram(`const hB = resident()\ncomponent c(i) { return hB[i] * 2 }\nc`, { host, env: residentEnv(sB.value) });
    const kC = resC.value as UserFn;
    const sC = await settle(kC, { output: [N], backend: 'webgl2' });
    expect(sC.pending).toBe(false);
    // The engine pools ONE backend per `requested` config, so all three stages share it. The pipeline is
    // CORRECT either way (each handle's descriptor readback is the always-correct fallback), on whichever
    // backend actually ran (webgl2 when live, else the cpu floor).
    expect(sC.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    for (const k of [0, 1, 7, 42, 127]) expect((sC.value as number[])[k]).toBeCloseTo((k + 1) * 2, 3);   // ((i)+1)*2
    engine[Symbol.dispose]();
  });

  it('residency fires through a POOLED engine pipeline on WebGL2: a producer resident texture binds directly on the next stage', async () => {
    const host = new RuntimeReactiveHost();
    const engine = new GpuEngine(host, deps);
    const N = 128;
    const residentEnv = (handle: unknown): HostEnvironment => ({
      resolveCall(head: string, _k: string, _a: Arg[], _c: HostValue[], _s: SourceSpan) {
        return head === 'resident' ? { handled: true as const, value: handle as HostValue, kind: 'value' as const } : { handled: false as const };
      },
    });
    const settle = async (kernel: UserFn, cfg: Parameters<GpuEngine['gpu']>[1]): Promise<ReturnType<GpuEngine['gpu']>> => {
      change(() => { engine.gpu(kernel, cfg); });
      await new Promise((r) => setTimeout(r, 300));
      let s!: ReturnType<GpuEngine['gpu']>;
      change(() => { s = engine.gpu(kernel, cfg); });
      return s;
    };
    // Reset the resident-bind counter, then run a 2-stage engine pipeline on the SAME pooled backend.
    mlgpuResetResidentBinds();
    // Stage A: a[i] = i + 1, retained as a resident gpu-buffer.
    const kA = kernelOf(`const N = ${N}\nconst x = f32(N, (i) => i)\ncomponent a(i) { return x[i] + 1 }\na`, host);
    const sA = await settle(kA, { output: [N], backend: 'webgl2', outputType: 'gpu-buffer' });
    // Stage B: b[i] = hA[i] * 2, closing over A's resident handle. On the POOLED engine, A + B share the
    // WebGL2 context, so B binds A's resident texture directly (token match) — no readback/re-upload.
    const resB = evaluateProgram(`const hA = resident()\ncomponent b(i) { return hA[i] * 2 }\nb`, { host, env: residentEnv(sA.value) });
    const kB = resB.value as UserFn;
    const sB = await settle(kB, { output: [N], backend: 'webgl2' });
    expect(sB.pending).toBe(false);
    expect(sB.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    for (const k of [0, 1, 7, 42, 127]) expect((sB.value as number[])[k]).toBeCloseTo((k + 1) * 2, 3);   // (i+1)*2
    // The empirical payoff: on a LIVE WebGL2 adapter the pooled backend bound A's resident texture on stage B
    // (residency fired). On the CPU floor there's no WebGL2 resident-texture path, so the counter stays 0
    // (correctness is still proven above via the fallback).
    if (webgl2Live) expect(mlgpuResidentBinds()).toBeGreaterThan(0);
    engine[Symbol.dispose]();
  });

  it('buffer output round-trips on WebGL2 (a frozen f32 handle whose cells match the oracle)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`
      const N = 64
      const x = f32(N, (i) => i)
      component k(i) { return x[i] * 3 }
      k`, host);
    const engine = new GpuEngine(host, deps);
    const cfg = { output: [64], backend: 'webgl2' as const, outputType: 'buffer' as const, verify: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, cfg); });
    expect(s.pending).toBe(false);
    expect(s.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    expect(s.match?.ok).toBe(true);
    const { descriptorOf, isTypedArray } = await import('@metael/lang');
    expect(isTypedArray(s.value)).toBe(true);
    expect(descriptorOf(s.value)!.getIndex!(s.value, 5)).toBeCloseTo(15, 3);   // x[5]*3
  });

  it('sums a 1024-element buffer via a multi-pass WebGL2 tree reduction (matches the oracle)', async () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const xs = evaluateProgram(`f32(1024, (i) => i + 1)`, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] }).value as object;
    const engine = new GpuEngine(host, deps);
    const cfg = { input: xs, identity: 0, backend: 'webgl2' as const, verify: true };
    change(() => { engine.gpuReduce(reducer, cfg); });
    await new Promise((r) => setTimeout(r, 400));
    let s!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { s = engine.gpuReduce(reducer, cfg); });
    expect(s.pending).toBe(false);
    // On a live WebGL2 adapter the tree reduction runs on webgl2; else the ladder falls to the cpu oracle floor.
    expect(s.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    expect(s.value).toBeCloseTo(524800, 0);   // sum(1..1024) = 1024*1025/2 = 524800 (exact-int in f32 range)
    expect(s.match?.ok).toBe(true);            // the GPU tree fold within tolerance of the linear oracle
  });

  it('sums a 777-element buffer (a non-tile-multiple length exercises the partial-tile identity guard)', async () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const xs = evaluateProgram(`f32(777, (i) => i + 1)`, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] }).value as object;
    const engine = new GpuEngine(host, deps);
    const cfg = { input: xs, identity: 0, backend: 'webgl2' as const, verify: true };
    change(() => { engine.gpuReduce(reducer, cfg); });
    await new Promise((r) => setTimeout(r, 400));
    let s!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { s = engine.gpuReduce(reducer, cfg); });
    expect(s.pending).toBe(false);
    expect(s.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    // sum(1..777) = 777*778/2 = 302253. The partial last tile must fold identity (0), not garbage texels.
    expect(s.value).toBeCloseTo(302253, 0);
    expect(s.match?.ok).toBe(true);
  });

  it('folds a max over a buffer via a comparison reducer on WebGL2 (the tree fold ≡ the linear oracle)', async () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component mx(a, b) { return a > b ? a : b }\nmx`, host);
    // A shuffled range so the max is not the last element — the tree fold must still find it.
    const xs = evaluateProgram(`f32(500, (i) => (i * 37) % 500)`, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] }).value as object;
    const engine = new GpuEngine(host, deps);
    const cfg = { input: xs, identity: -1e30, backend: 'webgl2' as const, verify: true };
    change(() => { engine.gpuReduce(reducer, cfg); });
    await new Promise((r) => setTimeout(r, 400));
    let s!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { s = engine.gpuReduce(reducer, cfg); });
    expect(s.pending).toBe(false);
    expect(s.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    expect(s.value).toBe(499);          // max of (i*37)%500 over i in 0..499 is 499
    expect(s.match?.ok).toBe(true);
  });

  it('reuses a cached program across two dispatches with different buffers (results stay correct)', async () => {
    // TWO kernels with IDENTICAL body/shape but DIFFERENT input buffers, dispatched on ONE engine. The
    // buffer initializer is NOT baked into the emitted GLSL (buffers are texture inputs, referenced by name),
    // so both kernels emit the SAME shader source → the SAME cached, linked WebGLProgram is reused across the
    // two dispatches. That the SECOND result is correct proves the cached program was rebound to a FRESH input
    // texture per dispatch, not left bound to the first dispatch's (stale) buffer. `evaluateProgram` gives
    // each kernel its own root env, so both can define `const x` on one host without colliding; the engine
    // resolves each kernel's buffer from ITS OWN closure (readClosureValue), independent of the engine's host.
    const host = new RuntimeReactiveHost();
    const engine = new GpuEngine(host, deps);
    const kA = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] + 1 }\nk`, host);
    const kB = kernelOf(`const x = f32(4, (i) => i * 5)\ncomponent k(i) { return x[i] + 1 }\nk`, host);
    const cfg = { output: [4], backend: 'webgl2' as const };

    // Dispatch A (buffer [0,1,2,3]) — settle via the file's inline change → setTimeout → re-read pattern.
    change(() => { engine.gpu(kA, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let rA!: ReturnType<GpuEngine['gpu']>;
    change(() => { rA = engine.gpu(kA, cfg); });
    expect(rA.pending).toBe(false);
    expect(rA.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');

    // Dispatch B (buffer [0,5,10,15]) on the SAME engine — the identical GLSL hits the cached program.
    change(() => { engine.gpu(kB, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let rB!: ReturnType<GpuEngine['gpu']>;
    change(() => { rB = engine.gpu(kB, cfg); });
    expect(rB.pending).toBe(false);
    expect(rB.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');

    expect(rA.value).toEqual([1, 2, 3, 4]);      // x[i] + 1 for [0,1,2,3]
    expect(rB.value).toEqual([1, 6, 11, 16]);    // x[i] + 1 for [0,5,10,15] — cached program, fresh buffer
    expect(rA.glsl).not.toBe('');                // a real shader was emitted
    expect(rA.glsl).toBe(rB.glsl);               // identical shader → same cache entry (proves reuse, not stale)
    engine[Symbol.dispose]();
  });
});
