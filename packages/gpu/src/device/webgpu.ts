// The WebGPU backend: verify a REAL adapter/device (navigator.gpu truthy ≠ working), create storage
// buffers + a uniform (dims + scalars), one compute pipeline, dispatch, read back. Compute-only.
//
// TypeScript's DOM lib ships every WebGPU *interface* (GPUDevice/GPUAdapter/GPUBuffer/…) but omits the two
// runtime flag-constant objects (GPUBufferUsage/GPUMapMode) — real globals in Chromium. Declare just those
// two module-locally rather than pulling in @webgpu/types (which would collide with the DOM lib's own GPU
// interfaces). No tsconfig lib change is needed: the base config already includes DOM.
import type { Backend, DispatchInput, DispatchResult, ReduceDispatchInput, ReduceDispatchResult, HistogramDispatchInput, HistogramDispatchResult } from './index.ts';
import type { DeviceLimits } from '../cost.ts';
// f16 packing/alignment lives in a device-free module so its 4-byte-alignment invariant is unit-testable in
// node (a real f16 dispatch needs a shader-f16 adapter, absent here). `packF16` rounds to an even element
// count (byteLength % 4 === 0); `align4` rounds a byte count up to a multiple of 4 — both required because
// WebGPU's writeBuffer data size + copyBufferToBuffer size must be multiples of 4, and an `array<f16>`
// element is 2 bytes (so an ODD element count would be 2-mod-4 and fault on a real device).
import { packF16, unpackF16, align4 } from '../f16-pack.ts';
import { REDUCE_TILE } from '../emit-glsl.ts';
import { HISTOGRAM_WORKGROUP } from '../emit-wgsl.ts';
import { makePipelineCache } from './pipeline-cache.ts';

declare const GPUBufferUsage: {
  readonly STORAGE: number; readonly UNIFORM: number;
  readonly COPY_DST: number; readonly COPY_SRC: number; readonly MAP_READ: number;
};
declare const GPUMapMode: { readonly READ: number };

// The opaque resident-buffer shape this backend produces + recognizes: a GPUBuffer tagged with the
// producing instance's token. Only a matching-token buffer may be bound directly (a foreign GPUDevice's
// buffer is unusable). Passed through `unknown` in the device contract.
interface WebGpuResident { readonly token: object; readonly buffer: GPUBuffer }
function asOwnResident(v: unknown, token: object): WebGpuResident | null {
  return v && typeof v === 'object' && (v as WebGpuResident).token === token ? (v as WebGpuResident) : null;
}

/** Probe for a working WebGPU backend, verifying a REAL adapter + device (a truthy `navigator.gpu` is NOT
 *  enough). Resolves the backend on success, or `null` when WebGPU is absent or adapter/device acquisition
 *  fails — so the engine cleanly re-ladders down to WebGL2 / CPU. */
export async function tryWebGpuBackend(): Promise<Backend | null> {
  const gpu = (globalThis.navigator as { gpu?: GPU } | undefined)?.gpu;
  if (!gpu) return null;
  let adapter: GPUAdapter | null;
  try { adapter = await gpu.requestAdapter(); } catch { return null; }
  if (!adapter) return null;
  // Request the `shader-f16` device feature when the adapter advertises it, so an `enable f16;` shader can
  // run correctly (a device WITHOUT the feature would reject an f16 shader module). When absent, the engine
  // downgrades an f16 request to f32 (a note on the resource), so a non-f16 device still works — this flag is
  // surfaced as `features.f16` below for that engine-side fallback decision.
  const hasF16 = adapter.features.has('shader-f16');
  let device: GPUDevice;
  try { device = await adapter.requestDevice(hasF16 ? { requiredFeatures: ['shader-f16'] } : {}); } catch { return null; }
  // A per-instance identity token. A resident object THIS instance creates carries this token; a resident
  // input is bound directly only when its token === INSTANCE (its GPUBuffer belongs to THIS device — a
  // foreign device's buffer is unusable). A foreign/untoken'd resident is ignored → the CPU-fallback
  // `inputs` data is uploaded fresh, keeping correctness regardless of whether residency actually fires.
  const INSTANCE: object = device;
  // Compile a WGSL module → compute pipeline ONCE per distinct shader source, reused across dispatches
  // (the backend is pooled, so this persists for the device's lifetime). A GPUComputePipeline has no
  // explicit destroy; dropping the reference (map.clear on dispose) lets it GC when the device is destroyed.
  const pipelineCache = makePipelineCache<GPUComputePipeline>(
    (code) => device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } }),
    () => { /* pipelines are freed when device.destroy() runs; nothing per-pipeline to free */ },
  );
  const limits: DeviceLimits = {
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
  };
  return {
    kind: 'webgpu',
    limits,
    features: { f16: hasF16 },
    async dispatch(input: DispatchInput): Promise<DispatchResult> {
      const total = input.dims.reduce((a, b) => a * b, 1);
      const comps = input.outputComps ?? 1;
      const outLen = total * comps;   // `_out` is a flat array<S> of length total*comps (the interleaved layout IS the normalized layout)
      const rows = input.dims[0]!; const cols = input.dims[1] ?? 1; const deps = input.dims[2] ?? 1;
      // f16 storage path: each `array<f16>` element is 2 bytes, so inputs are packed as binary16 + the output
      // is read back as binary16 and unpacked. Only reached when the device HAS shader-f16 (the engine
      // downgraded an f16 request to f32 otherwise, so `input.precision` here is f16 only on a capable device).
      const f16 = input.precision === 'f16';
      const elemBytes = f16 ? 2 : 4;
      // Output/readback byte size, rounded UP to a multiple of 4 (WebGPU requires copyBufferToBuffer size %
      // 4 === 0). For f32 this is a NO-OP (outLen*4 is already a multiple of 4) so the f32 path is
      // byte-identical; for f16 (elemBytes 2) an ODD outLen would be 2-mod-4 without the align, so it rounds
      // up by one 2-byte pad slot the shader never writes (its output store is bounds-guarded per invocation).
      const outBytes = Math.max(align4(outLen * elemBytes), 4);
      // Track which storage buffers THIS dispatch created (so it destroys exactly those); a directly-bound
      // resident input is owned by its producing dispatch's handle and must NOT be destroyed here.
      const created: GPUBuffer[] = [];
      const bufs = input.inputs.map(({ name, data }) => {
        const resident = asOwnResident(input.residentInputs?.get(name), INSTANCE);
        if (resident) return { name, gpu: resident.buffer };   // bind the resident buffer directly — no re-upload, not destroyed here
        // f16 → packF16 rounds to an even element count so packed.byteLength is a multiple of 4 (WebGPU's
        // writeBuffer size requirement); f32 → `data` is a Float32Array (byteLength always a multiple of 4).
        const packed: Float32Array | Uint16Array = f16 ? packF16(data) : data;
        const gbuf = device.createBuffer({ size: Math.max(packed.byteLength, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(gbuf, 0, packed);
        created.push(gbuf);
        return { name, gpu: gbuf };
      });
      const outBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const scalars = input.scalars;
      // The `_Params` uniform is THREE u32 dispatch dims (rows, cols, deps @ bytes 0/4/8) then each closed-over
      // scalar as an f32 (@ bytes 12,16,…) — the struct member order emitWgsl declares for EVERY rank. So the
      // field count is 3 + scalars.length; rounded UP to a multiple of 16 (the WGSL uniform-address-space
      // struct-size rule). `Math.max(…, 16)` keeps the WGSL 16-byte uniform minimum (fieldCount≥3 already
      // satisfies it, but the clamp is explicit).
      const fieldCount = 3 + scalars.length;
      const uniBytes = Math.max(Math.ceil((fieldCount * 4) / 16) * 16, 16);
      const uni = new ArrayBuffer(uniBytes);
      new Uint32Array(uni, 0, 3).set([rows, cols, deps]);
      if (scalars.length) new Float32Array(uni, 12, scalars.length).set(scalars.map((s) => s.value));
      const uniBuf = device.createBuffer({ size: uni.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(uniBuf, 0, uni);
      const pipeline = pipelineCache.get(input.wgsl);
      const entries: GPUBindGroupEntry[] = [];
      let bi = 0;
      for (const b of bufs) entries.push({ binding: bi++, resource: { buffer: b.gpu } });
      entries.push({ binding: bi++, resource: { buffer: outBuf } });
      entries.push({ binding: bi, resource: { buffer: uniBuf } });
      const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      const start = performance.now();
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
      // The dispatch grid per rank MUST match emitWgsl's `@workgroup_size` so `dispatchWorkgroups × workgroup`
      // covers every cell: rank-3 → (4,4,4) so ceil per axis / 4; rank-2 → (8,8) so ceil / 8 in x,y; rank-1 →
      // (64) so ceil / 64 in x. A rank-3 grid that omitted the z divisor (wgZ defaulting to 1) would cover only
      // deps 0..3 and silently drop the rest of the D axis. rows=dims[0]=W, cols=dims[1]=H, deps=dims[2]=D.
      const wgX = input.dims.length === 3 ? Math.ceil(rows / 4) : input.dims.length === 2 ? Math.ceil(rows / 8) : Math.ceil(rows / 64);
      const wgY = input.dims.length === 3 ? Math.ceil(cols / 4) : input.dims.length === 2 ? Math.ceil(cols / 8) : 1;
      const wgZ = input.dims.length === 3 ? Math.ceil(deps / 4) : 1;
      pass.dispatchWorkgroups(wgX, wgY, wgZ); pass.end();
      const readBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBytes);
      device.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      // f16 → read back the (possibly pad-rounded) binary16 bit patterns + unpack ONLY the first outLen to f32
      // (any trailing pad slot is ignored); f32 → the mapped range IS the Float32Array (outBytes === outLen*4).
      const output = f16
        ? unpackF16(new Uint16Array(readBuf.getMappedRange().slice(0)), outLen)
        : new Float32Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();
      const ms = performance.now() - start;
      // Destroy exactly the buffers THIS dispatch created (never a directly-bound resident input — that's
      // owned by its producing dispatch's handle). The uniform + readback buffers are always this
      // dispatch's. `outBuf` is retained (returned as `resident`) when asked, else destroyed now.
      for (const b of created) b.destroy(); uniBuf.destroy(); readBuf.destroy();
      const out = output.subarray(0, outLen);
      if (input.retainOutput) {
        // outBuf carries STORAGE usage → it binds directly as a `var<storage, read>` input to a later
        // same-instance dispatch. The disposer frees it once its owning handle is disposed/evicted.
        return { output: out, ms, resident: { gpuBuffer: { token: INSTANCE, buffer: outBuf }, dispose: () => outBuf.destroy() } };
      }
      outBuf.destroy();
      return { output: out, ms };
    },
    // A REDUCTION as a MULTI-PASS workgroup-shared tree reduction. Unlike WebGL2's fragment ping-pong, WebGPU
    // has a real compute stage with workgroup-shared memory: each workgroup of G=256 threads loads G elements
    // into `var<workgroup> _scratch`, tree-folds them with `workgroupBarrier()` between halving steps, and
    // thread 0 writes the workgroup's partial to `_out[workgroup_id]`. The driver folds in passes: N →
    // ceil(N/G) partials → … → 1. The SAME compute shader (input.wgsl, from emitReduceWgsl) + pipeline run
    // EVERY pass; only the `_in`/`_out` storage buffers + the `_RParams` uniform (inLen + identity + scalars)
    // change. Each pass frees the just-consumed input buffer + that pass's uniform (mirroring the map
    // dispatch's buffer discipline + the webgl2 reduce's per-pass free) — no leak. G is baked in the shader.
    //
    // NOTE: there is NO WebGPU adapter in this environment, so this leg is NOT runtime-exercised here — the
    // WGSL is structurally snapshotted + compiled on a real device via the gated browser test. The buffer
    // lifecycle below is written correct-by-inspection (a real device would leak/fault otherwise).
    async dispatchReduce(input: ReduceDispatchInput): Promise<ReduceDispatchResult> {
      const start = performance.now();
      const G = REDUCE_TILE;   // the SHARED constant — must match emitReduceWgsl's @workgroup_size(G) + array<f32, Gu>
      const n0 = input.inputValues.length;
      // A zero-length input folds to the identity — no buffers, no pass (mirrors the webgl2 early return).
      if (n0 === 0) return { value: input.identity, ms: performance.now() - start };

      const pipeline = pipelineCache.get(input.wgsl);

      // Pack the `_RParams` uniform for a pass: inLen (u32 @0) + identity (f32 @4) + each closed-over scalar
      // constant (f32 @8,12,…), the struct member order emitReduceWgsl declares. Rounded UP to a multiple of 16
      // (the WGSL uniform-address-space struct alignment — same rounding the map path's _Params uses).
      const scalars = input.scalars;
      const fieldCount = 2 + scalars.length;
      const uniBytes = Math.max(Math.ceil((fieldCount * 4) / 16) * 16, 16);
      const makeUniform = (inLen: number): GPUBuffer => {
        const ab = new ArrayBuffer(uniBytes);
        new Uint32Array(ab, 0, 1)[0] = inLen;
        new Float32Array(ab, 4, 1)[0] = input.identity;
        if (scalars.length) new Float32Array(ab, 8, scalars.length).set(scalars.map((s) => s.value));
        const ub = device.createBuffer({ size: uniBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(ub, 0, ab);
        return ub;
      };

      // The current input buffer this driver owns (the initial upload, then each pass's prior output). Held in
      // an OUTER binding so the `finally` frees it if an exception interrupts the fold mid-pass; nulled the
      // instant the loop frees the consumed input so the happy path never double-frees (mirrors webgl2's
      // `liveTex`). The readback buffer is likewise tracked so an interrupted pass frees it too.
      let liveIn: GPUBuffer | null = null;
      let readBuf: GPUBuffer | null = null;
      try {
        // Upload the initial input as a STORAGE buffer (bound as `_in`). Float32Array.byteLength is a multiple
        // of 4 (WebGPU's writeBuffer requirement); min 4 bytes.
        const inBuf = device.createBuffer({ size: Math.max(input.inputValues.byteLength, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(inBuf, 0, input.inputValues);
        liveIn = inBuf;
        let curLen = n0;
        // Fold in passes until ONE partial remains. At least one pass always runs (even a single-element input
        // folds `reduce(identity, x[0])` — the identity is neutral, so a no-op — matching the webgl2 leg).
        for (;;) {
          const outLen = Math.ceil(curLen / G);                 // this pass emits ceil(curLen/G) partials
          const outBuf = device.createBuffer({ size: Math.max(outLen * 4, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
          const uniBuf = makeUniform(curLen);
          const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: liveIn } },
              { binding: 1, resource: { buffer: outBuf } },
              { binding: 2, resource: { buffer: uniBuf } },
            ],
          });
          const enc = device.createCommandEncoder();
          const pass = enc.beginComputePass();
          pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(outLen);   // one workgroup per output partial (each writes _out[wid.x])
          pass.end();
          device.queue.submit([enc.finish()]);

          // Free the just-consumed input buffer + this pass's uniform (this driver owns every buffer it
          // creates). Null `liveIn` in the same breath so the finally can't re-delete it. The output buffer
          // becomes the next pass's input (and the next `liveIn`).
          liveIn.destroy(); liveIn = null; uniBuf.destroy();
          if (outLen === 1) {
            // The single surviving partial (element 0 of outBuf) is the scalar. Copy it to a MAP_READ buffer,
            // map, read, then free outBuf + the readback buffer.
            readBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            const enc2 = device.createCommandEncoder();
            enc2.copyBufferToBuffer(outBuf, 0, readBuf, 0, 4);
            device.queue.submit([enc2.finish()]);
            await readBuf.mapAsync(GPUMapMode.READ);
            const value = new Float32Array(readBuf.getMappedRange().slice(0))[0]!;
            readBuf.unmap();
            readBuf.destroy(); readBuf = null; outBuf.destroy();
            return { value, ms: performance.now() - start };
          }
          liveIn = outBuf;
          curLen = outLen;
        }
      } finally {
        // On the happy path `liveIn`/`readBuf` were nulled after being freed, so this is a no-op. On an
        // exception mid-pass they free the in-flight buffers (no leak). The shader module + pipeline are GC'd.
        if (liveIn) liveIn.destroy();
        if (readBuf) readBuf.destroy();
      }
    },
    // A HISTOGRAM as a SINGLE-PASS atomic scatter. WebGPU has real storage atomics, so one thread per input
    // element maps it to a bin index (`_binOf`, in input.wgsl from emitHistogramWgsl) and `atomicAdd`s that
    // bin. `_bins` is a `var<storage, read_write> array<atomic<u32>>` of `bins` counts; a fresh WebGPU storage
    // buffer created with mappedAtCreation:false is ZERO-INITIALIZED per spec, so no explicit clear is needed
    // (a note in emitHistogramWgsl documents this). Read back the u32 counts → a number[]. This driver frees
    // every buffer it creates in the `finally` (no leak on the happy path or an exception).
    //
    // NOTE: there is NO WebGPU adapter in this environment, so this leg is NOT runtime-exercised here — the
    // WGSL is structurally snapshotted + compiled on a real device via the gated browser test. The buffer
    // lifecycle below is written correct-by-inspection (a real device would leak/fault otherwise).
    async dispatchHistogram(input: HistogramDispatchInput): Promise<HistogramDispatchResult> {
      const start = performance.now();
      const G = HISTOGRAM_WORKGROUP;   // must match emitHistogramWgsl's @workgroup_size(G)
      const n = input.inputValues.length;
      const bins = Math.max(1, input.bins);   // at least one bin slot (a zero-bin histogram would fault createBuffer)

      const pipeline = pipelineCache.get(input.wgsl);

      // Pack the `_HParams` uniform: inLen (u32 @0) + bins (u32 @4) + each closed-over scalar constant
      // (f32 @8,12,…), the struct member order emitHistogramWgsl declares. Rounded UP to a multiple of 16
      // (WGSL uniform-address-space struct alignment — same rounding the map/reduce paths use).
      const scalars = input.scalars;
      const fieldCount = 2 + scalars.length;
      const uniBytes = Math.max(Math.ceil((fieldCount * 4) / 16) * 16, 16);

      let inBuf: GPUBuffer | null = null;
      let binsBuf: GPUBuffer | null = null;
      let uniBuf: GPUBuffer | null = null;
      let readBuf: GPUBuffer | null = null;
      try {
        // Upload the input as a STORAGE buffer (bound as `_in`). Float32Array.byteLength is a multiple of 4
        // (WebGPU's writeBuffer requirement); min 4 bytes.
        inBuf = device.createBuffer({ size: Math.max(input.inputValues.byteLength, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(inBuf, 0, input.inputValues);
        // The bins buffer: `bins` u32 counts, ZERO-INITIALIZED (createBuffer with mappedAtCreation false is
        // zero-initialized per WebGPU spec — no explicit clear needed). COPY_SRC to read the counts back.
        const binsBytes = Math.max(bins * 4, 4);
        binsBuf = device.createBuffer({ size: binsBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        uniBuf = device.createBuffer({ size: uniBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const ab = new ArrayBuffer(uniBytes);
        new Uint32Array(ab, 0, 2).set([n, bins]);
        if (scalars.length) new Float32Array(ab, 8, scalars.length).set(scalars.map((s) => s.value));
        device.queue.writeBuffer(uniBuf, 0, ab);

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: inBuf } },
            { binding: 1, resource: { buffer: binsBuf } },
            { binding: 2, resource: { buffer: uniBuf } },
          ],
        });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
        // One thread per input element; ceil(n/G) workgroups (each lane past inLen early-returns in the shader).
        pass.dispatchWorkgroups(Math.max(1, Math.ceil(n / G)));
        pass.end();
        readBuf = device.createBuffer({ size: binsBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        enc.copyBufferToBuffer(binsBuf, 0, readBuf, 0, binsBytes);
        device.queue.submit([enc.finish()]);
        await readBuf.mapAsync(GPUMapMode.READ);
        const raw = new Uint32Array(readBuf.getMappedRange().slice(0));
        readBuf.unmap();
        // Only the first `bins` slots are meaningful (binsBytes may round a 1-bin case up to 4 bytes = 1 slot).
        const counts = Array.from(raw.subarray(0, bins), (v) => v);
        return { counts, ms: performance.now() - start };
      } finally {
        // Free every buffer this driver created (the happy path + any exception mid-flight). The shader module
        // + pipeline are GC'd.
        if (inBuf) inBuf.destroy();
        if (binsBuf) binsBuf.destroy();
        if (uniBuf) uniBuf.destroy();
        if (readBuf) readBuf.destroy();
      }
    },
    [Symbol.dispose]() { pipelineCache[Symbol.dispose](); device.destroy(); },
  };
}
