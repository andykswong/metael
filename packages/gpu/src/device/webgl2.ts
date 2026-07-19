// The WebGL2 backend: run a compute kernel as a fullscreen-quad FRAGMENT shader (WebGL2 has no compute
// stage), reading inputs from float textures and writing one output cell per fragment into an RGBA32F
// framebuffer, read back with readPixels. Consumes EXACTLY the same seam as the WebGPU backend —
// `input.inputs` (pre-resolved Float32Arrays) + `input.scalars` (numbers) — never a custom-value store.
//
// TEXEL-PACKING CONTRACT (must match emit-glsl.ts's _fetch + the flat-index → fragment map):
//   • Every buffer is packed row-major into an R32F texture of width `texW = min(len, MAX_TEX_W)` and
//     height `ceil(len / texW)`; element idx lives at texel (idx % texW, idx / texW). The width is passed
//     as `<name>_texW` and the element count as `<name>_len`.
//   • Output: a 2-D kernel renders into a `cols × rows` RGBA32F target (fragment (x=col, y=row)); a 1-D
//     kernel renders into a `texW × ceil(total/texW)` target with `_texW` = that width. `_rows`/`_cols`
//     are set so the emitter's `_flat` and the bounds `discard` are correct.
import type { Backend, DispatchInput, DispatchResult, ReduceDispatchInput, ReduceDispatchResult } from './index.ts';
import type { DeviceLimits } from '../cost.ts';
import { makePipelineCache } from './pipeline-cache.ts';

const MAX_TEX_W = 2048;

/** The width of the square-ish texture that packs `len` elements (capped at MAX_TEX_W). */
function texWidth(len: number): number { return Math.max(1, Math.min(len, MAX_TEX_W)); }

// Test-only instrumentation: counts how many times a resident input was bound DIRECTLY (same-instance token
// match) rather than uploaded fresh. A test resets it, runs a pooled multi-stage pipeline, then asserts it
// incremented — proving a producer's on-device texture was actually fed into a consumer stage (residency),
// not merely that the values happen to match via the CPU-fallback readback. MLGPU_-namespaced test scaffolding.
let MLGPU_RESIDENT_BINDS = 0;
export function mlgpuResidentBinds(): number { return MLGPU_RESIDENT_BINDS; }
export function mlgpuResetResidentBinds(): void { MLGPU_RESIDENT_BINDS = 0; }

// The opaque resident-texture shape this backend produces + recognizes: an output RGBA32F texture tagged
// with the producing instance's token, plus the packing metadata (`texW`, `len`) a later stage needs to set
// the `<name>_texW`/`<name>_len` uniforms. Only a matching-token texture may be bound directly (a
// WebGLTexture belongs to its GL context). Passed through `unknown` in the device contract.
interface WebGl2Resident { readonly token: object; readonly texture: WebGLTexture; readonly texW: number; readonly len: number }
function asOwnResident(v: unknown, token: object): WebGl2Resident | null {
  return v && typeof v === 'object' && (v as WebGl2Resident).token === token ? (v as WebGl2Resident) : null;
}

/** Try to acquire a WebGL2 context with float-color-buffer support. Returns null when WebGL2, the
 *  EXT_color_buffer_float extension, or a float texture is unavailable (→ the ladder falls to CPU). */
export function tryWebGl2Backend(): Backend | null {
  const makeCanvas = (): { getContext(id: 'webgl2'): WebGL2RenderingContext | null } | null => {
    if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(1, 1) as unknown as { getContext(id: 'webgl2'): WebGL2RenderingContext | null };
    if (typeof document !== 'undefined') return document.createElement('canvas');
    return null;
  };
  const canvas = makeCanvas();
  if (!canvas) return null;
  const gl = canvas.getContext('webgl2');
  if (!gl) return null;
  // RGBA32F as a color-renderable/readable target requires this extension in WebGL2.
  if (!gl.getExtension('EXT_color_buffer_float')) return null;

  const limits: DeviceLimits = {
    maxStorageBufferBindingSize: MAX_TEX_W * MAX_TEX_W * 4,
    maxComputeWorkgroupsPerDimension: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
  };

  // A per-instance identity token (the GL context itself). A resident texture THIS instance creates carries
  // this token; a resident input is bound directly only when its token === INSTANCE (its WebGLTexture lives
  // in THIS context). A foreign resident is ignored → the CPU-fallback `inputs` data is uploaded fresh.
  const INSTANCE: object = gl;

  // Link a GLSL program ONCE per distinct fragment-shader source, reused across dispatches (the backend is
  // pooled). Freed in dispose() via gl.deleteProgram (parity with the per-dispatch delete this replaces).
  const programCache = makePipelineCache<WebGLProgram>(
    (frag) => buildProgram(gl, frag),
    (p) => gl.deleteProgram(p),
  );

  return {
    kind: 'webgl2',
    limits,
    async dispatch(input: DispatchInput): Promise<DispatchResult> {
      const start = performanceNow();
      const total = input.dims.reduce((a, b) => a * b, 1);
      // The dispatch dims: W=dims[0], H=dims[1], D=dims[2]. `_cols` (=H) and `_deps` (=D) feed the shader's
      // rank-3 flat-index decomposition (_x=_flat/(H*D), _y=(_flat/D)%H, _z=_flat%D), so a rank-3 dispatch must
      // set cols=dims[1] (NOT 1) and deps=dims[2]. rank 2 keeps cols=dims[1], deps=1; rank 1 keeps cols=1.
      const rows = input.dims[0] ?? 1;
      const cols = input.dims.length >= 2 ? input.dims[1]! : 1;
      const deps = input.dims.length === 3 ? input.dims[2]! : 1;
      // Output target dims. 2-D → cols×rows (fragment x=col, y=row). 1-D/3-D → a FLAT texW×ceil(total/texW)
      // texture (no 3-D render target in WebGL2; the shader decomposes the flat texel index back to x,y,z).
      const outW = input.dims.length === 2 ? cols : texWidth(total);
      const outH = input.dims.length === 2 ? rows : Math.ceil(total / outW);

      // The cost gate runs against a static limits HINT before backend selection, so it can't know THIS
      // device's real MAX_TEXTURE_SIZE. Re-check the derived output + input texture dims here and fail with
      // an ALLOC-shaped message (not a generic framebuffer-incomplete) if they exceed the device.
      const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
      const tooBig = (w: number, h: number): boolean => w > maxTex || h > maxTex;
      if (tooBig(outW, outH)) throw new Error(`MLGPU-ALLOC: output texture ${outW}x${outH} exceeds the device max texture size ${maxTex}`);
      for (const buf of input.inputs) { const w = texWidth(buf.data.length); if (tooBig(w, Math.ceil(buf.data.length / w))) throw new Error(`MLGPU-ALLOC: input '${buf.name}' texture exceeds the device max texture size ${maxTex}`); }

      const program = programCache.get(input.glsl);
      gl.useProgram(program);

      // Fullscreen clip-space quad (two triangles) → drives one fragment per output texel.
      const quad = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
      const posLoc = gl.getAttribLocation(program, '_pos');
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

      // Bind each input to a texture unit + set the sampler + its _texW/_len. A resident input owned by THIS
      // instance binds its existing texture directly (no upload); it is NOT pushed to `created` — its
      // producing dispatch's handle owns it. Everything else uploads a fresh R32F texture as before.
      const created: WebGLTexture[] = [];   // textures THIS dispatch created → destroyed at the end
      input.inputs.forEach((buf, unit) => {
        gl.activeTexture(gl.TEXTURE0 + unit);
        const resident = asOwnResident(input.residentInputs?.get(buf.name), INSTANCE);
        if (resident) {
          // A prior stage's RGBA32F output texture stores element i at texel (i % texW, i / texW), value in
          // .r — the exact layout emit-glsl's `_fetch` reads. So `texW = resident.texW`, `len = resident.len`,
          // and reading `.r` makes it bind-compatible as an input with no re-upload.
          gl.bindTexture(gl.TEXTURE_2D, resident.texture);
          gl.uniform1i(gl.getUniformLocation(program, buf.name), unit);
          gl.uniform1i(gl.getUniformLocation(program, `${buf.name}_texW`), resident.texW);
          gl.uniform1i(gl.getUniformLocation(program, `${buf.name}_len`), resident.len);
          MLGPU_RESIDENT_BINDS++;   // a same-instance resident input was bound directly (no re-upload)
          return;
        }
        const w = texWidth(buf.data.length);
        const h = Math.ceil(buf.data.length / w);
        const padded = new Float32Array(w * h);   // pad up to a full w×h grid (unused texels read as 0)
        padded.set(buf.data);
        const tex = gl.createTexture()!;
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, padded);
        gl.uniform1i(gl.getUniformLocation(program, buf.name), unit);
        gl.uniform1i(gl.getUniformLocation(program, `${buf.name}_texW`), w);
        gl.uniform1i(gl.getUniformLocation(program, `${buf.name}_len`), buf.data.length);
        created.push(tex);
      });
      // Scalars are declared `_u_<name>` by the emitter (namespaced away from the reserved dispatch uniforms).
      for (const s of input.scalars) gl.uniform1f(gl.getUniformLocation(program, `_u_${s.name}`), s.value);
      gl.uniform1i(gl.getUniformLocation(program, '_rows'), rows);
      gl.uniform1i(gl.getUniformLocation(program, '_cols'), cols);
      gl.uniform1i(gl.getUniformLocation(program, '_texW'), outW);
      gl.uniform1i(gl.getUniformLocation(program, '_deps'), deps);

      // RGBA32F render target sized to the output grid.
      const outTex = gl.createTexture()!;
      gl.activeTexture(gl.TEXTURE0 + input.inputs.length);
      gl.bindTexture(gl.TEXTURE_2D, outTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, outW, outH, 0, gl.RGBA, gl.FLOAT, null);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outTex, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('WebGL2 framebuffer incomplete (RGBA32F render target unsupported)');

      gl.viewport(0, 0, outW, outH);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      // Read back the whole RGBA grid, then gather `comps` channels per cell (row-major) into the
      // FLAT-INTERLEAVED output: cell i's component k lives in RGBA channel k of texel i → output[i*comps+k].
      // comps=1 → the R channel only (output[i] = rgba[i*4]), unchanged. The emitter packed the cell's N
      // components into the texel's RGBA (R,G,B,A) so channel k is component k.
      const comps = input.outputComps ?? 1;
      const rgba = new Float32Array(outW * outH * 4);
      gl.readPixels(0, 0, outW, outH, gl.RGBA, gl.FLOAT, rgba);
      const output = new Float32Array(total * comps);
      for (let i = 0; i < total; i++) for (let k = 0; k < comps; k++) output[i * comps + k] = rgba[i * 4 + k]!;

      // Delete only the textures THIS dispatch created (never a directly-bound resident input). The FBO +
      // quad are always this dispatch's; the program is owned by `programCache` (freed on dispose, reused
      // across dispatches). `outTex` is retained (returned as `resident`) when asked, else deleted now.
      for (const t of created) gl.deleteTexture(t);
      gl.deleteFramebuffer(fbo); gl.deleteBuffer(quad);
      if (input.retainOutput) {
        // Carry texW = outW + len = total so a later stage sets `<name>_texW`/`<name>_len` correctly when it
        // binds this RGBA32F texture as an input (its `_fetch` reads the R channel at (idx % texW, idx / texW)).
        return { output, ms: performanceNow() - start, resident: { gpuBuffer: { token: INSTANCE, texture: outTex, texW: outW, len: total }, dispose: () => gl.deleteTexture(outTex) } };
      }
      gl.deleteTexture(outTex);
      return { output, ms: performanceNow() - start };
    },
    // A REDUCTION as a MULTI-PASS ping-pong tree reduction. WebGL2 has no compute/shared-memory, so we fold
    // in passes: pass 1 reads the N-element input texture, each output texel folding a TILE of consecutive
    // elements (seeded by the identity) → ceil(N/TILE) partials in texture 1; pass k reads texture k-1 (M
    // partials) → ceil(M/TILE) → texture k; loop until 1 element remains → readback that single texel.
    // The SAME reducer-fold-over-a-tile fragment shader (input.glsl) runs every pass — only the input texture,
    // element count (`_inLen`), texel width (`_inTexW`), output width (`_outTexW`), and framebuffer size
    // change. Each pass's intermediate texture + FBO are freed the pass after they're consumed (no leak).
    async dispatchReduce(input: ReduceDispatchInput): Promise<ReduceDispatchResult> {
      const start = performanceNow();
      const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
      const tile = input.tile;

      const program = programCache.get(input.glsl);
      gl.useProgram(program);
      const quad = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
      const posLoc = gl.getAttribLocation(program, '_pos');
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
      // The identity + any closed-over scalar constant are set ONCE (they don't change across passes).
      gl.uniform1f(gl.getUniformLocation(program, '_identity'), input.identity);
      for (const s of input.scalars) gl.uniform1f(gl.getUniformLocation(program, `_u_${s.name}`), s.value);

      // Upload the initial input as an R32F texture (packed row-major into a texW×ceil(len/texW) grid). The
      // reduce shader's `_fetch` reads element i at texel (i % texW, i / texW).r — the same packing as a map
      // input. This texture is THIS driver's (never a resident input), so it is freed after pass 1 consumes it.
      const uploadInput = (data: Float32Array, len: number): { tex: WebGLTexture; texW: number; len: number } => {
        const w = texWidth(len);
        const h = Math.ceil(len / w);
        if (w > maxTex || h > maxTex) throw new Error(`MLGPU-ALLOC: reduce input texture ${w}x${h} exceeds the device max texture size ${maxTex}`);
        const padded = new Float32Array(w * h);
        padded.set(data.subarray(0, len));
        const tex = gl.createTexture()!;
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, padded);
        return { tex, texW: w, len };
      };

      const inLoc = gl.getUniformLocation(program, '_in');
      const inLenLoc = gl.getUniformLocation(program, '_inLen');
      const inTexWLoc = gl.getUniformLocation(program, '_inTexW');
      const outTexWLoc = gl.getUniformLocation(program, '_outTexW');

      // A zero-length input folds to the identity — no texture, no pass. The quad is freed BEFORE the try
      // (it allocates no texture) on this early return; the program is owned by `programCache`.
      if (input.inputValues.length === 0) { gl.deleteBuffer(quad); return { value: input.identity, ms: performanceNow() - start }; }

      // The single in-flight input texture this driver owns (the initial upload, then each pass's prior
      // output). Tracked in an OUTER binding so the `finally` frees it if an exception interrupts the fold
      // mid-pass — a `try`-scoped `cur` would be invisible there. Set to null the instant the loop frees the
      // consumed input, so the happy path never double-frees. The `uploadInput` + the loop's per-pass allocs
      // all live INSIDE the try, so their throws (an oversized initial input, an oversized intermediate, a
      // framebuffer-incomplete) are covered by the finally that frees the quad + program + this live texture.
      let liveTex: WebGLTexture | null = null;
      try {
        let cur = uploadInput(input.inputValues, input.inputValues.length);
        liveTex = cur.tex;
        // Fold in passes until ONE partial remains. At least one pass always runs (even for a single-element
        // input, so the result is `reduce(identity, x[0])`, not the raw x[0]), which also converts the R32F
        // input into an RGBA32F partial texture → the final readback is always from an RGBA32F attachment.
        for (;;) {
          const outLen = Math.ceil(cur.len / tile);       // this pass emits ceil(curLen/TILE) partials
          const outW = texWidth(outLen);
          const outH = Math.ceil(outLen / outW);
          if (outW > maxTex || outH > maxTex) throw new Error(`MLGPU-ALLOC: reduce output texture ${outW}x${outH} exceeds the device max texture size ${maxTex}`);
          const outTex = gl.createTexture()!;
          gl.bindTexture(gl.TEXTURE_2D, outTex);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, outW, outH, 0, gl.RGBA, gl.FLOAT, null);
          const fbo = gl.createFramebuffer();
          gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
          gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outTex, 0);
          if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) { gl.deleteTexture(outTex); gl.deleteFramebuffer(fbo); throw new Error('WebGL2 framebuffer incomplete (RGBA32F reduce target unsupported)'); }

          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, cur.tex);
          gl.uniform1i(inLoc, 0);
          gl.uniform1i(inLenLoc, cur.len);
          gl.uniform1i(inTexWLoc, cur.texW);
          gl.uniform1i(outTexWLoc, outW);
          gl.viewport(0, 0, outW, outH);
          gl.drawArrays(gl.TRIANGLES, 0, 6);

          // Free the just-consumed input texture (THIS driver owns every texture it creates — the initial
          // upload + every intermediate). Clear `liveTex` in the same breath so the finally can't re-delete it.
          // The output texture becomes the next pass's input (and the next `liveTex`).
          gl.deleteTexture(cur.tex);
          liveTex = null;
          if (outLen === 1) {
            // The single surviving partial texel's .r is the scalar. Read it, then free the fbo + texture.
            const px = new Float32Array(4);
            gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
            gl.deleteFramebuffer(fbo);
            gl.deleteTexture(outTex);
            return { value: px[0]!, ms: performanceNow() - start };
          }
          gl.deleteFramebuffer(fbo);
          cur = { tex: outTex, texW: outW, len: outLen };
          liveTex = cur.tex;
        }
      } finally {
        // The quad is always this dispatch's (freed exactly once here); the program is owned by
        // `programCache` (reused across passes/dispatches, freed on dispose). `liveTex` is the in-flight
        // input texture only when an exception interrupted a pass before the loop freed it — the happy path
        // nulled it, so this is a no-op on success (no double-free).
        if (liveTex) gl.deleteTexture(liveTex);
        gl.deleteBuffer(quad);
      }
    },
    // Explicitly drop the GL context (textures/FBOs are already freed per dispatch). The LRU eviction +
    // the [Symbol.dispose]() path re-acquire a fresh context per engine, so relying on GC would let live
    // WebGL2 contexts accumulate under sustained reactive re-dispatch (browsers cap concurrent contexts).
    // Parity with the WebGPU backend's device.destroy().
    [Symbol.dispose]() { programCache[Symbol.dispose](); gl.getExtension('WEBGL_lose_context')?.loseContext(); },
  };
}

// A pass-through vertex shader (the emitter produces only the fragment shader); it forwards the clip-space
// quad so gl_FragCoord spans the output grid.
const VERT = `#version 300 es
in vec2 _pos;
void main() { gl_Position = vec4(_pos, 0.0, 1.0); }`;

function buildProgram(gl: WebGL2RenderingContext, fragSrc: string): WebGLProgram {
  const vs = compile(gl, gl.VERTEX_SHADER, VERT);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fragSrc);
  const program = gl.createProgram()!;
  gl.attachShader(program, vs); gl.attachShader(program, fs); gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) { const log = gl.getProgramInfoLog(program); gl.deleteProgram(program); throw new Error(`WebGL2 link failed: ${log}`); }
  gl.deleteShader(vs); gl.deleteShader(fs);
  return program;
}
function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src); gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) { const log = gl.getShaderInfoLog(sh); gl.deleteShader(sh); throw new Error(`WebGL2 shader compile failed: ${log}`); }
  return sh;
}
function performanceNow(): number { return globalThis.performance?.now?.() ?? 0; }
