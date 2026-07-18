// The CPU backend: run the eval-free CPU emitter over every output cell. Always available; also the
// benchmark baseline (its ms feeds the GPU-vs-CPU race). No GPU needed.
import type { Backend, DispatchInput, DispatchResult } from './index.ts';
import { CPU_LIMITS } from '../cost.ts';

export function makeCpuBackend(): Backend {
  return {
    kind: 'cpu',
    limits: CPU_LIMITS,
    async dispatch(input: DispatchInput): Promise<DispatchResult> {
      const total = input.dims.reduce((a, b) => a * b, 1);
      const comps = input.outputComps ?? 1;
      // FLAT-INTERLEAVED output: cell i's component k at output[i*comps + k]. For comps=1 this is output[i].
      const output = new Float32Array(total * comps);
      const coordsOf = input.dims.length === 1 ? (i: number) => [i] : (i: number) => { const cols = input.dims[1]!; return [Math.floor(i / cols), i % cols]; };
      const start = performanceNow();
      for (let i = 0; i < total; i++) { const vals = input.cpuRun(coordsOf(i)); for (let k = 0; k < comps; k++) output[i * comps + k] = vals[k]!; }
      const ms = performanceNow() - start;
      // Trivially resident: the output Float32Array IS the buffer — portable + zero-copy, so a later stage
      // reads it straight back with a no-op dispose (nothing to free). The CPU backend never binds a
      // `residentInputs` entry: it computes via `cpuRun`, which reads each input through its descriptor.
      if (input.retainOutput) return { output, ms, resident: { gpuBuffer: output, dispose: () => { /* no-op */ } } };
      return { output, ms };
    },
    [Symbol.dispose]() { /* nothing to free */ },
  };
}
function performanceNow(): number { return (globalThis.performance?.now?.() ?? 0); }
