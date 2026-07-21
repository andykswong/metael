// @metael/gpu/lang — the metael-DSL binding for compute: author a kernel as a metael source string
// (compileKernel) + the head vocabulary (GpuHostEnv resolves gpu/gpuReduce/gpuHistogram) for driving
// compute from a metael program. This is the subpath that pulls the interpreter (evaluateProgram); the
// API-first core (@metael/gpu) does not.
export { GpuHostEnv } from './host-env.ts';
export { compileKernel } from './compile-kernel.ts';
