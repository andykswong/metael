// @metael/vdom/lang — the metael-DSL binding: derive a source string into a live vDOM. Built on the
// API-first core (@metael/vdom) — renderSource is `render` driven by compileToProducer.
export { renderSource } from './render-source.ts';
export type { VDomHandle, RenderSourceOptions } from './render-source.ts';
export { compileToProducer } from './compile.ts';
export type { CompileOptions, CompiledPass } from './compile.ts';
export { VDomHostEnv } from './host-env.ts';
export { materialize } from './materialize.ts';
export { vdomProfile } from './profile.ts';
