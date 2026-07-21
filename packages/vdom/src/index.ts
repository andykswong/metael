// @metael/vdom public barrel — the API-first core: the render loop, the hyperscript builder, the VNode
// type, and the output sanitizer. The metael-DSL binding (renderSource/VDomHostEnv/materialize) lives in
// the ./lang subpath (@metael/vdom/lang) — importing it, not this barrel, is what pulls the interpreter.
export { render } from './render.ts';
export type { RenderHandle, RenderProducer, RenderCoreHooks } from './render.ts';
export { normalizeNodes } from './normalize.ts';
export type { RenderNode } from './normalize.ts';
export type { VDomHandleBase } from './handle.ts';
export { h, Fragment } from './h.ts';
export type { Child, Props, Thunk } from './h.ts';
export { isVNode, textVNode, FRAGMENT, TEXT } from './vnode.ts';
export type { VNode, Handler } from './vnode.ts';
export { escapeText, safeAttrName, safeAttrValue } from './sanitize.ts';
