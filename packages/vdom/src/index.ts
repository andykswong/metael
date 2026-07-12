// @metael/vdom public barrel. The library surface: mount() + the handle + the VNode type + the sanitizer.
// Examples + the demo harness are NOT exported (a library stays app-free — the showcase apps import the
// examples from source, not the published surface).
export { mount } from './mount.ts';
export type { VDomHandle, MountOptions } from './mount.ts';
export { isVNode, textVNode, FRAGMENT, TEXT } from './vnode.ts';
export type { VNode, Handler } from './vnode.ts';
export { escapeText, safeAttrName, safeAttrValue } from './sanitize.ts';
