// Tiny host-DOM helpers for the tool shell + landing chrome (NOT the previews — those are real vdom mounts).
// Intentionally minimal: a create-element helper + a clear helper. No framework; the shell is imperative
// host TS because a sandboxed eval-free language cannot own the compiler/clipboard/URL wiring.
export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  children: (Node | string)[] = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

export function clear(node: Element): void {
  node.textContent = '';
}
