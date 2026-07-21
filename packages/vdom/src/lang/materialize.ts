import { isWrapper, type LangWrapper, type HostValue } from '@metael/runtime';
import { makeDiagnostic, type Diagnostic } from '@metael/lang';   // makeDiagnostic lives in lang, not runtime
import { FRAGMENT, isVNode, type VNode } from '../vnode.ts';

/** Convert the raw lowered tree (VNodes + LangWrappers) into a retained VNode tree, recording each captured
 *  handler keyed `${nodeKey}:${event}`. A 'component' wrapper (a declined in-DSL component) becomes a
 *  transparent FRAGMENT — its children splice into the parent (a component is not a DOM node). An 'unknown'
 *  wrapper (an unregistered head) is dropped with a diagnostic. */
export function materialize(
  value: HostValue,
  diagnostics: Diagnostic[],
  handlers: Map<string, (arg: unknown) => void>,
): VNode | null {
  if (value === null || value === undefined || typeof value !== 'object') return null;

  if (isWrapper(value)) {
    const w = value as LangWrapper;
    const children = materializeChildren(w.children, diagnostics, handlers);
    if (w.__mlWrap === 'unknown') {
      diagnostics.push(makeDiagnostic('ML-VDOM-UNKNOWN', `unknown element or component '${w.head}'`));
      return null;
    }
    return { tag: FRAGMENT, props: {}, children, key: w.key };   // 'component' → transparent fragment
  }

  if (isVNode(value)) {
    const node = value;
    captureHandlers(node, handlers);
    node.children = materializeChildren(node.children, diagnostics, handlers);
    return node;
  }
  return null;
}

function materializeChildren(raw: HostValue[], diagnostics: Diagnostic[], handlers: Map<string, (arg: unknown) => void>): VNode[] {
  const out: VNode[] = [];
  for (const c of raw) { const n = materialize(c, diagnostics, handlers); if (n) out.push(n); }
  return out;
}

function captureHandlers(node: VNode, handlers: Map<string, (arg: unknown) => void>): void {
  if (!node.handlers) return;
  for (const h of node.handlers) handlers.set(`${node.key}:${h.event}`, h.fn);
}
