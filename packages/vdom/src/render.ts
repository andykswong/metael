// packages/vdom/src/render.ts
// The API-first render driver: a mount()-shaped loop whose PRODUCER is a host callback returning a VNode
// tree (built with h()), not a metael source string. Same two-tier reactivity as mount: THE tracked pass
// re-runs on a structural signal write (a signal read in the producer body), while a value-only write fires
// only a leaf effect (bound by bindReactive) and patches one DOM node with no re-derive. Unlike mount, there
// is no fresh-host-per-pass GC, so leaf effects are disposed MANUALLY: the whole prior pass on re-derive,
// and a removed subtree via onRemove.
//
// DEPTH: the tree walks (assignKeys / bindReactive / disposeLeaf / createDom / reconcile) recurse once per
// tree level with no depth cap — the tree is HOST-AUTHORED TypeScript, so its depth is the caller's to
// bound. A host that maps UNTRUSTED deeply-nested data (e.g. an arbitrary comment/reply tree) to nested
// h() calls must cap the nesting itself; an unbounded depth overflows the JS stack. (This differs from the
// language mount() path, which threads a maxDepth budget because it walks attacker-influenceable source.)
import { effect, change, ReactiveFlushError } from '@metael/runtime';
import { makeDiagnostic, type Diagnostic } from '@metael/lang';
import { createDom } from './patch.ts';
import { reconcile, flattenFragments, type ReconcileHooks } from './reconcile.ts';
import { attachDelegation } from './delegate.ts';
import { assignKeys } from './keying.ts';
import { bindReactive, disposeLeaf } from './bind.ts';
import { normalizeNodes, type RenderNode } from './normalize.ts';
import { type VNode } from './vnode.ts';
import { type VDomHandleBase } from './handle.ts';

export type { RenderNode } from './normalize.ts';
/** The host callback render() drives: returns one node or a list, with conditional holes allowed. */
export type RenderProducer = () => RenderNode | RenderNode[];

/** The handle {@link render} returns: the shared {@link VDomHandleBase} (retained-tree view, diagnostics,
 *  handler firing, teardown, test-only probes) plus render's own reactive-write lever, `setState` — run a
 *  fn inside the render's `change()` boundary (vs mount's `updateData`, which pushes new data + re-derives). */
export interface RenderHandle extends VDomHandleBase {
  /** Run `fn` inside the render's change() boundary (drive a reactive write like a handler would). */
  setState(fn: () => void): void;
}

/** Optional overrides for {@link render}'s per-pass tree handling, for a caller whose producer emits nodes
 *  that are ALREADY keyed and whose leaf effects + handler registry were built upstream (the DSL front door).
 *  All fields are additive: a plain `render(producer, container)` with no hooks behaves exactly as before. */
export interface RenderCoreHooks {
  /** The nodes are ALREADY keyed (by an upstream minter) — skip the core's `assignKeys` re-keying. A caller
   *  whose handler registry is keyed by those same upstream keys sets this, so re-keying cannot desync the
   *  registry from the element `data-key` attributes. */
  preKeyed?: boolean;
  /** Supply the handler registry for THIS pass instead of deriving it from the tree via `bindReactive` (a
   *  caller that captured handlers upstream). When set, the core does NOT run `bindReactive` (that caller's
   *  leaf effects were bound upstream); it only swaps in these handlers. */
  onPassHandlers?: (nodes: VNode[]) => Map<string, (arg: unknown) => void>;
  /** Use THIS diagnostics array instead of a fresh one, so a caller that also pushes its own per-pass
   *  diagnostics shares ONE array with the core's convergence diagnostics (`ML-VDOM-CONVERGE`). */
  diagnostics?: Diagnostic[];
}

/** Mount a host-authored producer into `container` and keep it live. `producer` returns the {@link VNode}
 *  tree (built with {@link h}); its top-level signal reads subscribe the tracked pass, so a structural write
 *  re-runs it and reconciles the retained tree, while a value-only write fires only a leaf effect and patches
 *  one DOM node. Pass `undefined` for `container` to run headless (tree-only, no DOM). Returns a
 *  {@link RenderHandle}. */
export function render(producer: RenderProducer, container: Element | undefined, hooks?: RenderCoreHooks): RenderHandle {
  const diagnostics = hooks?.diagnostics ?? [];
  const index = new Map<string, Element>();
  const liveRegistry = new Map<string, (arg: unknown) => void>();
  let passDisposers: Array<() => void> = [];   // leaf-effect disposers for the CURRENT pass
  let currentRoot: VNode[] = [];
  let built = false;
  let passes = 0;
  let stopPass: (() => void) | null = null;
  let detach: (() => void) | null = null;

  // onRemove disposes a removed subtree's leaf effects (manual teardown — no GC to rely on).
  const reconcileHooks: ReconcileHooks = { onRemove: (v) => disposeLeaf(v) };

  const runProducer = (): VNode[] => {
    // Dispose the prior pass's leaf effects (a fresh pass re-binds them) BEFORE re-running the producer.
    for (const d of passDisposers) d();
    passDisposers = [];
    const raw = producer();
    // Coerce to an array and drop conditional holes — null/undefined/false/true — exactly as h() does for
    // children, so `() => null` (an empty root) and `() => [cond && node, ...]` (a top-level JSX-conditional)
    // are tolerated rather than crashing assignKeys on a non-object. tree() then returns null for an empty
    // result, matching the DSL mount() path.
    const tree = normalizeNodes(raw);
    // Skip re-keying for a pre-keyed producer: its nodes already carry an upstream minter's keys that its
    // handler registry is keyed by, so assignKeys would overwrite them and desync the registry.
    if (!hooks?.preKeyed) assignKeys(tree, '');
    // Rebuild the handler registry fresh each pass (mirrors mount()): otherwise a handler removed from — or
    // absent on — a surviving element leaves a stale entry that still fires on click. bindReactive only ever
    // set()s, never deletes; the synchronous repopulate below keeps delegation correct (events fire only
    // outside this producer).
    liveRegistry.clear();
    if (hooks?.onPassHandlers) {
      // A pre-built registry: this producer's leaf effects were bound upstream (not by bindReactive), so
      // swap in its handlers and do NOT bindReactive (which would double-bind / re-key).
      const passH = hooks.onPassHandlers(tree);
      for (const [k, v] of passH) liveRegistry.set(k, v);
    } else {
      bindReactive(tree, passDisposers, liveRegistry);
    }
    return tree;
  };

  const onPass = (): void => {
    passes++;
    const next = runProducer();
    if (!container) { currentRoot = next; return; }   // headless
    const doc = container.ownerDocument!;
    if (!built) {
      for (const c of flattenFragments(next)) container.appendChild(createDom(c, doc, index));
      built = true;
      currentRoot = next;
    } else {
      currentRoot = reconcile(container, currentRoot, next, doc, index, reconcileHooks);
    }
  };

  function runInChange(fn: () => void): void {
    try { change(fn); }
    catch (e) { if (e instanceof ReactiveFlushError) diagnostics.push(makeDiagnostic('ML-VDOM-CONVERGE', 'reactive flush did not converge')); else throw e; }
  }

  // THE tracked pass: its reads (the producer's top-level signal reads) subscribe it; a structural write
  // re-runs it through change()'s batch. Leaf effects bound inside bindReactive handle value-only writes.
  stopPass = effect(() => { runInChange(onPass); });

  if (container) detach = attachDelegation(container, liveRegistry, (fn, ev) => runInChange(() => fn(eventArg(ev))));

  function eventArg(ev: Event): unknown {
    const t = ev.target as HTMLInputElement | null;
    return { value: t?.value, key: (ev as KeyboardEvent).key };
  }

  const rootNode = (): VNode | null => {
    const flat = flattenFragments(currentRoot);
    return flat[0] ?? null;
  };

  return {
    tree: rootNode,
    diagnostics,
    setState: (fn) => runInChange(fn),
    invokeHandler: (nodeKey, event, arg) => { const fn = liveRegistry.get(`${nodeKey}:${event}`); if (fn) runInChange(() => fn(arg)); },
    hasHandler: (nodeKey, event) => liveRegistry.has(`${nodeKey}:${event}`),
    passCount: () => passes,
    unmount: () => {
      stopPass?.(); detach?.();
      for (const d of passDisposers) d(); passDisposers = [];
      // Also dispose any leaf effects still held on the retained tree (defensive; onRemove handled removals).
      for (const c of currentRoot) disposeLeaf(c);
      if (container) container.textContent = '';
      index.clear(); liveRegistry.clear();
    },
  };
}
