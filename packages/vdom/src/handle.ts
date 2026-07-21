// The shared handle contract for both mount drivers. mount() (the DSL path) and render() (the API-first
// path) return handles that differ only in their reactive-write lever (updateData vs setState); everything
// else — the retained-tree view, diagnostics, handler firing, teardown, and the test-only probes — is
// identical, so it lives here once and each driver's handle extends it with its own lever.
import type { Diagnostic } from '@metael/lang';
import type { VNode } from './vnode.ts';

/** The members common to {@link VDomHandle} (mount) and {@link RenderHandle} (render): an inspectable view
 *  of the retained tree, the collected diagnostics, handler firing, teardown, and the two test-only probes.
 *  Each driver extends this with its own reactive-write lever (mount's `updateData`, render's `setState`). */
export interface VDomHandleBase {
  /** The retained vnode tree, with a top-level component fragment unwrapped to its first real element, or
   *  `null` when nothing was produced. */
  tree(): VNode | null;
  /** Diagnostics collected across the mount's lifetime (e.g. an `ML-VDOM-CONVERGE` when a reactive flush
   *  fails to settle). */
  diagnostics: Diagnostic[];
  /** Fire a captured handler by node key + event, inside the runtime `change()` boundary. The reactive
   *  graph then decides the path: a value-only write fires only the leaf effect; a structural write
   *  re-derives. */
  invokeHandler(nodeKey: string, event: string, arg: unknown): void;
  /** Tear down the mount: stop the tracked pass, detach delegation, dispose leaf effects, clear the
   *  container, and drop the element index + handler registry. */
  unmount(): void;
  /** Test-only: does a handler exist for this node+event? (guards vacuous {@link VDomHandleBase.invokeHandler}
   *  no-ops). */
  hasHandler(nodeKey: string, event: string): boolean;
  /** Test-only: how many times the tracked structural pass ran (a fresh derive/producer + build/reconcile).
   *  A value-only change (a leaf effect) must NOT increment this — the direct proof the fine-grained path is
   *  real, not cosmetic; a structural change increments it once per pass. */
  passCount(): number;
}
