// packages/vdom/src/lang/render-source.ts
// The DSL front door: derive a metael source string into a live vDOM mount and keep it reactive. It is the
// render() core loop (the tracked pass, reconcile, delegation, per-pass + reconcile-removal disposal) driven
// by a PRODUCER built from compileToProducer. renderSource owns ONLY the DSL-specific cross-pass state — the
// host latch (priorState across passes), the handler-registry swap, a data signal (the updateData lever) —
// and the env capture for explicit disposal.
//
// DISPOSAL has two explicit owners (no reliance on whole-tree GC): the render core owns per-pass leaf-effect
// teardown + reconcile-removal teardown (its onRemove → disposeLeaf, a no-op here since these nodes' leaf
// effects live on the derive host, not the bind.ts WeakMap); this driver owns dropping the prior pass's
// derive host each pass (GC) and disposing the most recent pass's env on unmount (a gpu-backed env frees its
// WebGPU device via [Symbol.dispose]).
import { render, type RenderCoreHooks } from '../render.ts';
import { type VDomHandleBase } from '../handle.ts';
import { compileToProducer, type CompileOptions } from './compile.ts';
import { type VNode } from '../vnode.ts';
import { signal, change, type RuntimeReactiveHost, type Signal } from '@metael/runtime';
import { type Diagnostic, type BindableHostEnv } from '@metael/lang';
import { VDomHostEnv } from './host-env.ts';

/** Options for {@link renderSource}: the same DSL/derive inputs as {@link CompileOptions} (a named alias +
 *  doc anchor at the front door, and the seam for any future DSL-only options). */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type -- intentional named front-door alias for CompileOptions.
export interface RenderSourceOptions extends CompileOptions {}

/** The handle {@link renderSource} returns: the shared {@link VDomHandleBase} plus the DSL path's own
 *  reactive-write lever, `updateData` (a data-input push that re-derives). */
export interface VDomHandle extends VDomHandleBase {
  /** Re-derive with new data (only meaningful under `reactiveData`; a no-op otherwise). */
  updateData(next: unknown): void;
}

/** Derive a metael DSL `source` into a live vDOM mount inside `container` and keep it running. This is the
 *  DSL front door: it is {@link render} (the API-first core loop) driven by a producer built from
 *  {@link compileToProducer}. The core owns the tracked pass, reconcile, delegation, and disposal;
 *  `renderSource` adds only the DSL-specific cross-pass state (host latch, data signal) + `updateData`.
 *  Pass `undefined` for `container` to run headless (tree-only, no DOM). */
export function renderSource(source: string, container: Element | undefined, opts: RenderSourceOptions = {}): VDomHandle {
  const diagnostics: Diagnostic[] = [];
  const dataSignal: Signal<unknown> = signal(opts.data);
  let currentHost: RuntimeReactiveHost | null = null;
  let passHandlers = new Map<string, (arg: unknown) => void>();
  let lastEnv: BindableHostEnv | null = null;
  let firstPass = true;

  // Wrap the env factory so we capture each pass's env — unmount disposes the most recent one (a gpu-backed
  // env frees its WebGPU device on [Symbol.dispose]).
  const envFactory = (): BindableHostEnv => {
    const e = opts.envFactory ? opts.envFactory() : new VDomHostEnv();
    lastEnv = e;
    return e;
  };
  const { produce } = compileToProducer(source, { ...opts, envFactory, dataSignal });

  const producer = (): VNode[] => {
    const priorState = currentHost?.exportState();
    const pass = produce(priorState);
    currentHost = pass.host;
    passHandlers = pass.handlers;
    // Collect the derive + materialize diagnostics on the FIRST pass only — a re-derive re-produces the same
    // diagnostics, so appending every pass would duplicate them.
    if (firstPass) { for (const d of pass.diagnostics) diagnostics.push(d); firstPass = false; }
    return pass.nodes;
  };

  // preKeyed: the derive walk's minter already keyed these nodes (and the handler registry is keyed to match)
  // → skip the core's re-keying. onPassHandlers: this pass's handlers were captured during materialize and
  // its leaf effects were bound on the derive host → the core swaps in these handlers, never bindReactive.
  // diagnostics: shared so the handle carries BOTH the compile diagnostics AND the core's ML-VDOM-CONVERGE.
  const hooks: RenderCoreHooks = { preKeyed: true, onPassHandlers: () => passHandlers, diagnostics };
  const base = render(producer, container, hooks);

  return {
    ...base,
    updateData: (next: unknown) => {
      // Under reactiveData, push new data through the signal the producer read → the render core's tracked
      // pass re-runs (a fresh derive). When reactiveData is off, data is fixed at mount — a no-op.
      if (opts.reactiveData) change(() => dataSignal.set(next));
    },
    unmount: () => {
      base.unmount();
      // Dispose the most recent pass's env AFTER the core torn down (a gpu-backed env frees its device); a
      // plain VDomHostEnv is not Disposable, so the optional chain skips it.
      (lastEnv as Partial<Disposable> | null)?.[Symbol.dispose]?.();
    },
  };
}
