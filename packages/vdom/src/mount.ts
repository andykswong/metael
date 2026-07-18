import {
  derive, change, effect, ReactiveFlushError, PathKeyMinter, type RuntimeReactiveHost, type Diagnostic,
} from '@metael/runtime';
import { makeDiagnostic, type HostEnvironment, type ReactiveHost } from '@metael/lang';   // makeDiagnostic lives in lang, not runtime
import { VDomHostEnv } from './host-env.ts';
import { materialize } from './materialize.ts';
import { createDom } from './patch.ts';
import { reconcile, flattenFragments, type ReconcileHooks } from './reconcile.ts';
import { attachDelegation } from './delegate.ts';
import { FRAGMENT, type VNode } from './vnode.ts';

export interface MountOptions {
  data?: unknown;
  seed?: number;
  entry?: string;
  reactiveData?: boolean;
  maxSteps?: number;
  maxTimeMs?: number;
  maxDepth?: number;
  maxStringLength?: number;
  /** Supply a custom host-bindable environment instead of the default VDomHostEnv. A caller that needs
   *  ADDITIONAL heads beyond the display vocabulary (e.g. a composite that resolves an extra head, else
   *  delegates to a VDomHostEnv) provides a factory here. It is called per pass (each pass runs on a fresh
   *  host, then bindHost hands that host to the env before the walk); the returned env must expose
   *  bindHost. Default (omitted) → a new VDomHostEnv per pass, exactly as before (backward-compatible). */
  envFactory?: () => HostEnvironment & { bindHost(host: ReactiveHost): void };
}

export interface VDomHandle {
  /** The retained vnode tree, with a top-level component fragment unwrapped to its first real element. */
  tree(): VNode | null;
  diagnostics: Diagnostic[];
  /** Fire a captured handler by node key + event, inside the runtime change() boundary. The reactive graph
   *  then decides the path: a value-only write fires only the leaf effect; a structural write re-derives. */
  invokeHandler(nodeKey: string, event: string, arg: unknown): void;
  /** Re-derive with new data (only meaningful under reactiveData). */
  updateData(next: unknown): void;
  unmount(): void;
  /** Test-only: does a handler exist for this node+event? (guards vacuous invokeHandler no-ops). */
  hasHandler(nodeKey: string, event: string): boolean;
  /** Test-only: how many times the tracked walk-effect has run (a fresh derive + build/reconcile). A
   *  value-only change (a leaf effect) must NOT increment this — the direct proof the fine-grained path
   *  is real, not cosmetic; a structural change increments it once per re-derive. */
  passCount(): number;
}

export function mount(source: string, container: Element | undefined, opts: MountOptions): VDomHandle {
  const minter = new PathKeyMinter();
  const diagnostics: Diagnostic[] = [];
  const index = new Map<string, Element>();
  const liveRegistry = new Map<string, (arg: unknown) => void>();   // read live by delegation; swapped per pass

  let currentRoot: VNode | null = null;   // the retained tree (patched in place on reconcile)
  let currentHost: RuntimeReactiveHost | null = null;
  const currentData = opts.data;
  let built = false;
  let passes = 0;   // how many times the tracked walk-effect ran (structural passes) — instrumentation
  let stopWalkEffect: (() => void) | null = null;
  let detach: (() => void) | null = null;
  let liveData = opts.data;
  let lastEnv: unknown = null;   // the most recent pass's env — disposed on unmount if it exposes [Symbol.dispose]()

  const hooks: ReconcileHooks = { onRemove: () => {} };   // fresh-host-per-pass drops prior effects via GC

  /** One derive + materialize pass on a FRESH host (re-seeded identically → deterministic; the prior pass's
   *  host is dropped, its leaf effects GC'd — node identity is preserved by the keyed DOM reconcile, not
   *  host reuse). State latches via priorState so a surviving component instance keeps its mutated state. */
  const runPass = (data: unknown, collectDiag: boolean, priorState?: ReadonlyMap<string, unknown>): { root: VNode | null; host: RuntimeReactiveHost } => {
    const env = opts.envFactory ? opts.envFactory() : new VDomHostEnv();
    lastEnv = env;   // retained so unmount() can call [Symbol.dispose]() on a disposable env (e.g. a gpu engine → WebGPU device)
    const handlers = new Map<string, (arg: unknown) => void>();
    const res = derive(source, {
      env, minter, data, seed: opts.seed, entry: opts.entry, reactiveData: opts.reactiveData, priorState,
      maxSteps: opts.maxSteps, maxTimeMs: opts.maxTimeMs, maxDepth: opts.maxDepth, maxStringLength: opts.maxStringLength,
      onHost: (h) => { env.bindHost(h); },   // bind the host into the env before the walk runs resolveCall
    });
    if (collectDiag) for (const d of res.diagnostics) diagnostics.push(d);
    const root = materialize(res.value, collectDiag ? diagnostics : [], handlers);
    // Swap the live delegation registry to this pass's handlers (no listener re-attach needed).
    liveRegistry.clear();
    for (const [k, v] of handlers) liveRegistry.set(k, v);
    return { root, host: res.host };
  };

  /** Unwrap a top-level 'component' fragment (the entry Story) so tree()/build/reconcile operate on the
   *  first real element. A fragment root has no DOM node of its own. */
  const rootChildren = (root: VNode | null): VNode[] => (root ? flattenFragments([root]) : []);

  // THE tracked effect: its body reads the structural cells (via the walk's eager for/if reads), so a
  // structural write re-runs it; a value-only write is handled by a nested leaf effect and does NOT re-run
  // it. First run builds the DOM; later runs reconcile the retained tree in place.
  const onStructuralPass = (): void => {
    passes++;
    const priorState = currentHost?.exportState();
    const { root, host } = runPass(liveData, !built, priorState);
    const prevRoot = currentRoot;
    currentHost = host;
    if (!container) { currentRoot = root; return; }   // headless: no DOM; the fresh tree IS the current tree
    const doc = container.ownerDocument!;
    if (!built) {
      // First build creates DOM for EVERY node (each registers its element/text node), so the fresh tree
      // doubles as the retained tree for the next diff.
      for (const c of rootChildren(root)) container.appendChild(createDom(c, doc, index));
      built = true;
      currentRoot = root;
    } else {
      // reconcile returns the RETAINED top-level children: matched instances (reused in place, carrying
      // their already-registered DOM element/text nodes) + newly-created ones (registered during the
      // reconcile). Keep THESE as the tree for the next diff — NOT the fresh `root`, whose matched-node
      // vnodes never had DOM registered, so a later leaf/text patch on them would be a silent DOM no-op
      // (the retained instance is the one a leaf effect + the next reconcile must patch). Wrap them in a
      // fragment mirroring the entry root so tree()/rootChildren behave unchanged.
      const retained = reconcile(container, rootChildren(prevRoot), rootChildren(root), doc, index, hooks);
      currentRoot = { tag: FRAGMENT, props: {}, children: retained, key: root?.key ?? '' };
    }
  };

  // Wrap the pass in a tracked effect. The effect's reads (the walk's structural cell reads) subscribe it;
  // a structural write schedules a re-run through change()'s batch (a fresh pass + reconcile).
  stopWalkEffect = effect(() => { runInChange(onStructuralPass); });

  if (container) detach = attachDelegation(container, liveRegistry, (fn, ev) => runInChange(() => fn(eventArg(ev))));

  function runInChange(fn: () => void): void {
    try { change(fn); }
    catch (e) { if (e instanceof ReactiveFlushError) diagnostics.push(makeDiagnostic('ML-VDOM-CONVERGE', 'reactive flush did not converge')); else throw e; }
  }

  function eventArg(ev: Event): unknown {
    const t = ev.target as HTMLInputElement | null;
    return { value: t?.value, key: (ev as KeyboardEvent).key };
  }

  void currentData;
  return {
    tree: () => { const c = rootChildren(currentRoot); return currentRoot?.tag === FRAGMENT ? (c[0] ?? null) : currentRoot; },
    diagnostics,
    invokeHandler: (nodeKey, event, arg) => { const fn = liveRegistry.get(`${nodeKey}:${event}`); if (fn) runInChange(() => fn(arg)); },
    updateData: (next) => { liveData = next; if (opts.reactiveData && currentHost) runInChange(onStructuralPass); },
    unmount: () => { stopWalkEffect?.(); detach?.(); (lastEnv as Partial<Disposable> | null)?.[Symbol.dispose]?.(); if (container) container.textContent = ''; index.clear(); liveRegistry.clear(); },
    hasHandler: (nodeKey, event) => liveRegistry.has(`${nodeKey}:${event}`),
    passCount: () => passes,
  };
}
