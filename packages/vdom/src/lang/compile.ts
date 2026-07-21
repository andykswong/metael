import { derive, PathKeyMinter, type RuntimeReactiveHost, type Signal } from '@metael/runtime';   // PathKeyMinter is re-exported by runtime
import { type Diagnostic, type BindableHostEnv, type BuiltinModule } from '@metael/lang';
import { VDomHostEnv } from './host-env.ts';
import { materialize } from './materialize.ts';
import { flattenFragments } from '../reconcile.ts';
import { type VNode } from '../vnode.ts';

/** Options for {@link compileToProducer}: the derive inputs (data/seed/entry/reactiveData), the walk
 *  budgets, injected builtin modules, and an optional custom env factory (defaults to a fresh VDomHostEnv
 *  per pass). Mirrors the DSL-relevant subset of the mount options. */
export interface CompileOptions {
  /** Data made available to the program as its `data` binding (deep-frozen at the boundary). */
  data?: unknown;
  /** When set, each pass reads `data` through this signal so a `.set()` re-runs the render core's tracked
   *  pass (the reactive-data lever); falls back to {@link CompileOptions.data} when absent. */
  dataSignal?: Signal<unknown>;
  /** Seed for the intrinsic `rand()`/`range()` PRNG; each pass re-seeds identically for determinism. */
  seed?: number;
  /** Name of the entry component to instantiate as the root; omitted uses the program's default entry. */
  entry?: string;
  /** When `true`, the render core may re-derive with fresh data; forwarded to derive unchanged. */
  reactiveData?: boolean;
  /** Fuel budget (max evaluation steps) forwarded to the walk before it fails closed. */
  maxSteps?: number;
  /** Deadline budget in milliseconds forwarded to the walk. */
  maxTimeMs?: number;
  /** Recursion-depth budget forwarded to the walk (the source is attacker-influenceable, so depth is bounded). */
  maxDepth?: number;
  /** Cap on the length a string may grow to via `+`, forwarded to the walk. */
  maxStringLength?: number;
  /** Builtin modules the walk resolves unbound call heads against (e.g. a numeric library). */
  builtins?: readonly BuiltinModule[];
  /** Supply a custom host-bindable environment instead of the default VDomHostEnv. Called once per pass;
   *  the returned env must expose bindHost. Omitted → a new VDomHostEnv per pass. */
  envFactory?: () => BindableHostEnv;
}

/** One derived pass: the top-level VNodes (fragments flattened), the fresh host that owns the pass's cells
 *  + leaf effects, and the pass diagnostics. */
export interface CompiledPass {
  /** The top-level VNodes for this pass, with the entry-component fragment flattened away. */
  nodes: VNode[];
  /** The reactive host that owns this pass's cells + leaf effects (for state latch/export/teardown). */
  host: RuntimeReactiveHost;
  /** The diagnostics collected during this pass (derive + materialize). */
  diagnostics: Diagnostic[];
  /** The per-pass handler registry (nodeKey:event → fn), captured during materialize. */
  handlers: Map<string, (arg: unknown) => void>;
}

/** Compile a metael DSL `source` into a PRODUCER the render core can drive: each `produce(priorState?)`
 *  call runs ONE derive+materialize pass on a FRESH reactive host (re-seeded identically → deterministic;
 *  `priorState` latches surviving component instances) and returns the top-level VNodes + that host + the
 *  diagnostics. It does NOT own the reactive loop, the DOM, or reconciliation — the render core does; this
 *  is the pure "source → VNode tree" step, independently testable without a DOM. */
export function compileToProducer(source: string, opts: CompileOptions): {
  /** Run one derive+materialize pass on a fresh host, optionally latching `priorState`. */
  produce: (priorState?: ReadonlyMap<string, unknown>) => CompiledPass;
} {
  // derive() REQUIRES a KeyMinter (DeriveOptions.minter is non-optional). Construct ONE PathKeyMinter and
  // reuse it across every produce() pass so reconciliation keys are stable across re-derives.
  const minter = new PathKeyMinter();
  const produce = (priorState?: ReadonlyMap<string, unknown>): CompiledPass => {
    const env = opts.envFactory ? opts.envFactory() : new VDomHostEnv();
    const handlers = new Map<string, (arg: unknown) => void>();
    const diagnostics: Diagnostic[] = [];
    // Read data through the signal (if supplied) INSIDE produce, so the read subscribes the render core's
    // tracked pass — a later .set() then re-runs the pass. Fall back to the static opts.data otherwise.
    const data = opts.dataSignal ? opts.dataSignal.get() : opts.data;
    const res = derive(source, {
      env, minter,
      data, seed: opts.seed, entry: opts.entry, reactiveData: opts.reactiveData, priorState,
      maxSteps: opts.maxSteps, maxTimeMs: opts.maxTimeMs, maxDepth: opts.maxDepth, maxStringLength: opts.maxStringLength,
      builtins: opts.builtins,
      onHost: (h) => { env.bindHost(h); },
    });
    for (const d of res.diagnostics) diagnostics.push(d);
    const root = materialize(res.value, diagnostics, handlers);
    const nodes = root ? flattenFragments([root]) : [];
    return { nodes, host: res.host, diagnostics, handlers };
  };
  return { produce };
}
