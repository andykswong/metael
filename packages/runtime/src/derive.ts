// The one-shot composition root. derive() runs ONE change()-wrapped lowerEntry pass — a fresh
// RuntimeReactiveHost + the caller's HostEnvironment + a KeyMinter — and returns the raw host values +
// the host + diagnostics. It does NOT materialize a domain output, build a retained tree, or expose
// updateData/setState: those are DOMAIN-owned (the domain provides the patch target + the reconcile).
// derive owns only the generic slice: settle the initial flush and surface a non-converging flush as
// the ML-RT-CONVERGE diagnostic.
import {
  lowerEntry, makeDiagnostic, type HostEnvironment, type KeyMinter, type Diagnostic, type HostValue,
} from '@metael/lang';
import { RuntimeReactiveHost } from './reactive-host.ts';
import { change, ReactiveFlushError } from './reactive.ts';

export interface DeriveOptions {
  env: HostEnvironment;
  minter: KeyMinter;
  data?: unknown;
  seed?: number;
  entry?: string;
  /** Opt `data` into reactivity so a `data.x` read lowers to a trackable Region (Proxy-free). */
  reactiveData?: boolean;
  /** Prior settled state S (cellKey → value) to latch surviving component instances on a re-derive. */
  priorState?: ReadonlyMap<string, unknown>;
  /** Invoked with the freshly-created host BEFORE the walk runs, so a domain HostEnvironment can
   *  register leaf effects for reactive props during resolveCall. The resolveCall port carries no host
   *  param by design, so the host is handed over out-of-band, exactly once, through this callback. */
  onHost?: (host: RuntimeReactiveHost) => void;
  // Budget overrides forwarded to the lang evaluator/walk (fail-closed limits).
  maxSteps?: number;
  maxTimeMs?: number;
  maxDepth?: number;
  maxStringLength?: number;
}

export interface DeriveResult {
  /** The raw host value(s) the walk produced (opaque to metael; the domain materializes them). */
  value: HostValue;
  diagnostics: Diagnostic[];
  /** The reactive host that owns this derive's cells + leaf effects (for setState/exportState/teardown). */
  host: RuntimeReactiveHost;
}

export function derive(source: string, opts: DeriveOptions): DeriveResult {
  const host = new RuntimeReactiveHost(opts.priorState);
  opts.onHost?.(host);
  const diagnostics: Diagnostic[] = [];
  let value: HostValue = null;
  try {
    change(() => {
      // Build the walk options. `data` is forwarded ONLY when the caller supplied it: lowerEntry gates
      // the `data` binding on `'data' in opts` (an explicit `data: undefined` would still bind `data`),
      // so match that contract — omit the key entirely when opts.data is absent.
      const walkOpts: Parameters<typeof lowerEntry>[1] = {
        seed: opts.seed ?? 0, entry: opts.entry ?? 'Story',
        reactiveData: opts.reactiveData,
        host, env: opts.env, minter: opts.minter,
        maxSteps: opts.maxSteps, maxTimeMs: opts.maxTimeMs, maxDepth: opts.maxDepth, maxStringLength: opts.maxStringLength,
      };
      if ('data' in opts) walkOpts.data = opts.data;
      const res = lowerEntry(source, walkOpts);
      for (const d of res.diagnostics) diagnostics.push(d);
      value = res.value;
    });
  } catch (e) {
    if (e instanceof ReactiveFlushError) {
      diagnostics.push(makeDiagnostic('ML-RT-CONVERGE', 'reactive flush did not converge (cross-effect feedback past the cap)'));
    } else {
      throw e;   // a non-reactive throw is a genuine bug — do not swallow
    }
  }
  return { value, diagnostics, host };
}
