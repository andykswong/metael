// The one-shot composition root. derive() runs ONE change()-wrapped lowerEntry pass — a fresh
// RuntimeReactiveHost + the caller's HostEnvironment + a KeyMinter — and returns the raw host values +
// the host + diagnostics. It does NOT materialize a domain output, build a retained tree, or expose
// updateData/setState: those are DOMAIN-owned (the domain provides the patch target + the reconcile).
// derive owns only the generic slice: settle the initial flush and surface a non-converging flush as
// the ML-RT-CONVERGE diagnostic.
import {
  lowerEntry, makeDiagnostic, type HostEnvironment, type KeyMinter, type Diagnostic, type HostValue,
  type BuiltinModule,
} from '@metael/lang';
import { RuntimeReactiveHost } from './reactive-host.ts';
import { change, ReactiveFlushError } from './reactive.ts';

/** The inputs to {@link derive}: the host capabilities that resolve the vocabulary, the key minter for
 *  stable component/list identity, plus optional injected data, determinism seed, entry point, prior
 *  state to latch, an out-of-band host callback, and budget overrides. */
export interface DeriveOptions {
  /** The host environment that resolves an unbound call head to a host value (the domain vocabulary). */
  env: HostEnvironment;
  /** Mints the stable keys that identify component instances + keyed list items across a re-derive. */
  minter: KeyMinter;
  /** Data made available to the program as the `data` binding; forwarded to the walk only when the key
   *  is present (an explicit `undefined` still binds `data`). Deep-frozen at the boundary. */
  data?: unknown;
  /** Seed for the intrinsic `rand()`/`range()` PRNG; the same seed reproduces the same sequence. Defaults
   *  to `0`. */
  seed?: number;
  /** The name of the entry component to lower. Defaults to `'Story'`. */
  entry?: string;
  /** Opt `data` into reactivity so a `data.x` read lowers to a trackable Region (Proxy-free). */
  reactiveData?: boolean;
  /** Prior settled state S (cellKey → value) to latch surviving component instances on a re-derive. */
  priorState?: ReadonlyMap<string, unknown>;
  /** Invoked with the freshly-created host BEFORE the walk runs, so a domain HostEnvironment can
   *  register leaf effects for reactive props during resolveCall. The resolveCall port carries no host
   *  param by design, so the host is handed over out-of-band, exactly once, through this callback. */
  onHost?: (host: RuntimeReactiveHost) => void;
  /** Fuel budget override — max evaluation steps before the walk fails closed with `ML-LANG-BUDGET`. */
  maxSteps?: number;
  /** Deadline budget override in milliseconds before the walk fails closed with `ML-LANG-BUDGET`. */
  maxTimeMs?: number;
  /** Recursion-depth budget override before the walk fails closed with `ML-LANG-BUDGET`. */
  maxDepth?: number;
  /** Cap on the length a string may grow to via `+` before the walk fails closed with `ML-LANG-BUDGET`. */
  maxStringLength?: number;
  /** Builtin modules the walk resolves unbound call heads against (e.g. a numeric library so a `vec3(...)`
   *  / `f32(...)` in source dispatches). Forwarded verbatim to lowerEntry; omit for a builtin-free walk. */
  builtins?: readonly BuiltinModule[];
}

/** The outcome of one {@link derive} pass: the raw host value(s) produced, the diagnostics collected, and
 *  the reactive host that owns this pass's cells + leaf effects. */
export interface DeriveResult {
  /** The raw host value(s) the walk produced (opaque to metael; the domain materializes them). */
  value: HostValue;
  /** Every diagnostic collected during the parse + walk, including `ML-RT-CONVERGE` on a non-converging
   *  flush. Empty on a fully-successful pass. */
  diagnostics: Diagnostic[];
  /** The reactive host that owns this derive's cells + leaf effects (for setState/exportState/teardown). */
  host: RuntimeReactiveHost;
}

/**
 * The one-shot composition root: run a single {@link change}-wrapped lowering pass over `source` with a
 * fresh reactive host, the caller's host environment, and a key minter, and return the raw host value(s),
 * the host, and diagnostics.
 *
 * Owns only the generic slice — settle the initial flush and surface a non-converging flush as the
 * `ML-RT-CONVERGE` diagnostic. It does NOT materialize a domain output, build a retained tree, or expose
 * updateData/setState: those are domain-owned (the domain provides the patch target + the reconcile). A
 * non-reactive throw is treated as a genuine bug and propagates rather than being swallowed.
 *
 * @param source - the program source text.
 * @param opts - host capabilities, key minter, and the optional data/seed/entry/state/budget inputs
 *               ({@link DeriveOptions}).
 * @returns the produced value + diagnostics + the owning reactive host ({@link DeriveResult}).
 */
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
        host, env: opts.env, minter: opts.minter, builtins: opts.builtins,
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
