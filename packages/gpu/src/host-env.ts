// GpuHostEnv resolves the `gpu` head to a reactive resource. Follows the domain-package pattern: bindHost
// after construction, match the head in resolveCall, decline the rest. The resource is a pure value in
// expression position (kind:'value') so a program can read its fields (r.core, r.wgsl, r.gpuMs, …).
import type { HostEnvironment, Arg, HostValue, SourceSpan, ReactiveHost } from '@metael/lang';
import { isUserFn } from '@metael/lang';
import { GpuEngine, type GpuConfig, type ReduceConfig, type HistogramConfig, type GpuEngineDeps } from './resource.ts';

export class GpuHostEnv implements HostEnvironment {
  private engine: GpuEngine | null = null;
  private readonly deps: GpuEngineDeps;
  // THREE heads: `gpu` (a map kernel → the map-kernel gate), `gpuReduce` (a 2-arg associative reducer → the
  // DISTINCT reducer gate), and `gpuHistogram` (a 1-arg bin-mapper → the DISTINCT bin-mapper gate; an atomic
  // scatter). Registered together so a typo'd head fails loud with a did-you-mean over all three.
  readonly knownHeads = new Set(['gpu', 'gpuReduce', 'gpuHistogram']);
  constructor(deps: GpuEngineDeps) { this.deps = deps; }
  bindHost(host: ReactiveHost): void { this.engine = new GpuEngine(host, this.deps); }
  /** True if any resource dispatched on this env's engine is still pending — lets a headless driver await
   *  every DECLARED resource's settle, not only the ones reachable from the program's returned value. */
  anyPending(): boolean { return this.engine?.anyPending() ?? false; }
  [Symbol.dispose](): void { this.engine?.[Symbol.dispose](); }

  resolveCall(head: string, _key: string, args: Arg[], _children: HostValue[], _span: SourceSpan):
    { handled: true; value: HostValue; kind?: 'value' } | { handled: false } {
    if (head === 'gpu') {
      const kernel = args[0]?.value;
      const cfg = (args[1]?.value ?? {}) as GpuConfig;
      if (!isUserFn(kernel)) return { handled: true, value: null, kind: 'value' };
      return { handled: true, value: this.engine!.gpu(kernel, cfg), kind: 'value' };
    }
    if (head === 'gpuReduce') {
      const reducer = args[0]?.value;
      const cfg = (args[1]?.value ?? {}) as ReduceConfig;
      if (!isUserFn(reducer)) return { handled: true, value: null, kind: 'value' };
      return { handled: true, value: this.engine!.gpuReduce(reducer, cfg), kind: 'value' };
    }
    if (head === 'gpuHistogram') {
      const binMapper = args[0]?.value;
      const cfg = (args[1]?.value ?? {}) as HistogramConfig;
      if (!isUserFn(binMapper)) return { handled: true, value: null, kind: 'value' };
      return { handled: true, value: this.engine!.gpuHistogram(binMapper, cfg), kind: 'value' };
    }
    return { handled: false };
  }
}
