// packages/gpu/src/device/pipeline-cache.ts
// A per-backend compiled-pipeline cache keyed by shader source text. The emitted WGSL/GLSL is a pure
// function of (kernel, bindings, precision, comps), so the source string is a complete + correct key: a
// different kernel/precision yields a different string (fresh compile); an identical string is always safe
// to reuse (same layout, same `main` entry point). One cache per backend instance — its lifetime matches
// the pooled device, freed when the backend is disposed. Native Disposable: usable with `using` + composed
// into a backend's own [Symbol.dispose].
export interface PipelineCache<P> extends Disposable {
  /** The cached pipeline for `src`, compiling + storing it on first use. */
  get(src: string): P;
}

export function makePipelineCache<P>(compile: (src: string) => P, free: (p: P) => void): PipelineCache<P> {
  const map = new Map<string, P>();
  return {
    get(src) {
      const hit = map.get(src);
      if (hit !== undefined) return hit;
      const p = compile(src);
      map.set(src, p);
      return p;
    },
    [Symbol.dispose]() {
      // Idempotent: after the first dispose the map is empty, so a second call frees nothing.
      for (const p of map.values()) free(p);
      map.clear();
    },
  };
}
