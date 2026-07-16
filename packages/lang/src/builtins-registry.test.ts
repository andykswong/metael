import { describe, it, expect } from 'vitest';
import { BUILTINS, isBuiltin, IMPLEMENTED_BUILTINS } from './builtins-registry.ts';

describe('builtins registry', () => {
  it('every entry key matches its name field', () => {
    for (const [k, spec] of Object.entries(BUILTINS)) expect(spec.name).toBe(k);
  });
  it('every entry has a valid profile + portability + arity', () => {
    for (const spec of Object.values(BUILTINS)) {
      expect(['core', 'host']).toContain(spec.profile);
      expect(['exact', 'gpu-tolerant', 'cpu-only']).toContain(spec.portability);
      expect(spec.arity[0]).toBeLessThanOrEqual(spec.arity[1]);
      // A host-profile builtin is never cross-target exact/gpu-tolerant.
      if (spec.profile === 'host') expect(spec.portability).toBe('cpu-only');
    }
  });
  it('a closure-taking builtin is host, EXCEPT the linear-buffer constructors', () => {
    // A builtin that operates OVER a closure argument (map/filter/reduce) is inherently host. The
    // buffer constructors take an OPTIONAL generator closure but stay 'core' — they build a core
    // linear-memory type, and the generator is a construction convenience (a caller passing one is
    // independently classified non-core by the arrow arg, not by the constructor's profile).
    const bufferCtors = new Set(['f32', 'f64', 'i32', 'u32']);
    for (const spec of Object.values(BUILTINS)) {
      if (spec.takesClosure && !bufferCtors.has(spec.name)) expect(spec.profile).toBe('host');
    }
  });
  it('isBuiltin recognizes a known name and rejects an unknown one', () => {
    expect(isBuiltin('map')).toBe(true);
    expect(isBuiltin('sin')).toBe(true);           // future-declared still counts as a builtin name
    expect(isBuiltin('definitelyNotABuiltin')).toBe(false);
  });
  it('IMPLEMENTED_BUILTINS excludes future-only entries', () => {
    expect(IMPLEMENTED_BUILTINS.has('map')).toBe(true);
    expect(IMPLEMENTED_BUILTINS.has('sin')).toBe(false);
    expect(IMPLEMENTED_BUILTINS.has('mix')).toBe(false);
  });
});
