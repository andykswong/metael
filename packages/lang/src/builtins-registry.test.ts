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
  it('core-profile builtins never take a closure (closure ⇒ host)', () => {
    for (const spec of Object.values(BUILTINS)) {
      if (spec.takesClosure) expect(spec.profile).toBe('host');
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
