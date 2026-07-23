import { describe, it, expect } from 'vitest';
import { gpuProfile } from './index.ts';
describe('gpuProfile', () => {
  it('has exactly the three gpu heads, all returning values', () => {
    expect(new Set(gpuProfile.heads.keys())).toEqual(new Set(['gpu', 'gpuReduce', 'gpuHistogram']));
    for (const h of gpuProfile.heads.values()) expect(h.returns).toBe('value');
  });

  it('documents each head with a returnDoc + a doc on every param (for the hover card)', () => {
    // Iterate the live head set (not a hardcoded list) so a future head is forced through this doc guard too.
    for (const h of gpuProfile.heads.values()) {
      expect((h.returnDoc ?? '').length, `${h.name} returnDoc`).toBeGreaterThan(0);
      expect(h.params.length, `${h.name} params`).toBeGreaterThan(0);
      for (const p of h.params) expect((p.doc ?? '').length, `${h.name}.${p.name} doc`).toBeGreaterThan(0);
    }
  });
});
