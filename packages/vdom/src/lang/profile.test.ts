import { describe, it, expect } from 'vitest';
import { vdomProfile } from './index.ts';
describe('vdomProfile', () => {
  it('is a permissive (open) head set with documented common tags', () => {
    expect(vdomProfile.permissiveHeads).toBe(true);
    expect(vdomProfile.heads.has('div')).toBe(true);
  });

  it('documents div with props + children params and a returnDoc (for the hover card)', () => {
    const div = vdomProfile.heads.get('div')!;
    expect(div.params.length).toBeGreaterThanOrEqual(1);
    for (const p of div.params) expect((p.doc ?? '').length).toBeGreaterThan(0);
    expect(div.params.map((p) => p.name)).toEqual(['props', 'children']);
    expect((div.returnDoc ?? '').length).toBeGreaterThan(0);
  });

  it('documents every common tag with params-with-docs + a returnDoc', () => {
    for (const h of vdomProfile.heads.values()) {
      expect(h.params.length, `${h.name} params`).toBeGreaterThanOrEqual(1);
      for (const p of h.params) expect((p.doc ?? '').length, `${h.name}.${p.name} doc`).toBeGreaterThan(0);
      expect((h.returnDoc ?? '').length, `${h.name} returnDoc`).toBeGreaterThan(0);
    }
  });
});
