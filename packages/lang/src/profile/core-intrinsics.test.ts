import { describe, it, expect } from 'vitest';
import { coreIntrinsicsProfile } from './index.ts';

describe('coreIntrinsicsProfile', () => {
  it('publishes the range intrinsic spec so it stays classifiable/completable', () => {
    expect(coreIntrinsicsProfile.builtins.has('range')).toBe(true);
    expect(coreIntrinsicsProfile.builtins.get('range')!.profile).toBe('core');
  });

  it('documents range for editor tooling (doc + a documented param + returnDoc)', () => {
    const range = coreIntrinsicsProfile.builtins.get('range')!;
    expect((range.doc ?? '').length).toBeGreaterThan(0);
    expect(range.params?.length).toBe(1);
    expect(range.params?.[0]?.name).toBe('n');
    expect((range.params?.[0]?.doc ?? '').length).toBeGreaterThan(0);
    expect((range.returnDoc ?? '').length).toBeGreaterThan(0);
  });
});
