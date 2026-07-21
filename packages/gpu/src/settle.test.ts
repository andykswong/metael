// packages/gpu/src/settle.test.ts
import { describe, it, expect } from 'vitest';
import { settle, settled } from './settle.ts';
import type { GpuResource } from './resource.ts';

describe('settle (free helper over a re-dispatch thunk)', () => {
  it('re-dispatches until the resource is no longer pending', async () => {
    // A thunk that reports pending for 2 reads, then settled — models the engine's async settle.
    let n = 0;
    const thunk = (): GpuResource => ({ pending: n++ < 2, value: n >= 3 ? 42 : null, error: null } as unknown as GpuResource);
    const r = await settle(thunk);
    expect(settled(r)).toBe(true);
    expect(r.value).toBe(42);
  });

  it('settled() narrows pending only, not value-presence (a non-core run settles with value null)', () => {
    const nonCore = { pending: false, value: null, error: null } as unknown as GpuResource;
    expect(settled(nonCore)).toBe(true);   // settled, but value is still null — caller must null-check
  });
});
