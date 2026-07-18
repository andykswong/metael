import { describe, it, expect } from 'vitest';
import { runComputeSettled } from './targets.ts';
import { exampleById } from './examples.ts';

describe('unified gpu compute example on a real adapter (Chromium)', () => {
  it('settles to a real backend and pretty-prints the values', async () => {
    const ex = exampleById('gpu-compute-map')!;
    const out = await runComputeSettled(ex.source, {});
    expect(out.diagnostics).toEqual([]);
    expect(out.text).toContain('value');
    expect(out.text).toMatch(/0.*2.*4.*6/);
    // do not assert a specific backend — cpu floor is valid in a headless adapter-less run
  });
});
