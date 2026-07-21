import { describe, it, expect } from 'vitest';
import { exampleById } from './examples.ts';
import { runComputeSettled } from './targets.ts';

// Guards the object-building examples through the real playground compute pipeline (the same path the
// gallery runs). These examples use the `object` structural builtin (immutable rebuild from pairs); a
// regression in the std-builtin wiring or the `object` head would surface as a diagnostic here.
describe('structural object-building examples (compute pipeline)', () => {
  it('object-shape doubles a bumped price table via entries + map + object', async () => {
    const ex = exampleById('object-shape')!;
    expect(ex.target).toBe('compute');
    const out = await runComputeSettled(ex.source, {});
    expect(out.diagnostics).toEqual([]);
    expect(out.value).toEqual({ apple: 6, pear: 10, plum: 40 });
  });

  it('group-count tallies word frequency via reduce + includes + object', async () => {
    const ex = exampleById('group-count')!;
    expect(ex.target).toBe('compute');
    const out = await runComputeSettled(ex.source, {});
    expect(out.diagnostics).toEqual([]);
    expect(out.value).toEqual({ red: 3, blue: 2, green: 1 });
  });
});
