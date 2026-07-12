import { describe, it, expect } from 'vitest';
import { runTarget } from './targets.ts';

describe('runTarget', () => {
  it('ui target mounts headless and reports a tree + no diagnostics', () => {
    const run = runTarget('ui', 'component Story() { span("hi") }', undefined, {});
    expect(run.kind).toBe('ui');
    if (run.kind === 'ui') {
      expect(run.diagnostics).toEqual([]);
      expect(run.handle.tree()).not.toBeNull();
      run.handle.unmount();
    }
  });

  it('compute target evaluates to a pretty string + the raw value', () => {
    const run = runTarget('compute', 'map(range(4), (i) => i * i)', undefined, {});
    expect(run.kind).toBe('compute');
    if (run.kind === 'compute') {
      expect(run.value).toEqual([0, 1, 4, 9]);
      expect(run.text).toBe('[0, 1, 4, 9]');
      expect(run.diagnostics).toEqual([]);
    }
  });

  it('compute target injects data', () => {
    const run = runTarget('compute', 'map(data, (r) => r.n)', undefined, { data: [{ n: 1 }, { n: 2 }] });
    if (run.kind === 'compute') expect(run.value).toEqual([1, 2]);
  });

  it('surfaces diagnostics from a bad source (ui)', () => {
    const run = runTarget('ui', 'component Story( {', undefined, {});
    expect(run.diagnostics.length).toBeGreaterThan(0);
  });
});
