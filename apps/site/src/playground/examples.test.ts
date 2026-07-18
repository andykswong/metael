import { describe, it, expect } from 'vitest';
import { mount } from '@metael/vdom';
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { EXAMPLES, DEFAULT_EXAMPLE_ID, exampleById } from './examples.ts';
import { GpuUiEnv, runComputeSettled } from './targets.ts';

describe('curated examples', () => {
  it('ships >=5 UI, >=4 compute, and >=2 GPU examples', () => {
    expect(EXAMPLES.filter((e) => e.target === 'ui').length).toBeGreaterThanOrEqual(5);
    expect(EXAMPLES.filter((e) => e.target === 'compute').length).toBeGreaterThanOrEqual(4);
    expect(EXAMPLES.filter((e) => e.target === 'gpu').length).toBeGreaterThanOrEqual(2);
  });

  it('every example has a non-empty label + blurb + unique id', () => {
    const ids = new Set<string>();
    for (const e of EXAMPLES) {
      expect(e.label.length, `${e.id} label`).toBeGreaterThan(0);
      expect(e.blurb.length, `${e.id} blurb`).toBeGreaterThan(0);
      expect(ids.has(e.id), `${e.id} duplicate`).toBe(false);
      ids.add(e.id);
    }
  });

  it('the default example id resolves', () => {
    expect(exampleById(DEFAULT_EXAMPLE_ID)).toBeDefined();
  });

  it('keeps every source line short enough to fit the editor pane without horizontal scroll', () => {
    // The source pane is ~half the viewport; long lines force horizontal scroll + read as bad indentation.
    // Cap at 68 chars (comfortably fits the pane at the shipped 13.5px mono). A failure means an example
    // needs re-wrapping.
    const MAX = 68;
    for (const e of EXAMPLES) {
      for (const [i, line] of e.source.split('\n').entries()) {
        expect(line.length, `${e.id} line ${i + 1} (${line.length} chars): ${line}`).toBeLessThanOrEqual(MAX);
      }
    }
  });

  it('every UI example derives with zero diagnostics (headless)', () => {
    for (const e of EXAMPLES.filter((x) => x.target === 'ui')) {
      const h = mount(e.source, undefined, { data: e.data });
      expect(h.diagnostics, `${e.id} should derive clean — got ${JSON.stringify(h.diagnostics)}`).toEqual([]);
      expect(h.tree(), `${e.id} should produce a tree`).not.toBeNull();
    }
  });

  it('every GPU example derives with zero diagnostics (composite env, headless)', () => {
    // A GPU example calls the `gpu` head, which the plain display env does not provide — it derives clean
    // only under the composite GpuUiEnv (the playground's real 'gpu'-target env). The synchronous classify/
    // emit produces the resource on the first frame; the async dispatch is not awaited here (that settle is
    // exercised in gpu-example.browser.test.ts on a real adapter). ONE env per mount + immediate unmount:
    // the factory returns the same instance each pass so the engine's dispatch-memo persists (a fresh engine
    // per pass would re-enqueue a dispatch on every settle → an infinite re-derive loop), and unmount stops
    // the walk-effect before the deferred dispatch microtask fires, so this synchronous check never re-derives.
    for (const e of EXAMPLES.filter((x) => x.target === 'gpu')) {
      const env = new GpuUiEnv();
      const h = mount(e.source, undefined, { data: e.data, envFactory: () => env });
      expect(h.diagnostics, `${e.id} should derive clean — got ${JSON.stringify(h.diagnostics)}`).toEqual([]);
      expect(h.tree(), `${e.id} should produce a tree`).not.toBeNull();
      h.unmount();
    }
  });

  it('every compute example evaluates with zero diagnostics', () => {
    // A gpu-head compute example resolves the `gpu` head only under a gpu-aware env — the plain
    // RecordingHostEnv does not know it (that would be a spurious ML-LANG-UNKNOWN-CALL). Those examples
    // are proven clean on the settled compute path below; the plain-env sweep covers the rest.
    for (const e of EXAMPLES.filter((x) => x.target === 'compute' && !usesGpuHead(x.source))) {
      const res = evaluateProgram(e.source, {
        host: new PlainStorageHost(), env: new RecordingHostEnv(), data: e.data,
      });
      expect(res.diagnostics, `${e.id} should evaluate clean — got ${JSON.stringify(res.diagnostics)}`).toEqual([]);
    }
  });
});

// A compute source "uses a gpu head" if it calls gpu/gpuReduce/gpuHistogram (mirrors create.ts).
function usesGpuHead(source: string): boolean {
  return /\bgpu(Reduce|Histogram)?\s*\(/.test(source);
}

describe('unified gpu compute example (headless)', () => {
  it('gpu-compute-map settles to the expected values on the compute path', async () => {
    const ex = exampleById('gpu-compute-map')!;
    expect(ex.target).toBe('compute');
    const out = await runComputeSettled(ex.source, {});
    expect(out.diagnostics).toEqual([]);
    expect(out.text).toMatch(/\[0, ?2, ?4, ?6/);   // the settled array, pretty-printed
  });
});
