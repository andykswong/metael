import { describe, it, expect } from 'vitest';
import { mount } from '@metael/vdom';
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { EXAMPLES, DEFAULT_EXAMPLE_ID, exampleById } from './examples.ts';

describe('curated examples', () => {
  it('ships >=5 UI and >=4 compute examples', () => {
    expect(EXAMPLES.filter((e) => e.target === 'ui').length).toBeGreaterThanOrEqual(5);
    expect(EXAMPLES.filter((e) => e.target === 'compute').length).toBeGreaterThanOrEqual(4);
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

  it('every compute example evaluates with zero diagnostics', () => {
    for (const e of EXAMPLES.filter((x) => x.target === 'compute')) {
      const res = evaluateProgram(e.source, {
        host: new PlainStorageHost(), env: new RecordingHostEnv(), data: e.data,
      });
      expect(res.diagnostics, `${e.id} should evaluate clean — got ${JSON.stringify(res.diagnostics)}`).toEqual([]);
    }
  });
});
