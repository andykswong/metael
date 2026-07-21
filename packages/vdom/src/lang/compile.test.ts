import { describe, it, expect } from 'vitest';
import { compileToProducer } from './compile.ts';

describe('compileToProducer', () => {
  it('derives a DSL source into a VNode tree without running a DOM loop', () => {
    // Text content is a call arg (div("hi")) — a bare string statement in a block runs for effect and
    // is not collected as a child, so the arg form is the canonical way to place a text node.
    const { produce } = compileToProducer('component Story() { div("hi") }', {});
    const { nodes, diagnostics } = produce();
    expect(diagnostics).toEqual([]);
    // Story is a component fragment; its first real element is the div.
    const div = nodes.find((n) => n.tag === 'div');
    expect(div).toBeDefined();
    expect(div!.children.some((c) => c.text === 'hi')).toBe(true);
  });

  it('re-deriving with priorState latches a surviving component instance (fresh host each pass)', () => {
    const { produce } = compileToProducer('component Story() { let n = 1; div { n } }', {});
    const p1 = produce();
    const s = p1.host.exportState();
    const p2 = produce(s);           // second pass, same latched state
    expect(p2.host).not.toBe(p1.host);   // fresh host per pass
    expect(p2.diagnostics).toEqual([]);
  });
});
