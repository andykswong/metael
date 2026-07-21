import { describe, it, expect } from 'vitest';
import { createGpuEngine, settle, settled } from '@metael/gpu';
import { compileKernel } from '@metael/gpu/lang';
import { kernel, ret } from './index.ts';

// The payoff of the AST-equivalence spine (equivalence.test.ts): once a JS-built kernel produces the SAME
// kernel AST the parser emits, it flows through the identical gate/emit/oracle/dispatch path — so the two
// authoring front-ends must yield identical dispatched output. This runs both on the CPU backend (a NODE
// test, cpuOnly) and compares the settled values. Param names differ (p0/p1 vs row/col) — but they are just
// thread-coordinate bindings; the dispatched OUTPUT VALUES are what must match.
describe('a JS-built kernel dispatches identically to the DSL-authored one', () => {
  it('scalar map: same output on the CPU backend', async () => {
    using engine = createGpuEngine({ cpuOnly: true });
    // A kernel returns its per-cell value via an explicit `return` (a trailing bare expression is a
    // discarded statement, so a `{ row + col }` body dispatches all zeros — the OUTPUT is the returned
    // value). Both front-ends build a lowerable scalar map over the (row, col) thread coordinates; f32's
    // numeric model is already float, so no cast wrapper is needed (and `f32(...)` in a body is the
    // typed-array constructor, which is not shader-lowerable).
    const jsK = kernel((row, col) => {
      ret(row.add(col)); // UserFn built from the JS front-end
    });
    const dslK = compileKernel('component K(row, col) { return row + col }\nK', engine.host);
    const rjs = await settle(() => engine.dispatch(jsK, { output: [2, 2] }));
    const rdsl = await settle(() => engine.dispatch(dslK, { output: [2, 2] }));
    expect(settled(rjs) && settled(rdsl)).toBe(true);
    expect(rjs.value).not.toBeNull(); // guard: a non-core/cost-rejected/emit-errored run also settles pending:false with value null
    // The thread coords are (row, col) ∈ {0,1}×{0,1}, so each cell is row + col (row-major over [2,2]).
    expect(rjs.value).toEqual([0, 1, 1, 2]);
    expect(rjs.value).toEqual(rdsl.value);
  });

  it('arrow-return form dispatches the computed value (not all-zeros)', async () => {
    using engine = createGpuEngine({ cpuOnly: true });
    const jsK = kernel((row, col) => row.add(col));   // idiomatic arrow-return — no ret()
    const r = await settle(() => engine.dispatch(jsK, { output: [2, 2] }));
    expect(settled(r)).toBe(true);
    expect(r.value).not.toBeNull();
    expect(r.value).toEqual([0, 1, 1, 2]);   // row+col over [2,2], NOT [0,0,0,0]
  });
});
