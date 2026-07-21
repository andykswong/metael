import { describe, it, expect } from 'vitest';
import { kernelAst, letVar, set, forRange } from './index.ts';
import { call, param, lit } from './node.ts';
import { stripSpans, parseProgram } from '@metael/lang';

// The equivalent metael source, parsed to its decl AST (spans stripped). This is the ORACLE: the JS
// builder is correct iff it emits the exact node shape the parser does.
function parsedDecl(src: string): unknown {
  const { program, diagnostics } = parseProgram(src);
  expect(diagnostics).toEqual([]);
  const decl = program.stmts.find((s: { kind?: unknown }) => s.kind === 'component' || s.kind === 'function');
  expect(decl).toBeDefined();
  return stripSpans(decl);
}

// Corpus sources use the SYNTHESIZED param names p0/p1 and the fixed decl name K (the builder knows only
// arity), so the parsed AST matches what the builder emits. `a` is a FREE name (a buffer input) referenced
// via param('a') + the real `.at(...)` index chain (which emits { kind:'index', object, index } — matching
// a[i][j]); NOT call('at', ...), which would be a call node. Builder-arrow params infer as KNode from
// kernelAst's `(...params: KNode[]) => …` signature, so no `any` annotation is needed.
const CORPUS: { name: string; build: () => unknown; src: string }[] = [
  {
    name: 'scalar map',
    build: () => kernelAst((p0, p1) => call('f32', p0.add(p1))),
    src: 'component K(p0, p1) { return f32(p0 + p1) }',
  },
  {
    name: 'vec expression',
    build: () => kernelAst((p0, p1) => call('vec3', p0, p1, lit(0))),
    src: 'component K(p0, p1) { return vec3(p0, p1, 0) }',
  },
  {
    // matmul: `let acc = f32(0); for (const i of range(4)) { acc = acc + a[p0][i] * a[i][p1] } return acc`.
    // The JS builder RETURNS the `acc` KNode (a JS `return acc`), which kernelAst emits as a `return` stmt
    // — the dispatchable form the emitters write — matching the source's `return acc`. (A trailing bare
    // `acc` would parse to an `expr` stmt the emitters discard, dispatching all-zeros.)
    name: 'matmul with a range loop + accumulator',
    build: () =>
      kernelAst((p0, p1) => {
        const a = param('a'); // a FREE buffer name, referenced by index
        const acc = letVar('acc', call('f32', lit(0)));
        forRange(4, (i) => {
          set(acc, acc.add(a.at(p0, i).mul(a.at(i, p1)))); // a[p0][i] * a[i][p1]
        });
        return acc; // arrow-return → `return` stmt, matching the source's `return acc`
      }),
    src: 'component K(p0, p1) { let acc = f32(0)\nfor (const i of range(4)) { acc = acc + a[p0][i] * a[i][p1] }\nreturn acc }',
  },
];

describe('JS builder → AST ≡ parser → AST (the same-path guarantee)', () => {
  for (const c of CORPUS) {
    it(`${c.name}: kernelAst matches parseProgram`, () => {
      expect(stripSpans(c.build())).toEqual(parsedDecl(c.src));
    });
  }
});
