// Parser/lexer hazard hardening: source that a JS-literate author might write which used to mis-parse
// SILENTLY (a surprising shape, no diagnostic) or emit a misleading multi-diagnostic cascade is now
// guarded — either parsed the expected way, or rejected with ONE clear diagnostic + clean recovery.
// Each `describe` block is one root cause; the letter tags (A/B/C/D/E/F/G/H/I/J) map to the design's
// root-cause labels. Two rounds of adversarial review are baked in here as regression guards.
import { describe, it, expect } from 'vitest';
import { parseProgram, parseExpr } from './parser.ts';
import { lex } from './lexer.ts';
import type { Expr } from './ast.ts';

const prog = (src: string) => parseProgram(src);
const codes = (src: string) => parseProgram(src).diagnostics.map((d) => d.code);
type St = ReturnType<typeof prog>['program']['stmts'][number];
type Call = Extract<Expr, { kind: 'call' }>;

// ── B: the number lexer rejects malformed numbers instead of silently producing NaN / truncating ──
describe('number lexer rejects malformed literals (no silent NaN, no silent truncation)', () => {
  it('a multi-dot number emits ML-LANG-LEX (not a silent NaN)', () => {
    expect(lex('1.2.3').diagnostics.map((d) => d.code)).toContain('ML-LANG-LEX');
    expect(lex('10.0.0.1').diagnostics.map((d) => d.code)).toContain('ML-LANG-LEX');
  });
  it('a number immediately followed by identifier chars emits ML-LANG-LEX (0xFF / 1e3 / 1_000 / 10n)', () => {
    for (const src of ['0xFF', '1e3', '1_000', '10n']) {
      expect(lex(src).diagnostics.map((d) => d.code), src).toContain('ML-LANG-LEX');
    }
  });
  it('a well-formed number after the malformed prefix does not cascade into orphan idents', () => {
    // `1e3` used to lex as number `1` + ident `e3`; now it is one malformed number + one diagnostic.
    const toks = lex('1e3').tokens.filter((t) => t.type !== 'eof');
    expect(toks.every((t) => t.type === 'number')).toBe(true);
  });
  it('valid integers and decimals still lex cleanly (no false positives)', () => {
    for (const src of ['0', '42', '2.5', '0.5', '100', '3.14159']) {
      expect(lex(src).diagnostics, src).toEqual([]);
      expect(lex(src).tokens[0]).toMatchObject({ type: 'number' });
    }
  });
});

// ── C: an Object.prototype member name is a valid identifier (no prototype-chain keyword leak) ───
describe('reserved-word lookup does not leak the object prototype', () => {
  it('`toString` is a plain identifier, not a native-function token type', () => {
    const { program, diagnostics } = prog('let toString = 5');
    expect(diagnostics).toEqual([]);
    expect(program.stmts[0]).toMatchObject({ kind: 'let', name: 'toString', init: { kind: 'number', value: 5 } });
  });
  it('`valueOf` / `hasOwnProperty` / `constructor` work as identifiers and binding names', () => {
    for (const name of ['valueOf', 'hasOwnProperty', 'constructor']) {
      expect(lex(name).tokens[0], name).toMatchObject({ type: 'ident', value: name });
    }
    expect(prog('function valueOf() { 1 }').diagnostics).toEqual([]);
  });
});

// ── J: relational binds tighter than equality (JS-conformant precedence) ─────────────────────────
describe('relational operators bind tighter than equality', () => {
  it('`0 == 1 > 2` parses as `0 == (1 > 2)`, matching JS', () => {
    expect(parseExpr('0 == 1 > 2').expr).toMatchObject({
      kind: 'binary', op: '==',
      left: { kind: 'number', value: 0 },
      right: { kind: 'binary', op: '>' },
    });
  });
  it('`a < b == c` parses as `(a < b) == c`', () => {
    expect(parseExpr('a < b == c').expr).toMatchObject({
      kind: 'binary', op: '==', left: { kind: 'binary', op: '<' }, right: { kind: 'ident', name: 'c' },
    });
  });
});

// ── I: `.` requires a name-like property (ident / keyword / quoted string), not a number/operator ─
describe('member access requires a property name after "."', () => {
  it('a numeric "property" emits ML-LANG-PARSE', () => {
    expect(codes('let a = obj.5')).toContain('ML-LANG-PARSE');
  });
  it('a dangling "." emits ML-LANG-PARSE', () => {
    expect(codes('let a = obj.')).toContain('ML-LANG-PARSE');
  });
  it('an identifier, keyword, or quoted-string property is still accepted (round-trip forms)', () => {
    expect(parseExpr('obj.field').diagnostics).toEqual([]);
    expect(parseExpr('obj.if').diagnostics).toEqual([]);          // keyword property (JS-familiar)
    expect(parseExpr('obj."x-y"').diagnostics).toEqual([]);        // quoted-dot (printer round-trip)
  });
});

// ── D: an assignment target that is not an lvalue is a parse error (catches `==` typed as `=`) ────
describe('assignment target is validated at parse time', () => {
  it('a binary LHS (a `==` mistyped as `=`) is an invalid assignment target', () => {
    expect(codes('a + b = 1')).toContain('ML-LANG-PARSE');
    expect(codes('ready == loaded = true')).toContain('ML-LANG-PARSE');
  });
  it('a call LHS is an invalid assignment target', () => {
    expect(codes('f() = 1')).toContain('ML-LANG-PARSE');
  });
  it('ident / member / index targets remain valid (no false positive)', () => {
    expect(codes('x = 1')).toEqual([]);
    expect(codes('o.a = 1')).toEqual([]);
    expect(codes('a[0] = 1')).toEqual([]);
  });
});

// ── A: a newline before a `(` / `[` ends the statement (postfix does not cross a line) ──────────
describe('newline-guarded postfix (call / index do not glom onto the previous line)', () => {
  it('a paren-led next line is a NEW statement, not a call applied to the previous value', () => {
    const { program, diagnostics } = prog('let spec = pow(diffuse, s)\n(albedo * diffuse + spec) * falloff');
    expect(diagnostics).toEqual([]);
    expect(program.stmts).toHaveLength(2);
    expect(program.stmts[0]).toMatchObject({ kind: 'let', name: 'spec', init: { kind: 'call' } });
    // the second statement is the parenthesized product, NOT swallowed into a pow(...)(...) chain
    expect(program.stmts[1]).toMatchObject({ kind: 'expr', expr: { kind: 'binary', op: '*' } });
  });
  it('a bracket-led next line is a NEW array-literal statement, not an index of the previous value', () => {
    const { program, diagnostics } = prog('let xs = base\n[0, 1]');
    expect(diagnostics).toEqual([]);
    expect(program.stmts).toHaveLength(2);
    expect(program.stmts[0]).toMatchObject({ kind: 'let', name: 'xs', init: { kind: 'ident', name: 'base' } });
    expect(program.stmts[1]).toMatchObject({ kind: 'expr', expr: { kind: 'array' } });
  });
  it('an implicit-last-expression return is NOT destroyed by a paren-led final line', () => {
    // function body: `let w = getW()` then `(w * w)` — the two must stay separate so the last
    // expression (the product) is the implicit return, not an argument list applied to getW().
    const { program, diagnostics } = prog('function area() {\n  let w = getW()\n  (w * w)\n}');
    expect(diagnostics).toEqual([]);
    const fn = program.stmts[0] as Extract<St, { kind: 'function' }>;
    expect(fn.body).toHaveLength(2);
    expect(fn.body[1]).toMatchObject({ kind: 'expr', expr: { kind: 'binary', op: '*' } });
  });
  it('same-line call / index still chain (the guard is newline-only)', () => {
    expect(parseExpr('getArr(i)[0]').expr).toMatchObject({ kind: 'index', object: { kind: 'call' } });
    // a leading-dot method chain across lines is idiomatic and still works (only ( and [ are guarded)
    const { diagnostics } = prog('let x = foo(a)\n  .bar(b)');
    expect(diagnostics).toEqual([]);
  });
  it('the newline guard fires only at STATEMENT level, not inside an open grouping', () => {
    // Inside an open ( … ) / [ … ] there is no statement boundary, so a line-leading ( / [ must still
    // continue the postfix chain — a newline there is not a statement break. (Regression guard: the
    // guard must not fire universally.) All must parse as ONE clean statement, no diagnostics.
    expect(prog('f(g(x)\n(y))')).toMatchObject({ program: { stmts: [{ kind: 'expr' }] }, diagnostics: [] });
    expect(prog('render(\n items\n [0]\n)')).toMatchObject({ program: { stmts: [{ kind: 'expr' }] }, diagnostics: [] });
    expect(prog('const r = (\n base(x)\n (y)\n)')).toMatchObject({ program: { stmts: [{ kind: 'const' }] }, diagnostics: [] });
    // ...and likewise inside other non-statement contexts a newline is not a boundary: a ternary
    // THEN branch (bounded by `:`), an if-condition, and an array literal all keep a line-leading ( / [ chained.
    expect(prog('let v = c ? f(x)\n(y) : z')).toMatchObject({ program: { stmts: [{ kind: 'let' }] }, diagnostics: [] });
    expect(prog('if (ready(s)\n(t)) { go() }')).toMatchObject({ diagnostics: [] });
    expect(prog('let arr = [f(x)\n(y)]')).toMatchObject({ program: { stmts: [{ kind: 'let' }] }, diagnostics: [] });
  });
  it('a ternary ELSE branch at statement level is the statement tail — the guard still fires there', () => {
    // The else branch has no closing delimiter (unlike `then`, bounded by `:`), so at statement level
    // it IS the statement tail: a line-leading ( / [ after it must start a NEW statement, not glue onto
    // the else operand. (Regression guard: parseCond must not raise groupDepth across the else branch.)
    expect(prog('let a = c ? d : e\n(y)')).toMatchObject({
      program: { stmts: [{ kind: 'let', init: { kind: 'cond', else: { kind: 'ident', name: 'e' } } }, { kind: 'expr' }] },
      diagnostics: [],
    });
    expect(prog('let a = c ? d : e\n[0]')).toMatchObject({ program: { stmts: [{ kind: 'let' }, { kind: 'expr' }] }, diagnostics: [] });
    // but INSIDE a grouping the else tail still chains (no statement boundary there)
    expect(prog('f(c ? d : e\n(y))')).toMatchObject({ program: { stmts: [{ kind: 'expr' }] }, diagnostics: [] });
  });
});

// ── E: the `head { … }` wrap shorthand works in VALUE position (const/let/return/assign RHS) ──────
describe('wrap shorthand works in value position', () => {
  it('a const RHS wrap produces a wrapping call with the block attached', () => {
    const { program, diagnostics } = prog('const root = group {\n  text("x")\n}');
    expect(diagnostics).toEqual([]);
    const decl = program.stmts[0] as Extract<St, { kind: 'const' }>;
    expect(decl.init).toMatchObject({ kind: 'call', callee: { kind: 'ident', name: 'group' }, args: [] });
    expect((decl.init as Call).block).toHaveLength(1);
  });
  it('a return-value wrap attaches the block (not silently dropped)', () => {
    const { program, diagnostics } = prog('function f() {\n  return group {\n    text("x")\n  }\n}');
    expect(diagnostics).toEqual([]);
    const fn = program.stmts[0] as Extract<St, { kind: 'function' }>;
    const ret = fn.body[0] as Extract<St, { kind: 'return' }>;
    expect(ret.value).toMatchObject({ kind: 'call', callee: { kind: 'ident', name: 'group' } });
    expect((ret.value as Call).block).toHaveLength(1);
  });
  it('an assignment RHS wraps too, consistent with const/let/return', () => {
    const { program, diagnostics } = prog('root = group {\n  header()\n}');
    expect(diagnostics).toEqual([]);
    const asn = program.stmts[0] as Extract<St, { kind: 'assign' }>;
    expect(asn.value).toMatchObject({ kind: 'call', callee: { kind: 'ident', name: 'group' } });
    expect((asn.value as Call).block).toHaveLength(1);
  });
  it('an object-literal RHS is still an object, not a wrap (the same-line-{ shorthand needs a bare ident head)', () => {
    // `const o = { a: 1 }` — the `{` follows `=`, not a bare ident, so it is an object literal.
    const { program, diagnostics } = prog('const o = { a: 1 }');
    expect(diagnostics).toEqual([]);
    expect(program.stmts[0]).toMatchObject({ kind: 'const', init: { kind: 'object' } });
  });
  it('a bare ident RHS followed by a NEXT-line brace is two statements, not a wrap', () => {
    // `const x = value` then `{ … }` on the next line: the wrap fires only on a SAME-line brace.
    const { program } = prog('const x = value\n{ a: 1 }');
    expect(program.stmts[0]).toMatchObject({ kind: 'const', init: { kind: 'ident', name: 'value' } });
    expect(program.stmts).toHaveLength(2);
  });
});

// ── F: brace-less same-line wrap — the exact documented rules (LANGUAGE.md §5) ────────────────────
describe('brace-less same-line wrap rules', () => {
  it('a same-line trailing wrap child attaches (`transform(...) shape(...)`)', () => {
    const { program, diagnostics } = prog('transform({ x: 10 }) shape({ kind: "rect" })');
    expect(diagnostics).toEqual([]);
    const e = (program.stmts[0] as Extract<St, { kind: 'expr' }>).expr as Call;
    expect(e).toMatchObject({ kind: 'call', callee: { kind: 'ident', name: 'transform' } });
    expect(e.block).toHaveLength(1);
  });
  it('a `{` block after a call on the NEXT line does NOT wrap (newline-guarded)', () => {
    // `render()` then `{ compute() }` on the next line → two statements (the block does not attach).
    const { program } = prog('render()\n{ compute() }');
    expect(program.stmts).toHaveLength(2);
    expect(program.stmts[0]).toMatchObject({ kind: 'expr', expr: { kind: 'call', callee: { kind: 'ident', name: 'render' } } });
    const first = (program.stmts[0] as Extract<St, { kind: 'expr' }>).expr as Call;
    expect(first.block).toBeUndefined();
  });
  it('a same-line `{` block after a call still wraps (`render() { compute() }`)', () => {
    const e = (prog('render() { compute() }').program.stmts[0] as Extract<St, { kind: 'expr' }>).expr as Call;
    expect(e.block).toHaveLength(1);
  });
  it('a member/index head does NOT wrap (`ui.panel { … }` is a member expr + a separate statement)', () => {
    const { program } = prog('ui.panel { text("x") }');
    expect(program.stmts[0]).toMatchObject({ kind: 'expr', expr: { kind: 'member', property: 'panel' } });
    expect(program.stmts.length).toBeGreaterThanOrEqual(2);
  });
  it('a next-line `{` that is not a valid object emits ONE diagnostic, not a cascade, and resyncs', () => {
    // `render()⏎{ compute() }` → render() (call, no block) + a malformed object literal (one diagnostic).
    const one = codes('render()\n{ compute() }').filter((c) => c === 'ML-LANG-PARSE');
    expect(one).toHaveLength(1);
    // a following clean statement still parses
    const { program } = prog('render()\n{ compute() }\nconst ok = 5');
    expect(program.stmts.some((s) => s.kind === 'const' && (s as { name: string }).name === 'ok')).toBe(true);
  });
});

// ── G: an arrow with an assignment body is a fail-loud error; require a block body ────────────────
describe('arrow inline-assignment body is diagnosed (assignment is statement-only)', () => {
  it('`() => count = count + 1` emits a single targeted ML-LANG-PARSE (not a cascade)', () => {
    expect(codes('const h = () => count = count + 1').filter((c) => c === 'ML-LANG-PARSE')).toHaveLength(1);
  });
  it('a block-body arrow assignment is the accepted form (no diagnostic)', () => {
    const { program, diagnostics } = prog('const h = () => { count = count + 1 }');
    expect(diagnostics).toEqual([]);
    const decl = program.stmts[0] as Extract<St, { kind: 'const' }>;
    expect(decl.init).toMatchObject({ kind: 'arrow' });
    expect(Array.isArray((decl.init as Extract<typeof decl.init, { kind: 'arrow' }>).body)).toBe(true);
  });
  it('a pure-expression arrow body is unaffected', () => {
    expect(parseExpr('(x) => x + 1').diagnostics).toEqual([]);
    expect(parseExpr('(i) => !i.done').diagnostics).toEqual([]);
  });
  it('an arrow as a bare binary operand `1 + x => 2` is rejected (JS forbids it)', () => {
    expect(codes('const a = 1 + x => 2')).toContain('ML-LANG-PARSE');
  });
  it('a PARENTHESIZED arrow operand is FINE — the default-callback pattern must not be flagged', () => {
    // `handler || (() => fallback)` and `1 + (x => 2)` are valid: the parens make the arrow a value.
    expect(parseExpr('handler || (() => fallback)').diagnostics).toEqual([]);
    expect(parseExpr('props.onClick || ((e) => e)').diagnostics).toEqual([]);
    expect(parseExpr('1 + (x => 2)').diagnostics).toEqual([]);
    expect(parseExpr('cond && (() => run())').diagnostics).toEqual([]);
  });
});

// ── H: unsupported-JS-feature recovery — ONE named diagnostic + resync, not a cascade ─────────────
describe('eat()-cascade recovery: unsupported features give one clear diagnostic, not a cascade', () => {
  const parseErrs = (src: string) => codes(src).filter((c) => c === 'ML-LANG-PARSE').length;
  it('destructuring in a declaration → a single ML-LANG-PARSE, not a 7-diagnostic cascade', () => {
    expect(parseErrs('const { title, body } = props')).toBe(1);
  });
  it('multi-declarator `const a = 1, b = 2` → a single diagnostic, no phantom `EXPR null` statement', () => {
    const { program, diagnostics } = prog('const a = 1, b = 2');
    expect(diagnostics.filter((d) => d.code === 'ML-LANG-PARSE')).toHaveLength(1);
    // recovery must not emit a phantom `EXPR null` (a bare `null` expression statement from the stray comma)
    expect(program.stmts.some((s) => s.kind === 'expr' && (s as { expr: { kind: string } }).expr.kind === 'null')).toBe(false);
  });
  it('call-arg spread `f(...args)` → a single diagnostic, not a phantom block + cascade', () => {
    expect(parseErrs('f(...args)')).toBe(1);
  });
  it('`if (a = b)` (assignment in condition) → a single diagnostic', () => {
    expect(parseErrs('if (a = b) { x() }')).toBe(1);
  });
  it('a following clean statement still parses after a recovered unsupported feature', () => {
    // recovery must resync to the statement boundary so the NEXT statement is clean.
    const { program } = prog('const { a } = props\nconst ok = 5');
    const okDecl = program.stmts.find((s) => s.kind === 'const' && (s as { name: string }).name === 'ok');
    expect(okDecl).toBeTruthy();
  });
  it('two destructures do not spuriously report `\'\' already declared` (empty recovery-name is not a binding)', () => {
    expect(codes('const { a } = p\nconst { b } = q')).not.toContain('ML-LANG-REDECL');
  });
});

// ── regression guard: object literals accept reserved-word keys (printer round-trip) ─────────────
describe('object literals accept keyword-named keys (bare) — round-trip preserved', () => {
  it('`{ if: 1 }` / `{ true: 2 }` / `{ for: 3 }` parse with no diagnostic', () => {
    for (const src of ['const o = { if: 1 }', 'const o = { true: 2 }', 'const o = { for: 3, return: 4 }']) {
      expect(codes(src), src).toEqual([]);
    }
  });
  it('a genuinely malformed object still gets ONE diagnostic (the recovery is intact)', () => {
    expect(codes('const o = { compute() }').filter((c) => c === 'ML-LANG-PARSE')).toHaveLength(1);
  });
});
