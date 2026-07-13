import { describe, it, expect } from 'vitest';
import { printExpr, printString, stripSpans, printProgram, PrintDepthError } from './print.ts';
import { parseExpr, parseProgram } from './parser.ts';
import type { Expr } from './ast.ts';

const s = { start: 0, end: 0 };

describe('printExpr — literals & identifiers', () => {
  it('number', () => { expect(printExpr({ kind: 'number', value: 42, span: s })).toBe('42'); });
  it('bool / null', () => {
    expect(printExpr({ kind: 'bool', value: true, span: s })).toBe('true');
    expect(printExpr({ kind: 'null', span: s })).toBe('null');
  });
  it('ident', () => { expect(printExpr({ kind: 'ident', name: 'x', span: s })).toBe('x'); });
  it('string is quoted and escaped', () => {
    expect(printExpr({ kind: 'string', value: 'a"b\nc\t\\d', span: s })).toBe('"a\\"b\\nc\\t\\\\d"');
  });
});

describe('printString — inverts the lexer escapes', () => {
  it('escapes backslash, quote, newline, tab', () => {
    expect(printString('x')).toBe('"x"');
    expect(printString('a\nb')).toBe('"a\\nb"');
    expect(printString('a\\b')).toBe('"a\\\\b"');
    expect(printString('a"b')).toBe('"a\\"b"');
  });
});

describe('stripSpans — recursively removes span fields for structural comparison', () => {
  it('a parsed expr with spans equals the same tree stripped', () => {
    const { expr } = parseExpr('1 + 2');
    const stripped = stripSpans(expr) as Expr;
    expect((stripped as { left: unknown }).left).toEqual({ kind: 'number', value: 1 });
  });
});

/** Expression-level conservation: parse → print → parse yields a structurally identical Expr. */
function exprRoundTrips(src: string): void {
  const first = parseExpr(src);
  expect(first.diagnostics).toEqual([]);
  const second = parseExpr(printExpr(first.expr));
  expect(second.diagnostics).toEqual([]);
  expect(stripSpans(second.expr)).toEqual(stripSpans(first.expr));
}

describe('printExpr — compound expressions round-trip', () => {
  it('member / index', () => { exprRoundTrips('obj.field[0]'); });
  it('call with args', () => { exprRoundTrips('f(1, "x", y)'); });
  it('unary / binary with precedence', () => { exprRoundTrips('-x + y * 2 == 3 && !flag'); });
  it('ternary (right-assoc)', () => { exprRoundTrips('p ? q : r ? s : t'); });
  it('object literal with spread', () => { exprRoundTrips('{ x: 1, ...rest, y: z }'); });
  it('array literal with spread', () => { exprRoundTrips('[1, ...xs, 3]'); });
  it('arrow — expression body', () => { exprRoundTrips('(a, b) => a + b'); });
  it('arrow — single param, expression body', () => { exprRoundTrips('(h) => h + 1'); });
  // NOTE: arrow BLOCK bodies (`(h) => { x = h }`) and destructuring PARAMS (`{ a, b }`) are exercised in
  // the statement task — a block body needs statement printing (`printStmt`, added there), and the grammar
  // admits destructuring patterns only on `function`/`component` declarations, not on arrow expressions.
});

// A non-atomic head under postfix `.`/`[`/`(` — or an object-literal arrow body — must be parenthesized
// or reprinting would rebind the operators (`(a + b).c` → `a + b.c`) or reparse `=>{…}` as a statement block.
describe('printExpr — operand/callee/body parenthesization round-trips', () => {
  it('arrow returning an object literal', () => { exprRoundTrips('(x) => ({ id: x, n: x + 1 })'); });
  it('member on a parenthesized binary', () => { exprRoundTrips('(a + b).c'); });
  it('member on a unary', () => { exprRoundTrips('(-x).y'); });
  it('member on a ternary', () => { exprRoundTrips('(a ? b : c).d'); });
  it('index on a parenthesized binary', () => { exprRoundTrips('(a + b)[0]'); });
  it('IIFE — call of a parenthesized arrow', () => { exprRoundTrips('(() => 1)()'); });
});

/** The full conservation law: parse → print → parse again yields a structurally identical Program. */
function roundTrips(src: string): void {
  const first = parseProgram(src);
  expect(first.diagnostics).toEqual([]);
  const second = parseProgram(printProgram(first.program));
  expect(second.diagnostics).toEqual([]);
  expect(stripSpans(second.program)).toEqual(stripSpans(first.program));
}

describe('printProgram — statements round-trip', () => {
  it('const / let', () => { roundTrips('const x = 1\nlet y = 2'); });
  it('assign', () => { roundTrips('component Story() {\n  let n = 0\n  n = n + 1\n}'); });
  it('function declaration', () => { roundTrips('function add(a, b) {\n  a + b\n}'); });
  it('component with reactive let + child calls', () => {
    roundTrips('component Story() {\n  let n = 0\n  box(n)\n}');
  });
  it('if / else', () => { roundTrips('component Story() {\n  if (n < 3) {\n    text("a")\n  } else {\n    text("b")\n  }\n}'); });
  it('for-of', () => { roundTrips('component Story() {\n  for (const x of xs) {\n    box(x)\n  }\n}'); });
  it('while', () => { roundTrips('component Story() {\n  let i = 0\n  while (i < 3) {\n    i = i + 1\n  }\n}'); });
  it('return with and without value', () => { roundTrips('function f() {\n  if (x) {\n    return\n  }\n  return y\n}'); });
  it('wrapping element with a child block', () => { roundTrips('component Story() {\n  layout(1) {\n    box(2)\n  }\n}'); });
  it('head { } wrap shorthand reprints as a call with block', () => {
    roundTrips('component Story() {\n  group {\n    box(1)\n  }\n}');
  });
  it('destructuring params on a function declaration', () => { roundTrips('function f({ a, b }) {\n  a\n}'); });
  it('array-destructuring params on a function declaration', () => { roundTrips('function g([x, y]) {\n  x\n}'); });
  it('arrow with a statement block body', () => { roundTrips('const f = (h) => {\n  x = h\n}'); });
  it('else-if chain', () => { roundTrips('component Story() {\n  if (a) {\n    text("a")\n  } else if (b) {\n    text("b")\n  } else {\n    text("c")\n  }\n}'); });
  it('map with object-returning arrow', () => { roundTrips('component Story() {\n  const rows = map(items, (x) => ({ id: x }))\n}'); });
});

// A corpus of representative programs. Each must survive parse → print → parse structurally. This is
// the load-bearing conservation proof.
const CORPUS: string[] = [
  'const greeting = "hello \\"world\\"\\n"',
  'function double(n) {\n  n * 2\n}',
  'component Story() {\n  let count = 0\n  layout(1) {\n    button({ onClick: () => { count = count + 1 } }, "+")\n    span(count)\n  }\n}',
  'component List(items) {\n  for (const it of items) {\n    row({ key: it.id }, it.label)\n  }\n}',
  'const config = { a: 1, nested: { b: [1, 2, ...more] }, ...defaults }',
  'const pick = cond ? left : right',
  'component Story() {\n  if (a && b || !c) {\n    text("yes")\n  }\n}',
];

describe('printProgram — corpus conservation law', () => {
  for (const src of CORPUS) {
    it(`round-trips: ${src.slice(0, 40).replace(/\n/g, ' ')}…`, () => { roundTrips(src); });
  }
});

// A key that is not a bare identifier (hyphenated / digit-leading) must be quoted, or reprinting would
// re-lex it as several tokens and break the conservation law — realistic for aria-*/data-* attribute keys.
describe('printExpr — non-identifier object keys round-trip', () => {
  it('object with a hyphenated (non-identifier) key', () => { exprRoundTrips('{ "aria-label": 1, "data-x": y }'); });
  it('object with a digit-leading quoted key', () => { exprRoundTrips('{ "123abc": 1 }'); });
});

// String(n) prints large/small magnitudes in exponent form (1e21 / 1e-7), which the lexer splits into
// several tokens; printNumber must emit a plain decimal that round-trips.
describe('printExpr — numbers round-trip as plain decimals (never exponent form)', () => {
  it('a plain integer', () => { exprRoundTrips('1000000'); });
  it('a small fraction that String() would render as 1e-7', () => { exprRoundTrips('0.0000001'); });
  it('a large integer that String() would render as 1e+21', () => { exprRoundTrips('1000000000000000000000'); });
  it('ordinary decimals and zero', () => { exprRoundTrips('3.14'); exprRoundTrips('0'); });
  // Decimal-point shifting (not toFixed) preserves the shortest round-trip digits exactly, so even
  // high-significant-digit sub-1e-6 fractions and very small magnitudes round-trip without re-rounding.
  it('a high-significant-digit sub-1e-6 fraction', () => { exprRoundTrips('0.0000001234567890123456'); });
  it('a very small magnitude (would be 5e-21)', () => { exprRoundTrips('0.000000000000000000005'); });
  it('a very large integer (would be 1e300)', () => { exprRoundTrips('1' + '0'.repeat(300)); });
});

// An empty object-destructuring param must print `{}` (not `{  }`) and round-trip.
describe('printProgram — empty destructuring pattern', () => {
  it('empty object-pattern param on a function declaration', () => { roundTrips('function f({}) {\n  1\n}'); });
});

// Adversarial-review-caught round-trip breakers, fixed at root + locked here.
describe('printExpr — access-head + member-property + arrow-param edge cases round-trip', () => {
  // A number head under `.member` MUST be parenthesized: the lexer greedily eats the `.` into the
  // number token, so a bare `1.foo` re-lexes as the number `1.` then `foo`. (index/call/binary positions
  // don't have this hazard, so a number stays bare there — asserted below.)
  it('member access on a number literal head', () => { exprRoundTrips('(1).foo'); });
  it('member access on a float literal head', () => { exprRoundTrips('(1.5).foo'); });
  it('member-then-index chain on a number head', () => { exprRoundTrips('(1).foo[0]'); });
  it('a number in index / call / binary position stays bare (no over-parenthesizing)', () => {
    expect(printExpr(parseExpr('a[1]').expr)).toBe('a[1]');
    expect(printExpr(parseExpr('f(1)').expr)).toBe('f(1)');
    expect(printExpr(parseExpr('1 + 2').expr)).toBe('1 + 2');
  });
  // A non-identifier member property is emitted as a quoted dot-access `a."x-y"` (which the parser reads
  // to the same `member` node), NOT bare `a.x-y` (re-lexes as subtraction) nor `a["x-y"]` (a different
  // node kind, `index`).
  it('member access with a hyphenated (non-identifier) property', () => { exprRoundTrips('a."x-y"'); });
  it('member access with a whitespace property', () => { exprRoundTrips('a."fo o"'); });
});

// A `function` EXPRESSION with destructuring params parses to an `arrow` node whose params are patterns;
// arrow SYNTAX can't reparse patterns, so it must print back as a function expression to round-trip.
describe('printProgram — function-expression with destructuring params round-trips', () => {
  it('object-destructuring params', () => { roundTrips('const f = function({ a, b }) {\n  a\n}'); });
  it('array-destructuring params', () => { roundTrips('const g = function([x, y]) {\n  x\n}'); });
  it('a name-only function expression still prints as a concise arrow', () => {
    expect(printExpr(parseExpr('function(a, b) { a }').expr)).toBe('(a, b) => {\n  a\n}');
  });
});

// The printer must be a TOTAL function: the parser accepts unbounded left-spine chains (its depth guard
// counts only nested-expression recursion, not the iterative postfix/binary loops), so a valid AST can be
// arbitrarily deep. The printer bounds its own recursion and fails closed with PrintDepthError instead of
// letting a raw stack-overflow RangeError escape into the host.
describe('printer totality — a pathologically deep AST fails closed, never a raw stack overflow', () => {
  it('a deep member chain throws the typed PrintDepthError from printExpr and stripSpans', () => {
    const deep = parseExpr('a' + '.b'.repeat(6000));
    expect(deep.diagnostics).toEqual([]);   // the parser accepts it (unbounded postfix loop)
    expect(() => printExpr(deep.expr)).toThrow(PrintDepthError);
    expect(() => stripSpans(deep.expr)).toThrow(PrintDepthError);
  });
  it('a realistically-deep chain still prints fine (the cap is far above real nesting)', () => {
    exprRoundTrips('a' + '.b'.repeat(100));
  });
});

// The printer is "canonical": a block nested inside an expression (a wrapping-element child block) indents
// by its depth rather than resetting to column 0. Whitespace is insignificant to the lexer (so every case
// above round-trips regardless), but the exact indentation is a legibility property worth locking.
describe('printProgram — nested-block indentation is canonical', () => {
  it('a block nested inside an expression indents by depth, not column 0', () => {
    const src = 'component Story() {\n  layout(1) {\n    box(2)\n  }\n}';
    expect(printProgram(parseProgram(src).program)).toBe(src);
  });
});

describe('print public exports', () => {
  it('printProgram/printExpr/printStmt/printString/stripSpans are exported from the package barrel', async () => {
    const barrel = await import('./index.ts');
    expect(typeof barrel.printProgram).toBe('function');
    expect(typeof barrel.printExpr).toBe('function');
    expect(typeof barrel.printStmt).toBe('function');
    expect(typeof barrel.printString).toBe('function');
    expect(typeof barrel.stripSpans).toBe('function');
  });
});
