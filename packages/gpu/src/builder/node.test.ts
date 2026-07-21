import { describe, it, expect } from 'vitest';
import { lit, param, call, type KNode } from './node.ts';
import { stripSpans, parseExpr } from '@metael/lang';

// Compare a builder-produced Expr to the parser's, spans normalized out.
const eq = (k: KNode, src: string) =>
  expect(stripSpans(k.expr)).toEqual(stripSpans(parseExpr(src).expr));

describe('KNode expression builder', () => {
  it('lit + add build the same binary AST as the parser', () => {
    eq(param('x').add(lit(1)), 'x + 1');
  });
  it('chained arithmetic nests like the parser (left-assoc)', () => {
    eq(param('a').mul(param('b')).add(param('c')), 'a * b + c');
  });
  it('a builtin call matches a parsed call', () => {
    eq(call('dot', param('a'), param('b')), 'dot(a, b)');
  });
  it('index (.at) matches parsed index chaining', () => {
    eq(param('a').at(param('i'), param('j')), 'a[i][j]');
  });
  it('every op method emits the parser operator (no silent op-string drift)', () => {
    const a = param('a'),
      b = param('b');
    eq(a.sub(b), 'a - b');
    eq(a.div(b), 'a / b');
    eq(a.mod(b), 'a % b');
    eq(a.lt(b), 'a < b');
    eq(a.le(b), 'a <= b');
    eq(a.gt(b), 'a > b');
    eq(a.ge(b), 'a >= b');
    eq(a.eq(b), 'a == b');
    eq(a.ne(b), 'a != b');
    eq(a.add(b), 'a + b');
    eq(a.mul(b), 'a * b');
  });
  it('member() emits a parsed member-access node', () => {
    eq(param('r').member('value'), 'r.value');
  });
});
