// A TSL-style JS builder for metael kernel expressions. Chainable methods author the SAME expression
// AST the parser emits, so builder-authored kernels inherit the existing gate/emit/oracle/dispatch
// path unchanged. Imports ONLY the AST types from @metael/lang — no runtime, no evaluator.
import type { Expr, BinOp } from '@metael/lang';

/** The zero span every builder node carries. The builder has no source text, so there is nothing to
 *  point diagnostics at; equivalence with parser output is compared via `stripSpans`, which drops the
 *  `span` field, so the exact value never affects a compare. */
const ZERO_SPAN = { start: 0, end: 0 };

/**
 * A builder wrapper over a metael AST expression node.
 *
 * Chainable methods build `binary` / `index` / `member` nodes structurally identical to what the parser
 * emits — JS has no operator overloading, so arithmetic (`add`/`sub`/`mul`/`div`/`mod`) and comparison
 * (`lt`/`le`/`gt`/`ge`/`eq`/`ne`) are methods. Each takes a `KNode` or a bare `number` (coerced to a
 * numeric literal) and returns a fresh `KNode`, so chains build left-associatively — matching the
 * parser's left-to-right nesting.
 */
export class KNode {
  /** Wrap an already-built expression node. Prefer the factory helpers ({@link lit}/{@link param}/
   *  {@link call}) or the chainable methods over calling this directly.
   *  @param expr The wrapped expression node — the AST this builder node has authored so far. */
  constructor(readonly expr: Expr) {}

  /** Build a left-associative `binary` node with `this` on the left and `o` (coerced) on the right. */
  private bin(op: BinOp, o: KNode | number): KNode {
    return new KNode({ kind: 'binary', op, left: this.expr, right: toExpr(o), span: ZERO_SPAN });
  }

  /** Addition, `this + o`. */
  add(o: KNode | number): KNode {
    return this.bin('+', o);
  }
  /** Subtraction, `this - o`. */
  sub(o: KNode | number): KNode {
    return this.bin('-', o);
  }
  /** Multiplication, `this * o`. */
  mul(o: KNode | number): KNode {
    return this.bin('*', o);
  }
  /** Division, `this / o`. */
  div(o: KNode | number): KNode {
    return this.bin('/', o);
  }
  /** Remainder, `this % o`. */
  mod(o: KNode | number): KNode {
    return this.bin('%', o);
  }
  /** Less-than, `this < o`. */
  lt(o: KNode | number): KNode {
    return this.bin('<', o);
  }
  /** Less-than-or-equal, `this <= o`. */
  le(o: KNode | number): KNode {
    return this.bin('<=', o);
  }
  /** Greater-than, `this > o`. */
  gt(o: KNode | number): KNode {
    return this.bin('>', o);
  }
  /** Greater-than-or-equal, `this >= o`. */
  ge(o: KNode | number): KNode {
    return this.bin('>=', o);
  }
  /** Equality, `this == o`. */
  eq(o: KNode | number): KNode {
    return this.bin('==', o);
  }
  /** Inequality, `this != o`. */
  ne(o: KNode | number): KNode {
    return this.bin('!=', o);
  }

  /**
   * Chained computed index access: `a.at(i, j)` builds `a[i][j]`.
   *
   * Reduces the index list left-to-right, wrapping each step in an `index` node (the AST field is
   * `object`, not `target`) so the nesting matches the parser's left-associative index chaining.
   */
  at(...idx: (KNode | number)[]): KNode {
    return idx.reduce<KNode>(
      (acc, i) => new KNode({ kind: 'index', object: acc.expr, index: toExpr(i), span: ZERO_SPAN }),
      this,
    );
  }

  /** Static member access, `this.property` (the AST fields are `object` + `property`). */
  member(property: string): KNode {
    return new KNode({ kind: 'member', object: this.expr, property, span: ZERO_SPAN });
  }
}

/** Coerce a builder value into an `Expr`: a bare `number` becomes a numeric-literal node; a `KNode`
 *  yields its wrapped expression. Lets chainable ops accept `x.add(1)` as well as `x.add(y)`. */
export function toExpr(x: KNode | number): Expr {
  return typeof x === 'number' ? { kind: 'number', value: x, span: ZERO_SPAN } : x.expr;
}

/** A numeric-literal node, e.g. `lit(1)` → `1`. */
export function lit(n: number): KNode {
  return new KNode({ kind: 'number', value: n, span: ZERO_SPAN });
}

/** A bound-identifier reference — a kernel param or a `let` binding, e.g. `param('x')` → `x`. */
export function param(name: string): KNode {
  return new KNode({ kind: 'ident', name, span: ZERO_SPAN });
}

/**
 * A call against a head name: `call('dot', a, b)` → `dot(a, b)`.
 *
 * The callee is an `ident` naming the head (a builtin like `f32` / `vec3` / `dot`, or a component);
 * arguments are coerced via {@link toExpr}. A plain kernel call carries no wrapping `block`, so the
 * field is omitted entirely (rather than set to `undefined`) to match the parser's node exactly.
 */
export function call(head: string, ...args: (KNode | number)[]): KNode {
  return new KNode({
    kind: 'call',
    callee: { kind: 'ident', name: head, span: ZERO_SPAN },
    args: args.map(toExpr),
    span: ZERO_SPAN,
  });
}
