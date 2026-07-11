import { lex, type Token, type TokenType } from './lexer.ts';
import type { Diagnostic } from './diagnostics.ts';
import { makeDiagnostic } from './diagnostics.ts';
import type { Expr, Stmt, BinOp, Pattern, Program } from './ast.ts';   // Stmt: the statement layer (decls + control flow) + arrow block bodies
import { FORBIDDEN_KEYS } from './ast.ts';

export interface ParseExprResult { readonly expr: Expr; readonly diagnostics: Diagnostic[] }

// Recursion-depth cap for the recursive-descent parser. Deeply-nested source (e.g. thousands of
// `(` or a long unclosed chain) would otherwise overflow the JS call stack with an uncaught
// RangeError. The parser is a TOTAL function (like the lexer/evaluator): past the cap it emits a
// ML-LANG-PARSE diagnostic and fails closed, so the public parseProgram/parseExpr never throw.
export const MAX_PARSE_DEPTH = 512;
/** Thrown internally when the nesting cap trips; caught by the public entrypoints → diagnostic. */
class ParseDepthSignal extends Error {}

// Precedence tiers (low→high). Each entry: matching token types → BinOp.
const BINARY_TIERS: Array<Partial<Record<TokenType, BinOp>>> = [
  { or: '||' }, { and: '&&' },
  { eq: '==', neq: '!=', lt: '<', le: '<=', gt: '>', ge: '>=' },
  { plus: '+', minus: '-' }, { star: '*', slash: '/', percent: '%' },
];

export class Parser {
  protected toks: Token[];
  protected pos = 0;
  private depth = 0;
  readonly diagnostics: Diagnostic[] = [];
  constructor(src: string) { const r = lex(src); this.toks = r.tokens; this.diagnostics.push(...r.diagnostics); }

  protected peek(): Token { return this.toks[this.pos]!; }
  /** Return the current token and advance — but NEVER past the trailing `eof` (lex always appends
   *  exactly one). Clamping the cursor at the last token guarantees `peek()` always reads a real
   *  token, so error-recovery paths (e.g. `parsePrimary` skipping a bad trailing token) can never
   *  read `undefined.type` and throw. Malformed input yields diagnostics, never a TypeError. */
  protected next(): Token { const t = this.peek(); if (this.pos < this.toks.length - 1) this.pos++; return t; }
  protected eat(type: TokenType): Token | undefined {
    if (this.peek().type === type) return this.next();
    this.diagnostics.push(makeDiagnostic('ML-LANG-PARSE', `expected ${type}, got ${this.peek().type}`, this.peek().span));
    return undefined;
  }

  parseExpression(): Expr {
    // Depth guard: `parseExpression` is the entry for every nested-expression recursion (parens,
    // args, ternary branches, arrow bodies). Trip the cap here so pathological nesting fails closed
    // with a diagnostic instead of overflowing the stack. The public entrypoints catch the signal.
    if (++this.depth > MAX_PARSE_DEPTH) {
      this.diagnostics.push(makeDiagnostic('ML-LANG-PARSE', 'expression nesting too deep', this.peek().span));
      throw new ParseDepthSignal();
    }
    try { return this.parseCond(); } finally { this.depth--; }
  }

  /** Ternary `test ? then : else` — lower precedence than every binary op, right-associative
   *  (`a ? b : c ? d : e` = `a ? b : (c ? d : e)`). */
  private parseCond(): Expr {
    const test = this.parseBinary(0);
    if (this.peek().type !== 'question') return test;
    const span = this.peek().span; this.next();
    const then = this.parseExpression();   // middle is a full expression
    this.eat('colon');
    const els = this.parseExpression();     // right-assoc: recurse for chained ternaries
    return { kind: 'cond', test, then, else: els, span };
  }

  private parseBinary(tier: number): Expr {
    if (tier >= BINARY_TIERS.length) return this.parseUnary();
    let left = this.parseBinary(tier + 1);
    for (;;) {
      const op = BINARY_TIERS[tier]![this.peek().type];
      if (!op) return left;
      const span = this.peek().span; this.next();
      const right = this.parseBinary(tier + 1);
      left = { kind: 'binary', op, left, right, span };
    }
  }

  private parseUnary(): Expr {
    const t = this.peek();
    if (t.type === 'minus' || t.type === 'not') {
      this.next();
      return { kind: 'unary', op: t.type === 'minus' ? '-' : '!', operand: this.parseUnary(), span: t.span };
    }
    return this.parsePostfix();
  }

  private parsePostfix(): Expr {
    let e = this.parsePrimary();
    for (;;) {
      const t = this.peek();
      if (t.type === 'dot') {
        this.next(); const name = this.next();
        if (FORBIDDEN_KEYS.has(name.value)) this.diagnostics.push(makeDiagnostic('ML-LANG-FORBIDDEN', `forbidden property '${name.value}'`, name.span));
        e = { kind: 'member', object: e, property: name.value, span: t.span };
      } else if (t.type === 'lbracket') {
        this.next(); const index = this.parseExpression(); this.eat('rbracket');
        e = { kind: 'index', object: e, index, span: t.span };
      } else if (t.type === 'lparen') {
        this.next(); const args: Expr[] = [];
        while (this.peek().type !== 'rparen' && this.peek().type !== 'eof') {
          args.push(this.parseExpression());
          if (this.peek().type === 'comma') this.next(); else break;
        }
        this.eat('rparen');
        e = { kind: 'call', callee: e, args, span: t.span };
        // NOTE: a trailing `{}` child block or single trailing statement is attached by parseWrappable.
      } else return e;
    }
  }

  private parsePrimary(): Expr {
    const t = this.peek();
    switch (t.type) {
      case 'number': this.next(); return { kind: 'number', value: Number(t.value), span: t.span };
      case 'string': this.next(); return { kind: 'string', value: t.value, span: t.span };
      case 'true': this.next(); return { kind: 'bool', value: true, span: t.span };
      case 'false': this.next(); return { kind: 'bool', value: false, span: t.span };
      case 'null': this.next(); return { kind: 'null', span: t.span };
      case 'ident': {
        // arrow: ident => expr   OR   ident => { … block }
        if (this.toks[this.pos + 1]?.type === 'arrow') {
          this.next(); this.next();
          return { kind: 'arrow', params: [{ kind: 'name', name: t.value }], body: this.parseArrowBody(), span: t.span };
        }
        this.next(); return { kind: 'ident', name: t.value, span: t.span };
      }
      case 'function': {
        // Function EXPRESSION (e.g. an IIFE `(function make() { … })()`). Modeled as an `arrow`
        // with a Stmt[] block body (implicit-last-expr return) — no separate AST kind. The optional
        // name is parsed for JS-familiarity but discarded (self-reference is not a feature); a
        // `function` STATEMENT is handled by parseStatement, not here.
        this.next();
        if (this.peek().type === 'ident') this.next();
        const params = this.parseParams();
        const body = this.parseBlock();
        return { kind: 'arrow', params, body, span: t.span };
      }
      case 'lparen': return this.parseParenOrArrow();
      case 'lbrace': return this.parseObject();
      case 'lbracket': return this.parseArray();
      default:
        this.diagnostics.push(makeDiagnostic('ML-LANG-PARSE', `unexpected token '${t.type}'`, t.span));
        this.next();
        return { kind: 'null', span: t.span };
    }
  }

  private parseParenOrArrow(): Expr {
    const start = this.peek().span; this.next(); // (
    // arrow params: (a, b) =>
    const save = this.pos;
    const names: string[] = [];
    let looksLikeParams = true;
    while (this.peek().type !== 'rparen' && this.peek().type !== 'eof') {
      if (this.peek().type === 'ident') { names.push(this.next().value); if (this.peek().type === 'comma') this.next(); }
      else { looksLikeParams = false; break; }
    }
    if (looksLikeParams && this.peek().type === 'rparen' && this.toks[this.pos + 1]?.type === 'arrow') {
      this.next(); this.next(); // ) =>
      return { kind: 'arrow', params: names.map((n) => ({ kind: 'name', name: n })), body: this.parseArrowBody(), span: start };
    }
    this.pos = save;
    const e = this.parseExpression(); this.eat('rparen'); return e;
  }

  /** Arrow body after `=>` (`Arrow ::= … "=>" (Expr | Block)`). JS-identical rule: a leading `{`
   *  is a STATEMENT BLOCK (returns Stmt[], so a state-mutating `hover = h` assignment is legal),
   *  NOT an object literal — to return an object from an arrow, wrap it: `() => ({ … })`. Any other
   *  token is a single expression body. */
  private parseArrowBody(): Expr | Stmt[] {
    if (this.peek().type === 'lbrace') return this.parseBlock();
    return this.parseExpression();
  }

  private parseObject(): Expr {
    const start = this.peek().span; this.next(); // {
    const entries: { key: string; value: Expr }[] = [];
    while (this.peek().type !== 'rbrace' && this.peek().type !== 'eof') {
      const key = this.next().value;
      if (FORBIDDEN_KEYS.has(key)) this.diagnostics.push(makeDiagnostic('ML-LANG-FORBIDDEN', `forbidden key '${key}'`, this.peek().span));
      this.eat('colon');
      entries.push({ key, value: this.parseExpression() });
      if (this.peek().type === 'comma') this.next(); else break;
    }
    this.eat('rbrace');
    return { kind: 'object', entries, span: start };
  }

  private parseArray(): Expr {
    const start = this.peek().span; this.next(); // [
    const elements: Expr[] = [];
    while (this.peek().type !== 'rbracket' && this.peek().type !== 'eof') {
      elements.push(this.parseExpression());
      if (this.peek().type === 'comma') this.next(); else break;
    }
    this.eat('rbracket');
    return { kind: 'array', elements, span: start };
  }

  // --- statement layer ---
  parseProgramBody(): Stmt[] {
    const stmts: Stmt[] = [];
    const declared = new Set<string>();
    while (this.peek().type !== 'eof') {
      const s = this.parseStatement();
      if (s) {
        if ((s.kind === 'const' || s.kind === 'let' || s.kind === 'function' || s.kind === 'component')) {
          const name = (s as { name: string }).name;
          if (declared.has(name)) this.diagnostics.push(makeDiagnostic('ML-LANG-REDECL', `'${name}' already declared in this scope`, s.span));
          declared.add(name);
        }
        stmts.push(s);
      } else break;
    }
    return stmts;
  }

  private parseBlock(): Stmt[] {
    this.eat('lbrace');
    const stmts: Stmt[] = [];
    while (this.peek().type !== 'rbrace' && this.peek().type !== 'eof') {
      const s = this.parseStatement(); if (s) stmts.push(s); else break;
    }
    this.eat('rbrace');
    return stmts;
  }

  /** A control-flow body (`Body ::= Block | Stmt`): a braced block, OR a single brace-less
   *  statement (JS-like `for (…) KPI(k)` / `if (…) foo()`). Returns a Stmt[] either way so the
   *  AST for/if/while bodies stay uniform. */
  private parseBody(): Stmt[] {
    if (this.peek().type === 'lbrace') return this.parseBlock();
    const s = this.parseStatement();
    return s ? [s] : [];
  }

  private parsePattern(): Pattern {
    if (this.peek().type === 'lbrace') {
      this.next(); const fields: string[] = [];
      while (this.peek().type !== 'rbrace' && this.peek().type !== 'eof') { fields.push(this.next().value); if (this.peek().type === 'comma') this.next(); else break; }
      this.eat('rbrace'); return { kind: 'objectPattern', fields };
    }
    if (this.peek().type === 'lbracket') {
      this.next(); const elements: string[] = [];
      while (this.peek().type !== 'rbracket' && this.peek().type !== 'eof') { elements.push(this.next().value); if (this.peek().type === 'comma') this.next(); else break; }
      this.eat('rbracket'); return { kind: 'arrayPattern', elements };
    }
    return { kind: 'name', name: this.next().value };
  }

  private parseParams(): Pattern[] {
    this.eat('lparen'); const params: Pattern[] = [];
    while (this.peek().type !== 'rparen' && this.peek().type !== 'eof') { params.push(this.parsePattern()); if (this.peek().type === 'comma') this.next(); else break; }
    this.eat('rparen'); return params;
  }

  private parseStatement(): Stmt | undefined {
    const t = this.peek();
    switch (t.type) {
      case 'const': case 'let': {
        this.next(); const name = this.eat('ident')?.value ?? ''; this.eat('assign');
        const init = this.parseExpression(); if (this.peek().type === 'semi') this.next();
        return { kind: t.type, name, init, span: t.span };
      }
      case 'function': case 'component': {
        this.next(); const name = this.eat('ident')?.value ?? ''; const params = this.parseParams(); const body = this.parseBlock();
        return { kind: t.type, name, params, body, span: t.span };   // token type === Stmt kind for both
      }
      case 'if': {
        this.next(); this.eat('lparen'); const test = this.parseExpression(); this.eat('rparen');
        const then = this.parseBody();   // Body ::= Block | Stmt
        let elseBody: Stmt[] | undefined;
        if (this.peek().type === 'else') { this.next(); elseBody = this.peek().type === 'if' ? [this.parseStatement()!] : this.parseBody(); }
        return { kind: 'if', test, then, else: elseBody, span: t.span };
      }
      case 'for': {
        this.next(); this.eat('lparen'); this.eat('const'); const binding = this.eat('ident')?.value ?? ''; this.eat('of');
        const iterable = this.parseExpression(); this.eat('rparen'); const body = this.parseBody();   // brace-less single stmt OK
        return { kind: 'for', binding, iterable, body, span: t.span };
      }
      case 'while': {
        this.next(); this.eat('lparen'); const test = this.parseExpression(); this.eat('rparen'); const body = this.parseBody();
        return { kind: 'while', test, body, span: t.span };
      }
      case 'return': {
        // A value-less `return` is terminated by `;`, `}` (end of block), or eof — none of which start
        // an expression. Only parse a return value when a real expression token follows.
        this.next();
        const tt = this.peek().type;
        const value = (tt === 'semi' || tt === 'rbrace' || tt === 'eof') ? undefined : this.parseExpression();
        if (this.peek().type === 'semi') this.next();
        return { kind: 'return', value, span: t.span };
      }
      case 'ident': {
        // expression / assignment / wrapping element (no root-block production — the root is an entry component)
        const expr = this.parseWrappable();
        if (this.peek().type === 'assign') { this.next(); const value = this.parseExpression(); if (this.peek().type === 'semi') this.next(); return { kind: 'assign', target: expr, value, span: t.span }; }
        if (this.peek().type === 'semi') this.next();
        return { kind: 'expr', expr, span: t.span };
      }
      default: {
        const expr = this.parseWrappable(); if (this.peek().type === 'semi') this.next(); return { kind: 'expr', expr, span: t.span };
      }
    }
  }

  /** A call expression optionally followed by a `{}` child block OR a single trailing statement
   *  (the uniform wrapping rule). Attaches `block` to the outermost call node.
   *  GUARD: the brace-less single-trailing-statement wrap fires ONLY when the next token is on
   *  the SAME logical line (no newline, no `;`). A newline/`;` between `)` and the next call makes
   *  them siblings — otherwise `KPI(a)⏎KPI(b)` would mis-nest b inside a. A `{` block always wraps. */
  private parseWrappable(): Expr {
    const e = this.parseExpression();
    if (e.kind === 'call') {
      if (this.peek().type === 'lbrace') { e.block = this.parseBlock(); }
      else if (this.startsStatement() && !this.peek().newlineBefore) { e.block = [this.parseStatement()!]; }
    }
    return e;
  }

  private startsStatement(): boolean {
    const t = this.peek().type;
    return t === 'ident' || t === 'if' || t === 'for' || t === 'while';
  }
}

/** Convert a fail-closed parse abort into a diagnostic. The `parseExpression` depth guard trips first
 *  on paren/arg/ternary nesting (ParseDepthSignal); a RangeError is the belt-and-suspenders fallback
 *  for any OTHER unguarded recursion path (e.g. `parseUnary` on `!`×N), so the PUBLIC parser is TOTAL
 *  — it never throws a stack overflow into the host (mirrors the lexer/evaluator never-throw contract). */
function isParseAbort(e: unknown): boolean {
  return e instanceof ParseDepthSignal || e instanceof RangeError;
}

export function parseExpr(src: string): ParseExprResult {
  const p = new Parser(src);
  try {
    const expr = p.parseExpression();
    return { expr, diagnostics: p.diagnostics };
  } catch (e) {
    if (!isParseAbort(e)) throw e;
    p.diagnostics.push(makeDiagnostic('ML-LANG-PARSE', 'expression nesting too deep'));
    return { expr: { kind: 'null', span: { start: 0, end: 0 } }, diagnostics: p.diagnostics };
  }
}

export interface ParseProgramResult { readonly program: Program; readonly diagnostics: Diagnostic[] }
export function parseProgram(src: string): ParseProgramResult {
  const p = new Parser(src);
  try {
    const stmts = p.parseProgramBody();
    return { program: { stmts }, diagnostics: p.diagnostics };
  } catch (e) {
    if (!isParseAbort(e)) throw e;
    p.diagnostics.push(makeDiagnostic('ML-LANG-PARSE', 'source nesting too deep'));
    return { program: { stmts: [] }, diagnostics: p.diagnostics };
  }
}
