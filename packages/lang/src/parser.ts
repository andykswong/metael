import { lex, type Token, type TokenType } from './lexer.ts';
import type { Diagnostic, SourceSpan } from './diagnostics.ts';
import { makeDiagnostic } from './diagnostics.ts';
import type { Expr, Stmt, BinOp, Pattern, Program, ArrayElement, ObjectEntry } from './ast.ts';   // Stmt: the statement layer (decls + control flow) + arrow block bodies
import { FORBIDDEN_KEYS } from './ast.ts';

/** The outcome of {@link parseExpr}: the parsed expression and every diagnostic collected while lexing
 *  and parsing it, in order. */
export interface ParseExprResult {
  /** The parsed expression node. Best-effort on error — a `null` literal node when the source could
   *  not be parsed as an expression at all. */
  readonly expr: Expr;
  /** Every diagnostic collected while lexing and parsing, in order. Empty on a clean parse. */
  readonly diagnostics: Diagnostic[];
}

/** Recursion-depth cap for the recursive-descent parser. Deeply-nested source (e.g. thousands of `(`
 *  or a long unclosed chain) would otherwise overflow the JS call stack with an uncaught `RangeError`.
 *  The parser is a TOTAL function: past the cap it emits an `ML-LANG-PARSE` diagnostic and fails
 *  closed, so the public {@link parseProgram} / {@link parseExpr} never throw. */
export const MAX_PARSE_DEPTH = 512;
/** Thrown internally when the nesting cap trips; caught by the public entrypoints → diagnostic. */
class ParseDepthSignal extends Error {}

// Reserved-word token types that are still legal as a `.member` property name (JS allows `x.if`,
// `x.const`, etc.). A property after `.` must be one of these, an `ident`, or a quoted `string`.
const KEYWORD_TOKENS: ReadonlySet<TokenType> = new Set<TokenType>([
  'component', 'function', 'const', 'let', 'if', 'else', 'for', 'of', 'while', 'return', 'true', 'false', 'null',
]);

// Precedence tiers (low→high). Each entry: matching token types → BinOp.
const BINARY_TIERS: Array<Partial<Record<TokenType, BinOp>>> = [
  { or: '||' }, { and: '&&' },
  { eq: '==', neq: '!=' },                                 // equality — looser than relational (JS-conformant)
  { lt: '<', le: '<=', gt: '>', ge: '>=' },               // relational — binds tighter than == / != so `a == b > c` = `a == (b > c)`
  { plus: '+', minus: '-' }, { star: '*', slash: '/', percent: '%' },
];

/**
 * The recursive-descent parser for the language surface: it lexes the source, then parses tokens into
 * the {@link Expr} / {@link Stmt} AST.
 *
 * A TOTAL, non-throwing function of its input like the lexer and evaluator: malformed source yields
 * {@link Diagnostic}s (never an exception) accumulated in {@link diagnostics}, and pathological nesting
 * fails closed via {@link MAX_PARSE_DEPTH} rather than overflowing the JS stack. Prefer the free
 * functions {@link parseExpr} / {@link parseProgram} for one-shot parses; the class is exposed for
 * callers that need incremental access to its methods or wish to subclass it.
 *
 * @remarks Newline handling is context-sensitive: at statement/block level a newline before a postfix
 * `(` / `[` or a wrap `{` starts a fresh statement, while inside an open grouping it never breaks — see
 * {@link parseExpression} and the wrap rules.
 */
export class Parser {
  /** The token stream produced by lexing the source, terminated by exactly one trailing `eof`. */
  protected toks: Token[];
  /** The cursor: the index into {@link toks} of the next token to read. Clamped at the last token so a
   *  read never runs past `eof`. */
  protected pos = 0;
  private depth = 0;
  // Open-grouping nesting depth: >0 while parsing INSIDE a `( … )` / `[ … ]` / `{ … }` / ternary
  // branch / arrow body — any context where a newline is NOT a statement boundary. The newline guards
  // on postfix `(` / `[` (and the wrap rule) fire ONLY at `groupDepth === 0` (statement/block level),
  // so a line-leading `(` / `[` still continues a postfix chain inside a grouping (`f(g(x)⏎(y))`).
  private groupDepth = 0;
  // Expr nodes written inside grouping parens `( … )`. A parenthesized arrow is a legitimate value
  // (`handler || (() => x)`), so it is NOT flagged as a bare binary operand; only an un-parenthesized
  // arrow (`1 + x => 2`) is. Tracked by identity so the AST stays clean (no `parenthesized` field).
  private parenthesized = new WeakSet<object>();
  /** Every diagnostic collected while lexing and parsing, in order. Seeded with the lexer's diagnostics
   *  by the constructor, then appended to as parsing proceeds. */
  readonly diagnostics: Diagnostic[] = [];
  /** Lex `src` and prime the parser: the resulting tokens become {@link toks} and the lexer's
   *  diagnostics seed {@link diagnostics}.
   *  @param src - the source text to lex and parse. */
  constructor(src: string) { const r = lex(src); this.toks = r.tokens; this.diagnostics.push(...r.diagnostics); }

  /** Return the current token WITHOUT advancing the cursor. Always a real token (never `undefined`) —
   *  the cursor is clamped at the trailing `eof`. */
  protected peek(): Token { return this.toks[this.pos]!; }
  /** Return the current token and advance — but NEVER past the trailing `eof` (lex always appends
   *  exactly one). Clamping the cursor at the last token guarantees `peek()` always reads a real
   *  token, so error-recovery paths (e.g. `parsePrimary` skipping a bad trailing token) can never
   *  read `undefined.type` and throw. Malformed input yields diagnostics, never a TypeError. */
  protected next(): Token { const t = this.peek(); if (this.pos < this.toks.length - 1) this.pos++; return t; }
  /** Consume the current token if it matches `type` and return it; otherwise leave the cursor put, push
   *  an `ML-LANG-PARSE` diagnostic, and return `undefined`. The fail-soft counterpart to {@link next}
   *  used wherever a specific delimiter/keyword is required.
   *  @param type - the token type the current token must have.
   *  @returns the consumed token, or `undefined` if the current token did not match. */
  protected eat(type: TokenType): Token | undefined {
    if (this.peek().type === type) return this.next();
    this.diagnostics.push(makeDiagnostic('ML-LANG-PARSE', `expected ${type}, got ${this.peek().type}`, this.peek().span));
    return undefined;
  }

  /** Does a newline before `t` end the current statement? Only at STATEMENT/BLOCK level (`groupDepth
   *  === 0`): a newline before a postfix `(` / `[` there starts a fresh statement. Inside an open
   *  `( … )` / `[ … ]` / ternary branch / arrow body there is no statement boundary, so a leading
   *  newline never breaks — a postfix chain (e.g. a curried call `f(g(x)⏎(y))`) continues. */
  private newlineBreaks(t: Token): boolean { return t.newlineBefore && this.groupDepth === 0; }

  /** Emit ONE named diagnostic for an unsupported-but-natural-JS construct at STATEMENT level, then
   *  skip forward to the next statement boundary so the rest of the source still parses cleanly —
   *  instead of letting a later `eat()` mismatch cascade into a pile of misleading diagnostics +
   *  phantom statements. "Boundary" = a `;` (consumed), a statement-boundary newline, a `}` that would
   *  close the enclosing block (not consumed), or `eof`. Any `([{` opened while skipping is balanced so
   *  we don't stop on a nested delimiter. Fails closed: bounded by eof, so it always terminates. */
  private recoverToStatementBoundary(message: string, span: SourceSpan): void {
    this.diagnostics.push(makeDiagnostic('ML-LANG-PARSE', message, span));
    let depth = 0;
    for (;;) {
      const t = this.peek();
      if (t.type === 'eof') return;
      if (depth === 0) {
        if (t.type === 'semi') { this.next(); return; }               // consume the terminator
        if (t.type === 'rbrace') return;                              // let the enclosing block close it
        if (t.newlineBefore) return;                                  // a statement-boundary newline
      }
      if (t.type === 'lparen' || t.type === 'lbracket' || t.type === 'lbrace') depth++;
      else if (t.type === 'rparen' || t.type === 'rbracket' || t.type === 'rbrace') { if (depth > 0) depth--; }
      this.next();
    }
  }

  /** Emit ONE named diagnostic for a malformed construct that occurs INSIDE an already-open grouping
   *  (a call-arg list, an `if`/`while` condition), then skip to — but do NOT consume — that grouping's
   *  matching close delimiter, so the caller's normal `eat(')')` closes it and parsing continues past
   *  the whole grouping. Balances any nested `([{` opened while skipping; stops at our grouping's close,
   *  or `eof`. Fails closed: bounded by eof. */
  private skipMalformedToGroupClose(message: string, span: SourceSpan): void {
    this.diagnostics.push(makeDiagnostic('ML-LANG-PARSE', message, span));
    let depth = 0;   // counts groupings opened AFTER this point; a close at depth 0 is OUR grouping's close
    for (;;) {
      const t = this.peek();
      if (t.type === 'eof') return;
      if (t.type === 'lparen' || t.type === 'lbracket' || t.type === 'lbrace') { depth++; this.next(); continue; }
      if (t.type === 'rparen' || t.type === 'rbracket' || t.type === 'rbrace') {
        if (depth === 0) return;   // our grouping's close — leave it for the caller
        depth--; this.next(); continue;
      }
      this.next();
    }
  }

  /** Parse a single expression (the full precedence ladder down through the ternary, binary tiers,
   *  unary, postfix, and primary forms) and return its AST node.
   *  @returns the parsed {@link Expr}.
   *  @remarks The entry point for every nested-expression recursion (parens, call args, ternary
   *  branches, arrow bodies), so the {@link MAX_PARSE_DEPTH} guard is enforced here: pathological
   *  nesting fails closed via an internally-thrown signal that the public {@link parseExpr} /
   *  {@link parseProgram} catch, rather than overflowing the JS stack. */
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
    // The THEN branch is bounded by the mandatory `:`, so a newline in it is not a statement boundary —
    // raise groupDepth across it. The ELSE branch has NO closing delimiter: at statement level it IS the
    // statement tail, so it must inherit the caller's groupDepth (0 → the postfix `(`/`[` guard still
    // fires, starting a fresh statement; >0 inside a real grouping → still chains). Hence decrement
    // BEFORE parsing `els`, not after — raising across the else would reopen the leading-paren footgun.
    this.groupDepth++;
    const then = this.parseExpression();   // middle is a full expression
    this.eat('colon');
    this.groupDepth--;
    const els = this.parseExpression();     // right-assoc: recurse for chained ternaries; inherits caller depth
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
      // A BARE arrow as a binary operand (`1 + x => 2`) is invalid — JS forbids it, and here it would
      // silently absorb the right side as `1 + (x => 2)`. Flag it. A PARENTHESIZED arrow (`1 + (x => 2)`,
      // the idiomatic default-callback `handler || (() => x)`) is a legitimate value — never flagged.
      if (right.kind === 'arrow' && !this.parenthesized.has(right)) this.diagnostics.push(makeDiagnostic('ML-LANG-PARSE', 'an arrow function cannot be an operand of a binary operator; wrap it in parentheses', right.span));
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
        // The property must be a NAME-like token: an identifier, a keyword (JS allows `x.if`), or a
        // quoted string (the printer's round-trip form `a."x-y"`). A number/operator/punctuation after
        // `.` is a fail-loud parse error, not a silent bogus member (`obj.5`, `obj.` → ML-LANG-PARSE).
        const nameLike = name.type === 'ident' || name.type === 'string' || KEYWORD_TOKENS.has(name.type);
        if (!nameLike) this.diagnostics.push(makeDiagnostic('ML-LANG-PARSE', `expected property name after '.', got ${name.type}`, name.span));
        if (FORBIDDEN_KEYS.has(name.value)) this.diagnostics.push(makeDiagnostic('ML-LANG-FORBIDDEN', `forbidden property '${name.value}'`, name.span));
        e = { kind: 'member', object: e, property: name.value, span: t.span };
      } else if (t.type === 'lbracket' && !this.newlineBreaks(t)) {
        // The newline guard mirrors the wrap guards at parseWrappable: at STATEMENT level a `[` opening
        // the NEXT line is a fresh array-literal statement, not an index of the previous value (without
        // it, `let xs = base⏎[0, 1]` silently mis-parses as `base[0]` + an error cascade). Inside an open
        // grouping (groupDepth>0) there is no statement boundary, so the guard is suspended and the `[`
        // still indexes (`render(⏎items⏎[0]⏎)`).
        this.next(); this.groupDepth++; const index = this.parseExpression(); this.groupDepth--; this.eat('rbracket');
        e = { kind: 'index', object: e, index, span: t.span };
      } else if (t.type === 'lparen' && !this.newlineBreaks(t)) {
        // Same rationale as the `[` branch: at statement level a `(` opening the NEXT line is a fresh
        // parenthesized statement (the classic leading-paren hazard — `let s = pow(a,b)⏎(c+d)*e` must be
        // two statements, and an implicit-last-expression return must not be silently swallowed); inside
        // an open grouping the guard is suspended so a curried call still chains (`f(g(x)⏎(y))`).
        this.next(); this.groupDepth++; const args: Expr[] = [];
        while (this.peek().type !== 'rparen' && this.peek().type !== 'eof') {
          // Call-argument spread (`f(...args)`) is not supported (spread is literals-only). Emit ONE
          // named diagnostic + skip to the call's `)` rather than letting `...` derail parsePrimary.
          if (this.peek().type === 'ellipsis') { this.skipMalformedToGroupClose('spread is not allowed in a call argument (spread is limited to array/object literals)', this.peek().span); break; }
          args.push(this.parseExpression());
          if (this.peek().type === 'comma') this.next(); else break;
        }
        this.groupDepth--; this.eat('rparen');
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
    this.groupDepth++; const e = this.parseExpression(); this.groupDepth--; this.eat('rparen');
    this.parenthesized.add(e);   // mark as grouped so a parenthesized arrow is a legit operand/value
    return e;
  }

  /** Arrow body after `=>` (`Arrow ::= … "=>" (Expr | Block)`). JS-identical rule: a leading `{`
   *  is a STATEMENT BLOCK (returns Stmt[], so a state-mutating `hover = h` assignment is legal),
   *  NOT an object literal — to return an object from an arrow, wrap it: `() => ({ … })`. Any other
   *  token is a single expression body. */
  private parseArrowBody(): Expr | Stmt[] {
    if (this.peek().type === 'lbrace') return this.parseBlock();
    const body = this.parseExpression();
    // Assignment is a STATEMENT, not an expression — so an expression-bodied arrow whose body is
    // followed by `=` (`() => count = count + 1`) is a mistake. Emit ONE targeted diagnostic pointing
    // at the block-body form, and consume the `= …` RHS so it does not derail the caller.
    if (this.peek().type === 'assign') {
      this.diagnostics.push(makeDiagnostic('ML-LANG-PARSE', 'assignment is a statement; use a block body for a state-mutating arrow: () => { x = v }', this.peek().span));
      // Discard the whole assignment tail (incl. a chain `a = b = c`) so no stray `= …` leaks out as
      // phantom statements — one diagnostic, clean recovery.
      while (this.peek().type === 'assign') { this.next(); this.parseExpression(); }
    }
    return body;
  }

  private parseObject(): Expr {
    const start = this.peek().span; this.next(); this.groupDepth++; // {  (a literal body is not a statement boundary)
    const entries: ObjectEntry[] = [];
    while (this.peek().type !== 'rbrace' && this.peek().type !== 'eof') {
      if (this.peek().type === 'ellipsis') {
        this.next();
        entries.push({ key: '', value: this.parseExpression(), spread: true });
      } else {
        // An entry must be `key : value` — key = ident, quoted string, OR a keyword used as a name
        // (`{ if: 1 }`, `{ true: 2 }`; JS-familiar, and the printer emits keyword keys bare, so this
        // must round-trip). If the key is not name-like or is not followed by `:`, this is not a valid
        // object literal — emit ONE diagnostic + skip to the closing `}` rather than flailing per-token
        // into a cascade (e.g. a stray `{ compute() }` after a call). One clear message, clean recovery.
        const keyTok = this.peek();
        const keyIsName = keyTok.type === 'ident' || keyTok.type === 'string' || KEYWORD_TOKENS.has(keyTok.type);
        if (!keyIsName || this.toks[this.pos + 1]?.type !== 'colon') {
          this.skipMalformedToGroupClose('malformed object literal — an entry must be `key: value` (a bare block/wrap is not valid in value position)', keyTok.span);
          break;
        }
        const key = this.next().value;
        if (FORBIDDEN_KEYS.has(key)) this.diagnostics.push(makeDiagnostic('ML-LANG-FORBIDDEN', `forbidden key '${key}'`, this.peek().span));
        this.eat('colon');
        entries.push({ key, value: this.parseExpression(), spread: false });
      }
      if (this.peek().type === 'comma') this.next(); else break;
    }
    this.groupDepth--; this.eat('rbrace');
    return { kind: 'object', entries, span: start };
  }

  private parseArray(): Expr {
    const start = this.peek().span; this.next(); this.groupDepth++; // [  (a literal body is not a statement boundary)
    const elements: ArrayElement[] = [];
    while (this.peek().type !== 'rbracket' && this.peek().type !== 'eof') {
      const spread = this.peek().type === 'ellipsis';
      if (spread) this.next();
      elements.push({ value: this.parseExpression(), spread });
      if (this.peek().type === 'comma') this.next(); else break;
    }
    this.groupDepth--; this.eat('rbracket');
    return { kind: 'array', elements, span: start };
  }

  // --- statement layer ---
  /** Parse the top-level statement sequence until `eof` and return the list of statements. Also runs
   *  the top-level redeclaration check, emitting `ML-LANG-REDECL` when a `const`/`let`/`function`/
   *  `component` name is declared twice in this scope.
   *  @returns the parsed top-level {@link Stmt} list (the body of the {@link Program}). */
  parseProgramBody(): Stmt[] {
    const stmts: Stmt[] = [];
    const declared = new Set<string>();
    while (this.peek().type !== 'eof') {
      const s = this.parseStatement();
      if (s) {
        if ((s.kind === 'const' || s.kind === 'let' || s.kind === 'function' || s.kind === 'component')) {
          const name = (s as { name: string }).name;
          // An empty name is a recovery placeholder (e.g. destructuring-in-declaration recovery), never a
          // real binding — skip it so two recovered decls don't spuriously report `'' already declared`.
          if (name) {
            if (declared.has(name)) this.diagnostics.push(makeDiagnostic('ML-LANG-REDECL', `'${name}' already declared in this scope`, s.span));
            declared.add(name);
          }
        }
        stmts.push(s);
      } else break;
    }
    return stmts;
  }

  private parseBlock(): Stmt[] {
    this.eat('lbrace');
    // A block body is a STATEMENT context even when nested inside a grouping (an arrow/function body
    // passed as a call arg, e.g. `map(xs, (x) => { let a = f(x)⏎(y) })`). Reset groupDepth to 0 for the
    // block's statements so the newline guard fires again inside it, then restore on exit.
    const saved = this.groupDepth; this.groupDepth = 0;
    const stmts: Stmt[] = [];
    while (this.peek().type !== 'rbrace' && this.peek().type !== 'eof') {
      const s = this.parseStatement(); if (s) stmts.push(s); else break;
    }
    this.groupDepth = saved; this.eat('rbrace');
    return stmts;
  }

  /** The `( … )` condition head of `if` / `while`. The parens are a grouping (no statement boundary
   *  inside), so raise groupDepth around the test. If the test is immediately followed by `=`, the
   *  author wrote an assignment in a condition (`if (a = b)`) — unsupported (assignment is a statement,
   *  and `==` was almost certainly meant); emit ONE named diagnostic + recover past the `= …)` tail. */
  private parseParenCondition(): Expr {
    this.eat('lparen');
    this.groupDepth++;
    const test = this.parseExpression();
    if (this.peek().type === 'assign') {
      // `if (a = b)` — assignment in a condition (unsupported; `==` was almost certainly meant).
      this.skipMalformedToGroupClose('assignment is not allowed in a condition (did you mean `==`?)', this.peek().span);
    }
    this.groupDepth--; this.eat('rparen');
    return test;
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
        this.next();
        // Destructuring in a declaration (`const { a } = props` / `const [x] = xs`) is not supported.
        // Detect the pattern-opening `{`/`[` before eating the name and emit ONE named diagnostic +
        // recover, rather than letting `eat('ident')` mismatch cascade through the whole pattern.
        if (this.peek().type === 'lbrace' || this.peek().type === 'lbracket') {
          this.recoverToStatementBoundary('destructuring in a declaration is not supported; bind a name and read fields (e.g. `const p = props`)', t.span);
          return { kind: t.type, name: '', init: { kind: 'null', span: t.span }, span: t.span };
        }
        const name = this.eat('ident')?.value ?? '';
        this.eat('assign');
        const init = this.parseWrappable();   // RHS may be a wrapping element (`const root = group { … }`)
        // Multi-declarator (`const a = 1, b = 2`) is not supported: emit ONE named diagnostic + recover
        // to the statement boundary, rather than letting the stray `, b = 2` fall through as phantom
        // statements (the eat()-cascade this replaces).
        if (this.peek().type === 'comma') this.recoverToStatementBoundary('multiple declarators in one statement are not supported; use one `const`/`let` per line', t.span);
        else if (this.peek().type === 'semi') this.next();
        return { kind: t.type, name, init, span: t.span };
      }
      case 'function': case 'component': {
        this.next(); const name = this.eat('ident')?.value ?? ''; const params = this.parseParams(); const body = this.parseBlock();
        return { kind: t.type, name, params, body, span: t.span };   // token type === Stmt kind for both
      }
      case 'if': {
        this.next(); const test = this.parseParenCondition();
        const then = this.parseBody();   // Body ::= Block | Stmt
        let elseBody: Stmt[] | undefined;
        if (this.peek().type === 'else') { this.next(); elseBody = this.peek().type === 'if' ? [this.parseStatement()!] : this.parseBody(); }
        return { kind: 'if', test, then, else: elseBody, span: t.span };
      }
      case 'for': {
        this.next(); this.eat('lparen'); this.eat('const'); const binding = this.eat('ident')?.value ?? ''; this.eat('of');
        this.groupDepth++; const iterable = this.parseExpression(); this.groupDepth--; this.eat('rparen'); const body = this.parseBody();   // brace-less single stmt OK
        return { kind: 'for', binding, iterable, body, span: t.span };
      }
      case 'while': {
        this.next(); const test = this.parseParenCondition(); const body = this.parseBody();
        return { kind: 'while', test, body, span: t.span };
      }
      case 'return': {
        // A value-less `return` is terminated by `;`, `}` (end of block), or eof — none of which start
        // an expression. Only parse a return value when a real expression token follows.
        this.next();
        const tt = this.peek().type;
        const value = (tt === 'semi' || tt === 'rbrace' || tt === 'eof') ? undefined : this.parseWrappable();   // return value may be a wrapping element
        if (this.peek().type === 'semi') this.next();
        return { kind: 'return', value, span: t.span };
      }
      case 'ident': {
        // expression / assignment / wrapping element (no root-block production — the root is an entry component)
        const expr = this.parseWrappable();
        if (this.peek().type === 'assign') {
          // An assignment TARGET must be an lvalue (ident / member / index). Rejecting a non-lvalue
          // LHS at parse time catches the common `==` mistyped as `=` (`ready == loaded = true`) and
          // `f() = 1`, which would otherwise build a valid-looking assign node and only fail at eval.
          if (expr.kind !== 'ident' && expr.kind !== 'member' && expr.kind !== 'index') {
            this.diagnostics.push(makeDiagnostic('ML-LANG-PARSE', `invalid assignment target (${expr.kind})`, expr.span));
          }
          this.next(); const value = this.parseWrappable(); if (this.peek().type === 'semi') this.next(); return { kind: 'assign', target: expr, value, span: t.span };   // RHS may be a wrapping element (`root = group { … }`)
        }
        if (this.peek().type === 'semi') this.next();
        return { kind: 'expr', expr, span: t.span };
      }
      default: {
        const expr = this.parseWrappable(); if (this.peek().type === 'semi') this.next(); return { kind: 'expr', expr, span: t.span };
      }
    }
  }

  /** Parse an expression that may be a WRAPPING ELEMENT — a head applied to children via either a
   *  `{ … }` child block or a single same-line trailing statement. Attaches `block` to the head call
   *  node. Used in every position a wrap is legal: statement position AND a value RHS (const/let/return/
   *  assign) so `const root = group { … }` wraps identically to a statement-level `group { … }`.
   *
   *  The exact rules (all SAME-LINE only — a newline before the `{`/child is a statement boundary,
   *  consistent across every wrap form so identical-looking lines never nest differently by whitespace):
   *   1. `ident { … }`      — a bare identifier + a SAME-LINE `{` is a zero-arg wrapping call
   *                           (`group { … }` ≡ `group() { … }`). A NEXT-line `{` → two statements.
   *   2. `call() { … }`     — a `{` block on the SAME line as the `)` wraps. A NEXT-line `{` does NOT
   *                           wrap (it starts a fresh statement — no silent cross-newline swallow).
   *   3. `call() child()`   — a single same-line trailing statement wraps (`transform(…) shape(…)`).
   *                           A newline/`;` makes them siblings (else `KPI(a)⏎KPI(b)` would mis-nest).
   *  A non-ident, non-call head (a member/index like `ui.panel`) does NOT wrap — only a bare ident or
   *  a call head is a wrapping head; `ui.panel { … }` is a member expr then a separate `{ … }`. */
  private parseWrappable(): Expr {
    let e = this.parseExpression();
    // Rule 1 — bare-ident + same-line `{` → synthesized zero-arg wrapping call.
    if (e.kind === 'ident' && this.peek().type === 'lbrace' && !this.peek().newlineBefore) {
      e = { kind: 'call', callee: e, args: [], span: e.span };
    }
    // Rules 2 & 3 — a call head wraps a SAME-LINE `{` block or a single same-line trailing statement.
    // The `!newlineBefore` guard now applies to the `{` branch too (rule 2): a next-line `{` after a
    // call is a fresh statement, matching the bare-ident rule — no silent cross-newline block swallow.
    if (e.kind === 'call') {
      if (this.peek().type === 'lbrace' && !this.peek().newlineBefore) { e.block = this.parseBlock(); }
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

/**
 * Parse a single expression from source, returning its AST node and diagnostics.
 *
 * Total and non-throwing: a lex/parse error or nesting past {@link MAX_PARSE_DEPTH} surfaces as a
 * diagnostic with a safe `null` literal node rather than an exception.
 *
 * @param src - the expression source text.
 * @returns the parsed expression + the diagnostics collected while lexing and parsing
 *          ({@link ParseExprResult}).
 */
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

/** The outcome of {@link parseProgram}: the parsed program and every diagnostic collected while lexing
 *  and parsing it, in order. */
export interface ParseProgramResult {
  /** The parsed program (its top-level statement list). Best-effort on error — an empty statement list
   *  when the source could not be parsed (e.g. it nested past {@link MAX_PARSE_DEPTH}). */
  readonly program: Program;
  /** Every diagnostic collected while lexing and parsing, in order. Empty on a clean parse. */
  readonly diagnostics: Diagnostic[];
}
/**
 * Parse a whole program from source, returning its AST and diagnostics.
 *
 * The parsing entry point of the language kernel. Total and non-throwing: a lex/parse error or nesting
 * past {@link MAX_PARSE_DEPTH} surfaces as a diagnostic with a safe empty program rather than an
 * exception.
 *
 * @param src - the program source text.
 * @returns the parsed program + the diagnostics collected while lexing and parsing
 *          ({@link ParseProgramResult}).
 */
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
