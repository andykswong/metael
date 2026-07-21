/**
 * The metadata stored alongside a binding, recording whether it is an immutable
 * `const`/`function`/parameter binding or a reactive `let` bound to a host cell.
 *
 * @remarks
 * A `const`/`function`/parameter binding holds its VALUE directly in the scope. A
 * component-scoped reactive `let` binding instead holds an opaque cell reference here — its live
 * value lives in the reactive host's cell, so {@link Environment.assign} is NEVER used to write a
 * `let` (the evaluator routes `let` writes through the host). The `cell` reference is deliberately
 * typed `unknown` — an opaque handle — so this module carries no dependency on the host capability
 * interfaces; the evaluator casts it back to the concrete cell type at the boundary.
 */
export type BindingMeta =
  | {
      /** Discriminant for an immutable binding — a `const`, `function`, or parameter — whose value is stored directly in the scope. */
      kind: 'const';
    }
  | {
      /** Discriminant for a reactive `let` binding, whose live value lives in a host cell rather than in the scope. */
      kind: 'let';
      /** Opaque reference to the reactive host cell holding this binding's value; `undefined` before a cell is allocated. */
      cell?: unknown;
    };

interface Cell { value: unknown; meta: BindingMeta }

/**
 * A chained lexical scope: a name→binding map linked to an optional enclosing parent scope.
 *
 * Map-based (never a plain JS object), so there is no prototype chain to pollute and no inherited
 * key can masquerade as a binding. There is ONE namespace — `function`, `const`, and `let` all
 * share it — matching the surface grammar, which has no separate sigil to distinguish them.
 *
 * @remarks
 * Name resolution walks from this scope outward through {@link Environment.has} / {@link Environment.get}
 * / {@link Environment.meta}, returning the nearest binding. A binding's value is stored at
 * {@link Environment.define} time, EXCEPT a reactive `let`, whose live value lives in a host cell
 * referenced by its {@link BindingMeta} — see {@link Environment.assign} for the write path.
 */
export class Environment {
  private readonly cells = new Map<string, Cell>();
  private readonly parent?: Environment;

  /**
   * Create a scope, optionally nested inside an enclosing `parent`.
   * @param parent - the enclosing scope to resolve names against when this scope lacks a binding;
   *                 omit for a root scope.
   */
  constructor(parent?: Environment) { this.parent = parent; }

  /**
   * Bind `name` to `value` with the given metadata in THIS scope, shadowing any binding of the same
   * name in an enclosing scope.
   * @param name - the identifier to bind.
   * @param value - the stored value; for a reactive `let` the live value lives in a host cell
   *                instead (see {@link BindingMeta}).
   * @param meta - the binding kind + optional reactive-cell reference ({@link BindingMeta}).
   */
  define(name: string, value: unknown, meta: BindingMeta): void {
    this.cells.set(name, { value, meta });
  }
  /**
   * Whether `name` is bound in THIS scope specifically, NOT consulting the parent chain.
   * @param name - the identifier to test.
   * @returns `true` if this scope has its own binding for `name`.
   * @remarks Used by the redeclaration guard, which must distinguish a shadowing redeclaration in
   *          the current scope from an inherited binding of the same name resolved via
   *          {@link Environment.has}.
   */
  hasOwn(name: string): boolean { return this.cells.has(name); }
  /**
   * Whether `name` is bound in this scope OR any enclosing scope.
   * @param name - the identifier to test.
   * @returns `true` if the name resolves anywhere along the scope chain.
   */
  has(name: string): boolean { return this.cells.has(name) || (this.parent?.has(name) ?? false); }
  /**
   * Resolve `name` to its stored value, searching this scope then its parents.
   * @param name - the identifier to resolve.
   * @returns the nearest binding's stored value, or `undefined` if unbound anywhere.
   * @remarks Returns the value recorded at {@link Environment.define} time. A reactive `let`'s live
   *          value is NOT read here — it lives in a host cell referenced by its {@link BindingMeta},
   *          which the evaluator reads through the host.
   */
  get(name: string): unknown {
    const c = this.cells.get(name);
    return c ? c.value : this.parent?.get(name);
  }
  /**
   * Resolve `name` to its binding metadata, searching this scope then its parents.
   * @param name - the identifier to resolve.
   * @returns the nearest binding's {@link BindingMeta}, or `undefined` if unbound anywhere.
   * @remarks The evaluator inspects this to route a reactive `let` read/write through its host cell
   *          rather than the scope's own stored value.
   */
  meta(name: string): BindingMeta | undefined {
    const c = this.cells.get(name);
    return c ? c.meta : this.parent?.meta(name);
  }
  /** Assign to the nearest EXISTING binding's stored value. Returns false if unbound anywhere.
   *  NOTE: only used for `const`-initialization internals / non-reactive slots — a reactive
   *  `let` write goes through host.writeCell, NOT here. */
  assign(name: string, value: unknown): boolean {
    const c = this.cells.get(name);
    if (c) { c.value = value; return true; }
    return this.parent?.assign(name, value) ?? false;
  }
}
