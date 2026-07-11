// Chained lexical scope. Map-based, no prototype chain to pollute. ONE namespace (JS/ES has no
// `$`, so function/const/let share a scope).
//
// Reactive-binding contract: a `const`/`function`/param binding holds its VALUE in `.value`. A
// component-scoped reactive `let` binding holds an opaque `CellRef` in its meta — its value lives
// in the ReactiveHost cell, so `Environment.assign` is NEVER used to write a `let` (the evaluator
// routes let writes through host.writeCell). `CellRef` stays opaque here (typed `unknown`) so this
// module has no dependency on the ports.
export type BindingMeta =
  | { kind: 'const' }
  | { kind: 'let'; cell?: unknown };   // cell: CellRef (opaque)

interface Cell { value: unknown; meta: BindingMeta }

export class Environment {
  private readonly cells = new Map<string, Cell>();
  private readonly parent?: Environment;

  constructor(parent?: Environment) { this.parent = parent; }

  define(name: string, value: unknown, meta: BindingMeta): void {
    this.cells.set(name, { value, meta });
  }
  hasOwn(name: string): boolean { return this.cells.has(name); }
  has(name: string): boolean { return this.cells.has(name) || (this.parent?.has(name) ?? false); }
  get(name: string): unknown {
    const c = this.cells.get(name);
    return c ? c.value : this.parent?.get(name);
  }
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
