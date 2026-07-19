// Example components as DSL source — a shared test fixture for the browser tests. NOT public API (not
// exported from the barrel). Uses the collections capability this package depends on: spread ([...a],
// {...o}) + filter() for immutable list rebuilds, and the `head { }` wrap shorthand. Arrays/objects are
// immutable — a list "update" REASSIGNS a new array (items = [...items, x]); it never mutates in place.

/** COUNTER — a reactive `let n` read only by `span(n)` → the FINE-GRAINED leaf path (no re-render;
 *  the walk-effect is not subscribed to `n`, so a click patches only the text node). */
export const COUNTER = `
component Story() {
  let n = 0
  div({ class: "counter" }) {
    button({ onClick: () => { n = n + 1 } }, "+")
    span(n)
  }
}`;

/** TODO — a reactive `let items` iterated by a `for` → the STRUCTURAL keyed path. Add = reassign a new
 *  array via spread ([...items, newRow]); remove = reassign via filter(items, keep). Both are immutable
 *  rebuilds (identifier-LHS reassignment of `items` — the structural cell the walk-effect subscribes to). */
export const TODO = `
component Story() {
  let items = [{ id: 0, label: "first" }, { id: 1, label: "second" }]
  let nextId = 2
  div({ class: "todo" }) {
    ul {
      for (const it of items) {
        li({ key: it.id }) {
          span(it.label)
          button({ onClick: () => { items = filter(items, (r) => r.id != it.id) } }, "x")
        }
      }
    }
    button({ onClick: () => { items = [...items, { id: nextId, label: "new" }]; nextId = nextId + 1 } }, "add")
  }
}`;

/** FORM — an input whose value binds a reactive `let name`; proves focus + selection survive a re-derive. */
export const FORM = `
component Story() {
  let name = ""
  form {
    input({ value: name, onInput: (e) => { name = e.value }, id: "name-input" })
    span(name)
  }
}`;
