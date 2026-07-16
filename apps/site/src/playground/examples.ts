// The curated starter gallery — authored metael source strings the playground ships. A built-in authored
// set (distinct from any future hosted user-snippet gallery). Every example parses + derives with zero
// diagnostics (asserted in examples.test.ts) so the picker never ships a broken snippet; the flagship
// TodoMVC is additionally proven interactively (examples.browser.test.ts).
// Collections capability in use: spread ([...a]/{...o}) for immutable rebuilds, map/filter/reduce/entries.
// Immutable-update idiom: reassign a `let` with a rebuilt collection — never a member write.

export type Target = 'ui' | 'compute';

export interface Example {
  readonly id: string;
  readonly label: string;
  readonly target: Target;
  readonly blurb: string;      // one line shown under the picker: what this example demonstrates
  readonly source: string;
  readonly data?: unknown;     // injected as `data` for data-bound examples
}

// ─── UI examples ─── (entry component MUST be named `Story` — the derive's default entry).

// COUNTER — the fine-grained leaf path: `n` is read only by span(n), so a click patches ONLY that text node.
const COUNTER = `component Story() {
  let n = 0
  div({ class: "counter" }) {
    button({ onClick: () => { n = n - 1 } }, "-")
    span({ class: "count" }, n)
    button({ onClick: () => { n = n + 1 } }, "+")
  }
}`;

// TODMVC — THE FLAGSHIP. Multi-component (TodoRow + Story), per-row reactive state (inline edit), callback
// props (toggle/delete/rename passed down), structural keyed reconcile (add/remove/filter), derived counts,
// reactive class binding. Every feature below was proven with real clicks in the browser.
const TODO = `component TodoRow({ item, onToggle, onDelete, onRename }) {
  let editing = false
  let draft = item.label
  li({ class: item.done ? "done" : "active" }) {
    if (editing) {
      input({
        class: "edit",
        value: draft,
        onInput: (e) => { draft = e.value },
        onKeyDown: (e) => {
          if (e.key == "Enter") {
            onRename(draft)
            editing = false
          }
        }
      })
    } else {
      button(
        { class: "toggle", onClick: onToggle },
        item.done ? "done" : "todo"
      )
      span(
        { class: "label", onClick: () => { editing = true } },
        item.label
      )
      button({ class: "del", onClick: onDelete }, "delete")
    }
  }
}

component Story() {
  let items = [
    { id: 0, label: "learn metael", done: false },
    { id: 1, label: "build a thing", done: false }
  ]
  let draft = ""
  let nextId = 2
  let mode = "all"
  const visible =
      mode == "active" ? filter(items, (i) => !i.done)
    : mode == "done" ? filter(items, (i) => i.done)
    : items
  const remaining = filter(items, (i) => !i.done).length

  div({ class: "todoapp" }) {
    input({
      class: "new",
      value: draft,
      onInput: (e) => { draft = e.value },
      onKeyDown: (e) => {
        if (e.key == "Enter") {
          const row = { id: nextId, label: draft, done: false }
          items = [...items, row]
          nextId = nextId + 1
          draft = ""
        }
      }
    })
    ul {
      for (const it of visible) {
        TodoRow({
          item: it,
          key: it.id,
          onToggle: () => {
            items = map(items, (i) =>
              i.id == it.id ? ({ ...i, done: !i.done }) : i
            )
          },
          onDelete: () => {
            items = filter(items, (i) => i.id != it.id)
          },
          onRename: (name) => {
            items = map(items, (i) =>
              i.id == it.id ? ({ ...i, label: name }) : i
            )
          }
        })
      }
    }
    div({ class: "footer" }) {
      span({ class: "count" }, remaining + " left")
      button({ onClick: () => { mode = "all" } }, "all")
      button({ onClick: () => { mode = "active" } }, "active")
      button({ onClick: () => { mode = "done" } }, "done")
      button(
        {
          onClick: () => {
            items = filter(items, (i) => !i.done)
          }
        },
        "clear done"
      )
    }
  }
}`;

// FORM — reactive input binding + live derived preview; proves focus + caret survive a re-derive.
const FORM = `component Story() {
  let first = ""
  let last = ""
  const full = first + " " + last
  const greeting =
    full == " " ? "type your name" : "Hello, " + full
  form({ class: "namecard" }) {
    input({
      class: "f",
      value: first,
      onInput: (e) => { first = e.value }
    })
    input({
      class: "l",
      value: last,
      onInput: (e) => { last = e.value }
    })
    p({ class: "greeting" }, greeting)
  }
}`;

// TABS — conditional rendering: a reactive `let` selects which panel shows via if/else-if/else.
const TABS = `component Story() {
  let tab = "home"

  function Tab({ id, label }) {
    button(
      {
        class: tab == id ? "on" : "",
        onClick: () => { tab = id }
      },
      label
    )
  }

  div({ class: "tabs" }) {
    div({ class: "tabbar" }) {
      Tab({ id: "home", label: "Home" })
      Tab({ id: "docs", label: "Docs" })
      Tab({ id: "about", label: "About" })
    }
    div({ class: "panel" }) {
      if (tab == "home") { p("Welcome home.") }
      else if (tab == "docs") { p("Read the docs.") }
      else { p("About metael.") }
    }
  }
}`;

// BUFFER_UI — the interactive custom-value demo: a reactive `let` typed array
// mutated IN PLACE by buttons. It shows both element access (for-of over the
// cells) AND whole-buffer display ("" + buf), so the per-value generation
// signal drives a re-render on an in-place write.
const BUFFER_UI = `component Story() {
  let buf = f32([0, 0, 0, 0])
  let cursor = 0
  div({ class: "buffers" }) {
    p("in-place typed-array mutation is reactive")
    div({ class: "cells" }) {
      for (const x of buf) { span({ class: "cell" }, x) }
    }
    p({ class: "disp" }, "" + buf)
    div({ class: "row" }) {
      button(
        { onClick: () => { cursor = (cursor + 1) % 4 } },
        "cell " + cursor
      )
      button(
        { onClick: () => { buf[cursor] = buf[cursor] + 1 } },
        "+1"
      )
      button({ onClick: () => { buf[cursor] = 0 } }, "reset")
    }
  }
}`;

// DASHBOARD — data-driven UI: a stat grid + a nested map (rows within cards), a Stat sub-component,
// derived totals via reduce. Demonstrates function sub-components rendering into a component tree.
const DASHBOARD = `function Stat({ label, value }) {
  div({ class: "stat" }) {
    span({ class: "stat-value" }, value)
    span({ class: "stat-label" }, label)
  }
}
component Story() {
  let teams = [
    { id: 0, name: "Alpha", members: ["Ann", "Bo"], pts: 42 },
    { id: 1, name: "Bravo", members: ["Cy", "Dee", "El"], pts: 37 }
  ]
  const points = reduce(teams, (a, t) => a + t.pts, 0)
  const people = reduce(teams, (a, t) => a + t.members.length, 0)
  div({ class: "dashboard" }) {
    div({ class: "stats" }) {
      Stat({ label: "teams", value: teams.length })
      Stat({ label: "members", value: people })
      Stat({ label: "points", value: points })
    }
    ul({ class: "teamlist" }) {
      for (const t of teams) {
        li({ key: t.id, class: "team" }) {
          span({ class: "team-name" }, t.name + " (" + t.pts + ")")
          ul({ class: "members" }) {
            for (const m of t.members) {
              li({ key: m }) { span(m) }
            }
          }
        }
      }
    }
  }
}`;

// ─── Compute examples ─── (pure programs; the last expression is the value).

const FIB = `// the classic: recursion + a pure map over a range
function fib(n) {
  if (n < 2) { return n }
  fib(n - 1) + fib(n - 2)
}

map(range(12), (i) => fib(i))`;

const DATA_TRANSFORM = `// data = [{ name, score }] injected below.
// Keep the passing rows, project each to a derived grade.
const passing = filter(data, (r) => r.score >= 60)

function gradeOf(score) {
  if (score >= 90) { return "A" }
  if (score >= 80) { return "B" }
  if (score >= 70) { return "C" }
  "D"
}

map(passing, (r) => ({
  name: r.name,
  score: r.score,
  grade: gradeOf(r.score)
}))`;

const GROUP_COUNT = `// Word frequency — now with includes() for membership.
// (Object keys are still built from [key, value] pairs via
// fromEntries — the deterministic, immutable-rebuild idiom.)
const words = ["red", "blue", "red", "green", "blue", "red"]
const uniq = reduce(
  words,
  (acc, w) => includes(acc, w) ? acc : [...acc, w],
  []
)

fromEntries(
  map(uniq, (w) => [w, filter(words, (x) => x == w).length])
)`;

const OBJECT_SHAPE = `// Invert/transform a lookup table via entries + fromEntries
// + spread (immutable rebuild, deterministic key order).
const prices = { apple: 3, pear: 5, plum: 2 }
const bumped = { ...prices, plum: 20 }

fromEntries(
  map(entries(bumped), (kv) => [kv[0], kv[1] * 2])
)`;

// CUSTOM_MATH — vec/mat operators + swizzles + dot/cross/length, plus a
// generator-filled typed-array buffer. Every custom value is rendered via
// indexing or display (a bare custom value would print as its display too).
const CUSTOM_MATH = `const a = vec3(1, 2, 3)
const b = vec3(4, 5, 6)
const squares = f32(6, (i) => i * i)
{
  sum: (a + b).x,
  scaled: (a * 2).z,
  dot: dot(a, b),
  cross_z: cross(a, b).z,
  length: length(vec3(3, 4, 0)),
  mat_vec: (mat3() * vec3(7, 8, 9)).y,
  swizzle: a.xy.y,
  vec_display: "" + a,
  buffer: "" + squares,
  first_four: [squares[0], squares[1], squares[2], squares[3]]
}`;

export const EXAMPLES: readonly Example[] = [
  { id: 'counter', label: 'Counter', target: 'ui', blurb: 'fine-grained updates — a click patches one text node, no re-render', source: COUNTER },
  { id: 'todo', label: 'Todo (full-featured)', target: 'ui', blurb: 'multi-component, per-row edit, callback props, keyed reconcile, filters', source: TODO },
  { id: 'form', label: 'Form binding', target: 'ui', blurb: 'reactive inputs + a live derived value; focus survives re-derive', source: FORM },
  { id: 'tabs', label: 'Tabs', target: 'ui', blurb: 'conditional rendering via if / else-if / else on a reactive let', source: TABS },
  { id: 'dashboard', label: 'Data dashboard', target: 'ui', blurb: 'data-driven UI: sub-components, nested lists, reduce-derived totals', source: DASHBOARD },
  { id: 'buffer', label: 'Live buffer (reactive)', target: 'ui', blurb: 'in-place typed-array mutation re-renders via the per-value generation signal', source: BUFFER_UI },
  { id: 'fib', label: 'Fibonacci', target: 'compute', blurb: 'recursion + a pure map over a range', source: FIB },
  {
    id: 'data-transform', label: 'Grade transform', target: 'compute',
    blurb: 'filter + map over injected data with a derived grade', source: DATA_TRANSFORM,
    data: [{ name: 'Ann', score: 91 }, { name: 'Bo', score: 58 }, { name: 'Cy', score: 73 }, { name: 'Dee', score: 84 }],
  },
  { id: 'group-count', label: 'Word frequency', target: 'compute', blurb: 'reduce + a hand-written has() + fromEntries — a groupBy without the builtin', source: GROUP_COUNT },
  { id: 'object-shape', label: 'Object reshape', target: 'compute', blurb: 'entries / map / fromEntries + spread — immutable object rebuild', source: OBJECT_SHAPE },
  { id: 'custom-math', label: 'Vector & buffer math', target: 'compute', blurb: 'vec/mat operators, swizzles, dot/cross/length + a typed-array buffer', source: CUSTOM_MATH },
];

export const DEFAULT_EXAMPLE_ID = 'todo';

// The representative example to load when the user switches the run target — a target switch is treated as
// an example switch (a UI component run through the compute backend would just yield null, and vice versa).
const DEFAULT_BY_TARGET: Record<Target, string> = { ui: 'todo', compute: 'fib' };

export function exampleById(id: string): Example | undefined {
  return EXAMPLES.find((e) => e.id === id);
}

/** The default example for a target (used when the target selector flips). Falls back to the first example
 *  of that target if the pinned id is somehow missing. */
export function defaultExampleForTarget(target: Target): Example {
  return exampleById(DEFAULT_BY_TARGET[target]) ?? EXAMPLES.find((e) => e.target === target) ?? EXAMPLES[0]!;
}
