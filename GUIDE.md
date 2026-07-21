# The metael Language — a Guide

metael is a small, **eval-free**, deterministic scripting language with a JS/ES-familiar surface. You read JS, you read metael. It runs by a tree-walking interpreter (never `eval`/`new Function`), is budgeted and sandbox-safe, and is a **pure function of its inputs** — the same source + data + seed always produces the same result. That makes it safe to run arbitrary user source inline, and machine-verifiable.

This guide is the practical, example-driven tour of the language and its AST. For the API of the packages that run it, see the [README](./README.md).

> **One kernel, many uses.** metael has no built-in vocabulary of its own — no `div`, no `sphere`, no `chart`. A *host* supplies the words (heads) a program can call and decides what they build. The same language drives a virtual DOM, a scene graph, or a pure data pipeline, depending on the host. This guide covers the language you always get; your host's docs cover its vocabulary.

---

## 1. Two kinds of declaration

Everything starts with a top-level declaration. There are two producing forms:

- **`function`** — pure. Returns a **value**. Use it for computation.
- **`component`** — stateful. Returns a **host subgraph** (whatever the host builds). Its `let` bindings are **reactive state**.

```js
// A pure function: the last expression is the return value (no `return` keyword needed).
function add(a, b) {
  a + b
}
add(2, 3)        // → 5
```

The one deviation from JS: **the last expression in a `function` body is its return value** (Rust-style implicit return). An explicit `return` also works and stops the body early.

```js
function classify(n) {
  const label = n < 0 ? "negative" : n == 0 ? "zero" : "positive"
  label          // implicit return
}
```

---

## 2. Values & bindings

### Literals

```js
42            // number
3.14          // number (integer / float, one numeric type)
"hello"       // string  ('single' or "double" quotes; escapes \n \t \\ \" \')
true  false   // booleans
null          // the ONLY "absence" value — there is no `undefined`
[1, 2, 3]     // array
{ a: 1, b: 2 }// object
```

### `const` and `let`

```js
const x = 10          // constant, any scope. Reassigning it is an error.
```

- **`const`** — an immutable binding, usable anywhere.
- **`let`** — **reactive state, and only inside a `component`.** A `let` at the top level or inside a `function` is an error (`ML-LANG-LET-SCOPE`). Pure functions don't need mutable locals — you compute with `const` + recursion + the collection builtins (below).

```js
component Counter() {
  let count = 0                 // reactive: a write schedules a re-render
  // ... host vocabulary reads `count` and a handler writes it
}
```

### The null-only model

metael has exactly one "absence" value: `null`. There is no `undefined`.

- A missing object property reads as `null`: `({ a: 1 }).b` → `null`.
- An out-of-range index is an **error**, never a silent `null`: `[1, 2][5]` → `ML-LANG-INDEX-RANGE`.
- A bare unknown identifier is an error: `foo` (unbound) → `ML-LANG-UNKNOWN-VAR`.

This strictness is deliberate — you never chase an `undefined` through your program.

---

## 3. Expressions & operators

Familiar precedence, familiar operators:

```js
1 + 2 * 3                 // → 7   arithmetic: + - * / %
"a" + 1                   // → "a1"  (+ concatenates if either side is a string)
5 > 3 && 2 < 4            // → true  logical: && || !  (short-circuit)
n == 0 ? "zero" : "n"     // ternary ?:  (untaken branch not evaluated)
-x                        // unary minus
!done                     // unary not
```

- **Comparison:** `== != < <= > >=`. `==` is a lenient equality (number/string coercion like JS `==`); there is no `===`.
- **Short-circuit:** `&&`, `||`, and `?:` don't evaluate the branch they don't need.
- **Member & index:** `obj.prop`, `arr[i]`, `str[i]`, `obj[key]`. The keys `__proto__`, `constructor`, and `prototype` are **forbidden** on every access path (`ML-LANG-FORBIDDEN`) — a prototype-pollution guard.

### Arrow functions

```js
const double = (x) => x * 2           // expression body
const label = (r) => {                // block body (runs statements, implicit last-expr return)
  const name = toUpperCase(r.name)
  name
}
```

Arrows are values — you pass them to the collection builtins.

---

## 4. Control flow

```js
if (n > 0) {
  positive()
} else if (n == 0) {
  zero()
} else {
  negative()
}

for (const x of [1, 2, 3]) {   // for-of iterates arrays…
  use(x)
}

for (const ch of "hello") {    // …and strings (by Unicode code point)
  use(ch)
}

while (cond) {
  step()
}
```

- **`for-of`** iterates an **array**'s elements or a **string**'s code points. Any other value is `ML-LANG-FOR-ITER`. (Note: string `for-of` yields code points, while `s.length` and `s[i]` are UTF-16 code units — they differ for astral characters like emoji.)
- Every loop iteration is budget-charged, so an infinite loop fails closed (`ML-LANG-BUDGET`), never hangs.
- A single-statement body may drop the braces: `if (x) foo()`.

---

## 5. Immutability — update by rebuilding

Every array and object metael creates is **deep-frozen**. You never mutate in place — you build a new value.

```js
const o = { a: 1 }
o.a = 2                       // ERROR: ML-LANG-IMMUTABLE

const o2 = { ...o, a: 2 }     // ✓ rebuild with spread
const bigger = [...xs, 4]     // ✓ append
const merged = { ...defaults, ...overrides }   // ✓ merge (last wins)
```

**Spread** works in array and object literals (`[...a, x, ...b]`, `{ ...o, k: v }`). A reactive `let` reassignment (`count = count + 1`, an identifier on the left) is fine — that's state, not mutation; only member/index writes are blocked.

---

## 6. Built-in functions

metael ships a small, orthogonal set of **pure** builtins. They are free functions (call `map(xs, fn)`, not `xs.map(fn)`), each: returns a **new frozen** value, never mutates its input; is deterministic; is budget-charged per element; and **fails loud** on a wrong-shape argument (`ML-LANG-BUILTIN-ARG`) instead of throwing. A callback may be an arrow **or** a named `function`. Any builtin name can be shadowed by declaring your own `function` of that name.

### Collection — transform & aggregate

```js
map([1, 2, 3], (x) => x * 2)              // → [2, 4, 6]
filter([1, 2, 3, 4], (x) => x > 2)        // → [3, 4]
reduce([1, 2, 3, 4], (acc, x) => acc + x, 0)   // → 10
```

`map`/`filter`/`reduce` compose into pipelines:

```js
filter(map([1, 2, 3, 4], (x) => x * 10), (x) => x > 20)   // → [30, 40]
```

### Query & predicate

```js
some([1, 2, 3], (x) => x > 2)       // → true   (any match; short-circuits)
every([1, 2, 3], (x) => x > 0)      // → true   (all match; short-circuits)
find([1, 2, 3], (x) => x > 1)       // → 2      (first match, or null)
findIndex([1, 2, 3], (x) => x > 1)  // → 1      (first-match index, or -1)
includes([1, 2, 3], 2)              // → true   (value membership, no callback)
```

### Ordering & slicing (non-mutating)

```js
sort([3, 1, 2])                     // → [1, 2, 3]   (a new sorted array)
sort([3, 1, 2], (a, b) => b - a)    // → [3, 2, 1]   (with a comparator)
slice([1, 2, 3, 4, 5], 1, 3)        // → [2, 3]      (negative indices ok: slice(xs, -2))
reverse([1, 2, 3])                  // → [3, 2, 1]
```

`sort` without a comparator uses a **total, stable, deterministic** order: values are grouped by type (`null < boolean < number < string < object`), ordered within each group (numbers ascending, strings lexicographically), with `NaN` pinned to the end of the numbers. So even mixed arrays sort predictably:

```js
sort([2, "a", null, true, 1])       // → [null, true, 1, 2, "a"]
```

### Objects ⇄ arrays

```js
keys({ a: 1, b: 2 })                // → ["a", "b"]
values({ a: 1, b: 2 })              // → [1, 2]
entries({ a: 1, b: 2 })             // → [["a", 1], ["b", 2]]
object([["a", 1], ["b", 2]])        // → { a: 1, b: 2 }
has({ a: 1 }, "a")                  // → true    (own-property presence)
```

A common idiom — transform a lookup table immutably:

```js
// Double every price. entries → map → object.
object(map(entries({ apple: 3, pear: 5 }), (kv) => [kv[0], kv[1] * 2]))
// → { apple: 6, pear: 10 }
```

### Strings

Strings are indexable (`s[0]`), have `.length`, and are `for-of`-iterable. The collection builtins are **array-only**, so you bridge to/from strings explicitly:

```js
split("a,b,c", ",")                 // → ["a", "b", "c"]   (split("abc", "") → per character)
join(["a", "b", "c"], "-")          // → "a-b-c"
chars("hi")                         // → ["h", "i"]
slice("hello", 1, 3)                // → "el"    (substring; negatives count from the end)
codePointAt("ABC", 0)               // → 65      (Unicode code point at an index)
toUpperCase("hi")                   // → "HI"    (locale-independent)
toLowerCase("HI")                   // → "hi"
trim("  x  ")                       // → "x"
```

```js
// Uppercase every word:
join(map(split("hello world", " "), (w) => toUpperCase(w)), " ")   // → "HELLO WORLD"
```

### Numbers

```js
min(3, 7)  max(3, 7)                // → 3, 7   (binary)
abs(-5)  sign(-3)                   // → 5, -1
floor(2.9)  ceil(2.1)              // → 2, 3
round(2.5)                         // → 2      (round-half-to-EVEN, so round(3.5) → 4)
clamp(15, 0, 10)                   // → 10
sqrt(9)  pow(2, 8)                 // → 3, 256
format(3.14159, 2)                 // → "3.14" (fixed-decimal string)
```

`round` uses banker's rounding (half-to-even), which differs from JS `Math.round` — this keeps results reproducible across execution targets. `sqrt` of a negative number, or any non-numeric argument, fails loud rather than returning `NaN`.

### Deterministic randomness & ranges

```js
rand()                             // → a number in [0, 1), from a seeded generator
range(5)                           // → [0, 1, 2, 3, 4]
```

`rand()` is **seeded** — the same seed always produces the same sequence — so a program using randomness is still a pure function of `(source, data, seed)`.

---

## 7. Data in, result out

A host runs a program with injected `data` (read-only) and gets a value or a host subgraph back:

```js
// with data = { items: [{ label: "a" }, { label: "b" }] }
map(data.items, (it) => it.label)   // → ["a", "b"]
```

Injected `data` is frozen at the boundary — a program can read it freely but cannot mutate it.

A pure program (a `function`, or any top-level expression) yields a value; a `component` yields whatever the host builds from its body. Recursion is fully supported:

```js
function fib(n) {
  n < 2 ? n : fib(n - 1) + fib(n - 2)
}
fib(10)                             // → 55
```

---

## 8. Composition (components)

A host provides **heads** — named things you call. A call with a trailing block is a **wrapping element**: the head wraps the children collected from its block.

```js
// `group`, `text` here are HOST vocabulary — your host defines what they build.
component Card({ title, body }) {
  group {
    text(title, { emphasis: true })   // leaf: attributes/handlers are props (the { … } arg)
    text(body)
  }
}
```

- **Wrapping** — a head applied to a block of children: `group { … }`. A bare `head { … }` is shorthand for `head() { … }`.
- **Props** — leaf attributes and event handlers are ordinary arguments (typically an object): `text(title, { emphasis: true })`. Handler names like `onClick` are just a host convention.
- Which head is the entry point, and what each head builds, is the host's decision — the language treats `group`, `text`, or any name uniformly.

Inside a `component`, a `let` is reactive: when a handler writes it, only the affected part of the output updates. How that update manifests is the host's job; the language guarantees the reactivity.

---

## 9. Safety & determinism (what you can rely on)

- **Eval-free.** Source is walked as data; no `eval`, no `new Function`, no dynamic code. A program cannot reach `window`, `globalThis`, `Function`, `import`, or any host internal — an unbound name simply fails closed.
- **Budgeted.** Steps, wall-clock time, recursion depth, and string growth are all capped; exceeding any fails closed with `ML-LANG-BUDGET`. No program can hang or exhaust memory.
- **Deterministic.** `result = f(source, data, seed, state)`. The only randomness is seeded. Same inputs → same output, every time.
- **Immutable.** Everything a program creates is frozen; it cannot mutate injected `data`. "It cannot change anything it sees" holds structurally.
- **Fail-loud.** Errors are structured diagnostics (`ML-LANG-*`), never exceptions thrown into the host. Evaluation always returns a value (possibly `null`) plus a diagnostics list.

Diagnostics you'll meet: `ML-LANG-PARSE` (syntax), `ML-LANG-LET-SCOPE` (`let` outside a component), `ML-LANG-IMMUTABLE` (a member/index write), `ML-LANG-BUILTIN-ARG` (wrong-shape builtin argument), `ML-LANG-FOR-ITER` (`for-of` over a non-iterable), `ML-LANG-INDEX-RANGE` (out-of-range index), `ML-LANG-FORBIDDEN` (a forbidden key), `ML-LANG-UNKNOWN-VAR` / `ML-LANG-UNKNOWN-CALL` (unbound name/head), `ML-LANG-BUDGET` (a limit tripped).

---

## 10. The AST (for tooling)

A program parses into a small discriminated-union **AST** — inert, serializable data. Each node has a `kind` and a source `span` (start/end offsets), so tools can map between text and tree. This is what an editor mutates and what the interpreter walks.

- **Expressions** (`Expr`): `number`, `string`, `bool`, `null`, `ident`, `member`, `index`, `object`, `array`, `arrow`, `call`, `unary`, `binary`, `cond`. Array/object literals carry a `spread` flag per element/entry. A `call` with a `block` is a wrapping element.
- **Statements** (`Stmt`): `const`, `let`, `assign`, `function`, `component`, `if`, `for`, `while`, `return`, `expr`.
- **A `Program`** is just a list of statements.
- Notably absent: there are **no** domain node kinds (`shape`, `chart`, …). A domain word is just a `call` whose head the host resolves — which is why a host's vocabulary changes need zero grammar or AST change.

Because the AST is plain data and every node is span-tagged, the surface is amenable to structured editing, linting, and code-generation — not just interpretation.

### Capability profiles (for advanced hosts)

Each builtin is tagged with a **capability profile**: `core` (closure-free, numeric/scalar — expressible even on restricted compile targets) vs `host` (takes a closure and/or touches strings/objects/dynamic arrays — interpreter-backed). A static classifier can decide, from a function's AST alone, whether that function is `core`-compliant. This lets an advanced host route eligible functions to alternative execution backends while running everything else on the interpreter. For typical use you can ignore this — every builtin works on the interpreter; the profile metadata just keeps the door open.

---

*Next: the [README](./README.md) for installing and embedding the packages that lex, parse, evaluate, and render metael.*
