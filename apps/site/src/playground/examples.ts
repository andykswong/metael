// The curated starter gallery — authored metael source strings the playground ships. A built-in authored
// set (distinct from any future hosted user-snippet gallery). Every example parses + derives with zero
// diagnostics (asserted in examples.test.ts) so the picker never ships a broken snippet; the flagship
// TodoMVC is additionally proven interactively (examples.browser.test.ts).
// Collections capability in use: spread ([...a]/{...o}) for immutable rebuilds, map/filter/reduce/entries.
// Immutable-update idiom: reassign a `let` with a rebuilt collection — never a member write.

export type Target = 'ui' | 'compute' | 'gpu';

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

// ITERABLE_BUILTINS — the collection builtins accept a typed array directly:
// map/filter/slice/reduce/sort iterate an f32 buffer the same as a plain
// array, each returning a plain array. No copy-to-array step needed.
const ITERABLE_BUILTINS = `const buf = f32([3, 1, 4, 1, 5, 9, 2, 6])
{
  doubled: join(map(slice(buf, 0, 4), (x) => x * 2), ", "),
  evens: filter(buf, (x) => x % 2 == 0),
  total: reduce(buf, (a, x) => a + x, 0),
  sorted: sort(buf)
}`;

// GPU_COMPUTE_MAP — a gpu kernel on the pure COMPUTE path: the program's value is the resource's fields,
//  pretty-printed like any compute story (no div.gpu-demo, no vdom). Runs headless via runComputeSettled.
const GPU_COMPUTE_MAP = `
const x = f32(4, (i) => i)
component k(i) { return x[i] * 2 }
const r = gpu(k, { output: [4], backend: "cpu" })
{ result: r }`;

// ─── GPU examples ─── (entry component `Story`; the kernel is itself a
// `component` so its `let` accumulator works end-to-end. `gpu(kernel, {output})`
// returns a reactive resource; its fields render via vdom display heads. The
// resource settles asynchronously — the panel shows "computing…" first, then
// the backend, the first computed cells, and the generated WGSL. `verify` +
// `benchmark` are OPT-IN — a dispatch is GPU-only by default; these demos turn
// them on to show the correctness proof + the CPU race.)

// GPU_MATMUL — the race: an N×N matrix product. The kernel(row, col) reads two
// buffers with a `let sum` accumulator over range(N). We opt into verify +
// benchmark so the panel can show the first computed cells, the CPU-baseline
// timing, and the interpreter match — plus the emitted compute shader.
const GPU_MATMUL = `component Story() {
  const N = 32
  const a = f32(N * N, (i) => rand())
  const b = f32(N * N, (i) => rand())
  component product(row, col) {
    let sum = 0
    for (const k of range(N)) {
      sum = sum + a[row * N + k] * b[k * N + col]
    }
    return sum
  }
  const r = gpu(product, {
    output: [N, N], verify: true, benchmark: true
  })
  const head = r.value == null ? [] : slice(r.value, 0, 6)
  const cells = join(map(head, (x) => round(x * 100) / 100), ", ")
  div({ class: "gpu-demo" }) {
    p({ class: "title" }, N + "x" + N + " matmul")
    if (r.pending) {
      p({ class: "status" }, "computing on " + r.backend + "...")
    } else {
      p({ class: "result" }, "first cells: [" + cells + ", ...]")
      const gms = r.gpuMs == null ? "GPU n/a (CPU floor)"
        : "GPU " + round(r.gpuMs * 100) / 100 + "ms"
      const cms = "CPU " + round(r.cpuMs * 100) / 100 + "ms"
      const spd = r.speedup == null ? ""
        : " (" + round(r.speedup * 10) / 10 + "x)"
      const line = r.backend + " " + gms + " " + cms + spd
      p({ class: "status" }, line + " match=" + r.match.ok)
    }
    pre({ class: "shader" }, r.wgsl)
  }
}`;

// GPU_VECMATH — the correctness proof: a per-lane length(vec3(...)) map. vec3
// intermediates + a transcendental (sqrt via length) lower to the shader. We
// opt into verify so the oracle re-checks every sampled lane against the
// interpreter (r.match.ok) — and show the first computed lengths.
const GPU_VECMATH = `component Story() {
  const n = 64
  const xs = f32(n * 3, (i) => i)
  component lane(i) {
    const v = vec3(xs[i * 3], xs[i * 3 + 1], xs[i * 3 + 2])
    return length(v)
  }
  const r = gpu(lane, { output: [n], verify: true })
  const head = r.value == null ? [] : slice(r.value, 0, 6)
  const lens = join(map(head, (x) => round(x * 100) / 100), ", ")
  div({ class: "gpu-demo" }) {
    p({ class: "title" }, "length(vec3) x " + n)
    if (r.pending) {
      p({ class: "status" }, "computing on " + r.backend + "...")
    } else {
      p({ class: "result" }, "lengths: [" + lens + ", ...]")
      p({ class: "status" },
        r.backend + " - matches interpreter=" + r.match.ok)
    }
    pre({ class: "shader" }, r.wgsl)
  }
}`;

// GPU_BUFFER — buffer output: the kernel's result settles as a reusable f32
// buffer (a typed array), not a plain array. That handle is itself iterable,
// so the same slice/map/join display idiom reads its first cells — proving
// the collection builtins accept a typed array. verify re-checks the cells
// against the interpreter oracle (r.match.ok).
const GPU_BUFFER = `component Story() {
  const N = 48
  const x = f32(N, (i) => i)
  component k(i) { return x[i] * 3 }
  const r = gpu(k, {
    output: [N], outputType: "buffer", verify: true
  })
  const head = r.value == null ? [] : slice(r.value, 0, 6)
  const cells = join(map(head, (v) => round(v)), ", ")
  div({ class: "gpu-demo" }) {
    p({ class: "title" }, "buffer output x " + N)
    if (r.pending) {
      p({ class: "status" }, "computing on " + r.backend + "...")
    } else {
      p({ class: "result" }, "first cells: [" + cells + ", ...]")
      p({ class: "status" },
        r.backend + " - reusable f32 buffer, match=" + r.match.ok)
    }
    pre({ class: "shader" }, r.wgsl)
  }
}`;

// GPU_PIPELINE — pipelining: kernel A produces a resident "gpu-buffer" (the
// output stays on-device), and kernel B closes over A's handle (r.value) as
// an INPUT — a two-stage GPU pipeline. B re-dispatches whenever A settles a
// fresh handle (the resource memo keys off the handle, not a plain array). On
// the first synchronous frame A is still pending (rA.value == null), so we
// only build + dispatch B once A has a handle; until then the panel shows A's
// pending state. Reading B's handle back (slice/map) yields (i+1)*2.
const GPU_PIPELINE = `component Story() {
  const N = 48
  const seed = f32(N, (i) => i)
  component a(i) { return seed[i] + 1 }
  const rA = gpu(a, { output: [N], outputType: "gpu-buffer" })
  div({ class: "gpu-demo" }) {
    p({ class: "title" }, "2-stage pipeline x " + N)
    if (rA.value == null) {
      p({ class: "status" }, "stage A on " + rA.backend + "...")
    } else {
      const bufA = rA.value
      component b(i) { return bufA[i] * 2 }
      const rB = gpu(b, { output: [N] })
      const head = rB.value == null ? [] : slice(rB.value, 0, 6)
      const cells = join(map(head, (x) => round(x)), ", ")
      if (rB.pending) {
        p({ class: "status" }, "stage B on " + rB.backend + "...")
      } else {
        p({ class: "result" }, "B = A*2: [" + cells + ", ...]")
        p({ class: "status" }, "A -> B on " + rB.backend)
      }
      pre({ class: "shader" }, rB.wgsl)
    }
  }
}`;

// GPU_VECOUT — single vecN output: the kernel RETURNS a vec3 per cell, so the
// result is an N-wide interleaved buffer (cell c, component k at c*3 + k), not
// one value per cell. All three backends produce the SAME flat layout. We show
// the first two output vectors as (x, y, z) triples + the interpreter match.
const GPU_VECOUT = `component Story() {
  const n = 64
  const src = f32(n * 3, (i) => i)
  component k(i) {
    const v = vec3(src[i*3], src[i*3+1], src[i*3+2])
    return v + vec3(1, 2, 3)
  }
  const r = gpu(k, {
    output: [n], outputElement: "vec3", verify: true
  })
  const flat = r.value == null ? [] : slice(r.value, 0, 6)
  const v0 = join(map(slice(flat, 0, 3), (x) => round(x)), ", ")
  const v1 = join(map(slice(flat, 3, 6), (x) => round(x)), ", ")
  div({ class: "gpu-demo" }) {
    p({ class: "title" }, "vec3 output x " + n)
    if (r.pending) {
      p({ class: "status" }, "computing on " + r.backend + "...")
    } else {
      p({ class: "result" },
        "cell0=(" + v0 + ") cell1=(" + v1 + ")")
      p({ class: "status" },
        r.backend + " - N-wide buffer, match=" + r.match.ok)
    }
    pre({ class: "shader" }, r.wgsl)
  }
}`;

// GPU_MULTIOUT — multi-output: the kernel returns a NAMED OBJECT { sum, diff },
// so the run writes two output buffers instead of one. `outputs: { sum, diff }`
// declares them; `r.value` is null (no single primary value) and each named
// buffer settles into `r.outputs`. Under the hood each output is one dispatch
// over the proven single-output path, so verify re-checks every output against
// the interpreter (r.match.ok covers all of them). We show each buffer's first
// cells, guarding r.outputs == null while the dispatch is pending.
const GPU_MULTIOUT = `component Story() {
  const N = 32
  const a = f32(N, (i) => i + 1)
  const b = f32(N, (i) => i)
  component k(i) {
    return { sum: a[i] + b[i], diff: a[i] - b[i] }
  }
  const r = gpu(k, {
    output: [N], outputs: { sum: {}, diff: {} }, verify: true
  })
  const outs = r.outputs
  const sumH = outs == null ? [] : slice(outs.sum, 0, 6)
  const diffH = outs == null ? [] : slice(outs.diff, 0, 6)
  const sums = join(map(sumH, (x) => round(x)), ", ")
  const diffs = join(map(diffH, (x) => round(x)), ", ")
  div({ class: "gpu-demo" }) {
    p({ class: "title" }, "multi-output x " + N)
    if (r.pending) {
      p({ class: "status" }, "computing on " + r.backend + "...")
    } else {
      p({ class: "result" }, "sum:  [" + sums + ", ...]")
      p({ class: "result" }, "diff: [" + diffs + ", ...]")
      p({ class: "status" },
        r.backend + " - 2 named buffers, match=" + r.match.ok)
    }
    pre({ class: "shader" }, r.wgsl)
  }
}`;

// GPU_REDUCE — the reduction kernel kind: a 2-arg associative reducer folds a
// generated buffer to ONE scalar via `gpuReduce`. On WebGL2 this runs as a
// multi-pass ping-pong tree reduction (the generated shader folds a tile of
// elements per texel, over ping-pong textures); on the CPU floor it is the
// exact linear fold (the oracle). We opt into verify so the panel shows the
// GPU tree fold matches the linear oracle (a tree reorders the fold → a
// float-associativity tolerance) + the emitted reduction shader.
const GPU_REDUCE = `component Story() {
  const N = 1024
  const xs = f32(N, (i) => i + 1)
  component add(acc, x) { return acc + x }
  const r = gpuReduce(add, {
    input: xs, identity: 0, verify: true
  })
  div({ class: "gpu-demo" }) {
    p({ class: "title" }, "sum of 1.." + N)
    if (r.pending) {
      p({ class: "status" }, "reducing on " + r.backend + "...")
    } else {
      p({ class: "result" }, "sum = " + r.value)
      p({ class: "status" },
        r.backend + " tree fold, match=" + r.match.ok)
    }
    pre({ class: "shader" }, r.glsl)
  }
}`;

// GPU_HISTOGRAM — the histogram kernel kind: a 1-arg bin-mapper scatters each
// input element into a per-bin count via `gpuHistogram`. On WebGPU this runs
// as a single-pass atomic scatter (atomicAdd into a read_write bins buffer);
// WebGL2 has NO fragment-shader atomics, so a histogram FALLS TO the CPU
// oracle there (settled backend 'cpu' + a note). An out-of-range bin index is
// dropped (both backends agree). We opt into verify so the panel shows the
// counts match the exact CPU oracle + the emitted atomic-scatter shader.
const GPU_HISTOGRAM = `component Story() {
  const N = 256
  const xs = f32(N, (i) => i)
  component binOf(x) { return x % 4 }
  const r = gpuHistogram(binOf, {
    input: xs, bins: 4, verify: true
  })
  div({ class: "gpu-demo" }) {
    p({ class: "title" }, N + " values into 4 bins (x % 4)")
    if (r.pending) {
      p({ class: "status" }, "scattering on " + r.backend + "...")
    } else {
      p({ class: "result" },
        "counts = [" + join(r.value, ", ") + "]")
      p({ class: "status" },
        r.backend + " scatter, match=" + r.match.ok)
    }
    pre({ class: "shader" }, r.wgsl)
  }
}`;

// GPU_3D — a rank-3 dispatch: the kernel takes THREE coords (x, y, z), so
// `output: [W, H, D]` maps to a 3D grid (the WGSL bakes @workgroup_size(4,4,4)).
// The value encodes its own coords (x*100 + y*10 + z) so the flat row-major
// buffer (((x*H + y)*D + z) order) reads back legibly. Small dims (16 cells)
// keep it a quick, verifiable grid. verify re-checks each sampled cell against
// the interpreter (r.match.ok).
const GPU_3D = `component Story() {
  component k(x, y, z) {
    return x * 100 + y * 10 + z
  }
  const r = gpu(k, { output: [2, 2, 4], verify: true })
  const head = r.value == null ? [] : slice(r.value, 0, 6)
  const cells = join(map(head, (v) => round(v)), ", ")
  div({ class: "gpu-demo" }) {
    p({ class: "title" }, "rank-3 dispatch [2, 2, 4]")
    if (r.pending) {
      p({ class: "status" }, "computing on " + r.backend + "...")
    } else {
      p({ class: "result" }, "cells: [" + cells + ", ...]")
      p({ class: "status" },
        r.backend + " - matches interpreter=" + r.match.ok)
    }
    pre({ class: "shader" }, r.wgsl)
  }
}`;

// GPU_MATHFNS — the shader-intrinsic math builtins: a per-lane composition of
// tan / atan2 / exp2 / log2 / inverseSqrt, each lowered to its native WGSL
// intrinsic. The inputs are kept in-domain (log2/inverseSqrt args > 0) so the
// finite result matches the interpreter oracle (r.match.ok).
const GPU_MATHFNS = `component Story() {
  const n = 32
  const xs = f32(n, (i) => i * 0.04)
  component k(i) {
    const t = xs[i]
    return tan(t) + atan2(t, 2) + exp2(t)
      + log2(t + 1) + inverseSqrt(t + 1)
  }
  const r = gpu(k, { output: [n], verify: true })
  const head = r.value == null ? [] : slice(r.value, 0, 6)
  const cells = join(map(head, (x) => round(x * 100) / 100), ", ")
  div({ class: "gpu-demo" }) {
    p({ class: "title" }, "math intrinsics x " + n)
    if (r.pending) {
      p({ class: "status" }, "computing on " + r.backend + "...")
    } else {
      p({ class: "result" }, "cells: [" + cells + ", ...]")
      p({ class: "status" },
        r.backend + " - matches interpreter=" + r.match.ok)
    }
    pre({ class: "shader" }, r.wgsl)
  }
}`;

// GPU_QUAT — quaternion rotation: each lane builds a rotation quat about +z via
// qaxisangle(axis, angle) then applies it to (1, 0, 0) with qrotate. The angle
// varies per lane (i * 0.1) so every cell rotates by a different amount. The
// quat ops are hand-emitted WGSL (there is no native quat type); we return the
// rotated vector's y component (= sin(angle)) so a scalar output[n] works. No
// verify: quaternion ops are `gpu-tolerant` — the f32 shader reassociates the
// cross-product chain vs the f64 interpreter, so the values are correct but an
// exact per-cell match isn't the right claim for a transcendental rotation.
const GPU_QUAT = `component Story() {
  const n = 32
  component k(i) {
    const angle = i * 0.1
    const q = qaxisangle(vec3(0, 0, 1), angle)
    const v = qrotate(q, vec3(1, 0, 0))
    return v.y
  }
  const r = gpu(k, { output: [n] })
  const head = r.value == null ? [] : slice(r.value, 0, 6)
  const cells = join(map(head, (x) => round(x * 100) / 100), ", ")
  div({ class: "gpu-demo" }) {
    p({ class: "title" }, "quaternion rotate x " + n)
    if (r.pending) {
      p({ class: "status" }, "computing on " + r.backend + "...")
    } else {
      p({ class: "result" }, "rotated .y: [" + cells + ", ...]")
      p({ class: "status" },
        r.backend + " - quaternion ops (gpu-tolerant)")
    }
    pre({ class: "shader" }, r.wgsl)
  }
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
  { id: 'iterable-builtins', label: 'Builtins over a buffer', target: 'compute', blurb: 'map/filter/slice/reduce/sort accept a typed array directly', source: ITERABLE_BUILTINS },
  { id: 'gpu-compute-map', label: 'GPU map (compute)', target: 'compute', blurb: 'a gpu map kernel on the pure compute path — the settled resource, pretty-printed (no DOM)', source: GPU_COMPUTE_MAP },
  { id: 'gpu-matmul', label: 'GPU matmul (race)', target: 'gpu', blurb: 'N×N matmul on the GPU vs CPU — the generated WGSL + the backend + timing', source: GPU_MATMUL },
  { id: 'gpu-vecmath', label: 'GPU vec-math', target: 'gpu', blurb: 'a length(vec3(...)) map kernel with the correctness-vs-interpreter proof', source: GPU_VECMATH },
  { id: 'gpu-buffer', label: 'GPU buffer output', target: 'gpu', blurb: 'a kernel returns a reusable f32 buffer; slice/map/join over it', source: GPU_BUFFER },
  { id: 'gpu-vecout', label: 'GPU vec3 output', target: 'gpu', blurb: 'a vec3-returning kernel writes an N-wide interleaved buffer (cell*3 + k)', source: GPU_VECOUT },
  { id: 'gpu-multiout', label: 'GPU multi-output', target: 'gpu', blurb: 'a named-object return { sum, diff } writes two output buffers', source: GPU_MULTIOUT },
  { id: 'gpu-pipeline', label: 'GPU pipeline (A → B)', target: 'gpu', blurb: 'kernel A produces a resident gpu-buffer; kernel B consumes it as input', source: GPU_PIPELINE },
  { id: 'gpu-reduce', label: 'GPU reduction (sum)', target: 'gpu', blurb: 'a 2-arg associative reducer folds a buffer to one scalar (a multi-pass WebGL2 tree reduction)', source: GPU_REDUCE },
  { id: 'gpu-histogram', label: 'GPU histogram (scatter)', target: 'gpu', blurb: 'a 1-arg bin-mapper scatters values into per-bin counts (WebGPU atomicAdd; WebGL2 falls to the CPU oracle)', source: GPU_HISTOGRAM },
  { id: 'gpu-3d', label: 'GPU 3D dispatch', target: 'gpu', blurb: 'a rank-3 kernel k(x,y,z) over [W,H,D] — the 3D dispatch grid', source: GPU_3D },
  { id: 'gpu-mathfns', label: 'GPU math builtins', target: 'gpu', blurb: 'shader-intrinsic math builtins (tan/atan2/exp2/log2/inverseSqrt/…) lowered to WGSL', source: GPU_MATHFNS },
  { id: 'gpu-quat', label: 'GPU quaternion rotation', target: 'gpu', blurb: 'rotate a vector by a quaternion (qaxisangle/qrotate) — hand-emitted WGSL', source: GPU_QUAT },
];

export const DEFAULT_EXAMPLE_ID = 'todo';

// The representative example to load when the user switches the run target — a target switch is treated as
// an example switch (a UI component run through the compute backend would just yield null, and vice versa).
const DEFAULT_BY_TARGET: Record<Target, string> = { ui: 'todo', compute: 'fib', gpu: 'gpu-matmul' };

export function exampleById(id: string): Example | undefined {
  return EXAMPLES.find((e) => e.id === id);
}

/** The default example for a target (used when the target selector flips). Falls back to the first example
 *  of that target if the pinned id is somehow missing. */
export function defaultExampleForTarget(target: Target): Example {
  return exampleById(DEFAULT_BY_TARGET[target]) ?? EXAMPLES.find((e) => e.target === target) ?? EXAMPLES[0]!;
}
