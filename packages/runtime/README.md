# @metael/runtime

[![metael](https://img.shields.io/badge/project-metael-blueviolet.svg?style=flat-square&logo=github)](https://github.com/andykswong/metael)
[![npm](https://img.shields.io/npm/v/@metael/runtime?style=flat-square&logo=npm)](https://www.npmjs.com/package/@metael/runtime)
[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](./LICENSE)

**The fine-grained reactive runtime + the real host-injection port implementations** — the back half of the [metael](../../README.md) substrate. Depends only on [`@metael/lang`](../lang/README.md) + vendored `@vue/reactivity`.

`@metael/runtime` turns a parsed program into host values and keeps them reactive: it supplies the reactive core the language layer's port *interfaces* were designed around, the real `ReactiveHost`, the generic structural diff, and a one-shot composition root that drives the whole derive.

```
source ──lowerEntry (@metael/lang walk)──▶ host values          ← inside ONE change()
        │ instantiate entry · child-collect · resolveCall        │ reactive `let` → RuntimeReactiveHost cells
        │ mint keys · emit Region/Wrapper                        │ reactive prop (Region) → leaf effect (initial pipe + re-pipe)
derive(source, { env, minter, … }) ──▶ { value, diagnostics, host }
```

`derive()` is **one-shot**: it settles the initial reactive flush and returns the raw host values + the host that owns their cells/effects. The retained tree, materialization, and re-derive/reconcile are **domain-owned** — the runtime supplies the reusable machinery (the reactive core, the keyed diff, owner scopes) a domain composes into its own View.

## What's in the box

| Export | What it is |
|---|---|
| `signal`, `memo`, `effect`, `change` | the fine-grained reactive core — a read tracks a dependency, a write schedules dependents; `change()` is THE synchronous batch/flush boundary (writes accumulate, scheduled effects drain to a fixed point) |
| `ReactiveFlushError`, `Signal`, `Memo` | the converge-guard error (a cross-effect feedback loop fails closed past a fixed cap) + the cell/computed types |
| `RuntimeReactiveHost` | the real `ReactiveHost` over the signal core: native-`Disposable` `runLeafEffect`, `DisposableStack`-backed `scope()`, a cellKey **latch** (a surviving component instance keeps its state across a re-derive), `exportState()`, and cell-freeing on scope disposal |
| `diffKeyed`, `KeyedOp` | a pure, tree-shape-agnostic add/remove/move op list over two keyed sequences (complete, not LIS-minimized) |
| `applyKeyedDiff`, `KeyedReconcileHooks` | reconcile a retained list against a next-key order — reuse matched instances (consume-once), create new, and **dispose removed ones by identity** (the teardown contract) |
| `derive`, `DeriveOptions`, `DeriveResult` | the one-shot composition root — runs the `@metael/lang` walk inside one `change()`; surfaces a non-converging flush as an `ML-RT-CONVERGE` diagnostic |
| *(re-exported from `@metael/lang`)* `lowerEntry`, the port interfaces (`HostEnvironment`/`ReactiveHost`/`KeyMinter`), the value contract (`Arg`/`Region`/`LangWrapper`/`Scope`/…), the port helpers, and the test doubles (`PlainStorageHost`/`RecordingHostEnv`/`PathKeyMinter`) | so a domain imports the whole seam from a single place |

## How it composes with `@metael/lang`

The language package defines the three host-injection port *interfaces* and stays view-free. `@metael/runtime` provides the concrete pieces a domain wires together:

- **The reactive core** (`signal`/`memo`/`effect` + `change`) — vendored `@vue/reactivity` for signals/effects/scheduling, plus metael's own **synchronous** `change()` flush (the vendored core has no public synchronous flush) and a **converge guard** that fails closed on cross-effect feedback past a fixed cap.
- **`RuntimeReactiveHost`** — the real `ReactiveHost`. A reactive `let` is a cell; a reactive prop (a `Region`) is bound to a leaf effect that pipes its initial value synchronously and re-pipes on each dependent write. `runLeafEffect` returns a native `Disposable`; `scope()` is an owner boundary whose disposal tears down every cell + leaf effect allocated inside it — so a keyed **`remove`** can dispose a removed subtree without leaking (no lingering effects, no retained cell-key state).
- **The generic keyed-list diff** — the structural-reconciliation piece: given two keyed sequences it produces add/remove/move ops (`diffKeyed`), and `applyKeyedDiff` reconciles retained instances **plus owns disposal on removal, by instance identity**. A domain supplies the recursion over its own node shape and the patch application; the runtime supplies the diff + teardown.
- **`derive()`** — the composition root. It creates a fresh `RuntimeReactiveHost`, hands it to the caller's `HostEnvironment` (via `onHost`, before the walk, so `resolveCall` can register leaf effects), runs `lowerEntry` inside one `change()`, and returns `{ value, diagnostics, host }`. A domain builds its retained tree from the returned value, then drives updates through the host + the keyed diff.

## Usage

```ts
import { derive, RecordingHostEnv, PathKeyMinter } from '@metael/runtime';

const { value, diagnostics, host } = derive('component Story() { text("hi") }', {
  env: new RecordingHostEnv(),
  minter: new PathKeyMinter(),
});
// `value` = the raw host value(s) the walk produced (opaque to the runtime; the domain materializes them)
// `diagnostics` = [] on success; `host` owns the derive's cells + leaf effects
```

Determinism is inherited from `@metael/lang`: `result = f(source, data, seed, state)` — same source + same seed → identical host-value trace (seeded `rand`/`range` included), guarded by a conformance fixture that runs under the test doubles with no domain present. A companion disposal fixture proves a keyed `remove` tears down the removed subtree's cells + leaf effects with no post-removal effect and no retained cell-key state.

## Develop

```shell
npm run -w @metael/runtime typecheck
npm run -w @metael/runtime build     # → dist/ (.js + .d.ts, one per source module)
npx vitest run packages/runtime      # the suite (8 files / 55 tests)
```

Self-contained behind the port seam: `packages/runtime/src/` imports only `@metael/lang` + `@vue/reactivity` (nothing domain-specific) — enforced by an automated import-boundary test. See [../../AGENTS.md](../../AGENTS.md) for the load-bearing invariants and the editing guardrails.

## License

MIT.
