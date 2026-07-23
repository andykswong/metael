# @metael/runtime

[![metael](https://img.shields.io/badge/project-metael-blueviolet.svg?style=flat-square&logo=github)](https://github.com/andykswong/metael)
[![npm](https://img.shields.io/npm/v/@metael/runtime?style=flat-square&logo=npm)](https://www.npmjs.com/package/@metael/runtime)
[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](./LICENSE)

**The fine-grained reactive runtime + the real host-injection port implementations** — the back half of the [metael](../../README.md) substrate: the reactive core (`signal`/`memo`/`effect` + a synchronous `change()` + a converge guard), the generic keyed-list diff, the real `RuntimeReactiveHost`, the one-shot `derive()` composition root, and `composeEnvs` for merging single-vocabulary host environments. Depends only on [`@metael/lang`](../lang/README.md) + vendored `@vue/reactivity`.

## Install

```shell
npm install @metael/runtime    # the reactive runtime (pulls @metael/lang)
```

## Usage

The reactive core: a read tracks a dependency, a write schedules dependents, and `change()` is the synchronous batch/flush boundary (writes accumulate; scheduled effects drain to a fixed point):

```ts
import { signal, memo, effect, change } from '@metael/runtime';

const count = signal(1);
const doubled = memo(() => count.get() * 2);
effect(() => console.log('doubled =', doubled.get()));   // logs "doubled = 2"

change(() => count.set(5));                              // one flush → logs "doubled = 10"
```

`derive()` is the composition root: it runs the `@metael/lang` walk inside one `change()`, settles the initial reactive flush, and returns the raw host values + the host that owns their cells/effects:

```ts
import { derive, RecordingHostEnv, PathKeyMinter } from '@metael/runtime';

const { value, diagnostics, host } = derive('component Story() { text("hi") }', {
  env: new RecordingHostEnv(),
  minter: new PathKeyMinter(),
});
// `value` = the raw host value(s) the walk produced (opaque to the runtime; the domain materializes them)
// `diagnostics` = [] on success; `host` owns the derive's cells + leaf effects
```

The retained tree, materialization, and re-derive/reconcile are **domain-owned** — the runtime supplies the reusable machinery (reactive core, keyed diff, owner scopes) a domain composes into its own View.

## What's in the box

| Export | What it is |
|---|---|
| `signal`, `memo`, `effect`, `change` | the fine-grained reactive core; `change()` is THE synchronous batch/flush boundary |
| `ReactiveFlushError`, `Signal`, `Memo` | the converge-guard error (a cross-effect feedback loop fails closed past a fixed cap) + the cell/computed types |
| `RuntimeReactiveHost` | the real `ReactiveHost` over the signal core: native-`Disposable` `runLeafEffect`, `DisposableStack`-backed `scope()`, a cellKey **latch** (a surviving instance keeps its state across a re-derive), `exportState()`, and cell-freeing on scope disposal |
| `diffKeyed`, `KeyedOp` | a pure, tree-shape-agnostic add/remove/move op list over two keyed sequences |
| `applyKeyedDiff`, `KeyedReconcileHooks` | reconcile a retained list against a next-key order — reuse matched instances (consume-once), create new, and **dispose removed ones by identity** (the teardown contract) |
| `derive`, `DeriveOptions`, `DeriveResult` | the one-shot composition root — surfaces a non-converging flush as an `ML-RT-CONVERGE` diagnostic |
| `composeEnvs`, `ComposedHostEnv` | merge several single-vocabulary `HostEnvironment`s into one, reporting head-name collisions across vocabularies |
| *(re-exported from `@metael/lang`)* `lowerEntry`, the port interfaces (`HostEnvironment`/`ReactiveHost`/`KeyMinter`/`BindableHostEnv`), the value contract (`Region`/`LangWrapper`/`Scope`/…), and the test doubles (`PlainStorageHost`/`RecordingHostEnv`/`PathKeyMinter`) | so a domain imports the whole seam from a single place |

The reactive core is vendored `@vue/reactivity` for signals/effects/scheduling, plus metael's own **synchronous** `change()` flush (the vendored core has no public synchronous flush) and a converge guard. `RuntimeReactiveHost`'s `scope()` is an owner boundary whose disposal tears down every cell + leaf effect allocated inside it — so a keyed `remove` can dispose a removed subtree without leaking.

## Boundary

Determinism is inherited from `@metael/lang`: `result = f(source, data, seed, state)` — same source + same seed → identical host-value trace. `packages/runtime/src/` imports only `@metael/lang` + `@vue/reactivity` (nothing domain-specific), enforced by an automated import-boundary test.

## Develop

```shell
npm run -w @metael/runtime typecheck
npm run -w @metael/runtime build     # → dist/ (.js + .d.ts, one per source module)
npx vitest run packages/runtime      # the suite
```

From the repo root, `npm run docs:api:check` is the doc-coverage gate (every exported symbol needs a doc comment) and `npm run prepublishOnly` runs the full one-shot gate — `clean → build:packages → typecheck → lint → test → docs:api:check`.

See [../../AGENTS.md](../../AGENTS.md) for the load-bearing invariants and the editing guardrails.

## License

MIT — see [LICENSE](./LICENSE).
