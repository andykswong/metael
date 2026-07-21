# @metael/std

[![metael](https://img.shields.io/badge/project-metael-blueviolet.svg?style=flat-square&logo=github)](https://github.com/andykswong/metael)
[![npm](https://img.shields.io/npm/v/@metael/std?style=flat-square&logo=npm)](https://www.npmjs.com/package/@metael/std)
[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](./LICENSE)

**The general-purpose standard library for the [metael](../../README.md) kernel — the collection, string, structural, random, and datetime builtins as one injectable module.**

`@metael/lang` is a bare interpreter: it knows how to declare, compose, resolve, and react, but ships **no** general-purpose vocabulary of its own (only `range`, the bounded-loop primitive the compute-lowering gate + interpreter oracle depend on, stays a kernel intrinsic). `@metael/std` is that vocabulary — a `BuiltinModule` a consumer wires in at `evaluateProgram`, kept in lockstep with the language's own truthiness/equality/stringify/forbidden-key primitives rather than re-deriving them.

Every builtin is **pure and immutable-by-construction**: each returns a **new frozen** collection (or a scalar) and never mutates an input, ticks the budget per call + per element (a large collection fails closed with `ML-LANG-BUDGET`), and a wrong-shape argument is a fail-loud `ML-LANG-BUILTIN-ARG` (never a thrown exception). Callbacks may be an arrow or a user `function`.

## What's in the box

`STD_BUILTINS` is the module a consumer injects; it composes five sub-modules, each also exported on its own:

| Module | Builtins |
|---|---|
| `COLLECTION_BUILTINS` | `map`, `filter`, `reduce`, `some`, `every`, `find`, `findIndex`, `includes`, `sort` (total/stable/deterministic, NaN pinned), `slice`, `reverse` — array-only (strings bridge via `split`/`join`/`chars`). |
| `STRING_BUILTINS` | `split`, `join`, `chars`, `codePointAt`, `toUpperCase`, `toLowerCase`, `trim`, `format`. |
| `STRUCTURAL_BUILTINS` | `keys`, `values`, `entries`, `object` (array-of-pairs → object), `has` — `object`/`has` are `FORBIDDEN_KEYS`-guarded so a prototype-polluting key can never enter or be observed. |
| `RANDOM_BUILTINS` | `rand` — reads the seeded PRNG the language kernel owns, so a run is deterministic (`result = f(source, data, seed, state)`). |
| `DATETIME_BUILTINS` | `now` (wall-clock ms since the Unix epoch), `monotonic` (a high-resolution elapsed-time reading) — both read the host's **injected clock capability** via `ctx.clock()`, never an ambient `Date.now()`/`performance.now()`, so a run is replayable; a host that injects no clock makes them fail loud (`ML-LANG-NO-CLOCK`). |

`defaultCompare` + `stableSort` (the deterministic sort primitives behind `sort`) are also exported for reuse.

## Usage

```ts
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { STD_BUILTINS } from '@metael/std';

const { value, diagnostics } = evaluateProgram(
  `map(filter(data.items, (x) => x > 2), (x) => x * 10)`,
  {
    data: { items: [1, 2, 3, 4] },
    seed: 1,
    host: new PlainStorageHost(),
    env: new RecordingHostEnv(),
    builtins: [STD_BUILTINS],       // ← inject the standard library
  },
);

value;         // → [30, 40]
diagnostics;   // → []
```

`STD_BUILTINS` composes freely with other builtin modules (e.g. `@metael/math`'s `MATH_BUILTINS`) — pass them together in the `builtins` array.

## Boundary

`src/**` imports **only** `@metael/lang` — no third-party runtime dependency and no sibling `@metael/*` package (`runtime`/`vdom`/`gpu`/`math`). This is enforced by an automated `boundary.test.ts` (the build/publish contract rests on it), so the standard library stays a thin, deterministic layer over the kernel's own semantics.

## Develop

```shell
npm run -w @metael/std typecheck
npm run -w @metael/std build     # → dist/ (.js + .d.ts, one per source module)
npx vitest run packages/std      # the suite
```

See the root [README.md](../../README.md) for install + the package map, and [AGENTS.md](../../AGENTS.md) for the load-bearing invariants and editing guardrails.

## License

MIT.
