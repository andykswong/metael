# @metael/std

[![metael](https://img.shields.io/badge/project-metael-blueviolet.svg?style=flat-square&logo=github)](https://github.com/andykswong/metael)
[![npm](https://img.shields.io/npm/v/@metael/std?style=flat-square&logo=npm)](https://www.npmjs.com/package/@metael/std)
[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](./LICENSE)

**The general-purpose standard library for the [metael](../../README.md) kernel** — the collection, string, structural, random, and datetime builtins as one injectable `BuiltinModule` (`STD_BUILTINS`) plus its tooling `stdProfile`. `@metael/lang` is a bare interpreter with no vocabulary of its own; `@metael/std` is that vocabulary, wired in at `evaluateProgram`.

## Install

```shell
npm install @metael/std    # the general stdlib builtins (pulls @metael/lang)
```

## Usage

Inject `STD_BUILTINS` at `evaluateProgram` and the collection/string/structural builtins are in scope:

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

`STD_BUILTINS` composes freely with other builtin modules (e.g. `@metael/math`'s `MATH_BUILTINS`) — pass them together in the `builtins` array. Every builtin is **pure and immutable-by-construction**: it returns a **new frozen** collection (or a scalar), never mutates an input, ticks the budget per call + per element (a large collection fails closed with `ML-LANG-BUDGET`), and a wrong-shape argument is a fail-loud `ML-LANG-BUILTIN-ARG` (never a thrown exception).

## What's in the box

`STD_BUILTINS` composes five sub-modules, each also exported on its own:

| Module | Builtins |
|---|---|
| `COLLECTION_BUILTINS` | `map`, `filter`, `reduce`, `some`, `every`, `find`, `findIndex`, `includes`, `sort` (total/stable/deterministic, NaN pinned), `slice`, `reverse` — array-only (strings bridge via `split`/`join`/`chars`). |
| `STRING_BUILTINS` | `split`, `join`, `chars`, `codePointAt`, `toUpperCase`, `toLowerCase`, `trim`, `format`. |
| `STRUCTURAL_BUILTINS` | `keys`, `values`, `entries`, `object` (array-of-pairs → object), `has` — `object`/`has` are `FORBIDDEN_KEYS`-guarded so a prototype-polluting key can never enter or be observed. |
| `RANDOM_BUILTINS` | `rand` — reads the seeded PRNG the language kernel owns, so a run is deterministic. |
| `DATETIME_BUILTINS` | `now` (wall-clock ms since the Unix epoch), `monotonic` (high-resolution elapsed time) — both read the host's **injected clock capability** via `ctx.clock()`, never an ambient `Date.now()`, so a run is replayable; no injected clock ⇒ fail loud (`ML-LANG-NO-CLOCK`). |

`defaultCompare` + `stableSort` (the deterministic sort primitives behind `sort`) are also exported for reuse. `stdProfile` is the matching tooling **profile** — each builtin's capability spec (`core`/`host` × `exact`/`gpu-tolerant`/`cpu-only`) plus its editor-hover metadata (`doc`/`params`/`returnDoc`), for a static classifier / a language service; compose it with your own profile.

## Boundary

`src/**` imports **only** `@metael/lang` — no third-party runtime dependency and no sibling `@metael/*` package. Enforced by an automated `boundary.test.ts`, so the standard library stays a thin, deterministic layer over the kernel's own truthiness/equality/stringify/forbidden-key primitives rather than re-deriving them.

## Develop

```shell
npm run -w @metael/std typecheck
npm run -w @metael/std build     # → dist/ (.js + .d.ts, one per source module)
npx vitest run packages/std      # the suite
```

From the repo root, `npm run docs:api:check` is the doc-coverage gate (0 undocumented exported symbols), and `npm run prepublishOnly` is the full pre-publish gate (`clean → build:packages → typecheck → lint → test → docs:api:check`).

See the root [README.md](../../README.md) for the package map, and [AGENTS.md](../../AGENTS.md) for the load-bearing invariants and editing guardrails.

## License

MIT — see [LICENSE](./LICENSE).
