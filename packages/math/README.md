# @metael/math

[![metael](https://img.shields.io/badge/project-metael-blueviolet.svg?style=flat-square&logo=github)](https://github.com/andykswong/metael)
[![npm](https://img.shields.io/npm/v/@metael/math?style=flat-square&logo=npm)](https://www.npmjs.com/package/@metael/math)
[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](./LICENSE)

**The numeric standard library for the [metael](../../README.md) kernel.** A **zero-dependency core** of scalar/vec/mat/quaternion/transform math (column-major, out-param convention) that's usable with no language present, plus `@metael/math/lang` — the binding that boxes the same core as immutable/reactive custom values, the injectable `MATH_BUILTINS` module, and the tooling `mathProfile`.

## Install

```shell
npm install @metael/math    # core = zero deps; the @metael/math/lang subpath pulls @metael/lang
```

## Usage

The **core** — plain functions over flat number stores. Vector/matrix/transform ops use an out-param convention `out = op(...args, out?)`: omit `out` for a fresh array, pass one to write into it (aliasing an input as the output is always safe).

```ts
import { transformation, perspective, normalize } from '@metael/math';

const model = transformation([0, 1, 0], [0, 0, 0, 1], [2, 2, 2]);   // T·R·S, column-major, fresh number[]
const proj  = perspective(Math.PI / 4, 16 / 9, 0.1, 100);            // RH, Y-up, [-1,1] clip-z
const dir   = new Array(3);
normalize([3, 0, 4], dir);                                           // write into an out-param → [0.6, 0, 0.8]
```

The **language binding** — inject `MATH_BUILTINS` at `evaluateProgram` to get the vec/mat/quat/scalar vocabulary in metael source:

```ts
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { MATH_BUILTINS } from '@metael/math/lang';

const { value } = evaluateProgram(
  `dot(vec3(1, 2, 3), vec3(4, 5, 6))`,           // → 32
  { host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] },
);
```

## What's in the box

Two layers behind subpath exports, so a caller takes exactly what it needs:

| Subpath | What it is | Depends on |
|---|---|---|
| `@metael/math` | The **core** — a plain, zero-dependency numeric library. Usable with no language present. | nothing |
| `@metael/math/lang` | The **language binding** — the same core boxed in metael's immutable/reactive custom-value protocol, plus `MATH_BUILTINS` and `mathProfile`. | `@metael/lang` + `@metael/math` |

**Core** (`@metael/math`) — every op operates on a `Float32Array | Float64Array | number[]`, and **precision follows the store** (f32 rounds via `Math.fround`; f64 / `number[]` are exact):

- **Scalar** — `min`/`max`/`abs`/`sign`/`floor`/`ceil`/`round` (half-to-even)/`clamp`/`trunc`/`mod` (floored, GLSL/WGSL — sign follows the divisor)/`degrees`/`radians`, `sqrt`/`pow`/`exp`/`exp2`/`log`/`log2`/`inverseSqrt`/`fract`/`step`/`mix`/`smoothstep`, trig/inverse-trig/hyperbolic (`sin`…`atan2`…`atanh`). Domain-guarded functions return `NaN` out of domain (not `±Inf`), so vector results stay bit-identical to a shader.
- **vec** — componentwise `add`/`sub`/`mul`/`div`/`scale`, `dot`/`cross`/`normalize`/`length`/`distance`/`reflect`/`refract`/`faceforward`.
- **mat** — **column-major** flat storage (element `(row, col)` at flat index `col*rows + row`); `matmul`, `matColumn`, `transpose`/`determinant`/`inverse`, componentwise `matrixCompMult` + scalar `matScale`, plus `mat*vec` / `mat*mat`.
- **quat** — quaternions as a `vec4` laid out `(x, y, z, w)` (identity `[0,0,0,1]`): `qmul`/`qconj`/`qinvert`/`qaxisangle`/`qrotate`/`qslerp`/`qmat`.
- **transform / camera** — TRS `transformation` (M = T·R·S) + `decompose`, and camera matrices `perspective`/`ortho`/`lookAt` — **right-handed, Y-up, `[-1, 1]` clip-z**.
- **bits** — 32-bit unsigned bit ops (`countOneBits`/`reverseBits`), coercing via `x >>> 0` to match shading-language semantics.

**Language binding** (`@metael/math/lang`) — boxes results as immutable/reactive custom values, applies the fail-loud domain-guard diagnostics the surface expects, and threads element precision through every operator so an f64 chain never silently downcasts to f32:

| Export | What it is |
|---|---|
| `MATH_BUILTINS` | The numeric builtin **module** a consumer injects at `evaluateProgram` — the vec/mat/quat/scalar/bit vocabulary as one `BuiltinModule`. |
| `makeVec`, `makeMat`, `identityMat`, `vecStoreOf` | The vec/mat custom-value **builders** + descriptors (a `vec` is a single-column `mat`; immutable value types over the descriptor protocol). |
| `makeTypedArray`, `BUFFER_KINDS`, `TYPED_ARRAY_DESCRIPTORS`, `BufferKind` | The linear-buffer custom values — `f32`/`f64`/`i32`/`u32` typed arrays, the only in-place-mutable (and reactively-tracked) values (`let` = writable, `const` = frozen). |
| `mathProfile`, `isMathBuiltin` | The tooling **profile** — each builtin's capability spec (`core`/`host` × `exact`/`gpu-tolerant`/`cpu-only`) plus its editor-hover metadata (`doc`/`params`/`returnDoc`) and the `vec2`/`vec3`/`vec4` custom-type projections, for a static classifier / a language service. Compose it with your own profile. |

## Boundary

The **core** (`src/core/`) imports **nothing** — no bare specifier, no relative escape into the language layer. The **binding** (`src/lang/`) imports **only** `@metael/lang` (the descriptor protocol + registry seam) and `@metael/math` (the arithmetic core). Both invariants are enforced by an automated `boundary.test.ts`, keeping the core adoptable by a non-metael engine and the binding in lockstep with the language's own semantics.

## Develop

```shell
npm run -w @metael/math typecheck
npm run -w @metael/math build     # → dist/{core,lang} (.js + .d.ts, one per source module)
npx vitest run packages/math      # the suite
```

From the repo root, `npm run docs:api:check` is the doc-coverage gate (0 undocumented exported symbols), and `npm run prepublishOnly` is the full pre-publish gate (`clean → build:packages → typecheck → lint → test → docs:api:check`).

See the root [README.md](../../README.md) for the package map, and [AGENTS.md](../../AGENTS.md) for the load-bearing invariants and editing guardrails.

## License

MIT — see [LICENSE](./LICENSE).
