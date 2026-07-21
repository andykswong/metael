# @metael/math

[![metael](https://img.shields.io/badge/project-metael-blueviolet.svg?style=flat-square&logo=github)](https://github.com/andykswong/metael)
[![npm](https://img.shields.io/npm/v/@metael/math?style=flat-square&logo=npm)](https://www.npmjs.com/package/@metael/math)
[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](./LICENSE)

**The numeric standard library for the [metael](../../README.md) kernel — a zero-dependency core of scalar/vec/mat/quaternion/transform math, and the language binding that boxes it as reactive, immutable custom values.**

`@metael/math` ships in **two layers** behind subpath exports, so a caller takes exactly what it needs:

| Subpath | What it is | Depends on |
|---|---|---|
| `@metael/math` | The **core** — a plain, zero-dependency numeric library. Usable with no language present. | nothing |
| `@metael/math/lang` | The **language binding** — wraps the same core in metael's boxed/immutable/reactive custom-value protocol, plus the builtin module, the lowering catalog, and the capability classifier. | `@metael/lang` + `@metael/math` |

## `@metael/math` — the core (zero deps)

A plain numeric library over flat stores. Every op operates on a `Float32Array | Float64Array | number[]`, and **precision follows the store** (f32 rounds via `Math.fround`; f64 / `number[]` are exact) — no rounding is baked into the core itself. Vector/matrix/transform ops use an **out-param convention** `out = op(...args, out?)`: omit `out` for a fresh array, pass one to write into it, and aliasing an input as the output is always safe. It imports **nothing** — a scene-graph or 3D engine can adopt it as its transform-math library without pulling in the language.

- **Scalar** — `min`/`max`/`abs`/`sign`/`floor`/`ceil`/`round` (half-to-even, for cross-target bit-identity)/`clamp`/`trunc`/`degrees`/`radians`, `sqrt`/`pow`/`exp`/`exp2`/`log`/`log2`/`inverseSqrt`/`fract`/`step`/`mix`/`smoothstep`, trig `sin`/`cos`/`tan`, inverse-trig `asin`/`acos`/`atan`/`atan2`, hyperbolic `sinh`/`cosh`/`tanh` (+ `asinh`/`acosh`/`atanh`). Domain-guarded functions return `NaN` out of domain (not `±Inf`), so componentwise vector results stay bit-identical to a shader; the loud scalar-position diagnostics are the language binding's concern.
- **vec** — componentwise `add`/`sub`/`mul`/`div`/`scale`, `dot`/`cross`/`normalize`/`length`/`distance`/`reflect`/`refract`/`faceforward`.
- **mat** — **column-major** flat storage (element `(row, col)` at flat index `col*rows + row`); `matmul`, `matColumn`, `transpose`/`determinant`/`inverse`, plus `mat*vec` / `mat*mat`.
- **quat** — quaternions as a `vec4` laid out `(x, y, z, w)` (imaginary `xyz` + real `w`; identity `[0,0,0,1]`): `qmul`/`qconj`/`qinvert`/`qaxisangle`/`qrotate`/`qslerp`/`qmat`.
- **transform / camera** — TRS `transformation` (M = T·R·S) + `decompose`, and the camera matrices `perspective`/`ortho`/`lookAt` — **right-handed, Y-up, `[-1, 1]` clip-z** (an OpenGL-style projection; a `[0,1]` clip-z + Y-flip is a downstream normalization concern, deliberately not baked in).
- **bits** — 32-bit unsigned bit ops (`countOneBits`/`reverseBits`), each coercing via `x >>> 0` to match the shading-language semantics.

```ts
import { transformation, perspective, normalize } from '@metael/math';

const model = transformation([0, 1, 0], [0, 0, 0, 1], [2, 2, 2]);   // T·R·S, column-major, fresh number[]
const proj  = perspective(Math.PI / 4, 16 / 9, 0.1, 100);            // RH, Y-up, [-1,1] clip-z
const dir   = new Array(3);
normalize([3, 0, 4], dir);                                           // write into an out-param → [0.6, 0, 0.8]
```

## `@metael/math/lang` — the language binding

The same arithmetic, wrapped for the metael language surface. It boxes results as **immutable, reactive custom-value instances** through metael's Symbol-keyed descriptor protocol, applies the fail-loud domain-guard diagnostics the language surface expects (the core returns raw `NaN`; the binding raises the loud diagnostic in scalar position), and threads element precision through every operator so a chain of f64 operands never silently downcasts to f32.

| Export | What it is |
|---|---|
| `MATH_BUILTINS` | The numeric builtin **module** a consumer injects at `evaluateProgram` (`{ …, builtins: [MATH_BUILTINS] }`) — the vec/mat/quat/scalar/bit vocabulary as one `BuiltinModule`. |
| `makeVec`, `makeMat`, `identityMat`, `vecStoreOf` | The vec/mat custom-value **builders** + descriptors (a `vec` is a single-column `mat`; immutable value types over the descriptor protocol). |
| `makeTypedArray`, `BUFFER_KINDS`, `TYPED_ARRAY_DESCRIPTORS`, `BufferKind` | The linear-buffer custom values — `f32`/`f64`/`i32`/`u32` typed arrays, the only in-place-mutable (and reactively-tracked) values (a `let` buffer is writable, a `const` buffer is frozen). |
| `BUILTINS`, `isBuiltin`, `BuiltinSpec`, `BuiltinProfile`, `Portability` | The machine-readable **catalog** of every builtin's capability profile (`core`/`host`) + cross-target portability class (`exact`/`gpu-tolerant`/`cpu-only`) — the single source of truth for "which names are intrinsics" + the input a static classifier or codegen consumes. |
| `classifyProfile`, `ProfileResult` | The static **capability classifier** — decides a function's core-compliance from its AST (metadata + a pure classifier; no codegen/dispatch engine here). |

```ts
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { MATH_BUILTINS } from '@metael/math/lang';

const { value } = evaluateProgram(
  `dot(vec3(1, 2, 3), vec3(4, 5, 6))`,           // → 32
  { host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] },
);
```

## Boundary

The **core** (`src/core/`) imports **nothing** — no bare specifier, no relative escape into the language layer. The **binding** (`src/lang/`) imports **only** `@metael/lang` (the descriptor protocol + the registry seam) and `@metael/math` (the arithmetic core). Both invariants are enforced by an automated `boundary.test.ts`. This keeps the core adoptable by a non-metael engine and keeps the binding in lockstep with the language's own semantics rather than re-deriving them.

## Develop

```shell
npm run -w @metael/math typecheck
npm run -w @metael/math build     # → dist/{core,lang} (.js + .d.ts, one per source module)
npx vitest run packages/math      # the suite
```

See the root [README.md](../../README.md) for install + the package map, and [AGENTS.md](../../AGENTS.md) for the load-bearing invariants and editing guardrails.

## License

MIT.
