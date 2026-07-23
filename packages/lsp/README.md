# @metael/lsp

[![metael](https://img.shields.io/badge/project-metael-blueviolet.svg?style=flat-square&logo=github)](https://github.com/andykswong/metael)
[![npm](https://img.shields.io/npm/v/@metael/lsp?style=flat-square&logo=npm)](https://www.npmjs.com/package/@metael/lsp)
[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](./LICENSE)

**metael language services: a protocol-free analysis engine (`(source, Profile) → answers` in char offsets), a Language Server Protocol shell that marshals those answers to the wire, and a browser Web Worker transport.**

## Install

```shell
npm install @metael/lsp   # pulls @metael/lang + vscode-languageserver-protocol
```

## Usage

The engine is usable with **no transport** — speak char offsets and get plain `Svc*` records back. Open a document, inject a vocabulary `Profile`, and pull any analysis:

```ts
import { LanguageService } from '@metael/lsp/service';
import { composeProfiles, coreIntrinsicsProfile } from '@metael/lang/profile';
import { stdProfile } from '@metael/std';   // any vocabulary package publishes a Profile

const svc = new LanguageService();
svc.openDocument('mem://a.ml', 'map(xs, (x) => x * 2)', 1);
svc.setProfile('mem://a.ml', composeProfiles(coreIntrinsicsProfile, stdProfile));

const items = svc.completion('mem://a.ml', 3);   // → SvcCompletion[] at char offset 3
```

To speak LSP, wrap the engine in the shell over any transport — it maps `Position`↔offset and `Svc*`↔wire types for you:

```ts
import { createServer } from '@metael/lsp';
import type { Profile } from '@metael/lang/profile';

const server = createServer(reader, writer, {
  // Domain-agnostic: the host maps an opaque profile id → the Profile to analyse with.
  resolveProfile: (id: string | undefined): Profile => myProfile,
});
server.listen();
```

In a browser, `startWorkerServer(self, opts)` from `@metael/lsp/worker` binds `createServer` to a dedicated Web Worker's message port (via `vscode-languageserver-protocol/browser`).

## At a glance

Three layers behind subpath exports, so a caller takes exactly what it needs:

| Subpath | What it is | Depends on |
|---|---|---|
| `@metael/lsp/service` | The **protocol-free analysis engine** — `(source, Profile) → Svc*` answers in char offsets. Imports no LSP protocol type. | `@metael/lang` (+ `@metael/lang/profile`) |
| `@metael/lsp` | The **LSP wire-protocol shell** (main entry) — `createServer` over any transport, plus the sole offset↔`Position` / `Svc*`↔wire marshaler (`CAPABILITIES`, `TOKEN_LEGEND`). | `@metael/lsp/service` + `vscode-languageserver-protocol` |
| `@metael/lsp/worker` | The **browser Web Worker transport** — `startWorkerServer` binds `createServer` to the worker's message port. | `@metael/lsp` + `vscode-languageserver-protocol/browser` |

The engine answers nine offset-based analyses: **diagnostics · completion · hover · signature help · semantic tokens · folding · selection ranges · format · capability lens**. See [AGENTS.md](./AGENTS.md) for the per-analysis behavior, the architecture, and the editing guardrails.

## Boundary

The analysis engine (`src/service/`) imports **no** `vscode-languageserver*` type — it is offset-only and protocol-free (enforced by `boundary.test.ts`). All offset↔`Position` and `Svc*`↔wire marshaling lives in exactly one place, the shell (`src/`), keeping the protocol dependency at the edge and the engine reusable in any host with no LSP layer.

## Develop

```shell
npm run -w @metael/lsp typecheck
npm run -w @metael/lsp build      # → dist/ (.js + .d.ts, one per source module)
npx vitest run packages/lsp       # the suite (node + a Chromium worker-transport proof)
```

From the repo root, `npm run docs:api:check` is the doc-coverage gate (every exported symbol needs a doc comment) and `npm run prepublishOnly` runs the full one-shot gate — `clean → build:packages → typecheck → lint → test → docs:api:check`.

See the root [README.md](../../README.md) for the package map, [AGENTS.md](./AGENTS.md) for this package's architecture and change guidance, and the root [AGENTS.md](../../AGENTS.md) for the load-bearing kernel invariants.

## License

MIT — see [LICENSE](./LICENSE).
