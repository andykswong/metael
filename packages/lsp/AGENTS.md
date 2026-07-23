# @metael/lsp — Agent Guidelines

`@metael/lsp` = metael language services. Two layers sit behind subpath exports: **`@metael/lsp/service`** is a PROTOCOL-FREE analysis engine — it speaks char offsets and returns `Svc*` records, imports no LSP protocol; **`@metael/lsp`** (the main entry) is the LSP wire-protocol shell — the sole offset↔`Position` + `Svc*`↔wire marshaler — and **`@metael/lsp/worker`** is the browser Web Worker transport. The package CONSUMES the language via `@metael/lang` (its total lexer/parser/printer + the `@metael/lang/profile` vocabulary-metadata layer); it never re-implements a language fact. Read the root [AGENTS.md](../../AGENTS.md) first for the kernel invariants this builds on.

## Architecture / layer map

The `src/` tree is a clean split: everything under `src/service/` is offset-only and protocol-free; everything directly under `src/` is the protocol shell.

### `src/service/` — the protocol-free engine (imports NO `vscode-languageserver*`)

- **`line-index.ts`** — `LineIndex`: offset ↔ UTF-16 `{ line, character }` (0-based), built once per document version.
- **`document.ts`** — `Document`: an immutable per-version doc that memoizes `lex`/`parse` (both total in `@metael/lang`) lazily; an edit makes a fresh `Document`.
- **`results.ts`** — the `Svc*` result records (the engine's output vocabulary): `SvcSpan`/`SvcDiagnostic`/`SvcToken`/`SvcTokenKind`/`SvcCompletion`/`SvcHover`/`SvcSignature`/`SvcFold`/`SvcSelection`/`SvcEdit`/`SvcLens`. All positions are char offsets — no protocol type appears here.
- **`scope-model.ts`** — `ScopeModel`: a static scope tree (the resolver the language lacks — the language binds names at eval time). Walks the AST once, recording a `Binding` per `const`/`let`/`function`/`component`/`param`/`for` name with its `[scopeStart, scopeEnd]` visibility range. A block-body scope snaps its end to the **closing brace** (recovered from the token stream's matched brace pairs), so completion on a blank line above a `}` still sees the block's bindings; visibility is forward-only from the declaration.
- **`language-service.ts`** — `LanguageService`, the façade: `openDocument`/`updateDocument`/`closeDocument`/`setProfile`/`hasDocument`/`lineIndexFor` + the nine analyses. It owns the doc/scope/profile maps and lazily builds the `ScopeModel`. Analyses that need vocabulary short-circuit to empty when no `Profile` is set.
- **`service/analyses/`** — one pure `computeX(doc, [scope], [profile], [offset])` per analysis:
  - `diagnostics.ts` — reads `doc.parse.diagnostics` → `SvcDiagnostic[]` (span-less codes get a whole-doc range; `PROFILE` codes are info, else error). The `diagnostics(uri)` façade method APPENDS the scope-check pass when a scope+profile exist.
  - `scope-check.ts` — reads the AST + scope + profile → `SvcDiagnostic[]`: undeclared value-reads (`ML-LANG-UNKNOWN-VAR`) + block-scope redeclarations (`ML-LANG-REDECL`). Mirrors the evaluator (see the invariant below). Not a standalone façade method — it's folded into `diagnostics()`.
  - `completion.ts` — reads doc + scope + profile + offset → `SvcCompletion[]`: member candidates after a `.` (from a resolved receiver type's `profile.types` members), else visible bindings ∪ `profile.builtins` ∪ `profile.heads` ∪ `range` ∪ `KEYWORDS_SET`.
  - `hover.ts` — reads doc + scope + profile + offset → `SvcHover | null`: a builtin/head/type card from the profile, or a local-binding card, as markdown.
  - `signature.ts` — reads doc + profile + offset → `SvcSignature | null`: params + active-param index resolved from `profile.heads`/`profile.builtins` for the enclosing call.
  - `semantic-tokens.ts` — reads doc + scope + profile → `SvcToken[]`: every lexed token classified via `lexicalCategory(type)`, identifiers refined against the profile (builtin/head/type) then the scope.
  - `folding.ts` — reads doc (tokens + AST) → `SvcFold[]`: one fold per brace-delimited block body, snapped to the `{`..`}` pair. No profile.
  - `selection.ts` — reads doc + offsets → `SvcSelection[]`: the widening chain of enclosing subtree extents per offset. No profile.
  - `format.ts` — reads doc → `SvcEdit[]`: one whole-document edit via `printProgram` (empty when the parse carries errors or the text is already canonical). No profile.
  - `capability-lens.ts` — reads doc + profile → `SvcLens[]`: per top-level `function`/`component`, whether its body is GPU/WASM-lowerable, straight from `classifyProfile(fn, profile)` (in `@metael/lang/profile`).

### `src/` — the SHELL (imports the protocol; NOT under `service/`)

- **`marshal.ts`** — the SOLE offset↔`Position` / `Svc*`↔wire-type converters (`spanToRange`, `positionToOffset`, `toDiagnostic`, `toCompletionItem`, `toHover`, `toSignatureHelp`, `toFoldingRange`, `toSelectionRange`, `toTextEdit`, `encodeSemanticTokens`). Uses a document's `LineIndex` to turn offsets into positions.
- **`server.ts`** — `createServer(reader, writer, opts)`: wires a `LanguageService` to a JSON-RPC connection over any `MessageReader`/`MessageWriter`. Domain-agnostic via the injected `opts.resolveProfile(id) => Profile`. Advertises `CAPABILITIES` on `initialize`, keeps full-text sync, publishes diagnostics on open/change, answers the standard requests, and adds the custom `metael/setProfile` + `metael/capabilityLens` methods. Exports `ServerOptions` + `CapabilityLensItem`.
- **`capabilities.ts`** — `CAPABILITIES` (the advertised feature set) + `TOKEN_LEGEND` (the semantic-token wire legend).
- **`index.ts`** — the main-entry barrel: re-exports `createServer`, `ServerOptions`, `CapabilityLensItem`, `CAPABILITIES`, `TOKEN_LEGEND`.
- **`worker/index.ts`** — `startWorkerServer(scope, opts)`: binds `createServer` to `BrowserMessageReader`/`BrowserMessageWriter`.

## Load-bearing invariants

- **The service↛protocol boundary.** `src/service/**` NEVER imports `vscode-languageserver*` — guarded by `src/service/boundary.test.ts` (a source scan). The engine is offset-only + protocol-free so any host (an editor, a Web Worker, a headless test) can reuse it. ALL `Position`/wire marshaling lives ONLY in `src/marshal.ts` (the shell). Do not leak a protocol type into the engine, and do not put a second marshaler anywhere but `marshal.ts`.
- **Domain-agnosticism.** The shell NEVER imports a concrete profile. Vocabulary arrives ONLY via the injected `resolveProfile(id) => Profile` (from the client's `initializationOptions.profileId` or a `metael/setProfile` request). Do not couple the shell to a domain.
- **`Profile` is the vocabulary seam.** Every vocabulary-aware analysis is `(source, Profile) → answers`; the engine knows nothing of which heads/builtins/types exist except through the injected `@metael/lang/profile` `Profile`. Completion/hover/signature/semantic-tokens/scope-check all read `profile.builtins`/`profile.heads`/`profile.types`. (Diagnostics' lex/parse pass, folding, selection, and format are the profile-free analyses.)
- **Single source of truth for language facts — NO re-encoding.** The LSP must NOT hardcode a copy of any language fact; it derives them from `@metael/lang`:
  - reserved keywords ← `KEYWORDS_SET`; token→colour category ← `lexicalCategory(type)` (an exhaustive `Record<TokenType, LexicalCategory>` in the lexer — adding a `TokenType` is a compile error there until it's classified).
  - the semantic-token wire legend `TOKEN_LEGEND` (capabilities.ts) is pinned to the `SvcTokenKind` union via `satisfies readonly SvcTokenKind[]` + the exact-set assertion in `capabilities.test.ts`.
  - the undeclared-var/redecl scope-check MIRRORS the evaluator's `evalIdent`/`env.hasOwn` semantics and is deliberately CONSERVATIVE — its "declared" set is a SUPERSET of the evaluator's env, so it can only ever false-NEGATIVE, never false-POSITIVE on valid code. Do not make it flag anything the evaluator wouldn't (e.g. call heads, member/entry-key strings, `for` bindings, assign-target idents are never value-reads; `let` redecl is excluded because it's gated on `insideComponent`, which a static pass can't reproduce).

## When the language evolves, update these — the drift checklist

Keeping the LSP current as the language grows is the point of this package's design. Keyed to the kind of language change:

- **Add/remove a keyword or `TokenType`.** The source is `@metael/lang`'s lexer: `KEYWORDS_SET` + the `Record<TokenType, LexicalCategory>` update THERE. That `Record` FORCES a typecheck failure until the new token is classified — the LSP's completion keywords and semantic-token colouring then follow AUTOMATICALLY (no LSP edit for the sets). Just verify the `semantic-tokens` + `completion` tests still pass.
- **Add a `SvcTokenKind`** (a new semantic-token colour). Add it to the `SvcTokenKind` union in `results.ts` AND to `TOKEN_LEGEND` (capabilities.ts) — IN THE SAME ORDER, because the legend index is the wire contract. The `satisfies` + `capabilities.test.ts` catch a mismatch; `encodeSemanticTokens` uses `legend.indexOf` and silently DROPS a kind absent from the legend, so an unlisted kind just won't colour.
- **Add a builtin / head / custom type to a domain.** It flows in via that domain's `Profile`. The LSP needs NO change — completion/hover/signature/semantic-tokens/scope-check read the injected profile. (Exporting the `Profile` is the domain package's job, not the LSP's.)
- **Add an AST node kind / `Stmt` / `Expr` variant.** Update the AST walks that switch on `kind`: `scope-model.ts` (`walkStmt`/`walkExpr`) and `scope-check.ts` (the value-read walk) are the exhaustive per-kind switches; add a case if the new node opens a scope, binds a name, or holds value-read children. `folding.ts` switches on `kind` only to pick which field holds a block body (add a branch if the node introduces a new foldable block). `selection.ts` walks children generically (no per-kind switch) and needs no edit. `capability-lens.ts` delegates to `classifyProfile` — update that in `@metael/lang/profile`, not here. These walks are GUARD-TOLERANT of unknown kinds (a new kind won't crash — it's ignored until handled), so ADD a test when you add a node so the silent-ignore doesn't hide a gap.
- **Add a diagnostic code the evaluator emits statically-detectably.** Consider whether the `scope-check` pass should surface it too, keeping the no-false-positive discipline (only report what the evaluator would).
- **Add an LSP feature (a new analysis).** Touch all of: a `computeX` under `service/analyses/`; a method on `LanguageService`; a `Svc*` result type in `results.ts`; a marshaler in `marshal.ts`; a handler (+ any capability) in `server.ts`; an entry in `CAPABILITIES`; and an e2e round-trip in `server.test.ts`.

## Testing

Two layers. Per-analysis pure `computeX` unit tests (node) assert offset-based `Svc*` shapes directly. The e2e `server.test.ts` round-trips ALL protocol handlers through an in-memory JSON-RPC channel, asserting the MARSHALED wire shapes. `worker/worker.browser.test.ts` proves the browser transport in Chromium over a `MessageChannel`. `service/boundary.test.ts` enforces the protocol-free engine; `capabilities.test.ts` pins the legend↔union set. Keep them green and add a test with any change — especially an e2e round-trip for a new handler.

## Build / verify

`npm run -w @metael/lsp typecheck` / `build`; `npx vitest run packages/lsp` (node + the Chromium worker test). From the repo root, `npm run docs:api:check` is the doc-coverage gate (every exported symbol needs a doc comment) and `npm run prepublishOnly` runs the full one-shot gate — `clean → build:packages → typecheck → lint → test → docs:api:check`.

---

Root [AGENTS.md](../../AGENTS.md) — kernel invariants + editing guardrails. [README.md](./README.md) — install + the subpath surface.
