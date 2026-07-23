// The main-thread half of the playground's language intelligence: it spawns the LSP server as a module
// Web Worker and speaks JSON-RPC to it over `vscode-languageserver-protocol`. CodeMirror extensions call
// these thin promise methods with plain character offsets; the client owns the per-document `LineIndex`
// (protocol-free, safe on the main thread) that converts a CM6 offset → an LSP `Position` for a request and
// each response's `Range`/`Position` back to offset(s) CM6 can consume. The worker runs the analysis engine.
import { BrowserMessageReader, BrowserMessageWriter } from 'vscode-languageserver-protocol/browser';
import {
  createProtocolConnection,
  InitializeRequest,
  CompletionRequest,
  HoverRequest,
  SignatureHelpRequest,
  SemanticTokensRequest,
  DidOpenTextDocumentNotification,
  DidChangeTextDocumentNotification,
  DidCloseTextDocumentNotification,
  PublishDiagnosticsNotification,
} from 'vscode-languageserver-protocol';
import type {
  ProtocolConnection,
  CompletionItem,
  SignatureHelp,
} from 'vscode-languageserver-protocol';
import { LineIndex } from '@metael/lsp/service';
import { TOKEN_LEGEND } from '@metael/lsp';
import type { CapabilityLensItem } from '@metael/lsp';

/** A diagnostic reversed to CM6 char offsets. */
export interface ClientDiagnostic {
  /** The inclusive start offset (UTF-16 code-unit index). */
  readonly from: number;
  /** The exclusive end offset (UTF-16 code-unit index). */
  readonly to: number;
  /** The LSP numeric severity (1 = error … 4 = hint). */
  readonly severity: number;
  /** The stable machine-readable diagnostic code. */
  readonly code?: string | number;
  /** The human-readable message. */
  readonly message: string;
}

/** A hover reversed to CM6 char offsets. */
export interface ClientHover {
  /** The start offset of the described span. */
  readonly from: number;
  /** The end offset of the described span. */
  readonly to: number;
  /** The hover content as markdown. */
  readonly markdown: string;
}

/** One decoded semantic token: a CM6 offset span plus its legend kind. */
export interface ClientSemanticToken {
  /** The start offset of the token. */
  readonly from: number;
  /** The end offset of the token. */
  readonly to: number;
  /** The token kind, resolved from the server's {@link TOKEN_LEGEND}. */
  readonly kind: string;
}

/** A capability lens reversed to CM6 char offsets. */
export interface ClientLens {
  /** The start offset of the annotated span. */
  readonly from: number;
  /** The end offset of the annotated span. */
  readonly to: number;
  /** The label shown for the lens. */
  readonly label: string;
  /** Whether the covered construct can be lowered. */
  readonly lowerable: boolean;
  /** The explanations backing the `lowerable` verdict. */
  readonly reasons: readonly string[];
}

/** A listener notified whenever the server publishes diagnostics for a document. */
export type DiagnosticsListener = (uri: string, diagnostics: readonly ClientDiagnostic[]) => void;

/** A promise-based client that spawns the metael LSP server in a Web Worker and translates between CM6 char
 *  offsets and LSP positions. Construct it once per editor, `initialize` with a profile id, then keep the
 *  server's document view in sync via `didOpen`/`didChange`/`didClose` and query it with the request methods. */
export class LspClient {
  private readonly worker: Worker;
  private readonly conn: ProtocolConnection;
  /** Per-uri line boundaries for the latest synced text, so offsets ⇄ positions convert without the server. */
  private readonly lineIndexes = new Map<string, LineIndex>();
  private readonly diagListeners: DiagnosticsListener[] = [];
  /** Resolves once `initialize` has completed; request methods await it so an early call is still safe. */
  private ready: Promise<void> = Promise.resolve();
  private disposed = false;

  constructor() {
    this.worker = new Worker(new URL('./lsp-worker.ts', import.meta.url), { type: 'module' });
    this.conn = createProtocolConnection(
      new BrowserMessageReader(this.worker),
      new BrowserMessageWriter(this.worker),
    );
    // Register the diagnostics handler before listening so no early publish is dropped.
    this.conn.onNotification(PublishDiagnosticsNotification.type, (params) => {
      const li = this.lineIndexes.get(params.uri);
      const diagnostics: ClientDiagnostic[] = params.diagnostics.map((d) => ({
        from: li ? li.lineColToOffset(d.range.start) : 0,
        to: li ? li.lineColToOffset(d.range.end) : 0,
        severity: d.severity ?? 1,
        code: d.code,
        // `message` is `string | MarkupContent` on the wire; the metael server always sends a string.
        message: typeof d.message === 'string' ? d.message : d.message.value,
      }));
      for (const listener of this.diagListeners) listener(params.uri, diagnostics);
    });
    this.conn.listen();
  }

  /** Send `initialize` with the chosen vocabulary profile id and return once the server has acknowledged it.
   *  The server also reads the profile id from `initializationOptions`; no separate `initialized` is needed. */
  initialize(profileId: string): Promise<void> {
    this.ready = this.conn
      .sendRequest(InitializeRequest.type, {
        processId: null,
        rootUri: null,
        capabilities: {},
        initializationOptions: { profileId },
      })
      .then(() => undefined);
    return this.ready;
  }

  /** Open a document on the server, recording its line boundaries for offset conversion. */
  didOpen(uri: string, text: string, version: number): void {
    this.lineIndexes.set(uri, new LineIndex(text));
    void this.conn.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri, languageId: 'metael', version, text },
    });
  }

  /** Replace a document's full text on the server (full-sync), rebuilding its line boundaries. */
  didChange(uri: string, text: string, version: number): void {
    this.lineIndexes.set(uri, new LineIndex(text));
    void this.conn.sendNotification(DidChangeTextDocumentNotification.type, {
      textDocument: { uri, version },
      contentChanges: [{ text }],
    });
  }

  /** Close a document on the server and drop its line boundaries. */
  didClose(uri: string): void {
    this.lineIndexes.delete(uri);
    void this.conn.sendNotification(DidCloseTextDocumentNotification.type, {
      textDocument: { uri },
    });
  }

  /** Switch the active vocabulary profile for a document (re-resolve + re-publish diagnostics). */
  setProfile(uri: string, profileId: string): void {
    void this.conn.sendNotification('metael/setProfile', { uri, profileId });
  }

  /** Register a listener fed by every `publishDiagnostics` the server sends. */
  onDiagnostics(cb: DiagnosticsListener): void {
    this.diagListeners.push(cb);
  }

  /** Request completions at a CM6 char offset. */
  async completion(uri: string, offset: number): Promise<CompletionItem[]> {
    const li = await this.positionAt(uri, offset);
    if (!li) return [];
    const res = await this.conn.sendRequest(CompletionRequest.type, {
      textDocument: { uri },
      position: li,
    });
    return Array.isArray(res) ? res : (res?.items ?? []);
  }

  /** Request hover info at a CM6 char offset, reversed to offsets. */
  async hover(uri: string, offset: number): Promise<ClientHover | null> {
    const pos = await this.positionAt(uri, offset);
    if (!pos) return null;
    const res = await this.conn.sendRequest(HoverRequest.type, {
      textDocument: { uri },
      position: pos,
    });
    if (!res || !res.range) return null;
    const contents = res.contents;
    const markdown =
      typeof contents === 'string'
        ? contents
        : Array.isArray(contents)
          ? contents.map((c) => (typeof c === 'string' ? c : c.value)).join('\n\n')
          : contents.value;
    const li = this.lineIndexes.get(uri)!;
    return { from: li.lineColToOffset(res.range.start), to: li.lineColToOffset(res.range.end), markdown };
  }

  /** Request signature help at a CM6 char offset (exposed as the LSP shape). */
  async signatureHelp(uri: string, offset: number): Promise<SignatureHelp | null> {
    const pos = await this.positionAt(uri, offset);
    if (!pos) return null;
    return (
      (await this.conn.sendRequest(SignatureHelpRequest.type, {
        textDocument: { uri },
        position: pos,
      })) ?? null
    );
  }

  /** Request full-document semantic tokens, decoded to offset spans + legend kinds for CM6. */
  async semanticTokens(uri: string): Promise<ClientSemanticToken[]> {
    await this.ready;
    const li = this.lineIndexes.get(uri);
    if (!li) return [];
    const res = await this.conn.sendRequest(SemanticTokensRequest.type, { textDocument: { uri } });
    return res ? decodeSemanticTokens(res.data, li) : [];
  }

  /** Request the capability lenses for a document, reversed to offset spans. */
  async capabilityLens(uri: string): Promise<ClientLens[]> {
    await this.ready;
    const li = this.lineIndexes.get(uri);
    if (!li) return [];
    const items = await this.conn.sendRequest<CapabilityLensItem[]>('metael/capabilityLens', { uri });
    return items.map((l) => ({
      from: li.lineColToOffset(l.range.start),
      to: li.lineColToOffset(l.range.end),
      label: l.label,
      lowerable: l.lowerable,
      reasons: l.reasons,
    }));
  }

  /** Tear down the connection and terminate the worker. Idempotent. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.conn.dispose();
    this.worker.terminate();
  }

  /** Await initialization, then map a CM6 offset → an LSP `Position`; `undefined` if the doc isn't open. */
  private async positionAt(uri: string, offset: number): Promise<{ line: number; character: number } | undefined> {
    await this.ready;
    const li = this.lineIndexes.get(uri);
    return li ? li.offsetToLineCol(offset) : undefined;
  }
}

/** Decode the LSP 5-int-per-token delta stream `[deltaLine, deltaStartChar, length, tokenType, mods]` into
 *  absolute offset spans. `length` is UTF-16 units on the token's line for single-line tokens and the raw
 *  offset span otherwise, so `to = from + length` holds either way (the encoder's inverse). */
function decodeSemanticTokens(data: readonly number[], li: LineIndex): ClientSemanticToken[] {
  const out: ClientSemanticToken[] = [];
  let line = 0;
  let char = 0;
  for (let i = 0; i + 4 < data.length; i += 5) {
    const deltaLine = data[i]!;
    const deltaChar = data[i + 1]!;
    const length = data[i + 2]!;
    const type = data[i + 3]!;
    line += deltaLine;
    char = deltaLine === 0 ? char + deltaChar : deltaChar;
    const kind = TOKEN_LEGEND[type];
    if (kind === undefined) continue;
    const from = li.lineColToOffset({ line, character: char });
    out.push({ from, to: from + length, kind });
  }
  return out;
}
