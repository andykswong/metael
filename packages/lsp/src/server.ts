// The transport-agnostic protocol shell: it wires a metael `LanguageService` (the offset-based, protocol-
// free analysis engine) to a JSON-RPC connection over any `MessageReader`/`MessageWriter` transport. Every
// request handler marshals the doc's cursor Position → a char offset, calls the service, and maps the
// offset-based `Svc*` result back to its wire type via the marshalers. Domain vocabulary is injected: the
// host maps an opaque profile id → a (possibly composed) Profile, so the shell knows nothing of any domain.
// This file sits at `src/`, so it legitimately imports the wire types the engine under `src/service/` must not.
import {
  createProtocolConnection,
  InitializeRequest,
  DidOpenTextDocumentNotification,
  DidChangeTextDocumentNotification,
  DidCloseTextDocumentNotification,
  CompletionRequest,
  HoverRequest,
  SignatureHelpRequest,
  FoldingRangeRequest,
  SelectionRangeRequest,
  DocumentFormattingRequest,
  SemanticTokensRequest,
  PublishDiagnosticsNotification,
} from 'vscode-languageserver-protocol';
import type { MessageReader, MessageWriter, Range } from 'vscode-languageserver-protocol';
import { LanguageService } from './service/index.ts';
import type { Profile } from '@metael/lang/profile';
import { CAPABILITIES, TOKEN_LEGEND } from './capabilities.ts';
import {
  toDiagnostic,
  toCompletionItem,
  toHover,
  toSignatureHelp,
  toFoldingRange,
  toSelectionRange,
  toTextEdit,
  encodeSemanticTokens,
  spanToRange,
  positionToOffset,
} from './marshal.ts';

/** Options that keep the shell domain-agnostic: the host resolves an opaque profile id (from the client's
 *  `initializationOptions.profileId` or a `metael/setProfile` request) to a (possibly composed) Profile. */
export interface ServerOptions {
  /** Map an opaque profile id — or `undefined` when the client sent none — to the Profile to analyse with. */
  readonly resolveProfile: (id: string | undefined) => Profile;
}

/** A marshaled capability lens: the source `range` it annotates plus the verdict a Phase-4 client renders. */
export interface CapabilityLensItem {
  /** The source range the lens annotates. */
  readonly range: Range;
  /** The label shown for the lens. */
  readonly label: string;
  /** Whether the covered construct can be lowered. */
  readonly lowerable: boolean;
  /** The explanations backing the `lowerable` verdict (empty when lowerable). */
  readonly reasons: readonly string[];
}

/** Wire a metael {@link LanguageService} to a JSON-RPC connection over any transport. Returns a handle
 *  whose `listen()` starts consuming messages. The server advertises {@link CAPABILITIES} on `initialize`,
 *  keeps documents in full-text sync, publishes diagnostics on open/change, and answers completion, hover,
 *  signature help, folding, selection ranges, formatting, and full-document semantic tokens — plus the
 *  custom `metael/setProfile` (re-resolve + re-publish) and `metael/capabilityLens` methods. */
export function createServer(reader: MessageReader, writer: MessageWriter, opts: ServerOptions): { listen(): void } {
  const conn = createProtocolConnection(reader, writer);
  const svc = new LanguageService();
  let profile: Profile | undefined;

  /** Recompute + push diagnostics for a document (no-op when it isn't open). */
  const publish = (uri: string): void => {
    const li = svc.lineIndexFor(uri);
    if (!li) return;
    void conn.sendNotification(PublishDiagnosticsNotification.type, {
      uri,
      diagnostics: svc.diagnostics(uri).map((d) => toDiagnostic(d, li)),
    });
  };

  conn.onRequest(InitializeRequest.type, (p) => {
    profile = opts.resolveProfile((p.initializationOptions as { profileId?: string } | undefined)?.profileId);
    return { capabilities: CAPABILITIES };
  });

  conn.onNotification(DidOpenTextDocumentNotification.type, (p) => {
    svc.openDocument(p.textDocument.uri, p.textDocument.text, p.textDocument.version);
    if (profile) svc.setProfile(p.textDocument.uri, profile);
    publish(p.textDocument.uri);
  });

  conn.onNotification(DidChangeTextDocumentNotification.type, (p) => {
    // Full sync: the last content change carries the whole new text.
    const last = p.contentChanges[p.contentChanges.length - 1];
    if (!last) return;
    svc.updateDocument(p.textDocument.uri, last.text, p.textDocument.version ?? 0);
    if (profile) svc.setProfile(p.textDocument.uri, profile);
    publish(p.textDocument.uri);
  });

  conn.onNotification(DidCloseTextDocumentNotification.type, (p) => {
    svc.closeDocument(p.textDocument.uri);
  });

  conn.onRequest(CompletionRequest.type, (p) => {
    const li = svc.lineIndexFor(p.textDocument.uri);
    if (!li) return [];
    return svc.completion(p.textDocument.uri, positionToOffset(p.position, li)).map(toCompletionItem);
  });

  conn.onRequest(HoverRequest.type, (p) => {
    const li = svc.lineIndexFor(p.textDocument.uri);
    if (!li) return null;
    const h = svc.hover(p.textDocument.uri, positionToOffset(p.position, li));
    return h ? toHover(h, li) : null;
  });

  conn.onRequest(SignatureHelpRequest.type, (p) => {
    const li = svc.lineIndexFor(p.textDocument.uri);
    if (!li) return null;
    const s = svc.signatureHelp(p.textDocument.uri, positionToOffset(p.position, li));
    return s ? toSignatureHelp(s) : null;
  });

  conn.onRequest(FoldingRangeRequest.type, (p) => {
    const li = svc.lineIndexFor(p.textDocument.uri);
    if (!li) return [];
    return svc.foldingRanges(p.textDocument.uri).map((f) => toFoldingRange(f, li));
  });

  conn.onRequest(SelectionRangeRequest.type, (p) => {
    const li = svc.lineIndexFor(p.textDocument.uri);
    if (!li) return [];
    // The request carries multiple positions; answer one SelectionRange chain per position, in order.
    return p.positions.map((pos) =>
      toSelectionRange(svc.selectionRanges(p.textDocument.uri, [positionToOffset(pos, li)])[0] ?? { ranges: [] }, li),
    );
  });

  conn.onRequest(DocumentFormattingRequest.type, (p) => {
    const li = svc.lineIndexFor(p.textDocument.uri);
    if (!li) return [];
    return svc.format(p.textDocument.uri).map((e) => toTextEdit(e, li));
  });

  conn.onRequest(SemanticTokensRequest.type, (p) => {
    const li = svc.lineIndexFor(p.textDocument.uri);
    if (!li) return { data: [] };
    return encodeSemanticTokens(svc.semanticTokens(p.textDocument.uri), li, TOKEN_LEGEND);
  });

  // Custom methods (outside the standard LSP surface) — namespaced under `metael/`.
  conn.onNotification('metael/setProfile', (p: { uri?: string; profileId?: string }) => {
    profile = opts.resolveProfile(p.profileId);
    if (p.uri) {
      svc.setProfile(p.uri, profile);
      publish(p.uri);
    }
  });

  conn.onRequest('metael/capabilityLens', (p: { uri: string }): CapabilityLensItem[] => {
    const li = svc.lineIndexFor(p.uri);
    if (!li) return [];
    return svc.capabilityLens(p.uri).map((l) => ({
      range: spanToRange(l.span, li),
      label: l.label,
      lowerable: l.lowerable,
      reasons: l.reasons ?? [],
    }));
  });

  return {
    listen(): void {
      conn.listen();
    },
  };
}
