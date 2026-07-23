import { describe, it, expect } from 'vitest';
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
import type { PublishDiagnosticsParams, ProtocolConnection } from 'vscode-languageserver-protocol';
import { composeProfiles } from '@metael/lang/profile';
import type { Profile } from '@metael/lang/profile';
import { mathProfile } from '@metael/math/lang';
import { stdProfile } from '@metael/std';
import { vdomProfile } from '@metael/vdom/lang';
import { makePair } from './test-harness.ts';
import { createServer, TOKEN_LEGEND } from './index.ts';
import type { CapabilityLensItem } from './index.ts';

/** Wire a fresh server (with the given profile resolver) to a listening client over an in-memory pair. */
function connect(resolveProfile: (id: string | undefined) => Profile): ProtocolConnection {
  const { serverReader, serverWriter, clientReader, clientWriter } = makePair();
  createServer(serverReader, serverWriter, { resolveProfile }).listen();
  const client = createProtocolConnection(clientReader, clientWriter);
  client.listen();
  return client;
}

/** A give-the-async-notification-a-tick delay, matching the notification-driven assertions below. */
const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 20));

describe('the LSP server over an in-memory channel', () => {
  it('initializes, opens a doc, and answers completion with profile builtins', async () => {
    const { serverReader, serverWriter, clientReader, clientWriter } = makePair();
    const profile = composeProfiles(mathProfile, stdProfile);
    createServer(serverReader, serverWriter, { resolveProfile: () => profile }).listen();
    const client = createProtocolConnection(clientReader, clientWriter);
    client.listen();

    const init = await client.sendRequest(InitializeRequest.type, {
      processId: null,
      rootUri: null,
      capabilities: {},
      initializationOptions: { profileId: 'compute' },
    });
    expect(init.capabilities.completionProvider).toBeTruthy();
    expect(init.capabilities.semanticTokensProvider).toBeTruthy();

    client.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri: 'a', languageId: 'metael', version: 1, text: 'const x = ' },
    });
    const res = await client.sendRequest(CompletionRequest.type, {
      textDocument: { uri: 'a' },
      position: { line: 0, character: 10 },
    });
    const labels = (Array.isArray(res) ? res : (res?.items ?? [])).map((i) => i.label);
    expect(labels).toContain('map');
    expect(labels).toContain('sqrt');
  });

  it('publishes diagnostics on open and updates them on a full-sync change', async () => {
    const { serverReader, serverWriter, clientReader, clientWriter } = makePair();
    const profile = composeProfiles(mathProfile, stdProfile);
    createServer(serverReader, serverWriter, { resolveProfile: () => profile }).listen();
    const client = createProtocolConnection(clientReader, clientWriter);

    const published: PublishDiagnosticsParams[] = [];
    client.onNotification(PublishDiagnosticsNotification.type, (p) => {
      published.push(p);
    });
    client.listen();

    await client.sendRequest(InitializeRequest.type, { processId: null, rootUri: null, capabilities: {} });
    client.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri: 'b', languageId: 'metael', version: 1, text: 'const x = (' }, // a parse error
    });
    // Give the async notification a tick to arrive.
    await new Promise((r) => setTimeout(r, 20));
    expect(published.length).toBeGreaterThan(0);
    expect(published.at(-1)!.diagnostics.length).toBeGreaterThan(0);

    client.sendNotification(DidChangeTextDocumentNotification.type, {
      textDocument: { uri: 'b', version: 2 },
      contentChanges: [{ text: 'const x = 1' }], // now valid — no diagnostics
    });
    await new Promise((r) => setTimeout(r, 20));
    expect(published.at(-1)!.diagnostics).toEqual([]);
  });

  it('answers hover with null when the offset is not on a resolvable identifier', async () => {
    const { serverReader, serverWriter, clientReader, clientWriter } = makePair();
    const profile = composeProfiles(mathProfile, stdProfile);
    createServer(serverReader, serverWriter, { resolveProfile: () => profile }).listen();
    const client = createProtocolConnection(clientReader, clientWriter);
    client.listen();

    await client.sendRequest(InitializeRequest.type, { processId: null, rootUri: null, capabilities: {} });
    client.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri: 'c', languageId: 'metael', version: 1, text: 'const x = 1' },
    });
    const hover = await client.sendRequest(HoverRequest.type, { textDocument: { uri: 'c' }, position: { line: 0, character: 8 } });
    expect(hover).toBeNull();
  });

  it('answers signature help for the enclosing call with the active parameter selected', async () => {
    const client = connect(() => composeProfiles(mathProfile, stdProfile));
    await client.sendRequest(InitializeRequest.type, { processId: null, rootUri: null, capabilities: {} });
    const src = 'const r = dot(a, b)'; // dot is a 2-arg math builtin
    client.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri: 'sig', languageId: 'metael', version: 1, text: src },
    });
    // Cursor just after the comma → inside the second argument slot.
    const after = src.indexOf(',') + 1;
    const sig = await client.sendRequest(SignatureHelpRequest.type, {
      textDocument: { uri: 'sig' },
      position: { line: 0, character: after },
    });
    expect(sig).not.toBeNull();
    expect(sig!.signatures[0]!.label).toContain('dot');
    expect(sig!.activeParameter).toBe(1);
  });

  it('answers folding ranges as LINE-based ranges (not offset spans)', async () => {
    const client = connect(() => stdProfile);
    await client.sendRequest(InitializeRequest.type, { processId: null, rootUri: null, capabilities: {} });
    client.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri: 'fold', languageId: 'metael', version: 1, text: 'component App() {\n  const a = 1\n  div(a)\n}' },
    });
    const folds = await client.sendRequest(FoldingRangeRequest.type, { textDocument: { uri: 'fold' } });
    expect(folds).not.toBeNull();
    expect(folds!.length).toBeGreaterThanOrEqual(1);
    const f = folds![0]!;
    // The marshaler turned offsets into LINE numbers: startLine/endLine present as numbers, no offset fields.
    expect(typeof f.startLine).toBe('number');
    expect(typeof f.endLine).toBe('number');
    expect(f.startLine).toBeLessThan(f.endLine!);
    expect(f).not.toHaveProperty('start');
    expect(f).not.toHaveProperty('end');
  });

  it('answers selection ranges as a nested chain of wire Ranges per requested position', async () => {
    const client = connect(() => stdProfile);
    await client.sendRequest(InitializeRequest.type, { processId: null, rootUri: null, capabilities: {} });
    const src = 'const total = 1 + 2 * 3'; // a nested expression
    client.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri: 'sel', languageId: 'metael', version: 1, text: src },
    });
    const at = src.indexOf('2');
    const sels = await client.sendRequest(SelectionRangeRequest.type, {
      textDocument: { uri: 'sel' },
      positions: [{ line: 0, character: at }],
    });
    // One SelectionRange per requested position.
    expect(Array.isArray(sels)).toBe(true);
    expect(sels!.length).toBe(1);
    const head = sels![0]!;
    // Its `range` is a wire Range of line/char positions (NOT an offset span).
    expect(head.range.start).toEqual({ line: 0, character: expect.any(Number) });
    expect(head.range.end).toEqual({ line: 0, character: expect.any(Number) });
    // It chains outward to a strictly WIDER parent range that contains the head range.
    expect(head.parent).toBeDefined();
    const parent = head.parent!;
    expect(parent.range.start.character).toBeLessThanOrEqual(head.range.start.character);
    expect(parent.range.end.character).toBeGreaterThanOrEqual(head.range.end.character);
    const headWidth = head.range.end.character - head.range.start.character;
    const parentWidth = parent.range.end.character - parent.range.start.character;
    expect(parentWidth).toBeGreaterThan(headWidth);
  });

  it('answers formatting with a whole-document TextEdit, and [] on a parse error', async () => {
    const client = connect(() => stdProfile);
    await client.sendRequest(InitializeRequest.type, { processId: null, rootUri: null, capabilities: {} });
    // Non-canonical source (extra spaces, missing spaces) reprints to canonical.
    client.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri: 'fmt', languageId: 'metael', version: 1, text: 'const  x=1' },
    });
    const edits = await client.sendRequest(DocumentFormattingRequest.type, {
      textDocument: { uri: 'fmt' },
      options: { tabSize: 2, insertSpaces: true },
    });
    expect(edits!.length).toBe(1);
    const edit = edits![0]!;
    // The edit's range is a wire Range over the whole doc, starting at the origin.
    expect(edit.range.start).toEqual({ line: 0, character: 0 });
    expect(edit.newText).toContain('const x = 1');

    // A doc WITH a parse error yields no edit (never reprints a broken AST).
    client.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri: 'fmt-err', languageId: 'metael', version: 1, text: 'const x = (' },
    });
    const none = await client.sendRequest(DocumentFormattingRequest.type, {
      textDocument: { uri: 'fmt-err' },
      options: { tabSize: 2, insertSpaces: true },
    });
    expect(none).toEqual([]);
  });

  it('answers full-document semantic tokens as a 5-int-per-token delta array', async () => {
    const client = connect(() => composeProfiles(mathProfile, stdProfile, vdomProfile));
    await client.sendRequest(InitializeRequest.type, { processId: null, rootUri: null, capabilities: {} });
    const src = 'component App() { const x = 1 }';
    client.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri: 'tok', languageId: 'metael', version: 1, text: src },
    });
    const tokens = await client.sendRequest(SemanticTokensRequest.type, { textDocument: { uri: 'tok' } });
    expect(tokens).not.toBeNull();
    const data = tokens!.data;
    expect(data.length).toBeGreaterThan(0);
    expect(data.length % 5).toBe(0); // the LSP 5-int delta encoding
    // Decode the first token: `component` is at line 0, column 0, length 9, a keyword.
    const [deltaLine, deltaStart, length, tokenType] = data;
    expect(deltaLine).toBe(0);
    expect(deltaStart).toBe(src.indexOf('component'));
    expect(length).toBe('component'.length);
    expect(tokenType).toBe(TOKEN_LEGEND.indexOf('keyword'));
  });

  it('drops a closed document: completion on it returns []', async () => {
    const client = connect(() => composeProfiles(mathProfile, stdProfile));
    await client.sendRequest(InitializeRequest.type, { processId: null, rootUri: null, capabilities: {} });
    client.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri: 'closeme', languageId: 'metael', version: 1, text: 'const x = ' },
    });
    // Sanity: while open, completion answers.
    const open = await client.sendRequest(CompletionRequest.type, {
      textDocument: { uri: 'closeme' },
      position: { line: 0, character: 10 },
    });
    expect((Array.isArray(open) ? open : (open?.items ?? [])).length).toBeGreaterThan(0);

    client.sendNotification(DidCloseTextDocumentNotification.type, { textDocument: { uri: 'closeme' } });
    await tick(); // let the close notification arrive before we re-query
    const closed = await client.sendRequest(CompletionRequest.type, {
      textDocument: { uri: 'closeme' },
      position: { line: 0, character: 10 },
    });
    // The doc is gone → lineIndexFor is undefined → the handler returns [].
    expect(closed).toEqual([]);
  });

  it('retargets the vocabulary on metael/setProfile (re-resolve + re-publish)', async () => {
    // resolveProfile maps 'B' → a vdom+std profile (heads incl. `div`, NO sqrt); anything else → math+std (has sqrt).
    const profileA = composeProfiles(mathProfile, stdProfile);
    const profileB = composeProfiles(vdomProfile, stdProfile);
    const client = connect((id) => (id === 'B' ? profileB : profileA));
    await client.sendRequest(InitializeRequest.type, {
      processId: null,
      rootUri: null,
      capabilities: {},
      initializationOptions: { profileId: 'A' },
    });
    client.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri: 'prof', languageId: 'metael', version: 1, text: 'const y = ' },
    });
    const before = await client.sendRequest(CompletionRequest.type, {
      textDocument: { uri: 'prof' },
      position: { line: 0, character: 10 },
    });
    const beforeLabels = (Array.isArray(before) ? before : (before?.items ?? [])).map((i) => i.label);
    expect(beforeLabels).toContain('sqrt'); // profile A (math) has it
    expect(beforeLabels).not.toContain('div');

    // Retarget this uri to profile B via the custom notification.
    client.sendNotification('metael/setProfile', { uri: 'prof', profileId: 'B' });
    await tick();
    const after = await client.sendRequest(CompletionRequest.type, {
      textDocument: { uri: 'prof' },
      position: { line: 0, character: 10 },
    });
    const afterLabels = (Array.isArray(after) ? after : (after?.items ?? [])).map((i) => i.label);
    expect(afterLabels).toContain('div'); // profile B (vdom) head is now offered
    expect(afterLabels).not.toContain('sqrt'); // and the old math builtin is gone
  });

  it('answers metael/capabilityLens with wire ranges + lowerable verdicts', async () => {
    const client = connect(() => composeProfiles(mathProfile, stdProfile));
    await client.sendRequest(InitializeRequest.type, { processId: null, rootUri: null, capabilities: {} });
    const src = 'function pure() { return 1 + 2 }\nfunction hostuse() { return map(xs, (x)=>x) }';
    client.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri: 'lens', languageId: 'metael', version: 1, text: src },
    });
    const lenses = await client.sendRequest<CapabilityLensItem[]>('metael/capabilityLens', { uri: 'lens' });
    expect(Array.isArray(lenses)).toBe(true);
    expect(lenses.length).toBe(2);
    for (const l of lenses) {
      // Each item carries a wire Range (line/char positions) + a boolean verdict + reasons array.
      expect(l.range.start).toEqual({ line: expect.any(Number), character: expect.any(Number) });
      expect(l.range.end).toEqual({ line: expect.any(Number), character: expect.any(Number) });
      expect(typeof l.lowerable).toBe('boolean');
      expect(Array.isArray(l.reasons)).toBe(true);
    }
    const pure = lenses[0]!;
    const hostuse = lenses[1]!;
    // The two lenses sit on different lines (proves spanToRange used line/col, not raw offsets).
    expect(pure.range.start.line).toBe(0);
    expect(hostuse.range.start.line).toBe(1);
    expect(pure.lowerable).toBe(true);
    expect(pure.reasons).toEqual([]);
    expect(hostuse.lowerable).toBe(false);
    expect(hostuse.reasons.length).toBeGreaterThan(0);
  });
});
