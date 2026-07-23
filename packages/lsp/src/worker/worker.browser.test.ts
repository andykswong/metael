/// <reference lib="webworker" />
import { describe, it, expect } from 'vitest';
import {
  BrowserMessageReader,
  BrowserMessageWriter,
  createProtocolConnection,
  InitializeRequest,
  DidOpenTextDocumentNotification,
  CompletionRequest,
} from 'vscode-languageserver-protocol/browser';
import { composeProfiles } from '@metael/lang/profile';
import { mathProfile } from '@metael/math/lang';
import { stdProfile } from '@metael/std';
import { startWorkerServer } from './index.ts';

// The worker binding is a thin adapter over the already-round-tripped shell (see server.test.ts, the node
// gate). This proves the browser transport in Chromium: `startWorkerServer` drives one end of a
// `MessageChannel` — a `MessagePort` stands in for the worker scope, which `BrowserMessageReader`/`Writer`
// accept alongside `DedicatedWorkerGlobalScope` — while a client protocol connection drives the other end.
describe('the LSP worker binding over a MessageChannel', () => {
  it('starts the server on a port and answers completion end-to-end', async () => {
    const channel = new MessageChannel();
    const profile = composeProfiles(mathProfile, stdProfile);
    // A `MessagePort` cross-wires exactly like the worker scope: the reader listens via `onmessage`, the
    // writer emits via `postMessage`. Cast to the scope type the binding's signature declares.
    startWorkerServer(channel.port1 as unknown as DedicatedWorkerGlobalScope, { resolveProfile: () => profile });

    const client = createProtocolConnection(
      new BrowserMessageReader(channel.port2),
      new BrowserMessageWriter(channel.port2),
    );
    client.listen();

    const init = await client.sendRequest(InitializeRequest.type, {
      processId: null,
      rootUri: null,
      capabilities: {},
      initializationOptions: { profileId: 'compute' },
    });
    expect(init.capabilities.completionProvider).toBeTruthy();

    client.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri: 'w', languageId: 'metael', version: 1, text: 'const x = ' },
    });
    const res = await client.sendRequest(CompletionRequest.type, {
      textDocument: { uri: 'w' },
      position: { line: 0, character: 10 },
    });
    const labels = (Array.isArray(res) ? res : (res?.items ?? [])).map((i) => i.label);
    expect(labels).toContain('map');
    expect(labels).toContain('sqrt');
  });
});
