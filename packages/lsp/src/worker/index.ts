/// <reference lib="webworker" />
import { BrowserMessageReader, BrowserMessageWriter } from 'vscode-languageserver-protocol/browser';
import { createServer } from '../index.ts';
import type { ServerOptions } from '../server.ts';

/** Start the metael LSP server inside a dedicated Web Worker, bound to the worker's message port. The
 *  host posts JSON-RPC messages in; the server posts responses/notifications out. */
export function startWorkerServer(scope: DedicatedWorkerGlobalScope, opts: ServerOptions): void {
  const reader = new BrowserMessageReader(scope);
  const writer = new BrowserMessageWriter(scope);
  createServer(reader, writer, opts).listen();
}
