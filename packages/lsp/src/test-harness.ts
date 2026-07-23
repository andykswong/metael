// An in-memory transport pair for exercising the protocol shell end-to-end without a socket or child
// process. Two Node `PassThrough` streams cross-wire a client JSON-RPC connection to a server one: the
// client's writes flow to the server's reader and the server's writes flow back to the client's reader.
import { PassThrough } from 'node:stream';
import { StreamMessageReader, StreamMessageWriter } from 'vscode-languageserver-protocol/node';
import type { MessageReader, MessageWriter } from 'vscode-languageserver-protocol';

/** A cross-wired pair of JSON-RPC transports over two in-memory streams: one end for a server connection
 *  and one end for a client connection, so a test can drive a real request/response round-trip. */
export interface TransportPair {
  /** The reader a server connection consumes (carries client→server traffic). */
  readonly serverReader: MessageReader;
  /** The writer a server connection produces to (carries server→client traffic). */
  readonly serverWriter: MessageWriter;
  /** The reader a client connection consumes (carries server→client traffic). */
  readonly clientReader: MessageReader;
  /** The writer a client connection produces to (carries client→server traffic). */
  readonly clientWriter: MessageWriter;
}

/** Build a fresh in-memory {@link TransportPair} over two `PassThrough` streams — `c2s` carries
 *  client→server messages, `s2c` carries server→client — so both connections can `listen()` and talk. */
export function makePair(): TransportPair {
  const c2s = new PassThrough();
  const s2c = new PassThrough();
  return {
    serverReader: new StreamMessageReader(c2s),
    serverWriter: new StreamMessageWriter(s2c),
    clientReader: new StreamMessageReader(s2c),
    clientWriter: new StreamMessageWriter(c2s),
  };
}
