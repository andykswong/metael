// Lossless, backend-free share: the playground state is a pure function of the editor, so it round-trips
// through the URL #fragment (the fragment is never sent to servers → shared source stays out of logs).
// Primary scheme 'd': JSON -> deflate-raw (native CompressionStream) -> base64url. Fallback scheme 'j':
// JSON -> encodeURIComponent (when CompressionStream is unavailable). The 1-char scheme prefix lets decode
// dispatch, so both coexist and old links never break.
import { makeDiagnostic } from '@metael/lang';
import type { Diagnostic } from '@metael/lang';
import type { Target } from './examples.ts';

export interface ShareState {
  source: string;
  target: Target;
  data?: unknown;
  seed?: number;
}

export type DecodeResult =
  | { ok: true; state: ShareState }
  | { ok: false; diagnostic: Diagnostic };

const hasCompression = typeof CompressionStream !== 'undefined';

// A share link is a pure function of the editor state — a few KB even for a large program. These caps bound
// an adversarial fragment: deflate-raw's ~1032:1 max ratio means a small crafted payload could otherwise
// inflate to hundreds of MB (a decompression bomb) and OOM/hang the tab BEFORE the fail-loud path runs.
// Exceeding either cap throws → the decodeState try/catch turns it into an ML-PLAY-SHARE diagnostic.
const MAX_FRAGMENT_CHARS = 256 * 1024;   // reject an absurd fragment before even base64-decoding it
const MAX_INFLATED_BYTES = 2 * 1024 * 1024;   // bound the decompressed size (a real state is « this)

// NB: annotate byte buffers as Uint8Array<ArrayBuffer> (not the default Uint8Array<ArrayBufferLike>). Under
// TS 6 + strict, `new Blob([bytes])` requires an ArrayBuffer-backed view; the plain Uint8Array type widens
// to ArrayBufferLike and fails typecheck (TS2322). TextEncoder.encode() + new Uint8Array(number|ArrayBuffer)
// all already produce ArrayBuffer-backed views, so the annotation is accurate.
function bytesToB64url(bytes: Uint8Array<ArrayBuffer>): string {
  let bin = '';
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
function b64urlToBytes(s: string): Uint8Array<ArrayBuffer> {
  const b64 = s.replace(/-/g, '+').replace(/_/g, '/') + '==='.slice((s.length + 3) % 4);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
async function deflate(bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Inflate, reading chunks and aborting once the running total exceeds `max` — so a decompression bomb
 *  throws (→ a fail-loud diagnostic) instead of buffering unbounded output and hanging/OOM-ing the tab. */
async function inflateBounded(bytes: Uint8Array<ArrayBuffer>, max: number): Promise<string> {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let total = 0;
  let out = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > max) { await reader.cancel(); throw new Error('share payload too large'); }
    out += decoder.decode(value, { stream: true });
  }
  out += decoder.decode();
  return out;
}

/** Encode state to a #fragment payload (WITHOUT the leading '#'). Async because deflate is streaming. */
export async function encodeState(state: ShareState): Promise<string> {
  const json = JSON.stringify(state);
  if (!hasCompression) return 'j' + encodeURIComponent(json);
  const bytes: Uint8Array<ArrayBuffer> = new TextEncoder().encode(json);
  const deflated = await deflate(bytes);
  return 'd' + bytesToB64url(deflated);
}

/** Decode a #fragment payload (WITHOUT the leading '#'). Never throws — a bad payload yields a diagnostic. */
export async function decodeState(fragment: string): Promise<DecodeResult> {
  try {
    if (!fragment) return fail('empty share payload');
    if (fragment.length > MAX_FRAGMENT_CHARS) return fail('share link is too large');
    const scheme = fragment[0];
    const payload = fragment.slice(1);
    let json: string;
    if (scheme === 'j') {
      json = decodeURIComponent(payload);
    } else if (scheme === 'd') {
      if (typeof DecompressionStream === 'undefined') return fail('this browser cannot read a compressed share link');
      json = await inflateBounded(b64urlToBytes(payload), MAX_INFLATED_BYTES);
    } else {
      return fail(`unknown share encoding '${scheme ?? ''}'`);
    }
    const parsed = JSON.parse(json) as ShareState;
    if (typeof parsed?.source !== 'string' || !['ui', 'compute', 'gpu'].includes(parsed.target as string)) {
      return fail('share payload is not a valid playground state');
    }
    return { ok: true, state: parsed };
  } catch {
    return fail('could not decode the share link');
  }
}

function fail(message: string): DecodeResult {
  return { ok: false, diagnostic: makeDiagnostic('ML-PLAY-SHARE', message) };
}
