import { describe, it, expect } from 'vitest';
import { encodeState, decodeState, type ShareState } from './share.ts';

const CASES: ShareState[] = [
  { source: 'component Story() { span("hi") }', target: 'ui' },
  { source: 'map(range(5), (i) => i * i)', target: 'compute', seed: 7 },
  { source: 'filter(data, (r) => r.x)', target: 'compute', data: [{ x: true }, { x: false }] },
];

describe('share round-trip', () => {
  it('deflate scheme round-trips every case losslessly', async () => {
    for (const state of CASES) {
      const frag = await encodeState(state);
      expect(frag[0]).toBe('d'); // Node has CompressionStream
      const res = await decodeState(frag);
      expect(res.ok).toBe(true);
      if (res.ok) expect(res.state).toEqual(state);
    }
  });

  it('decodes a fallback (j) payload written by hand', async () => {
    const state = CASES[0]!;
    const frag = 'j' + encodeURIComponent(JSON.stringify(state));
    const res = await decodeState(frag);
    expect(res.ok).toBe(true);
    if (res.ok) expect(res.state).toEqual(state);
  });

  it('rejects an empty payload with an ML-PLAY diagnostic', async () => {
    const res = await decodeState('');
    expect(res.ok).toBe(false);
    if (!res.ok) expect(res.diagnostic.code).toBe('ML-PLAY-SHARE');
  });

  it('rejects an unknown scheme', async () => {
    const res = await decodeState('zGARBAGE');
    expect(res.ok).toBe(false);
    if (!res.ok) expect(res.diagnostic.code).toBe('ML-PLAY-SHARE');
  });

  it('rejects a well-formed-but-wrong-shape payload', async () => {
    const frag = 'j' + encodeURIComponent(JSON.stringify({ nope: 1 }));
    const res = await decodeState(frag);
    expect(res.ok).toBe(false);
  });

  it('rejects garbled base64 without throwing', async () => {
    const res = await decodeState('d!!!!not-base64!!!!');
    expect(res.ok).toBe(false);
  });

  it('rejects an oversize fragment before decoding (fail-loud, no hang)', async () => {
    const huge = 'd' + 'A'.repeat(300 * 1024);   // > MAX_FRAGMENT_CHARS
    const res = await decodeState(huge);
    expect(res.ok).toBe(false);
    if (!res.ok) expect(res.diagnostic.code).toBe('ML-PLAY-SHARE');
  });

  it('caps a decompression bomb instead of buffering it unbounded', async () => {
    // A tiny deflate-raw stream of many megabytes of zeros — the classic bomb. Build it with the same
    // native CompressionStream, then confirm decode fails loud (bounded) rather than inflating it all.
    const bombInput = new Uint8Array(8 * 1024 * 1024);   // 8 MB of 0x00 — compresses to a few KB
    const cs = new Blob([bombInput]).stream().pipeThrough(new CompressionStream('deflate-raw'));
    const deflated = new Uint8Array(await new Response(cs).arrayBuffer());
    let bin = '';
    for (const b of deflated) bin += String.fromCharCode(b);
    const b64url = btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    const frag = 'd' + b64url;
    // The compressed fragment is small (< the fragment cap), so this exercises the INFLATED-bytes cap.
    expect(frag.length).toBeLessThan(300 * 1024);
    const res = await decodeState(frag);
    expect(res.ok).toBe(false);   // bounded → fail-loud, not an OOM/hang
    if (!res.ok) expect(res.diagnostic.code).toBe('ML-PLAY-SHARE');
  });
});
