// A structural hash of a kernel: stripSpans(body) + the resolved values of captured constants (a uniform
// like N is part of the compiled kernel, so it belongs in the memo key). Reused by resource.ts's memo.
import type { UserFn } from '@metael/lang';
import { stripSpans } from '@metael/lang';
import type { BindingTable } from './binding.ts';

/** A structural hash of a kernel for the dispatch memo key: its span-stripped params + body, combined with
 *  the resolved values of its closed-over scalar constants (a uniform like `N` is baked into the compiled
 *  shader, so it belongs in the key). Two kernels with the same shape + constants produce the same hash. */
export function kernelHash(kernel: UserFn, bindings: BindingTable): string {
  const shape = JSON.stringify(stripSpans({ params: kernel.params, body: kernel.body }));
  const consts: Record<string, number> = {};
  for (const b of bindings.byName.values()) if (b.role === 'scalar') consts[b.name] = b.value;
  return `${shape}::${JSON.stringify(consts)}`;
}

/** A compact CONTENT digest of a buffer's values — the memo-key discriminator for a plain user buffer.
 *  Two DISTINCT same-length buffers read generation 0 (a fresh buffer) and their contents are ABSENT from
 *  kernelHash, so without this they'd share a memo key → a consumer returns the first buffer's stale result.
 *  The digest is CONTENT-DETERMINISTIC (identical element values → identical string), which is BOTH correct
 *  (different content → a different key → re-dispatch) AND convergent (a rebuilt identical-content buffer →
 *  the same key → a memo hit → a fixpoint, never an infinite re-dispatch loop). Two independent FNV-1a-style
 *  32-bit rolling hashes over each element's f64 bit pattern give a ~64-bit combined digest (very low
 *  collision probability); the length is folded in so a short view can't alias a longer one. This is a FULL
 *  O(n) hash (correctness over the marginal cost — a silent same-length collision is the whole bug this
 *  fixes); the buffer length is already bounded by the cost gate (MAX_GPU_ALLOC). NB: only the key hashing
 *  reads the values here — the dispatch data transfer stays zero-copy. */
export function bufferFingerprint(data: ArrayLike<number>): string {
  const scratch = new Float64Array(1);
  const words = new Uint32Array(scratch.buffer);   // the two 32-bit halves of each element's f64
  let h1 = 0x811c9dc5 >>> 0;   // FNV offset basis
  let h2 = 0x1000193 >>> 0;    // a distinct seed → a second independent lane
  const n = data.length;
  for (let i = 0; i < n; i++) {
    scratch[0] = data[i]!;
    const lo = words[0]!, hi = words[1]!;
    h1 = Math.imul(h1 ^ lo, 0x01000193) >>> 0;
    h1 = Math.imul(h1 ^ hi, 0x01000193) >>> 0;
    h2 = Math.imul(h2 ^ hi, 0x85ebca6b) >>> 0;
    h2 = Math.imul(h2 ^ lo, 0x85ebca6b) >>> 0;
  }
  return `${n.toString(16)}.${h1.toString(16)}.${h2.toString(16)}`;
}
