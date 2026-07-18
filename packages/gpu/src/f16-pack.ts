// Pure f32 ↔ IEEE-754 binary16 (half) bit conversion + 4-byte alignment helpers for the f16 storage-buffer
// path. Extracted here (no WebGPU device references) so the packing/alignment invariant is unit-testable in
// node — the real f16 dispatch is only reachable on a shader-f16 adapter, absent in this environment.
//
// WHY the even-length rounding matters: WebGPU requires `queue.writeBuffer` data size and
// `copyBufferToBuffer` size to be multiples of 4. Each `array<f16>` element is 2 bytes, so an ODD element
// count would produce a 2-mod-4 byte size and fault on a real device. `packF16` rounds the packed
// Uint16Array up to an EVEN length (a trailing 0 pad u16 that the shader never reads — reads are bounded by
// the kernel's own logic + arrayLength, and the output write is bounds-guarded per invocation), and
// `align4` rounds a byte count up to the next multiple of 4 for output/readback buffer sizing.
//
// NOTE: this value path is NOT runtime-tested here (no WebGPU adapter with shader-f16); the bit conversions
// are implemented per the IEEE-754 half spec + verified by the real-device compile gate. The unit tests lock
// the alignment invariant of the extracted helpers, not a real dispatch.

const HAS_FLOAT16 = typeof (globalThis as { Float16Array?: unknown }).Float16Array !== 'undefined';
type Float16ArrayCtor = { new (input: ArrayLike<number> | number): { readonly buffer: ArrayBufferLike; readonly length: number; [i: number]: number } };
const Float16ArrayRef = (globalThis as { Float16Array?: Float16ArrayCtor }).Float16Array;

const _f32scratch = new Float32Array(1);
const _u32scratch = new Uint32Array(_f32scratch.buffer);

/** Round a byte count up to the next multiple of 4 (WebGPU writeBuffer/copyBufferToBuffer size requirement). */
export function align4(n: number): number {
  return Math.ceil(n / 4) * 4;
}

/** f32 → the u16 bit pattern of its nearest binary16 value (round-to-nearest-even). */
export function f32ToF16(x: number): number {
  _f32scratch[0] = x;
  const bits = _u32scratch[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exp = (bits >>> 23) & 0xff;
  let mant = bits & 0x7fffff;
  if (exp === 0xff) return sign | (mant ? 0x7e00 : 0x7c00);   // NaN → a quiet NaN; ±Inf → half Inf
  const unbiased = exp - 127 + 15;                            // rebias 127 → 15
  if (unbiased >= 0x1f) return sign | 0x7c00;                 // overflow → ±Inf
  if (unbiased <= 0) {                                        // subnormal or zero
    if (unbiased < -10) return sign;                          // too small → signed zero
    mant |= 0x800000;                                         // restore the implicit leading 1
    const shift = 14 - unbiased;                              // shift the mantissa into the subnormal range
    let half = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);                    // round-to-nearest-even on the shifted-out bits
    const halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && (half & 1))) half += 1;
    return sign | half;
  }
  let half = (unbiased << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;                                  // round-to-nearest-even on the 13 dropped bits
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half += 1;   // carry may bump into the exponent — that is correct
  return sign | half;
}

/** The u16 bit pattern of a binary16 value → f32. */
export function f16ToF32(h: number): number {
  const sign = (h & 0x8000) << 16;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) {
    if (mant === 0) { _u32scratch[0] = sign; return _f32scratch[0]!; }   // ±0
    // subnormal: normalize
    let e = -14; let m = mant;
    while ((m & 0x400) === 0) { m <<= 1; e -= 1; }
    m &= 0x3ff;
    _u32scratch[0] = sign | ((e + 127) << 23) | (m << 13);
    return _f32scratch[0]!;
  }
  if (exp === 0x1f) { _u32scratch[0] = sign | 0x7f800000 | (mant << 13); return _f32scratch[0]!; }   // Inf / NaN
  _u32scratch[0] = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  return _f32scratch[0]!;
}

/**
 * Pack a Float32Array as a Uint16Array of binary16 bit patterns (Float16Array fast path when present),
 * rounded UP to an EVEN length so `byteLength % 4 === 0` (a WebGPU writeBuffer requirement). An odd input
 * gets a single trailing 0 pad u16 the shader never reads.
 */
export function packF16(data: Float32Array): Uint16Array {
  const evenLen = data.length + (data.length & 1);   // round up to even → 4-byte-aligned byteLength
  if (HAS_FLOAT16 && Float16ArrayRef) {
    const half = new Uint16Array(new Float16ArrayRef(data).buffer);   // length === data.length
    if (evenLen === data.length) return half;
    const padded = new Uint16Array(evenLen);   // trailing slot left 0 (pad)
    padded.set(half);
    return padded;
  }
  const out = new Uint16Array(evenLen);
  for (let i = 0; i < data.length; i++) out[i] = f32ToF16(data[i]!);
  return out;   // out[evenLen - 1] left 0 when odd (pad)
}

/** Unpack the first `outLen` binary16 bit patterns of `half` into a fresh Float32Array (ignoring any pad). */
export function unpackF16(half: Uint16Array, outLen: number): Float32Array {
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) out[i] = f16ToF32(half[i]!);
  return out;
}
