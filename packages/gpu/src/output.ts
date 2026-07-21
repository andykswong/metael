// The output-element vocabulary: a kernel writes either a scalar (`f32`, the default — one value per cell)
// or a small vector (`vec2`/`vec3`/`vec4` — N values per cell). `compsOf` maps an element to its component
// width N. Every backend produces the SAME normalized FLAT-INTERLEAVED layout: cell `c`, component `k`
// lives at `output[c * comps + k]`. For `f32` (comps=1) this is exactly `output[c]` (full back-compat).
// A single small file so the emitters, the backends, the oracle, and the resource all agree on the width.
/** A kernel's per-cell output element: a scalar (`'f32'`, one value per cell) or a small vector
 *  (`'vec2'`/`'vec3'`/`'vec4'`, N values per cell in the flat-interleaved layout). */
export type OutputElement = 'f32' | 'vec2' | 'vec3' | 'vec4';

/** The component width of an output element: f32→1, vec2→2, vec3→3, vec4→4. */
export function compsOf(el: OutputElement | undefined): number {
  switch (el) {
    case 'vec2': return 2;
    case 'vec3': return 3;
    case 'vec4': return 4;
    default: return 1;   // 'f32' or undefined → scalar
  }
}
