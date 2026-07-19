// The correctness oracle: run a SAMPLE of K output cells through the shipped interpreter (via makeCallable
// — a fresh, budget-raised Runner) and tolerance-check the produced output against that reference. NOT a
// proof — a sampled differential check (systematic emitter bugs caught with high probability).
import type { UserFn, ReactiveHost, HostEnvironment } from '@metael/lang';
import { makeCallable, descriptorOf, NOT_HANDLED, DEFAULT_MAX_STEPS, MAX_RANGE } from '@metael/lang';
import type { BindingTable } from './binding.ts';

export interface MatchVerdict { readonly ok: boolean; readonly kind: 'exact' | 'ulp'; readonly maxUlp: number }
export interface OracleInput {
  readonly fn: UserFn; readonly host: ReactiveHost; readonly bindings: BindingTable;
  readonly output: ArrayLike<number>; readonly dims: readonly number[];
  readonly precision: 'f16' | 'f32'; readonly sampleCount: number;
  /** The output element's component width (default 1). For a vecN output the interpreter returns a vec
   *  value per cell; each of its `comps` components is compared to output[flat*comps + k]. */
  readonly comps?: number;
}

function ulpDistance(a: number, b: number): number {
  if (a === b) return 0;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return Infinity;
  const buf = new ArrayBuffer(8); const f = new Float32Array(buf); const i = new Int32Array(buf);
  f[0] = Math.fround(a); const ia = i[0]!; f[0] = Math.fround(b); const ib = i[0]!;
  return Math.abs(ia - ib);
}

// Extract a cell's `comps` components from an interpreter return value: a scalar for comps=1; a vecN read
// component-wise (x,y,z,w) via the descriptor's getMember for comps>1 — mirrors emit-cpu's extractComps.
function extractComps(r: unknown, comps: number): number[] {
  if (comps === 1) return [Number(r)];
  const d = descriptorOf(r);
  const out: number[] = [];
  for (let k = 0; k < comps; k++) {
    let v: unknown = 0;
    if (d?.getMember) { try { const m = d.getMember(r, 'xyzw'[k] as string); v = m === NOT_HANDLED ? 0 : m; } catch { v = 0; } }
    out.push(Number(v ?? 0));
  }
  return out;
}

/** Verdict for a REDUCTION's GPU tree fold vs the linear-fold oracle (`cpuReduce`). A tree reduction REORDERS
 *  the fold, so a float sum differs from the linear oracle by float-ASSOCIATIVITY rounding — NOT the map path's
 *  tight ulp bound. An exact-integer sum within f32's 2^24 range is bit-identical (ulp 0, kind 'exact'); a
 *  general float reduction is accepted within a small RELATIVE tolerance (1e-4), which documents the reorder
 *  while still catching a real emitter bug (a wrong op / a dropped tile diverges by far more than 1e-4). */
export function checkReduceMatch(gpu: number, oracle: number): MatchVerdict {
  const ulp = ulpDistance(gpu, oracle);
  const relErr = Math.abs(gpu - oracle) / (Math.abs(oracle) + 1e-6);
  const ok = ulp === 0 || relErr <= 1e-4;
  return { ok, kind: ulp === 0 ? 'exact' : 'ulp', maxUlp: Number.isFinite(ulp) ? ulp : Number.MAX_SAFE_INTEGER };
}

export function checkMatch(input: OracleInput): MatchVerdict {
  const { fn, host, dims, output, precision, sampleCount } = input;
  const comps = input.comps ?? 1;
  const total = dims.reduce((a, b) => a * b, 1);
  // A per-SAMPLE step budget (a fresh Runner per cell below), sized for one cell's worst case — a full
  // range(MAX_RANGE) sweep. makeCallable's budget is AGGREGATE across calls of ONE callable, so reusing a
  // single callable for all ~256 samples would exhaust the budget partway through a heavy kernel (a big
  // matmul), throwing mid-sweep and reading as a spurious mismatch. A fresh callable per cell mirrors
  // emit-cpu's per-cell makeCallable.
  const perCellSteps = Math.max(DEFAULT_MAX_STEPS, MAX_RANGE * 4096);
  const declineEnv: HostEnvironment = { resolveCall: () => ({ handled: false }) };
  // Reconstruct a cell's coords from its flat index under the shared row-major flatten (the LAST dim varies
  // fastest): flat = (…(c0*d1 + c1)*d2 + c2)… For [W,H] this is x=flat/H, y=flat%H; for [W,H,D],
  // z=flat%D, y=(flat/D)%H, x=flat/(H*D) — matching the CPU emitter's coords + every shader's decomposition.
  // Decompose from the innermost dim outward so a rank-3 (or rank-1) dispatch verifies against the right cell.
  const coordsOf = (flat: number): number[] => {
    const c = new Array<number>(dims.length);
    let rem = flat;
    for (let i = dims.length - 1; i >= 0; i--) { c[i] = rem % dims[i]!; rem = Math.floor(rem / dims[i]!); }
    return c;
  };
  const tol = precision === 'f16' ? 64 : 4;
  let exact = true; let maxUlp = 0; let sampled = 0;
  const step = Math.max(1, Math.floor(total / Math.max(1, sampleCount)));
  for (let flat = 0; flat < total; flat += step) {
    const call = makeCallable(fn, { host, env: declineEnv, maxSteps: perCellSteps });   // fresh budget per cell
    const refComps = extractComps(call(...coordsOf(flat)), comps);   // the interpreter's cell (scalar or vecN)
    for (let k = 0; k < comps; k++) {
      const ref = refComps[k]!;
      const got = output[flat * comps + k] ?? NaN;
      const u = ulpDistance(ref, got);
      if (u !== 0) exact = false;
      if (u > maxUlp) maxUlp = u;
    }
    sampled++;
  }
  // Zero cells sampled (a degenerate/empty/NaN output dimension) is NOT "verified correct" — do not
  // rubber-stamp ok:true. A dispatch with no output to check fails the verdict loud.
  if (sampled === 0) return { ok: false, kind: 'ulp', maxUlp: Infinity };
  const ok = maxUlp <= tol;
  return { ok, kind: exact ? 'exact' : 'ulp', maxUlp };
}
