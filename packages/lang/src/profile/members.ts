import type { MemberSpec } from './types.ts';

const SWIZZLE = 'xyzw';

/** Generate the swizzle/component member set for a `rows`-component vector, matching the runtime
 *  member rule: single components `x…` within `rows`, plus every 2–4-length combination of those
 *  in-range letters. `rows` is clamped to `[0, 4]`. */
export function swizzleMembers(rows: number): readonly MemberSpec[] {
  const n = Math.max(0, Math.min(4, Math.trunc(rows)));
  const letters = SWIZZLE.slice(0, n).split('');
  const out: MemberSpec[] = letters.map((name) => ({ name, kind: 'component', doc: `component ${name}` }));
  const combos = (len: number): void => {
    const rec = (prefix: string): void => {
      if (prefix.length === len) { out.push({ name: prefix, kind: 'swizzle', doc: `${len}-component swizzle` }); return; }
      for (const c of letters) rec(prefix + c);
    };
    rec('');
  };
  for (let len = 2; len <= Math.min(4, n); len++) combos(len);
  return out;
}
