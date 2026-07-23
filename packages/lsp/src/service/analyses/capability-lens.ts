import { classifyProfile } from '@metael/lang/profile';
import type { Profile } from '@metael/lang/profile';
import type { Stmt } from '@metael/lang';
import type { Document } from '../document.ts';
import type { SvcLens } from '../results.ts';

/** A `function`/`component` statement — the two top-level declarations with a `body: Stmt[]` that can be
 *  classified for lowerability. */
type FunctionLike = Extract<Stmt, { kind: 'function' | 'component' }>;

/** Narrow a top-level statement to a {@link FunctionLike}, or `undefined` when it is neither a
 *  `function` nor a `component`. */
function asFunctionLike(s: Stmt): FunctionLike | undefined {
  return s.kind === 'function' || s.kind === 'component' ? s : undefined;
}

/**
 * Compute a capability lens for every top-level `function`/`component` in a document.
 *
 * @remarks
 * Each lens reports whether the declaration's body is GPU/WASM-lowerable under the active `profile` —
 * i.e. uses only core-compliant, closure-free, heap-free constructs and no host builtins. The verdict
 * and its backing reasons come straight from {@link classifyProfile}, which is handed the declaration
 * itself (it reads only `.body`) and the profile so a called name can be resolved to a core intrinsic
 * versus a host capability. `label` is `'GPU/WASM-lowerable'` when core, else `'not lowerable'`;
 * `reasons` carries one human-readable explanation per disqualifier (empty when lowerable). Only
 * top-level declarations are lensed; nested functions are surfaced as a disqualifier of their enclosing
 * body rather than as their own lens. Pure and total.
 */
export function computeCapabilityLens(doc: Document, profile: Profile): readonly SvcLens[] {
  const out: SvcLens[] = [];
  for (const s of doc.parse.program.stmts ?? []) {
    const fn = asFunctionLike(s);
    if (!fn) continue;
    const r = classifyProfile(fn, profile);
    out.push({
      span: { start: fn.span.start, end: fn.span.end },
      label: r.core ? 'GPU/WASM-lowerable' : 'not lowerable',
      lowerable: r.core,
      reasons: r.reasons.map((d) => d.message),
    });
  }
  return out;
}
