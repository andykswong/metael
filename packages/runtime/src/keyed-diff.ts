// The generic keyed-list diff. PURE + tree-shape-agnostic: given two keyed sequences, produce a
// complete add/remove/move op list that preserves identity (a move is emitted for every element whose
// index shifted — this is COMPLETE and correct, not LIS-minimized; an LIS pass to minimize moves is a
// later optimization if a domain measures it). applyKeyedDiff reconciles a retained list against a
// next-key sequence — reusing matched instances, creating new ones, and DISPOSING removed ones (the
// disposal contract runLeafEffect/scope() were built to serve). The domain supplies create + dispose
// (and its own recursion/patch); the runtime owns the diff + teardown-on-remove.

/** One reconciliation instruction produced by {@link diffKeyed} to turn a `prev` key sequence into a
 *  `next` one. */
export type KeyedOp =
  /** A key present in `next` but not matched in `prev` (a new element, or a duplicate key): create it at
   *  `index` in the next sequence. */
  | { type: 'add'; key: string; index: number }
  /** A key present in `prev` but absent from `next`: the element is gone and should be torn down. */
  | { type: 'remove'; key: string }
  /** A matched key whose position shifted: relocate the retained element from index `from` to `to`. */
  | { type: 'move'; key: string; from: number; to: number };

/** Pure op generation over two key sequences. Duplicate keys in `next` past the first occurrence are
 *  treated as adds (a keyed list has unique keys; a dup is a new slot, never an alias).
 *  Note: diffKeyed is not teardown-authoritative under duplicate keys (a collapsed duplicate in `prev`
 *  yields no remove op); a domain that needs teardown uses applyKeyedDiff, which disposes by identity. */
export function diffKeyed(prev: readonly string[], next: readonly string[]): KeyedOp[] {
  const ops: KeyedOp[] = [];
  const prevIndex = new Map<string, number>();
  prev.forEach((k, i) => { if (!prevIndex.has(k)) prevIndex.set(k, i); });
  const seen = new Set<string>();

  // Removes: any prev key absent from next.
  const nextKeySet = new Set(next);
  for (const k of prev) if (!nextKeySet.has(k)) ops.push({ type: 'remove', key: k });

  // Adds + moves: walk next; a first-seen matched key that shifted position is a move, an unmatched
  // (or duplicate) key is an add.
  next.forEach((k, to) => {
    const seenBefore = seen.has(k);
    seen.add(k);
    const from = prevIndex.get(k);
    if (from === undefined || seenBefore) {
      ops.push({ type: 'add', key: k, index: to });
    } else if (from !== to) {
      ops.push({ type: 'move', key: k, from, to });
    }
  });
  return ops;
}

/** The domain-supplied callbacks {@link applyKeyedDiff} uses to build and tear down list instances while
 *  reconciling. The runtime owns the diff + teardown-on-remove; the domain owns instance construction and
 *  disposal. */
export interface KeyedReconcileHooks<T> {
  /** Build a fresh instance for a `next` key with no reusable match, at its position `index` in the next
   *  sequence. */
  create: (key: string, index: number) => T;
  /** Tear down a `prev` instance that was not carried into the output (its subtree/effects are released). */
  dispose: (item: T) => void;
}

/** Reconcile `prev` against the `next` key order. Matched keys reuse the retained instance (identity
 *  preserved — consume-once so a duplicate key in next takes a fresh instance, never a hard alias);
 *  new keys are created; gone keys are disposed (teardown). Returns the next-ordered list. Order =
 *  the `next` sequence. */
export function applyKeyedDiff<T>(
  prev: readonly T[],
  next: readonly string[],
  keyOf: (item: T) => string,
  hooks: KeyedReconcileHooks<T>,
): T[] {
  const byKey = new Map<string, T>();
  for (const it of prev) { const k = keyOf(it); if (!byKey.has(k)) byKey.set(k, it); }

  const out: T[] = [];
  const reused = new Set<T>();          // the prev INSTANCES actually carried into out (by identity)
  const consumed = new Set<string>();
  next.forEach((k, index) => {
    const existing = byKey.get(k);
    if (existing !== undefined && !consumed.has(k)) {
      consumed.add(k);            // consume-once: a 2nd next-key gets a fresh instance below
      reused.add(existing);
      out.push(existing);
    } else {
      out.push(hooks.create(k, index));   // new key (or duplicate) → fresh instance
    }
  });

  // Dispose every prev instance NOT reused into the output (identity-based — sound even when two prev
  // items share a key, which consume-once can produce; disposing by key-absence would orphan the extra).
  for (const it of prev) if (!reused.has(it)) hooks.dispose(it);
  return out;
}
