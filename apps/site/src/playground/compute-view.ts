// Render a Depth-B value (the pure result of evaluateProgram) as a pretty-printed, honest string. Values are
// plain JSON-shaped data (numbers/strings/booleans/null/arrays/objects), all deep-frozen by the language.
// This is "pretty JSON, but readable": a value (or subtree) is printed COMPACT on one line when it fits the
// width budget; only when it overflows does it break onto multiple lines — and a long flat array of
// primitives greedy-fills each line rather than exploding one element per row. So `[0,1,1,2,3,5,8,13]`
// stays a single line, an array of objects breaks per-object, and everything still reads as valid JSON.
// The only non-JSON leaves a metael program can yield are functions and `undefined`; those render as a
// readable placeholder rather than being dropped/mangled.

import { descriptorOf } from '@metael/lang';

const MAX_WIDTH = 72;   // soft line-width budget (chars): compact under it, wrap over it

function isPrimitive(v: unknown): boolean {
  return v === null || v === undefined || typeof v === 'number' || typeof v === 'boolean'
    || typeof v === 'string' || typeof v === 'function';
}

/** A custom value (vec/mat/typed array) is an opaque leaf: it stores its data behind a hidden Symbol,
 *  so Object.entries sees nothing and it would render as `{}`. Render its own display string instead
 *  (e.g. `vec3(1, 2, 3)` / `f32[0, 1, 4] (len 3)`). Returns undefined for a plain value. */
function customLeaf(v: unknown): string | undefined {
  const d = descriptorOf(v);
  if (!d) return undefined;
  return d.display ? d.display(v) : `[${d.name}]`;
}

function leaf(v: unknown): string {
  if (v === null) return 'null';
  if (v === undefined) return '<undefined>';
  if (typeof v === 'function') return '<function>';
  if (typeof v === 'string') return JSON.stringify(v);
  return String(v);
}

/** One-line rendering of any value (no width check) — used to test whether a subtree fits. */
function compact(v: unknown): string {
  const custom = customLeaf(v);
  if (custom !== undefined) return custom;
  if (isPrimitive(v)) return leaf(v);
  if (Array.isArray(v)) return '[' + v.map(compact).join(', ') + ']';
  const entries = Object.entries(v as Record<string, unknown>);
  return entries.length
    ? '{ ' + entries.map(([k, x]) => `${JSON.stringify(k)}: ${compact(x)}`).join(', ') + ' }'
    : '{}';
}

/** Render `v` at `indent` levels of nesting: compact if it fits the width budget, else broken across lines. */
function render(v: unknown, indent: number): string {
  // A custom value is an opaque leaf: never break its display across lines.
  const custom = customLeaf(v);
  if (custom !== undefined) return custom;
  const oneLine = compact(v);
  if (indent * 2 + oneLine.length <= MAX_WIDTH) return oneLine;

  const pad = '  '.repeat(indent + 1);
  const close = '  '.repeat(indent);

  if (Array.isArray(v) && v.length > 0) {
    // A long flat array of primitives: greedy-fill each line up to the width budget (not one per line).
    if (v.every(isPrimitive)) {
      const lines: string[] = [];
      let cur = '';
      for (let i = 0; i < v.length; i++) {
        const tok = leaf(v[i]) + (i < v.length - 1 ? ',' : '');
        if (cur && pad.length + cur.length + 1 + tok.length > MAX_WIDTH) { lines.push(cur); cur = tok; }
        else cur = cur ? `${cur} ${tok}` : tok;
      }
      if (cur) lines.push(cur);
      return `[\n${lines.map((l) => pad + l).join('\n')}\n${close}]`;
    }
    // An array with nested structure: one element per line, each recursively rendered.
    return `[\n${v.map((x) => pad + render(x, indent + 1)).join(',\n')}\n${close}]`;
  }

  if (v && typeof v === 'object') {
    const entries = Object.entries(v as Record<string, unknown>);
    if (entries.length > 0) {
      return `{\n${entries.map(([k, x]) => `${pad}${JSON.stringify(k)}: ${render(x, indent + 1)}`).join(',\n')}\n${close}}`;
    }
  }
  return oneLine;
}

export function prettyValue(value: unknown): string {
  // Top-level non-JSON leaves render unquoted (a bare value, not a JSON string).
  if (typeof value === 'function') return '<function>';
  if (value === undefined) return 'undefined';
  return render(value, 0);
}
