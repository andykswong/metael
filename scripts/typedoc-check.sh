#!/usr/bin/env bash
# Per-package TypeDoc "notDocumented" gate.
#
# Why per-package and not one aggregate run: TypeDoc's multi-entry `resolve` strategy drops
# cross-imported reflections before validation, so an aggregate config over all packages validates
# NONE of them (a missing doc goes unreported). Each package is therefore validated in isolation via
# its own packages/<pkg>/typedoc.json (which `extends` typedoc.base.json — settings live in ONE place).
#
# The `.__type.` filter drops discriminated-union arm members: a union's arms are anonymous inline type
# literals, so TypeDoc reports each arm field as `<Union>.__type.<field>`. Those ARE documented — at the
# arm level (one `/** */` per variant), the only place a doc reads sensibly. Every REAL named
# interface/class property still counts.
#
# Pass --files to print each offending symbol.
cd "$(dirname "$0")/.." || exit 2
show_files=0
[ "${1:-}" = "--files" ] && show_files=1
total=0
for cfg in packages/*/typedoc.json; do
  out=$(npx typedoc --options "$cfg" --out "$(mktemp -d)/td" 2>&1) || true
  # A crash (bad path, resolver error) must not read as "0 undocumented" — require the success marker.
  if ! printf '%s' "$out" | grep -q "generated at"; then
    echo "✗ $cfg: TypeDoc did not complete — cannot trust the result:"
    printf '%s\n' "$out" | tail -5
    exit 2
  fi
  gaps=$(printf '%s\n' "$out" | grep "does not have any documentation" | grep -v "\.__type\." || true)
  hits=$(printf '%s' "$gaps" | grep -c . || true)
  if [ "$hits" -eq 0 ]; then
    echo "✓ $cfg: 0 undocumented"
  else
    echo "✗ $cfg: $hits undocumented"
    [ "$show_files" -eq 1 ] && printf '%s\n' "$gaps" | sed -E 's/^.*\] //; s/, does not have any documentation//; s/^/    /'
  fi
  total=$((total + hits))
done
echo ""
echo "TOTAL undocumented public symbols: $total"
[ "$total" -eq 0 ]
