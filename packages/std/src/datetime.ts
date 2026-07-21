// Datetime builtins — `now()` (wall-clock milliseconds since the Unix epoch) and `monotonic()` (a
// monotonically-non-decreasing high-resolution reading for measuring elapsed time). Both read the host's
// INJECTED clock capability via `ctx.clock()` — never an ambient `Date.now()`/`performance.now()` inside the
// builtin — so a run is replayable: a domain/test supplies a frozen or recorded clock and the same inputs
// reproduce the same trace. A host that injects NO clock makes these fail LOUD: `ctx.clock()` returns
// undefined → raise ML-LANG-NO-CLOCK + return null (fail-closed; NEVER a fabricated zero, which would read
// as a real timestamp of the epoch).
import type { Builtin, BuiltinSpec, BuiltinCtx } from '@metael/lang';

const spec = (name: string): BuiltinSpec =>
  ({ name, profile: 'host', portability: 'cpu-only', takesClosure: false, arity: [0, 0] });

const nowBuiltin: Builtin = {
  spec: spec('now'),
  invoke: (ctx: BuiltinCtx) => {
    ctx.tick();
    const clk = ctx.clock();
    if (!clk) { ctx.error('ML-LANG-NO-CLOCK', 'now() requires a host clock capability'); return null; }
    return clk.now();
  },
};

const monotonicBuiltin: Builtin = {
  spec: spec('monotonic'),
  invoke: (ctx: BuiltinCtx) => {
    ctx.tick();
    const clk = ctx.clock();
    if (!clk) { ctx.error('ML-LANG-NO-CLOCK', 'monotonic() requires a host clock capability'); return null; }
    return clk.monotonic();
  },
};

/** The datetime builtin module: `now()` (wall-clock milliseconds since the Unix epoch) and `monotonic()`
 *  (a monotonically-non-decreasing high-resolution reading for measuring elapsed time). Both read the
 *  host's INJECTED clock capability rather than an ambient `Date.now()`/`performance.now()`, so a run is
 *  replayable under a frozen or recorded clock. A host that injects no clock makes these fail loud with
 *  `ML-LANG-NO-CLOCK` (returning `null`, never a fabricated zero). Inject via a run's builtin modules to
 *  expose time-reading. */
export const DATETIME_BUILTINS: readonly Builtin[] = [nowBuiltin, monotonicBuiltin];
