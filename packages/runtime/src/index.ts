// @metael/runtime public barrel. Exports: the reactive core, the real ReactiveHost, the generic keyed
// diff, and the one-shot derive() composition root. HostEnvironment/ReactiveHost/KeyMinter INTERFACES +
// the test doubles live in @metael/lang (re-exported here for a domain's convenience).
export { signal, memo, effect, change, ReactiveFlushError } from './reactive.ts';
export type { Signal, Memo } from './reactive.ts';
export { RuntimeReactiveHost } from './reactive-host.ts';
export { diffKeyed, applyKeyedDiff } from './keyed-diff.ts';
export type { KeyedOp, KeyedReconcileHooks } from './keyed-diff.ts';
export { derive } from './derive.ts';
export type { DeriveOptions, DeriveResult } from './derive.ts';
export { composeEnvs } from './compose-envs.ts';
export type { ComposedHostEnv } from './compose-envs.ts';

// Convenience re-exports of the lang seam a domain needs alongside the runtime (single import site).
export {
  lowerEntry, region, isRegion, wrapper, isWrapper, didYouMean,
  PlainStorageHost, RecordingHostEnv, PathKeyMinter,
} from '@metael/lang';
export type {
  HostEnvironment, ReactiveHost, KeyMinter, HostValue, CellRef, EffectRegion,
  Region, LangWrapper, Arg, Scope, Diagnostic, SourceSpan, LowerOptions, LowerResult,
  BindableHostEnv,
} from '@metael/lang';
