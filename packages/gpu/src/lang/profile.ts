import type { Profile, HeadSpec } from '@metael/lang/profile';

const H = (name: string, doc: string, params: HeadSpec['params'], returnDoc: string): HeadSpec => ({ name, params, arity: [params.length, params.length], returns: 'value', doc, returnDoc });

const HEADS: readonly HeadSpec[] = [
  H('gpu', 'Dispatch a map kernel over the GPU (or the CPU fallback), producing a reactive result handle you read fields off (r.value, r.backend, r.wgsl, …).', [{ name: 'kernel', doc: 'the per-element kernel component' }, { name: 'cfg', doc: 'dispatch config' }], 'a reactive result handle (the mapped output buffer)'),
  H('gpuReduce', 'Dispatch a reduction (associative fold) kernel, producing a reactive result handle whose value is the reduced scalar once settled.', [{ name: 'reducer', doc: 'the 2-arg associative reducer component' }, { name: 'cfg', doc: 'dispatch config' }], 'a reactive result handle holding the reduction'),
  H('gpuHistogram', 'Dispatch a histogram (atomic scatter) kernel, producing a reactive result handle whose value is the per-bin count array once settled.', [{ name: 'binMapper', doc: 'the bin-mapper component' }, { name: 'cfg', doc: 'dispatch config' }], 'a reactive result handle holding the per-bin counts'),
];

/** The gpu tooling profile: a CLOSED head set (mirrors GpuHostEnv.knownHeads). Each head builds a
 *  value (a compute resource in expression position). */
export const gpuProfile: Profile = {
  id: 'gpu',
  builtins: new Map(),
  heads: new Map(HEADS.map((h) => [h.name, h])),
  types: new Map(),
};
