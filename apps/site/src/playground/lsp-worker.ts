// The playground's LSP server, running off the main thread. Vite bundles this as a Web Worker
// (imported via `new Worker(new URL('./lsp-worker.ts', import.meta.url), { type: 'module' })`).
import { startWorkerServer } from '@metael/lsp/worker';
import { composeProfiles, coreIntrinsicsProfile } from '@metael/lang/profile';
import type { Profile } from '@metael/lang/profile';
import { mathProfile } from '@metael/math/lang';
import { stdProfile } from '@metael/std';
import { vdomProfile } from '@metael/vdom/lang';
import { gpuProfile } from '@metael/gpu/lang';

/** Map the playground's target id to the composed vocabulary Profile the analysis engine should use. */
function resolveProfile(id: string | undefined): Profile {
  switch (id) {
    case 'ui': return composeProfiles(vdomProfile, stdProfile, coreIntrinsicsProfile);
    case 'gpu': return composeProfiles(gpuProfile, vdomProfile, mathProfile, stdProfile, coreIntrinsicsProfile);
    case 'compute': default: return composeProfiles(mathProfile, stdProfile, coreIntrinsicsProfile);
  }
}

startWorkerServer(self as unknown as DedicatedWorkerGlobalScope, { resolveProfile });
