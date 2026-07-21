import type { RuntimeReactiveHost } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv, type UserFn } from '@metael/lang';
import { MATH_BUILTINS } from '@metael/math/lang';   // the numeric builtins a kernel intrinsically uses (vec/mat/f32)

/** Compile a metael kernel snippet into the UserFn the engine lowers. Evaluate against `host` so the
 *  kernel's closure (its `const a = f32(...)` inputs, or a factory's captured params) lives on the same
 *  host the engine reads. Throws if the program's value is not a function/component. */
export function compileKernel(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
  if (!isUserFn(res.value)) throw new Error('kernel source must evaluate to a function or component');
  return res.value;
}
