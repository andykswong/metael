// packages/gpu/src/device/pipeline-cache.test.ts
import { describe, it, expect } from 'vitest';
import { makePipelineCache } from './pipeline-cache.ts';

describe('makePipelineCache — compile once per distinct shader source', () => {
  it('reuses the compiled artifact for the same source; recompiles for a different source', () => {
    let compiles = 0;
    const disposed: string[] = [];
    const cache = makePipelineCache<string>((src) => { compiles++; return `pipe:${src}`; }, (p) => disposed.push(p));
    expect(cache.get('AAA')).toBe('pipe:AAA');
    expect(cache.get('AAA')).toBe('pipe:AAA');   // memo hit
    expect(compiles).toBe(1);
    expect(cache.get('BBB')).toBe('pipe:BBB');
    expect(compiles).toBe(2);
    cache[Symbol.dispose]();                     // native Disposable
    expect(disposed.sort()).toEqual(['pipe:AAA', 'pipe:BBB']);
    // after dispose the cache is empty → a get recompiles
    expect(cache.get('AAA')).toBe('pipe:AAA');
    expect(compiles).toBe(3);
  });

  it('is idempotent: a second dispose is a no-op', () => {
    const disposed: string[] = [];
    const cache = makePipelineCache<string>((src) => `pipe:${src}`, (p) => disposed.push(p));
    cache.get('AAA');
    cache[Symbol.dispose]();
    cache[Symbol.dispose]();                      // no double-free
    expect(disposed).toEqual(['pipe:AAA']);
  });

  it('works with a `using` declaration', () => {
    const disposed: string[] = [];
    {
      using cache = makePipelineCache<string>((src) => `pipe:${src}`, (p) => disposed.push(p));
      cache.get('AAA');
    }   // block exit → [Symbol.dispose]() fires
    expect(disposed).toEqual(['pipe:AAA']);
  });
});

describe('makePipelineCache — WebGL2 program free semantics', () => {
  it('frees every distinct program exactly once on dispose', () => {
    const deleted: number[] = [];
    let next = 0;
    const cache = makePipelineCache<number>(() => ++next, (p) => deleted.push(p));
    cache.get('vsA|fsA'); cache.get('vsA|fsA'); cache.get('vsB|fsB');
    expect(next).toBe(2);            // two distinct sources → two programs
    cache[Symbol.dispose]();
    expect(deleted.sort((a, b) => a - b)).toEqual([1, 2]);
  });
});
