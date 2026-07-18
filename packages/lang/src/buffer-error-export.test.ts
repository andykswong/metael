// packages/lang/src/buffer-error-export.test.ts
import { describe, it, expect } from 'vitest';
import { BufferError } from './index.ts';

describe('BufferError is exported from the barrel (so a domain consumer throws the SAME class the interpreter catches)', () => {
  it('is a constructable Error subclass carrying code + detail', () => {
    const e = new BufferError('ML-LANG-INDEX-RANGE', 'index 9 is out of range (length 4)');
    expect(e).toBeInstanceOf(Error);
    expect(e.code).toBe('ML-LANG-INDEX-RANGE');
    expect(e.detail).toContain('out of range');
  });
});
