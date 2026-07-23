import { describe, it, expect } from 'vitest';
import { LanguageService } from './index.ts';
import { stdProfile } from '@metael/std';

describe('LanguageService document lifecycle', () => {
  it('opens, updates, and tracks per-uri profile', () => {
    const svc = new LanguageService();
    svc.openDocument('a.metael', 'const x = 1', 1);
    svc.setProfile('a.metael', stdProfile);
    svc.updateDocument('a.metael', 'const x = 2', 2);
    // no throw + the diagnostics call (added later) will read this doc; here just assert it opened:
    expect(svc.hasDocument('a.metael')).toBe(true);
  });
});
