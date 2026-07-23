import { describe, it, expect } from 'vitest';
import { LanguageService } from '../index.ts';
import { stdProfile } from '@metael/std';

describe('diagnostics', () => {
  it('surfaces parse errors as error-severity SvcDiagnostics with a span', () => {
    const svc = new LanguageService();
    svc.openDocument('a', 'const x = ', 1);
    svc.setProfile('a', stdProfile);
    const ds = svc.diagnostics('a');
    expect(ds.length).toBeGreaterThan(0);
    expect(ds[0]!.severity).toBe('error');
    expect(ds[0]!.span.end).toBeGreaterThanOrEqual(ds[0]!.span.start);
  });
  it('returns [] for a clean program', () => {
    const svc = new LanguageService();
    svc.openDocument('b', 'const x = 1', 1);
    svc.setProfile('b', stdProfile);
    expect(svc.diagnostics('b')).toEqual([]);
  });
  it('reports a lexer diagnostic exactly once (no lex/parse duplication)', () => {
    const svc = new LanguageService();
    svc.openDocument('c', 'const x = @', 1);
    svc.setProfile('c', stdProfile);
    const ds = svc.diagnostics('c');
    expect(ds.filter((d) => d.code === 'ML-LANG-LEX').length).toBe(1);
  });
});
