import { describe, it, expect } from 'vitest';
import { LspClient } from './lsp-client.ts';

describe('LspClient over the real Worker (Chromium)', () => {
  it('initializes and answers completion', async () => {
    const client = new LspClient();
    await client.initialize('compute');
    client.didOpen('a', 'const x = ', 1);
    const items = await client.completion('a', 'const x = '.length);
    expect(items.map((i) => i.label)).toContain('map');
    client.dispose();
  });

  it('delivers diagnostics for an undeclared identifier read', async () => {
    const client = new LspClient();
    await client.initialize('compute');
    // The listener races the didOpen-triggered publish, so capture into a promise.
    const arrived = new Promise<{ uri: string; diags: readonly { code?: string | number }[] }>((resolve) => {
      client.onDiagnostics((uri, diags) => {
        if (uri === 'b' && diags.length > 0) resolve({ uri, diags });
      });
    });
    // `bar` is read as a value but never declared → ML-LANG-UNKNOWN-VAR under the compute profile.
    client.didOpen('b', 'const y = bar', 1);
    const { uri, diags } = await arrived;
    expect(uri).toBe('b');
    expect(diags.length).toBeGreaterThan(0);
    expect(String(diags[0]!.code)).not.toBe('');
    client.dispose();
  });
});
