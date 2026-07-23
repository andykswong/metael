import type { Profile } from '@metael/lang/profile';
import { Document } from './document.ts';
import type { LineIndex } from './line-index.ts';
import { ScopeModel } from './scope-model.ts';
import { computeDiagnostics } from './analyses/diagnostics.ts';
import { computeScopeChecks } from './analyses/scope-check.ts';
import { computeCompletion } from './analyses/completion.ts';
import { computeHover } from './analyses/hover.ts';
import { computeSignature } from './analyses/signature.ts';
import { computeSemanticTokens } from './analyses/semantic-tokens.ts';
import { computeFolding } from './analyses/folding.ts';
import { computeSelection } from './analyses/selection.ts';
import { computeFormat } from './analyses/format.ts';
import { computeCapabilityLens } from './analyses/capability-lens.ts';
import type { SvcCompletion, SvcDiagnostic, SvcEdit, SvcFold, SvcHover, SvcLens, SvcSelection, SvcSignature, SvcToken } from './results.ts';

/** The pure metael analysis engine. Speaks char offsets + `Svc*` records; imports no LSP protocol.
 *  A host opens/updates documents, injects a (possibly composed) Profile per document, and pulls
 *  analyses. Analysis methods are added incrementally. */
export class LanguageService {
  private readonly docs = new Map<string, Document>();
  private readonly scopes = new Map<string, ScopeModel>();
  private readonly profiles = new Map<string, Profile>();

  /** Register a newly opened document. */
  openDocument(uri: string, text: string, version: number): void {
    this.setDoc(uri, new Document(text, version));
  }

  /** Replace a document's text (full sync). */
  updateDocument(uri: string, text: string, version: number): void {
    this.setDoc(uri, new Document(text, version));
  }

  /** Forget a closed document. */
  closeDocument(uri: string): void {
    this.docs.delete(uri);
    this.scopes.delete(uri);
  }

  /** Inject the active vocabulary Profile for a document (from the host's target/config). */
  setProfile(uri: string, profile: Profile): void {
    this.profiles.set(uri, profile);
  }

  /** True when a document is open. */
  hasDocument(uri: string): boolean {
    return this.docs.has(uri);
  }

  /** The offset↔line/col mapper for an open document (for a protocol shell to marshal positions), or
   *  `undefined` when the document is not open. Returns the protocol-free {@link LineIndex} so the engine
   *  stays offset-only while a shell gets the mapper it owns the use of. */
  lineIndexFor(uri: string): LineIndex | undefined {
    return this.docFor(uri)?.lineIndex;
  }

  /** Merged lex/parse diagnostics for a document, plus — when a Profile is set — the ScopeModel-driven
   *  scope checks (undeclared value-reads and block-scope redeclarations) the parser cannot produce. The
   *  scope checks need a Profile to recognise builtin/head/type names (else those would wrongly flag as
   *  undeclared), so without one only the lex/parse diagnostics are returned. */
  diagnostics(uri: string): readonly SvcDiagnostic[] {
    const doc = this.docFor(uri);
    if (!doc) return [];
    const base = computeDiagnostics(doc);
    const scope = this.scopeFor(uri);
    const profile = this.profileFor(uri);
    if (!scope || !profile) return base;
    return [...base, ...computeScopeChecks(doc, scope, profile)];
  }

  /** Completion candidates at `offset`: member candidates after a `.`, otherwise visible bindings ∪
   *  profile builtins ∪ heads (+ `range`). Empty when the document, scope, or profile is unavailable. */
  completion(uri: string, offset: number): readonly SvcCompletion[] {
    const doc = this.docFor(uri);
    const scope = this.scopeFor(uri);
    const profile = this.profileFor(uri);
    if (!doc || !scope || !profile) return [];
    return computeCompletion(doc, scope, profile, offset);
  }

  /** The hover card for the identifier at `offset`: a builtin/head/type card from the profile, or a
   *  local binding card, formatted as markdown. `null` when the offset isn't on a resolvable identifier
   *  or the document, scope, or profile is unavailable. */
  hover(uri: string, offset: number): SvcHover | null {
    const doc = this.docFor(uri);
    const scope = this.scopeFor(uri);
    const profile = this.profileFor(uri);
    if (!doc || !scope || !profile) return null;
    return computeHover(doc, scope, profile, offset);
  }

  /** Signature help for the call enclosing `offset`: the rendered signature, its parameters, and the
   *  active parameter index, resolved from the profile's heads/builtins. `null` when the offset is not
   *  inside a resolvable call or the document or profile is unavailable. */
  signatureHelp(uri: string, offset: number): SvcSignature | null {
    const doc = this.docFor(uri);
    const profile = this.profileFor(uri);
    if (!doc || !profile) return null;
    return computeSignature(doc, profile, offset);
  }

  /** Semantic tokens for the whole document: every lexed token classified into an {@link SvcTokenKind},
   *  with identifiers refined against the profile and scope. Empty when the document, scope, or profile
   *  is unavailable. */
  semanticTokens(uri: string): readonly SvcToken[] {
    const doc = this.docFor(uri);
    const scope = this.scopeFor(uri);
    const profile = this.profileFor(uri);
    if (!doc || !scope || !profile) return [];
    return computeSemanticTokens(doc, scope, profile);
  }

  /** Foldable regions of a document — one per brace-delimited block body (`function`/`component` bodies,
   *  `if`/`else`/`while`/`for` bodies, wrapping-call and arrow blocks), each spanning `{`..`}`. Empty when
   *  the document is unavailable. */
  foldingRanges(uri: string): readonly SvcFold[] {
    const doc = this.docFor(uri);
    return doc ? computeFolding(doc) : [];
  }

  /** Selection (expand-selection) ranges for each requested offset: the widening chain of enclosing AST
   *  spans, innermost-first. One {@link SvcSelection} per input offset, in order. Empty when the document
   *  is unavailable. */
  selectionRanges(uri: string, offsets: readonly number[]): readonly SvcSelection[] {
    const doc = this.docFor(uri);
    return doc ? computeSelection(doc, offsets) : [];
  }

  /** Whole-document format via the canonical printer — a single edit over `[0, length)`, or [] when the
   *  parse carries errors (don't reprint a broken AST), the text is already canonical, or the document is
   *  unavailable. */
  format(uri: string): readonly SvcEdit[] {
    const doc = this.docFor(uri);
    return doc ? computeFormat(doc) : [];
  }

  /** A capability lens per top-level `function`/`component`: whether its body is GPU/WASM-lowerable under
   *  the active profile, plus why-not reasons. Empty when the document or profile is unavailable. */
  capabilityLens(uri: string): readonly SvcLens[] {
    const doc = this.docFor(uri);
    const profile = this.profileFor(uri);
    return doc && profile ? computeCapabilityLens(doc, profile) : [];
  }

  private setDoc(uri: string, doc: Document): void {
    this.docs.set(uri, doc);
    this.scopes.delete(uri);
  }

  /** @internal */
  protected docFor(uri: string): Document | undefined {
    return this.docs.get(uri);
  }

  /** @internal */
  protected scopeFor(uri: string): ScopeModel | undefined {
    const doc = this.docs.get(uri);
    if (!doc) return undefined;
    let m = this.scopes.get(uri);
    if (!m) {
      m = new ScopeModel(doc);
      this.scopes.set(uri, m);
    }
    return m;
  }

  /** @internal */
  protected profileFor(uri: string): Profile | undefined {
    return this.profiles.get(uri);
  }
}
