/**
 * MCPL 0.5 capability grant and feature sets (SPEC §5, §6), on mcpl-core's helpers.
 *
 * Tools are MCP's own surface (§3.2, and mcpl-core's PATHS_NOT_ADVERTISED_IN_MCPL): they answer
 * under any policy, as heartbeat-mcpl's do and as mcpl-cc-bridge's `"grant": []` (plain MCP
 * passthrough) expects. The two MCPL behaviours each have a feature set, and each is active only
 * while every path its `uses` names is in the effective grant of an answered Request: absence is
 * denial, a malformed policy grants nothing, a Notification only narrows.
 */
import {
  CAPABILITY_PATHS,
  capabilityPatternMatches,
  validateUses,
} from './vendor/mcpl-core/index.js';
import type {
  CapabilityPath,
  FeatureSetDeclaration,
  FeatureSetsUpdateAccepted,
  McplCapabilities,
  UnavailableFeature,
} from './vendor/mcpl-core/index.js';

export const FS_FELT_SENSE = 'wuji.feltSense';
export const FS_SIGNALS = 'wuji.signals';

export const EVENT_KINDS = ['balance', 'breath', 'form', 'strain'] as const;

export const FEATURE_SETS: Record<string, FeatureSetDeclaration> = {
  [FS_FELT_SENSE]: {
    description: 'Felt sense before each turn: a few lines of proprioception (weight, alignment, effort, breath, heart) '
      + 'injected before the user message. Write-without-read: it never sees what the user said.',
    uses: ['contextHooks.beforeInference.inject.beforeUser'],
  },
  [FS_SIGNALS]: {
    description: 'Body signals: the body asks for attention (balance near the edge, the urge to breathe, '
      + 'form progress, strain). Rare by design.',
    uses: ['pushEvents'],
    tagOntology: {
      tags: {
        'wuji:balance': { desc: 'Balance: the centre of pressure nears the edge of the support, a foot unloads, or sway grows.', facet: 'content' },
        'wuji:breath': { desc: 'Breath: the urge to breathe rises during a hold, or the breath settles again.', facet: 'content' },
        'wuji:form': { desc: 'A step of a form begins, or the form ends.', facet: 'lifecycle' },
        'wuji:strain': { desc: 'A posture cannot be held without excessive effort.', facet: 'content' },
        'wuji:warning': { desc: 'A balance signal of severity "warning" (with wuji:balance).', facet: 'urgency' },
        'wuji:notice': { desc: 'A balance signal of severity "notice" (with wuji:balance).', facet: 'urgency' },
      },
      open: true,
    },
  },
};

/**
 * The manifest (`capabilities.experimental.mcpl`). Only the injection leaf is advertised, not
 * `observe`: the felt sense has no need of the user's text, so a host never sends it (§10.1).
 */
export const MCPL_MANIFEST: McplCapabilities = {
  version: '0.5',
  pushEvents: true,
  contextHooks: { beforeInference: { inject: { beforeUser: true } } },
  featureSets: FEATURE_SETS,
};

export interface Policy {
  /** Concrete capability paths granted (patterns resolved against the §6.2 vocabulary). */
  paths: ReadonlySet<CapabilityPath>;
  /** `enabled` from the Request: null constrains nothing; a list is an allowlist (§5.3). */
  enabled: readonly string[] | null;
  /** `disabled`, accumulated: always subtracts. */
  disabled: readonly string[];
  /** Established only by a well-formed Request, once answered (§5.3, §6.7). */
  ready: boolean;
}

export const NO_POLICY: Policy = { paths: new Set(), enabled: null, disabled: [], ready: false };

type Parsed = {
  effectiveCapabilities?: string[];
  deniedCapabilities?: string[];
  enabled?: string[];
  disabled?: string[];
};

function parseParams(raw: unknown): { ok: true; params: Parsed } | { ok: false; error: string } {
  if (raw === undefined || raw === null) return { ok: true, params: {} };
  if (typeof raw !== 'object' || Array.isArray(raw)) return { ok: false, error: 'params must be an object' };
  const out: Parsed = {};
  for (const field of ['effectiveCapabilities', 'deniedCapabilities', 'enabled', 'disabled'] as const) {
    const v = (raw as Record<string, unknown>)[field];
    if (v === undefined) continue;
    if (!Array.isArray(v) || !v.every((x) => typeof x === 'string')) return { ok: false, error: `"${field}" must be an array of strings` };
    out[field] = v as string[];
  }
  return { ok: true, params: out };
}

function resolvePaths(patterns: readonly string[]): Set<CapabilityPath> {
  return new Set(CAPABILITY_PATHS.filter((path) => patterns.some((p) => capabilityPatternMatches(p, path))));
}

/**
 * A featureSets/update Request: a full policy statement. Absent effectiveCapabilities grants
 * nothing; a path both granted and denied makes it malformed, and the caller fails closed.
 */
export function policyFromRequest(raw: unknown): { ok: true; policy: Policy } | { ok: false; error: string } {
  const parsed = parseParams(raw);
  if (!parsed.ok) return parsed;
  const p = parsed.params;
  const effective = p.effectiveCapabilities ?? [];
  const denied = p.deniedCapabilities ?? [];
  const both = effective.filter((x) => denied.includes(x));
  if (both.length > 0) return { ok: false, error: `in both effectiveCapabilities and deniedCapabilities: ${both.join(', ')}` };
  return {
    ok: true,
    policy: { paths: resolvePaths(effective), enabled: p.enabled ?? null, disabled: p.disabled ?? [], ready: true },
  };
}

/**
 * A featureSets/update Notification (§6.7): it can only narrow and never makes ready. `disabled`
 * subtracts; an effectiveCapabilities list narrows the grant to what both allow; an `enabled`
 * list narrows the selection. Nothing it carries can widen.
 */
export function narrowByNotification(current: Policy, raw: unknown): { ok: true; policy: Policy } | { ok: false; error: string } {
  const parsed = parseParams(raw);
  if (!parsed.ok) return parsed;
  const p = parsed.params;
  const paths = p.effectiveCapabilities
    ? new Set([...current.paths].filter((path) => resolvePaths(p.effectiveCapabilities!).has(path)))
    : current.paths;
  let enabled = current.enabled;
  if (p.enabled) enabled = enabled === null ? p.enabled : enabled.filter((n) => p.enabled!.some((pat) => nameMatches(pat, n)));
  return { ok: true, policy: { paths, enabled, disabled: [...current.disabled, ...(p.disabled ?? [])], ready: current.ready } };
}

/** §6.3: a feature-set name, `*`, or a `prefix.*` pattern. */
function nameMatches(pattern: string, name: string): boolean {
  return pattern === name || pattern === '*' || (pattern.endsWith('.*') && name.startsWith(pattern.slice(0, -1)));
}

export type InactiveReason = 'not_ready' | 'invalid_uses' | 'capability_denied' | 'host_disabled' | 'not_selected';

export function featureSetState(name: string, policy: Policy): { active: true } | { active: false; reason: InactiveReason; missing: CapabilityPath[] } {
  const decl = FEATURE_SETS[name];
  if (!decl || !validateUses(decl.uses).valid) return { active: false, reason: 'invalid_uses', missing: [] };
  const missing = decl.uses.filter((u) => !policy.paths.has(u));
  if (!policy.ready) return { active: false, reason: 'not_ready', missing };
  if (missing.length > 0) return { active: false, reason: 'capability_denied', missing };
  if (policy.disabled.some((pat) => nameMatches(pat, name))) return { active: false, reason: 'host_disabled', missing: [] };
  if (policy.enabled !== null && !policy.enabled.some((pat) => nameMatches(pat, name))) return { active: false, reason: 'not_selected', missing: [] };
  return { active: true };
}

const CONSEQUENCE: Record<string, string> = {
  [FS_FELT_SENSE]: 'no felt sense arrives before turns; the sense tool still answers',
  [FS_SIGNALS]: 'body signals (balance, breath, form, strain) are not pushed; sense and the tools still work',
};

/** The degradation receipt (§6.7): testimony about consequences, `mode` only when degraded. */
export function buildReceipt(policy: Policy): FeatureSetsUpdateAccepted {
  const unavailableFeatures: UnavailableFeature[] = [];
  const notes: string[] = [];
  for (const name of Object.keys(FEATURE_SETS)) {
    const s = featureSetState(name, policy);
    if (s.active) continue;
    unavailableFeatures.push({ featureSet: name, missingCapabilities: s.missing, effect: 'disabled' });
    notes.push(`${name}: ${s.reason === 'capability_denied' ? `${s.missing.join(', ')} not granted` : s.reason.replace('_', ' ')}; ${CONSEQUENCE[name]}`);
  }
  return {
    accepted: true,
    ...(unavailableFeatures.length > 0 ? { mode: 'degraded' } : {}),
    unavailableFeatures,
    notes,
  };
}
