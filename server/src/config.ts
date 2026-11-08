import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

/** The wuji repo: this file is built to <repo>/server/dist/config.js. */
export const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));

/**
 * The engine's argv. WUJI_ENGINE replaces the default with a JSON array of strings (no shell, so
 * a path with spaces is one element, as it is on every platform).
 */
export function engineCommand(env: NodeJS.ProcessEnv = process.env, repoRoot: string = REPO_ROOT): string[] {
  const raw = env.WUJI_ENGINE;
  if (raw === undefined || raw.trim() === '') {
    return ['uv', 'run', '--project', join(repoRoot, 'engine'), 'python', '-m', 'wuji.engine'];
  }
  let parsed: unknown;
  try { parsed = JSON.parse(raw); } catch (e) {
    throw new Error(`WUJI_ENGINE must be a JSON array of strings, e.g. ["python", "-m", "wuji.engine"]: ${(e as Error).message}`);
  }
  if (!Array.isArray(parsed) || parsed.length === 0 || !parsed.every((x) => typeof x === 'string' && x !== '')) {
    throw new Error('WUJI_ENGINE must be a non-empty JSON array of non-empty strings, e.g. ["python", "-m", "wuji.engine"]');
  }
  return parsed as string[];
}

/** A non-negative integer from the environment, or the default. */
export function envMs(env: NodeJS.ProcessEnv, name: string, fallback: number): number {
  const raw = env[name];
  if (raw === undefined || raw.trim() === '') return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? Math.round(n) : fallback;
}
