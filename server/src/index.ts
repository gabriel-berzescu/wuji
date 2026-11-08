#!/usr/bin/env node
/** wuji-server: MCPL over stdio; the engine as a child process. Logs go to stderr. */
import { readFileSync } from 'node:fs';
import { McplConnection } from './vendor/mcpl-core/index.js';
import { REPO_ROOT, engineCommand, envMs } from './config.js';
import { Engine } from './engine.js';
import { WujiServer } from './server.js';

const log = (...a: unknown[]) => console.error('[wuji]', ...a);

async function main(): Promise<void> {
  const env = process.env;
  const { version } = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')) as { version: string };
  const engine = new Engine({
    command: engineCommand(env),
    cwd: REPO_ROOT,
    log,
    startTimeoutMs: envMs(env, 'WUJI_ENGINE_START_TIMEOUT_MS', 300_000),
    restartMinMs: envMs(env, 'WUJI_RESTART_MIN_MS', 500),
    restartMaxMs: envMs(env, 'WUJI_RESTART_MAX_MS', 30_000),
  });
  const server = new WujiServer(engine, {
    version,
    log,
    senseBudgetMs: envMs(env, 'WUJI_SENSE_BUDGET_MS', 1_500),
    feltSense: env.WUJI_FELT_SENSE === 'brief' ? 'brief' : 'ambient',
    toolTimeoutMs: envMs(env, 'WUJI_TOOL_TIMEOUT_MS', 20_000),
    // a form calls at every movement (one every few seconds): one a second per kind, 40 a minute in all
    pushMinIntervalMs: envMs(env, 'WUJI_PUSH_MIN_INTERVAL_MS', 1_000),
    pushMaxPerMinute: envMs(env, 'WUJI_PUSH_MAX_PER_MINUTE', 40),
  });

  const conn = McplConnection.fromStreams(process.stdin, process.stdout);
  let exiting = false;
  const exit = async (code: number, why: string): Promise<void> => {
    if (exiting) return;
    exiting = true;
    log(`${why}; stopping the engine`);
    await engine.stop();
    conn.close();
    process.exit(code);
  };
  for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP'] as const) process.on(sig, () => void exit(0, sig));

  engine.start();
  await server.serve(conn);
  await exit(0, 'host disconnected');
}

main().catch((e) => {
  console.error('[wuji] fatal:', e instanceof Error ? e.message : e);
  process.exit(1);
});
