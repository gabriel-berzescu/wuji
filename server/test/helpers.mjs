// Test harness: runs the built server as a child process against the fake engine, and plays the
// MCPL host on its stdio with mcpl-core's own McplConnection.
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { McplConnection } from '../dist/vendor/mcpl-core/index.js';

export const SERVER = fileURLToPath(new URL('../dist/index.js', import.meta.url));
export const FAKE_ENGINE = fileURLToPath(new URL('./fake-engine.mjs', import.meta.url));

export const FULL_GRANT = ['tools', 'pushEvents', 'contextHooks.beforeInference.inject.beforeUser'];

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function tempDir() {
  // A path with a space in it, to check nothing splits on spaces.
  return mkdtempSync(join(tmpdir(), 'wuji test '));
}

/** Start the server with the fake engine; returns a host-side harness. */
export function startServer(fakeEnv = {}, serverEnv = {}) {
  const stateDir = tempDir();
  const child = spawn(process.execPath, [SERVER], {
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true,
    env: {
      ...process.env,
      WUJI_ENGINE: JSON.stringify([process.execPath, FAKE_ENGINE]),
      WUJI_RESTART_MIN_MS: '100',
      WUJI_PUSH_MIN_INTERVAL_MS: '0',
      FAKE_STATE_DIR: stateDir,
      ...fakeEnv,
      ...serverEnv,
    },
  });
  let stderr = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (d) => { stderr += d; if (process.env.WUJI_TEST_VERBOSE) process.stderr.write(d); });
  const exited = new Promise((r) => child.once('exit', (code) => r(code)));
  const conn = McplConnection.fromStreams(child.stdout, child.stdin);
  conn.on('error', () => {});
  const pushes = [];
  const unexpected = [];
  // Answer what the server asks of the host (push/event), and keep it.
  (async () => {
    for (;;) {
      let msg;
      try { msg = await conn.nextMessage(); } catch { return; }
      if (msg.type === 'request' && msg.request.method === 'push/event') {
        pushes.push(msg.request.params);
        conn.sendResponse(msg.request.id, { accepted: true });
      } else {
        unexpected.push(msg);
        if (msg.type === 'request') conn.sendError(msg.request.id, -32601, 'not a host method');
      }
    }
  })();

  const h = {
    child, conn, pushes, unexpected, stateDir, exited,
    get stderr() { return stderr; },
    request: (method, params, timeout = 10_000) => conn.sendRequest(method, params, timeout),
    async initialize(mcpl = true) {
      const result = await conn.sendRequest('initialize', {
        protocolVersion: '2024-11-05',
        capabilities: { tools: {}, ...(mcpl ? { experimental: { mcpl: { version: '0.5', pushEvents: true, featureSets: true, contextHooks: { beforeInference: true } } } } : {}) },
        clientInfo: { name: 'wuji-test-host', version: '0' },
      });
      conn.sendNotification('notifications/initialized', {});
      return result;
    },
    grant(effectiveCapabilities, extra = {}) {
      return conn.sendRequest('featureSets/update', { ...(effectiveCapabilities ? { effectiveCapabilities } : {}), ...extra });
    },
    call(name, args = {}, timeout) { return conn.sendRequest('tools/call', { name, arguments: args }, timeout ?? 15_000); },
    beforeInference(timeout = 5_000) {
      return conn.sendRequest('context/beforeInference', {
        inferenceId: `inf_${Math.random().toString(36).slice(2)}`, conversationId: 'conv_test', turnIndex: 0, userMessage: null,
        model: { id: 'test', vendor: 'test', contextWindow: 1000, capabilities: [] },
      }, timeout);
    },
    lives() {
      const f = join(stateDir, 'lives');
      return existsSync(f) ? Number(readFileSync(f, 'utf8')) : 0;
    },
    ops(life = 1) {
      const f = join(stateDir, `ops-${life}.jsonl`);
      return existsSync(f) ? readFileSync(f, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l)) : [];
    },
    enginePid(life = 1) { return Number(readFileSync(join(stateDir, `pid-${life}`), 'utf8')); },
    /** Close the host side: the server sees the end of its stdin. */
    async close() {
      conn.close();
      try { child.stdin.end(); } catch { /* closed */ }
      const code = await Promise.race([exited, sleep(8_000).then(() => 'timeout')]);
      if (code === 'timeout') child.kill();
      return code;
    },
  };
  return h;
}

/** Poll until fn returns a truthy value, or fail after timeout. */
export async function waitFor(fn, timeout = 5_000, what = 'condition') {
  const end = Date.now() + timeout;
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
    await sleep(25);
  }
}

export function isAlive(pid) {
  try { process.kill(pid, 0); return true; } catch (e) { return e.code === 'EPERM'; }
}
