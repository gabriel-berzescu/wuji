import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { Engine, EngineError } from '../dist/engine.js';
import { engineCommand } from '../dist/config.js';
import { FAKE_ENGINE, isAlive, sleep, tempDir, waitFor } from './helpers.mjs';
import { readFileSync } from 'node:fs';

function fake(env = {}, opts = {}) {
  const stateDir = tempDir();
  const logs = [];
  const engine = new Engine({
    command: [process.execPath, FAKE_ENGINE],
    env: { ...process.env, FAKE_STATE_DIR: stateDir, ...env },
    log: (...a) => logs.push(a.join(' ')),
    restartMinMs: 50,
    stopGraceMs: 2_000,
    ...opts,
  });
  return { engine, stateDir, logs };
}

describe('engine command', () => {
  it('defaults to uv run on the repo engine, as one argv (no shell)', () => {
    const argv = engineCommand({}, join('/some path', 'wuji'));
    assert.deepEqual(argv, ['uv', 'run', '--project', join('/some path', 'wuji', 'engine'), 'python', '-m', 'wuji.engine']);
  });
  it('takes WUJI_ENGINE as a JSON array', () => {
    assert.deepEqual(engineCommand({ WUJI_ENGINE: '["C:\\\\Program Files\\\\py.exe", "-m", "wuji.engine"]' }), ['C:\\Program Files\\py.exe', '-m', 'wuji.engine']);
  });
  it('rejects a WUJI_ENGINE that is not a non-empty array of strings', () => {
    assert.throws(() => engineCommand({ WUJI_ENGINE: 'python -m wuji.engine' }), /JSON array/);
    assert.throws(() => engineCommand({ WUJI_ENGINE: '[]' }), /non-empty/);
    assert.throws(() => engineCommand({ WUJI_ENGINE: '["python", 3]' }), /non-empty/);
  });
});

describe('engine process', () => {
  it('sends hello first, matches responses by id out of order, tolerates noise and CRLF', async () => {
    const { engine, stateDir } = fake({ FAKE_SENSE_DELAY_MS: '300', FAKE_NOISE: '1' });
    engine.start();
    try {
      const sense = engine.request('sense', { detail: 'brief' });
      const intend = engine.request('intend', { stance: 'horse' });
      const order = [];
      sense.then(() => order.push('sense'));
      intend.then(() => order.push('intend'));
      const [s, i] = await Promise.all([sense, intend]);
      assert.deepEqual(order, ['intend', 'sense']);
      assert.match(s.text, /Standing in/);
      assert.equal(i.accepted, true);
      assert.equal(engine.state, 'ready');
      assert.equal(engine.hello.muscles, 416);
      const ops = readFileSync(join(stateDir, 'ops-1.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l).op);
      assert.equal(ops[0], 'hello');
    } finally { await engine.stop(); }
  });

  it('turns ok:false into a refused EngineError carrying the engine message', async () => {
    const { engine } = fake();
    engine.start();
    try {
      await assert.rejects(engine.request('intend', { stance: 'crane' }), (e) => e instanceof EngineError && e.kind === 'refused' && /unknown stance 'crane'/.test(e.message));
    } finally { await engine.stop(); }
  });

  it('forwards events with the run they came from', async () => {
    const { engine } = fake({ FAKE_EVENT_MS: '50' });
    const events = [];
    engine.on('event', (e) => events.push(e));
    engine.start();
    try {
      await waitFor(() => events.length >= 2, 3_000, 'events');
      assert.equal(events[0].event, 'balance');
      assert.equal(events[0].data.severity, 'warning');
      assert.equal(events[0].run, engine.run);
      assert.equal(typeof events[0].t, 'number');
    } finally { await engine.stop(); }
  });

  it('restarts a dead engine, failing the requests it held, and says hello again', async () => {
    const { engine, stateDir } = fake({ FAKE_DIE_ON: 'reset' });
    engine.start();
    try {
      await engine.request('sense');
      const firstRun = engine.run;
      await assert.rejects(engine.request('reset'), (e) => e instanceof EngineError && e.kind === 'unavailable' && /exited/.test(e.message));
      const s = await engine.request('sense', {}, { timeoutMs: 5_000 }); // waits for the restart
      assert.equal(s.life, 2);
      assert.notEqual(engine.run, firstRun);
      assert.equal(engine.starts, 2);
      const ops = readFileSync(join(stateDir, 'ops-2.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l).op);
      assert.equal(ops[0], 'hello');
    } finally { await engine.stop(); }
  });

  it('stops with shutdown and the process is gone', async () => {
    const { engine, stateDir } = fake();
    engine.start();
    await engine.request('sense');
    const pid = Number(readFileSync(join(stateDir, 'pid-1'), 'utf8'));
    assert.ok(isAlive(pid));
    await engine.stop();
    assert.equal(engine.state, 'stopped');
    await waitFor(() => !isAlive(pid), 3_000, 'engine exit');
    const ops = readFileSync(join(stateDir, 'ops-1.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l).op);
    assert.equal(ops.at(-1), 'shutdown');
    await assert.rejects(engine.request('sense'), (e) => e.kind === 'unavailable');
  });

  it('reports a missing program clearly and keeps trying', async () => {
    const logs = [];
    const engine = new Engine({ command: ['wuji-no-such-program-here'], log: (...a) => logs.push(a.join(' ')), restartMinMs: 50 });
    engine.start();
    try {
      await waitFor(() => engine.starts >= 2, 3_000, 'a retry');
      assert.match(engine.lastError, /ENOENT|not found|cannot find/i);
      assert.match(engine.lastError, /WUJI_ENGINE/);
      await assert.rejects(engine.request('sense', {}, { timeoutMs: 100 }), (e) => e.kind === 'unavailable');
    } finally { await engine.stop(); }
    const starts = engine.starts;
    await sleep(200);
    assert.equal(engine.starts, starts, 'no restarts after stop');
  });
});
