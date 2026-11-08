import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { FULL_GRANT, isAlive, sleep, startServer, waitFor } from './helpers.mjs';

async function withServer(fakeEnv, serverEnv, fn) {
  const h = startServer(fakeEnv, serverEnv);
  try { await fn(h); } catch (e) {
    e.message += `\n--- server stderr ---\n${h.stderr}`;
    throw e;
  } finally { await h.close(); }
}

describe('initialize and grant', () => {
  it('advertises MCPL 0.5: push events, beforeUser injection without observe, valid feature sets', () => withServer({}, {}, async (h) => {
    const r = await h.initialize();
    assert.ok(r.capabilities.tools);
    const m = r.capabilities.experimental.mcpl;
    assert.equal(m.version, '0.5');
    assert.equal(m.pushEvents, true);
    assert.deepEqual(m.contextHooks, { beforeInference: { inject: { beforeUser: true } } });
    assert.deepEqual(Object.keys(m.featureSets).sort(), ['wuji.feltSense', 'wuji.signals']);
    assert.deepEqual(m.featureSets['wuji.feltSense'].uses, ['contextHooks.beforeInference.inject.beforeUser']);
    assert.deepEqual(m.featureSets['wuji.signals'].uses, ['pushEvents']);
    assert.equal(r.serverInfo.name, 'wuji');
  }));

  it('answers a plain MCP host without the MCPL block, and its tools still work', () => withServer({}, {}, async (h) => {
    const r = await h.initialize(false);
    assert.equal(r.capabilities.experimental, undefined);
    const s = await h.call('sense');
    assert.ok(!s.isError);
  }));

  it('returns a clean receipt for the full grant, and a degraded one naming what is missing', () => withServer({}, {}, async (h) => {
    await h.initialize();
    const full = await h.grant(FULL_GRANT);
    assert.equal(full.accepted, true);
    assert.equal(full.mode, undefined);
    assert.deepEqual(full.unavailableFeatures, []);

    const partial = await h.grant(['tools', 'pushEvents']);
    assert.equal(partial.mode, 'degraded');
    assert.deepEqual(partial.unavailableFeatures, [
      { featureSet: 'wuji.feltSense', missingCapabilities: ['contextHooks.beforeInference.inject.beforeUser'], effect: 'disabled' },
    ]);

    // §5.3: an absent effectiveCapabilities is a grant of nothing.
    const none = await h.grant(undefined);
    assert.equal(none.unavailableFeatures.length, 2);

    // `*` matches exactly one segment: contextHooks.* grants no depth-4 leaf.
    const wild = await h.grant(['*', 'contextHooks.*']);
    assert.deepEqual(wild.unavailableFeatures.map((u) => u.featureSet), ['wuji.feltSense']);

    // A present `enabled` is an allowlist.
    const selected = await h.grant(FULL_GRANT, { enabled: ['wuji.signals'] });
    assert.deepEqual(selected.unavailableFeatures.map((u) => u.featureSet), ['wuji.feltSense']);
    const disabled = await h.grant(FULL_GRANT, { disabled: ['wuji.*'] });
    assert.equal(disabled.unavailableFeatures.length, 2);
  }));

  it('fails closed on a malformed policy', () => withServer({}, {}, async (h) => {
    await h.initialize();
    await h.grant(FULL_GRANT);
    await assert.rejects(h.grant(['pushEvents'], { deniedCapabilities: ['pushEvents'] }), (e) => e.code === -32602);
    await assert.rejects(h.grant('pushEvents'), (e) => e.code === -32602);
    const r = await h.beforeInference();
    assert.deepEqual(r.contextInjections, [], 'nothing granted after a malformed policy');
  }));

  it('answers unknown methods with an error rather than silence', () => withServer({}, {}, async (h) => {
    await h.initialize();
    await assert.rejects(h.request('state/get', {}), (e) => e.code === -32601);
  }));
});

describe('tools', () => {
  it('lists the seven tools with schemas mirroring the protocol', () => withServer({}, {}, async (h) => {
    await h.initialize();
    const { tools } = await h.request('tools/list', {});
    assert.deepEqual(tools.map((t) => t.name), ['sense', 'intend', 'tension', 'breathe', 'form', 'reset', 'viewer']);
    const intend = tools.find((t) => t.name === 'intend');
    assert.deepEqual(intend.inputSchema.properties.stance.enum, ['wuji', 'horse']);
    assert.equal(intend.inputSchema.properties.weight.maximum, 1);
    const tension = tools.find((t) => t.name === 'tension');
    assert.ok(tension.inputSchema.properties.region.enum.includes('all'));
    assert.deepEqual(tension.inputSchema.required, ['region', 'level']);
    for (const t of tools) assert.ok(t.description.length > 30, `${t.name} has a description`);
  }));

  it('round-trips each tool: engine text first, then compact JSON', () => withServer({}, {}, async (h) => {
    await h.initialize();
    await h.grant(FULL_GRANT);
    const s = await h.call('sense', { detail: 'full' });
    assert.ok(!s.isError);
    assert.match(s.content[0].text, /^Standing in wuji/);
    assert.match(s.content[0].text, /Knees 12°/);
    const json = JSON.parse(s.content[1].text);
    assert.equal(json.text, undefined, 'text is not repeated in the JSON');
    assert.equal(json.body.stance, 'wuji');
    assert.ok(!s.content[1].text.includes('\n'), 'compact JSON');

    const i = await h.call('intend', { stance: 'horse', sink: 0.4, over: 3 });
    assert.equal(i.content[0].text, 'Moving, smoothly.');
    assert.deepEqual(JSON.parse(i.content[1].text), { accepted: true, arrives_in: 3 });
    assert.match((await h.call('tension', { region: 'shoulders', level: 0 })).content[0].text, /shoulders/);
    assert.match((await h.call('breathe', { pattern: 'abdominal', inhale: 4, exhale: 6 })).content[0].text, /abdominal/);
    assert.match((await h.call('form', { name: 'lift_sky' })).content[0].text, /lift_sky/);
    assert.match((await h.call('reset')).content[0].text, /wuji/);
    const v = await h.call('viewer');
    assert.match(v.content[0].text, /^The viewer is open at http:\/\/127\.0\.0\.1:7341\//);
    assert.deepEqual(JSON.parse(v.content[1].text), { url: 'http://127.0.0.1:7341/' });

    const ops = h.ops().map((o) => o.op);
    assert.deepEqual(ops, ['hello', 'sense', 'intend', 'tension', 'breathe', 'form', 'reset', 'viewer']);
  }));

  it('reports engine refusals and unknown tools as tool errors', () => withServer({}, {}, async (h) => {
    await h.initialize();
    const bad = await h.call('intend', { stance: 'crane', wings: 2 });
    assert.equal(bad.isError, true);
    assert.match(bad.content[0].text, /unknown stance 'crane'/);
    assert.match(bad.content[0].text, /Ignored: wings/);
    const nope = await h.call('fly');
    assert.equal(nope.isError, true);
    assert.match(nope.content[0].text, /No tool named "fly"/);
  }));
});

describe('felt sense on beforeInference', () => {
  it('injects the ambient sense (what changed) at beforeUser, tagged [wuji], when granted', () => withServer({}, {}, async (h) => {
    await h.initialize();
    await h.grant(FULL_GRANT);
    await h.call('sense'); // the engine is up
    const r = await h.beforeInference();
    assert.equal(r.featureSet, 'wuji.feltSense');
    assert.equal(r.contextInjections.length, 1);
    const inj = r.contextInjections[0];
    assert.equal(inj.position, 'beforeUser');
    assert.equal(inj.namespace, 'wuji');
    assert.match(inj.content[0].text, /^\[wuji\] Wuji stance as before; nothing new\./);
    assert.ok(h.ops().some((o) => o.op === 'sense' && o.args.detail === 'ambient'));
  }));

  it('injects the brief sense every turn with WUJI_FELT_SENSE=brief', () => withServer({}, { WUJI_FELT_SENSE: 'brief' }, async (h) => {
    await h.initialize();
    await h.grant(FULL_GRANT);
    await h.call('sense');
    const r = await h.beforeInference();
    assert.match(r.contextInjections[0].content[0].text, /^\[wuji\] Standing in wuji/);
    assert.ok(h.ops().some((o) => o.op === 'sense' && o.args.detail === 'brief'));
  }));

  it('injects nothing without the grant, or before any policy', () => withServer({}, {}, async (h) => {
    await h.initialize();
    await h.call('sense');
    assert.deepEqual((await h.beforeInference()).contextInjections, [], 'before policy');
    await h.grant(['tools', 'pushEvents']);
    assert.deepEqual((await h.beforeInference()).contextInjections, [], 'not granted');
    const senses = h.ops().filter((o) => o.op === 'sense').length;
    assert.equal(senses, 1, 'the engine is not even asked');
  }));

  it('fails open on a slow engine: answers within the budget with no injection', () => withServer({ FAKE_SENSE_DELAY_MS: '3000' }, { WUJI_SENSE_BUDGET_MS: '300' }, async (h) => {
    await h.initialize();
    await h.grant(FULL_GRANT);
    await h.call('reset'); // the engine is up
    const t0 = Date.now();
    const r = await h.beforeInference();
    const took = Date.now() - t0;
    assert.deepEqual(r.contextInjections, []);
    assert.ok(took < 1_500, `answered in ${took} ms`);
    // and the connection stays usable while the slow sense is still out
    assert.ok(!(await h.call('reset')).isError);
  }));

  it('fails open when the engine is down', () => withServer({}, { WUJI_ENGINE: JSON.stringify(['wuji-no-such-program-here']), WUJI_TOOL_TIMEOUT_MS: '500' }, async (h) => {
    await h.initialize();
    await h.grant(FULL_GRANT);
    const t0 = Date.now();
    assert.deepEqual((await h.beforeInference()).contextInjections, []);
    assert.ok(Date.now() - t0 < 1_000);
    // the tool waits for the engine up to its timeout, then says why
    const s = await h.call('sense');
    assert.equal(s.isError, true);
    assert.match(s.content[0].text, /sense did not reach the body: .*WUJI_ENGINE/s);
  }));
});

describe('push events', () => {
  it('pushes engine events as push/event with featureSet, stable eventId, origin, tags and the text', () => withServer({ FAKE_EVENT_MS: '60' }, {}, async (h) => {
    await h.initialize();
    await h.grant(FULL_GRANT);
    await waitFor(() => h.pushes.length >= 4, 5_000, 'four pushes');
    const p = h.pushes[0];
    assert.equal(p.featureSet, 'wuji.signals');
    assert.match(p.eventId, /^wuji-[0-9a-f]{8}-balance-/);
    assert.ok(!Number.isNaN(Date.parse(p.timestamp)));
    assert.equal(p.origin.server, 'wuji');
    assert.equal(p.origin.event, 'balance');
    assert.equal(p.origin.severity, 'warning');
    assert.deepEqual(p.tags, ['wuji:balance', 'wuji:warning']);
    assert.deepEqual(p.payload.content, [{ type: 'text', text: 'Weight is on the outer edge of the left foot (1).' }]);
    const kinds = new Set(h.pushes.slice(0, 4).map((x) => x.origin.event));
    assert.deepEqual([...kinds].sort(), ['balance', 'breath', 'form', 'strain']);
    const ids = h.pushes.map((x) => x.eventId);
    assert.equal(new Set(ids).size, ids.length, 'eventIds are unique');
  }));

  it('pushes nothing before the policy or without pushEvents, and stops when it is withdrawn', () => withServer({ FAKE_EVENT_MS: '50' }, {}, async (h) => {
    await h.initialize();
    await sleep(300);
    assert.equal(h.pushes.length, 0, 'before policy');
    await h.grant(['tools', 'contextHooks.beforeInference.inject.beforeUser']);
    await sleep(300);
    assert.equal(h.pushes.length, 0, 'not granted');
    await h.grant(FULL_GRANT);
    await waitFor(() => h.pushes.length >= 1, 3_000, 'a push once granted');
    // A Notification can only narrow (§6.7).
    h.conn.sendNotification('featureSets/update', { disabled: ['wuji.signals'] });
    await sleep(150);
    const n = h.pushes.length;
    await sleep(400);
    assert.ok(h.pushes.length <= n + 1, 'stopped after the narrowing');
  }));

  it('a Notification cannot widen the grant', () => withServer({ FAKE_EVENT_MS: '50' }, {}, async (h) => {
    await h.initialize();
    await h.grant(['tools']);
    h.conn.sendNotification('featureSets/update', { effectiveCapabilities: FULL_GRANT });
    await sleep(500);
    assert.equal(h.pushes.length, 0);
  }));

  it('rate-limits: one per kind per interval, the latest kept', () => withServer({ FAKE_EVENT_MS: '40', FAKE_EVENT_KINDS: 'balance' }, { WUJI_PUSH_MIN_INTERVAL_MS: '500' }, async (h) => {
    await h.initialize();
    await h.grant(FULL_GRANT);
    await sleep(1_300);
    const n = h.pushes.length;
    assert.ok(n >= 2 && n <= 4, `${n} pushes in 1.3 s at one per 500 ms`);
    // the trailing push is a later event than the one before it, not the first one held
    if (n >= 2) assert.notEqual(h.pushes[0].payload.content[0].text, h.pushes[1].payload.content[0].text);
  }));
});

describe('engine lifecycle', () => {
  it('restarts the engine when it dies, and the tools come back', () => withServer({ FAKE_DIE_ON: 'reset' }, {}, async (h) => {
    await h.initialize();
    await h.grant(FULL_GRANT);
    assert.equal(JSON.parse((await h.call('sense')).content[1].text).life, 1);
    const died = await h.call('reset');
    assert.equal(died.isError, true);
    assert.match(died.content[0].text, /did not reach the body: the engine exited/);
    const s = await h.call('sense'); // waits for the restart
    assert.ok(!s.isError, s.content[0].text);
    assert.equal(JSON.parse(s.content[1].text).life, 2);
    assert.equal(h.ops(2)[0].op, 'hello');
  }));

  it('stops the engine (shutdown) when the host goes away', async () => {
    const h = startServer();
    await h.initialize();
    await h.call('sense');
    const pid = h.enginePid(1);
    assert.ok(isAlive(pid));
    const code = await h.close();
    assert.equal(code, 0);
    await waitFor(() => !isAlive(pid), 3_000, 'engine exit');
    assert.equal(h.ops(1).at(-1).op, 'shutdown');
  });
});
