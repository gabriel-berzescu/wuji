// Live check: the real chain, host -> MCPL server -> Python engine (uv) -> MuJoCo. Not part of `npm test` (it needs uv and
// takes half a minute): run it with `npm run live`. Prints what Claude would see.
import { fileURLToPath } from 'node:url';
import { FULL_GRANT, sleep, startServer, waitFor } from './helpers.mjs';

const engineDir = fileURLToPath(new URL('../../engine', import.meta.url));
const h = startServer({}, {
  WUJI_ENGINE: JSON.stringify(['uv', 'run', '--project', engineDir, 'python', '-m', 'wuji.engine']),
  WUJI_PUSH_MIN_INTERVAL_MS: '0',
});
const text = (r) => (r.content ?? []).map((c) => c.text).join('\n');
const show = (label, s) => console.log(`\n== ${label}\n${s}`);

try {
  const init = await h.initialize();
  show('initialize', `${init.serverInfo?.name} ${init.serverInfo?.version}; mcpl ${JSON.stringify(init.capabilities?.experimental?.mcpl?.version)}`);
  const receipt = await h.grant(FULL_GRANT);
  show('grant', JSON.stringify(receipt));
  const tools = await h.request('tools/list', {});
  show('tools', tools.tools.map((t) => t.name).join(', '));

  show('sense (first call waits for the engine to start)', text(await h.call('sense', {}, 300_000)));
  show('intend', text(await h.call('intend', { sink: 0.4, arms: 'embrace', over: 3 })));
  show('breathe', text(await h.call('breathe', { pattern: 'abdominal', inhale: 4, exhale: 6 })));
  await sleep(4500);
  const inj = await h.beforeInference(5000);
  show('beforeInference (injected before Claude thinks)', JSON.stringify(inj, null, 1).slice(0, 1500));
  show('form', text(await h.call('form', { name: 'wuji' })));
  await waitFor(() => h.pushes.length > 0, 20_000, 'a push event');
  await sleep(1000);
  show('push events', h.pushes.map((p) => `${p.origin?.event ?? ''} ${JSON.stringify(p.payload?.content?.[0]?.text ?? p.payload)}`).join('\n'));
} catch (e) {
  console.error('LIVE CHECK FAILED:', e);
  console.error(h.stderr.slice(-2000));
  process.exitCode = 1;
} finally {
  await h.close();
}
