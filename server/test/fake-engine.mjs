#!/usr/bin/env node
// A stand-in for the Python engine: speaks docs/engine-protocol.md with canned answers.
//
// Behaviour, from the environment:
//   FAKE_STATE_DIR       a directory for: lives (count of starts), pid-<life>, ops-<life>.jsonl (every request)
//   FAKE_SENSE_DELAY_MS  delay every sense response (default 0)
//   FAKE_EVENT_MS        emit an event every N ms (default: none)
//   FAKE_EVENT_KINDS     comma list of event kinds to cycle (default balance,breath,form,strain)
//   FAKE_DIE_ON          an op that makes the FIRST life exit with code 3 instead of answering
//   FAKE_NOISE           write a non-JSON line and a CRLF-terminated response first, to test parsing

import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const env = process.env;
const dir = env.FAKE_STATE_DIR;
let life = 1;
if (dir) {
  mkdirSync(dir, { recursive: true });
  const f = join(dir, 'lives');
  life = (existsSync(f) ? Number(readFileSync(f, 'utf8')) : 0) + 1;
  writeFileSync(f, String(life));
  writeFileSync(join(dir, `pid-${life}`), String(process.pid));
}
const started = Date.now();
const t = () => Math.round((Date.now() - started) / 100) / 10 + 0.8;
const out = (msg, eol = '\n') => process.stdout.write(JSON.stringify(msg) + eol);
const log = (s) => process.stderr.write(`fake-engine[${life}]: ${s}\n`);

const state = { stance: 'wuji', weight: 0.5, sink: 0.2, arms: 'hang', head: 'suspended', breath: 'natural' };
const ENUMS = {
  stance: ['wuji', 'horse'],
  arms: ['hang', 'embrace', 'lift_sky', 'press_down', 'bow_extend', 'bow_pull', 'punch', 'fist_waist', 'earth_press', 'owl', 'hands_thighs'],
  head: ['suspended', 'tucked', 'turned_left', 'turned_right'],
};

function senseResult(detail) {
  const lines = [
    `Standing in ${state.stance}, knees soft. Weight ${Math.round(state.weight * 100)}% left, through the middle of both feet.`,
    'Spine long, crown lifted.',
    'Effort sits lightly in the calves.',
    `Breath ${state.breath}, slow, through the nose.`,
    'Heart 64.',
  ];
  if (detail === 'full') lines.push('Knees 12°, hips 6°.');
  if (detail === 'ambient') return { t: 1, text: 'Wuji stance as before; nothing new.', changed: [] };
  return {
    t: t(),
    life,
    text: lines.join('\n'),
    body: { stance: state.stance, weight: { left: state.weight, right: 1 - state.weight }, sway_mm: 3.1, feasible: true },
    breath: { pattern: state.breath },
    heart: { bpm: 64.0 },
  };
}

function handle(op, args) {
  switch (op) {
    case 'hello': return { version: '0.0.1', model: 'fake', muscles: 416, t: t(), life };
    case 'sense': return senseResult(args.detail ?? 'brief');
    case 'intend': {
      for (const k of ['stance', 'arms', 'head']) {
        if (k in args && !ENUMS[k].includes(args[k])) throw new Error(`unknown ${k} '${args[k]}'`);
      }
      Object.assign(state, args);
      return { accepted: true, arrives_in: args.over ?? 1, text: 'Moving, smoothly.' };
    }
    case 'tension': return { accepted: true, text: `Tension in the ${args.region} set to ${args.level}.` };
    case 'breathe': state.breath = args.pattern ?? state.breath; return { accepted: true, text: `Breathing ${state.breath}.` };
    case 'form': return { accepted: true, steps: 4, text: `Beginning ${args.name}.` };
    case 'reset': Object.assign(state, { stance: 'wuji', weight: 0.5, breath: 'natural' }); return { accepted: true, text: 'Back to wuji.' };
    case 'viewer': return { url: 'http://127.0.0.1:7341/', text: 'The viewer is open at http://127.0.0.1:7341/.' };
    default: throw new Error(`unknown op '${op}'`);
  }
}

if (env.FAKE_NOISE) process.stdout.write('this is not json\n');

let buf = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buf += chunk;
  let nl;
  while ((nl = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, nl);
    buf = buf.slice(nl + 1);
    if (line.trim()) onRequest(JSON.parse(line));
  }
});
process.stdin.on('end', () => { log('stdin ended; exiting'); process.exit(0); });

function onRequest({ id, op, args = {} }) {
  if (dir) appendFileSync(join(dir, `ops-${life}.jsonl`), JSON.stringify({ op, args }) + '\n');
  if (op === 'shutdown') { out({ id, ok: true, result: { ok: true } }); log('shutdown'); setTimeout(() => process.exit(0), 10); return; }
  if (env.FAKE_DIE_ON === op && life === 1) { log(`dying on ${op}`); process.exit(3); }
  const reply = () => {
    try { out({ id, ok: true, result: handle(op, args) }, env.FAKE_NOISE ? '\r\n' : '\n'); }
    catch (e) { out({ id, ok: false, error: e.message }); }
  };
  const delay = op === 'sense' ? Number(env.FAKE_SENSE_DELAY_MS ?? 0) : 0;
  if (delay > 0) setTimeout(reply, delay); else reply();
}

if (env.FAKE_EVENT_MS) {
  const kinds = (env.FAKE_EVENT_KINDS ?? 'balance,breath,form,strain').split(',');
  let i = 0;
  setInterval(() => {
    const kind = kinds[i % kinds.length];
    i += 1;
    const data = {
      balance: { severity: 'warning', text: `Weight is on the outer edge of the left foot (${i}).` },
      breath: { kind: 'urge', urge: 0.7, text: 'The urge to breathe is rising.' },
      form: { name: 'lift_sky', step: i, of: 4, done: false, text: `Lift the sky, step ${i}.` },
      strain: { text: 'The thighs are working hard to hold this.' },
    }[kind];
    out({ event: kind, t: t() + i / 1000, data });
  }, Number(env.FAKE_EVENT_MS));
}
log('ready');
