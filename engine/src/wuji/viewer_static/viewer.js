// Wuji, watching: the body's bones, its muscles at work along their paths, the weight in the feet, the breath.
// Data from the engine's viewer (viewer.py): /model.bin once, then /stream (Server-Sent Events). MuJoCo's world is z-up.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const REGION_NAMES = {
  feet: 'feet', calves: 'calves', thighs: 'thighs', hips: 'hips', lower_back: 'lower back', belly: 'belly',
  shoulders: 'shoulders', arms: 'arms', hands: 'hands',
};

// ---------- the scene ----------

const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x181b21, 4.5, 11); // the floor fades into the page before its edge

const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 60);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.6;
controls.maxDistance = 12;
controls.maxPolarAngle = Math.PI * 0.53;

scene.add(new THREE.HemisphereLight(0xe4e8ef, 0x2a2420, 0.85));
const key = new THREE.DirectionalLight(0xfff1de, 1.7);
key.position.set(2.2, -2.6, 3.6);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, { left: -1.2, right: 1.2, top: 1.2, bottom: -1.2, near: 0.5, far: 10 });
key.shadow.bias = -0.0004;
key.shadow.radius = 4;
scene.add(key, key.target);
const rim = new THREE.DirectionalLight(0x9fb6d6, 0.7);
rim.position.set(-2.5, 3.0, 2.2);
scene.add(rim);

// the floor: dark, a soft grid fading into the fog
const floor = new THREE.Mesh(new THREE.CircleGeometry(30, 64), new THREE.MeshStandardMaterial({ color: 0x1c1f25, roughness: 1 }));
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(20, 80, 0x5b5850, 0x3a3a38);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.0005;
grid.material.transparent = true;
grid.material.opacity = 0.35;
grid.material.depthWrite = false;
scene.add(grid);

// ---------- the model ----------

async function loadModel() {
  const buf = await (await fetch('model.bin')).arrayBuffer();
  const view = new DataView(buf);
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
  if (magic !== 'WUJI') throw new Error('model.bin: bad magic');
  const hlen = view.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hlen)));
  const base = 8 + hlen;
  const blob = (name, Type) => {
    const b = header.blobs[name];
    return new Type(buf, base + b.offset, b.shape.reduce((x, y) => x * y, 1));
  };
  return { header, vert: blob('vert', Float32Array), face: blob('face', Uint16Array) };
}

const boneMaterial = new THREE.MeshStandardMaterial({ color: 0xe9e0cb, roughness: 0.72, metalness: 0.0 });
let bodies = [];

function buildBones({ header, vert, face }) {
  const geoms = header.meshes.map((m) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(vert.slice(m.vert * 3, (m.vert + m.nvert) * 3), 3));
    g.setIndex(new THREE.BufferAttribute(face.slice(m.face * 3, (m.face + m.nface) * 3), 1));
    g.computeVertexNormals();
    return g;
  });
  bodies = Array.from({ length: header.nbody }, () => {
    const o = new THREE.Group();
    o.visible = false; // until the first frame places it
    scene.add(o);
    return o;
  });
  for (const g of header.geoms) {
    const mesh = new THREE.Mesh(geoms[g.mesh], boneMaterial);
    mesh.position.fromArray(g.pos);
    mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    bodies[g.body].add(mesh);
  }
}

// ---------- the muscles: thin tubes along each tendon's path, one instance per segment ----------

const muscle = { mesh: null, alpha: null, glow: null, ofTendon: null, radius: null, capacity: 0 };
const RAMP = [
  [0.0, new THREE.Color(0x7a6470), 0.2, 0.0],
  [0.35, new THREE.Color(0xc9824a), 0.62, 0.25],
  [0.7, new THREE.Color(0xe9572f), 0.9, 0.55],
  [1.0, new THREE.Color(0xff3220), 1.0, 0.9],
];

function ramp(s, color) {
  let i = 1;
  while (i < RAMP.length - 1 && s > RAMP[i][0]) i++;
  const [s0, c0, a0, g0] = RAMP[i - 1];
  const [s1, c1, a1, g1] = RAMP[i];
  const u = Math.min(1, Math.max(0, (s - s0) / (s1 - s0)));
  color.lerpColors(c0, c1, u);
  return [a0 + u * (a1 - a0), g0 + u * (g1 - g0)];
}

function buildMuscles(header) {
  const cap = header.wrap_capacity;
  const geo = new THREE.CylinderGeometry(1, 1, 1, 7, 1, true);
  const alpha = new THREE.InstancedBufferAttribute(new Float32Array(cap), 1);
  const glow = new THREE.InstancedBufferAttribute(new Float32Array(cap), 1);
  geo.setAttribute('instanceAlpha', alpha);
  geo.setAttribute('instanceGlow', glow);
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.55, transparent: true, depthWrite: false });
  mat.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float instanceAlpha;\nattribute float instanceGlow;\nvarying float vAlpha;\nvarying float vGlow;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvAlpha = instanceAlpha;\nvGlow = instanceGlow;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vAlpha;\nvarying float vGlow;')
      .replace('#include <color_fragment>', '#include <color_fragment>\ndiffuseColor.a *= vAlpha;')
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += diffuseColor.rgb * vGlow;');
  };
  const mesh = new THREE.InstancedMesh(geo, mat, cap);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.setColorAt(0, new THREE.Color());
  mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
  mesh.frustumCulled = false;
  mesh.count = 0;
  mesh.renderOrder = 2;
  scene.add(mesh);

  const ofTendon = new Int32Array(header.tendons.length).fill(-1);
  const fmax = Math.max(...header.muscles.map((m) => m.f0));
  const radius = new Float32Array(header.muscles.length);
  header.muscles.forEach((m, i) => {
    if (m.tendon >= 0) ofTendon[m.tendon] = i;
    radius[i] = 0.0016 + 0.0042 * Math.sqrt(m.f0 / fmax);
  });
  Object.assign(muscle, { mesh, alpha, glow, ofTendon, radius, capacity: cap });
}

const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _d = new THREE.Vector3(), _m = new THREE.Matrix4();
const _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _c = new THREE.Color();
const Y = new THREE.Vector3(0, 1, 0);

function updateMuscles(wrapnum, wrap, act) {
  const { mesh, alpha, glow, ofTendon, radius, capacity } = muscle;
  let n = 0, p = 0;
  for (let t = 0; t < wrapnum.length; t++) {
    const k = wrapnum[t];
    const mi = ofTendon[t];
    const a = mi >= 0 ? act[mi] : 0;
    const s = Math.pow(Math.min(1, a / 0.35), 0.7); // most muscles work at a few percent: lift the low end a little
    const [al, gl] = ramp(s, _c);
    const r = (mi >= 0 ? radius[mi] : 0.0015) * (1 + 0.5 * s);
    for (let i = 0; i + 1 < k && n < capacity; i++) {
      const j = 3 * (p + i);
      if (Number.isNaN(wrap[j]) || Number.isNaN(wrap[j + 3])) continue;
      _a.set(wrap[j], wrap[j + 1], wrap[j + 2]);
      _b.set(wrap[j + 3], wrap[j + 4], wrap[j + 5]);
      _d.subVectors(_b, _a);
      const len = _d.length();
      if (len < 1e-5) continue;
      _q.setFromUnitVectors(Y, _d.divideScalar(len));
      _s.set(r, len, r);
      _m.compose(_a.add(_b).multiplyScalar(0.5), _q, _s);
      mesh.setMatrixAt(n, _m);
      mesh.setColorAt(n, _c);
      alpha.array[n] = al;
      glow.array[n] = gl;
      n++;
    }
    p += k;
  }
  mesh.count = n;
  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.needsUpdate = true;
  alpha.needsUpdate = true;
  glow.needsUpdate = true;
}

// ---------- the feet: the centre of pressure under each, lit by its load ----------

function copMarker() {
  const g = new THREE.Group();
  const halo = new THREE.Mesh(new THREE.CircleGeometry(1, 40),
    new THREE.MeshBasicMaterial({ color: 0xf2b45e, transparent: true, opacity: 0.18, depthTest: false, depthWrite: false }));
  const dot = new THREE.Mesh(new THREE.CircleGeometry(1, 32),
    new THREE.MeshBasicMaterial({ color: 0xffe2b0, transparent: true, opacity: 0.9, depthTest: false, depthWrite: false, toneMapped: false }));
  halo.renderOrder = dot.renderOrder = 5;
  g.add(halo, dot);
  g.userData = { halo, dot };
  g.visible = false;
  scene.add(g);
  return g;
}
const cops = { left: copMarker(), right: copMarker() };

function updateFeet(feet) {
  for (const side of ['left', 'right']) {
    const f = feet?.[side];
    const g = cops[side];
    if (!f || !f.cop || f.share < 0.01) { g.visible = false; continue; }
    g.visible = true;
    g.position.set(f.cop[0], f.cop[1], 0.002);
    const { halo, dot } = g.userData;
    dot.scale.setScalar(0.009 + 0.018 * f.share);
    halo.scale.setScalar(0.03 + 0.07 * f.share);
    dot.material.opacity = 0.55 + 0.45 * f.share;
    halo.material.opacity = 0.1 + 0.25 * f.share;
  }
}

// ---------- the HUD ----------

let regionRows = {};
function buildRegions(names) {
  const box = $('regions');
  box.innerHTML = '';
  regionRows = {};
  for (const r of names) {
    const row = document.createElement('div');
    row.className = 'region';
    row.innerHTML = `<span class="label">${REGION_NAMES[r] ?? r}</span><div class="bar"><i></i><b title="held"></b></div><span class="num"></span>`;
    box.append(row);
    regionRows[r] = { bar: row.querySelector('i'), held: row.querySelector('b'), num: row.querySelector('.num') };
  }
}

const pct = (x) => `${Math.round(100 * x)}%`;
const fmt = (x, d = 1) => (x == null || Number.isNaN(x) ? '–' : Number(x).toFixed(d));
let frcL = 2.8, icL = 3.0;
let heartBpm = 0;

function updateHud(fr) {
  const mins = Math.floor(fr.t / 60), secs = Math.floor(fr.t % 60);
  $('status').textContent = `${fr.stance}${fr.moving ? ' · moving' : ''} · ${mins}:${String(secs).padStart(2, '0')}`;

  const wl = fr.weight?.left ?? 0.5, wr = fr.weight?.right ?? 0.5;
  $('w-left').textContent = pct(wl);
  $('w-right').textContent = pct(wr);
  $('w-bar-left').style.width = `${100 * wl}%`;

  const regions = fr.effort?.regions ?? {};
  for (const [r, row] of Object.entries(regionRows)) {
    const v = regions[r] ?? 0;
    row.bar.style.width = `${Math.min(100, 100 * Math.sqrt(Math.min(1, v / 0.5)))}%`; // same lift as the muscles' colour
    row.bar.style.opacity = String(0.35 + 0.65 * Math.min(1, v / 0.15));
    row.num.textContent = v < 0.005 ? '·' : pct(v);
    const held = fr.tension?.[r] ?? 0;
    row.held.style.display = held > 0 ? 'block' : 'none';
    row.held.style.left = `${100 * Math.sqrt(Math.min(1, held * 0.15 / 0.5))}%`;
  }

  const b = fr.breath;
  if (!b) { $('phase').textContent = 'no breath model'; return; }
  const phaseWords = { inhale: 'inhale', exhale: 'exhale', pause: 'pause', hold: 'holding' };
  $('phase').textContent = phaseWords[b.phase] ?? b.phase ?? '—';
  $('rate').textContent = fmt(b.rate_bpm, 1);
  $('tidal').textContent = fmt(b.tidal_l, 2);
  $('pattern').textContent = `${b.pattern}${b.paced ? ', paced' : ''}${b.mouth ? ', mouth open' : ', through the nose'}`;
  const frac = Math.min(1, Math.max(0, (b.lung_volume_l - (frcL - 0.2)) / (0.45 * icL + 0.2)));
  $('lung-fill').setAttribute('r', String(22 + 34 * frac));

  const diverge = (el, cm, range) => {
    const u = Math.max(-1, Math.min(1, cm / range)) * 50;
    el.style.left = `${u < 0 ? 50 + u : 50}%`;
    el.style.width = `${Math.abs(u)}%`;
  };
  diverge($('belly-bar'), b.belly_cm ?? 0, 4);
  diverge($('chest-bar'), b.chest_cm ?? 0, 2.5);
  $('belly').textContent = `${(b.belly_cm ?? 0) >= 0 ? '+' : ''}${fmt(b.belly_cm, 1)} cm`;
  $('chest').textContent = `${(b.chest_cm ?? 0) >= 0 ? '+' : ''}${fmt(b.chest_cm, 1)} cm`;

  for (const [side, id] of [['left', 'nl'], ['right', 'nr']]) {
    const n = b.nostrils?.[side];
    if (!n) continue;
    const bar = $(`${id}-bar`);
    bar.style.width = `${Math.min(100, 100 * Math.abs(n.flow_lps) / 0.4)}%`;
    bar.classList.toggle('warm', n.flow_lps < 0);
    $(id).textContent = `${fmt(n.temp_c, 1)} °C`;
  }
  $('co2').textContent = `${fmt(b.co2_mmhg, 1)} mmHg`;
  $('o2').textContent = `${fmt(b.o2_sat, 0)}%`;
  heartBpm = b.heart_bpm ?? 0;
  $('heart').textContent = `${fmt(heartBpm, 0)} bpm`;
  $('settled-bar').style.width = `${100 * (b.settled ?? 0)}%`;
  $('settled').textContent = pct(b.settled ?? 0);
}

// ---------- the stream ----------

function b64(s, Type) {
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

let latest = null, applied = null, lastFrameAt = 0, framed = false;
const target = new THREE.Vector3(0, 0, 0.95);

function applyFrame(fr) {
  const pose = b64(fr.bodies, Float32Array);
  for (let i = 0; i < bodies.length; i++) {
    const o = bodies[i], k = 7 * i;
    o.position.set(pose[k], pose[k + 1], pose[k + 2]);
    o.quaternion.set(pose[k + 4], pose[k + 5], pose[k + 6], pose[k + 3]);
    o.visible = true;
  }
  updateMuscles(b64(fr.wrapnum, Uint16Array), b64(fr.wrap, Float32Array), b64(fr.act, Float32Array));
  updateFeet(fr.feet);
  updateHud(fr);
  if (!framed) { // a 3/4 front view of the whole body: it faces -y, its left side is +x
    framed = true;
    const root = bodies[1].position;
    // far enough for the whole body (about 2.1 m tall with the floor, 1.1 m wide) at this aspect; on a narrow screen
    // the body sits higher, above the panels
    const narrow = camera.aspect < 0.8;
    const half = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const dist = Math.max(narrow ? 1.5 / half : 1.05 / half, 0.75 / (half * camera.aspect));
    target.set(root.x, root.y, narrow ? 0.45 : 0.9);
    const dir = new THREE.Vector3(1.75, -3.3, 0.55).normalize();
    camera.position.copy(target).addScaledVector(dir, dist);
    controls.target.copy(target);
    key.target.position.set(root.x, root.y, 0.8);
    controls.update();
    window.__READY__ = true; // the body is placed (for headless screenshots)
  }
}

function connect() {
  const es = new EventSource('stream');
  es.onmessage = (e) => { latest = JSON.parse(e.data); lastFrameAt = performance.now(); };
  es.onerror = () => { $('status').textContent = 'the body is out of reach; listening again…'; };
}

// ---------- the loop ----------

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

canvas.addEventListener('dblclick', () => { framed = false; if (latest) applyFrame(latest); });

let beatPhase = 0, lastTick = performance.now();
function tick(now) {
  const dt = Math.min(0.1, (now - lastTick) / 1000);
  lastTick = now;
  if (latest && latest !== applied) { applyFrame(latest); applied = latest; }
  if (lastFrameAt && now - lastFrameAt > 3000) $('status').textContent = 'waiting for the body…';
  if (heartBpm > 0) {
    beatPhase = (beatPhase + dt * heartBpm / 60) % 1;
    $('beat').style.opacity = String(0.25 + 0.75 * Math.exp(-beatPhase * 7));
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

window.wuji = { scene, camera, controls }; // for a look from the console

camera.position.set(1.75, -3.3, 1.45);
controls.target.copy(target);
loadModel().then((model) => {
  frcL = model.header.frc_l ?? frcL;
  icL = model.header.ic_l ?? icL;
  buildBones(model);
  buildMuscles(model.header);
  buildRegions(model.header.regions);
  connect();
  window.wujiReady = true;
}).catch((e) => {
  $('status').textContent = `could not load the body: ${e.message}`;
  console.error(e);
});
requestAnimationFrame(tick);
