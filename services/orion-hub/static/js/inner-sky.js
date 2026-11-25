// Inner Sky renderer. Data: /api/inner-sky/events + /api/inner-sky/state (scripts/inner_sky_routes.py).
// Pure logic lives in inner-sky-core.js (window.OrionInnerSkyCore). Every drawn light maps to one row;
// nothing here invents a value -- sizes are counts, pulse rates are measured count deltas.
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/+esm';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js/+esm';
import { EffectComposer } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/postprocessing/EffectComposer.js/+esm';
import { RenderPass } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/postprocessing/RenderPass.js/+esm';
import { UnrealBloomPass } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/postprocessing/UnrealBloomPass.js/+esm';
import { OutputPass } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/postprocessing/OutputPass.js/+esm';

const core = window.OrionInnerSkyCore;
const EMBED = document.body.classList.contains('embed');
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const WINDOW_SEC = 12 * 3600, LIVE_SEC = 3600;
const EVENTS_POLL_MS = 5000, STATE_POLL_MS = 20000;

const COL = {camera:'#3fa9ff',body:'#ff8a5c',prediction:'#5cf2c4',metacognition:'#c9d3ff',reverie:'#b48cff',dream:'#7a8cff',memory:'#f2a65a',reading:'#9be564',curiosity:'#ffd166',painting:'#ff7ac6',play:'#ff5d73',chat:'#7cf0ff',prime:'#ffe9a8',intention:'#a0ffea',personality:'#ff9ee5',day_letter:'#d9c49a',unregistered:'#888888'};
const colorOf = (o) => COL[o] || '#9aa3c7';
const nice = (s) => String(s).replace(/_/g, ' ');
const hhmm = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function ago(sec) { if (sec < 120) return `${Math.round(sec)} s`; if (sec < 7200) return `${Math.round(sec / 60)} min`; if (sec < 172800) return `${Math.round(sec / 3600)} h`; return `${Math.round(sec / 86400)} days`; }

// ---------------------------------------------------------------- clock (server time)
let clockOffset = 0; // server now - client now
const serverNow = () => Date.now() / 1000 + clockOffset;

// ---------------------------------------------------------------- renderer
const canvas = document.getElementById('sky');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05060d);
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 500);
camera.position.set(0, EMBED ? 18 : 15, EMBED ? 42 : 34);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.autoRotate = !REDUCED; controls.autoRotateSpeed = 0.35;
controls.minDistance = 12; controls.maxDistance = 80; controls.enablePan = false;
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 1.15, 0.75, 0.0));
composer.addPass(new OutputPass());
const uScale = { value: 1 };
const updateScale = () => { uScale.value = innerHeight * renderer.getPixelRatio() / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))); };
updateScale();

const pointMat = () => new THREE.ShaderMaterial({
  uniforms: { uScale },
  vertexShader: `attribute float size; attribute float alpha; attribute vec3 aColor;
    uniform float uScale; varying vec3 vC; varying float vA;
    void main(){ vC=aColor; vA=alpha; vec4 mv=modelViewMatrix*vec4(position,1.0);
      gl_PointSize=max(1.0, size*uScale/-mv.z); gl_Position=projectionMatrix*mv; }`,
  fragmentShader: `varying vec3 vC; varying float vA;
    void main(){ float d=length(gl_PointCoord-0.5); float a=smoothstep(0.5,0.0,d); a*=a;
      gl_FragColor=vec4(vC*a*vA, a*vA); }`,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
});
class Pool {
  constructor(n) {
    this.n = n; const g = new THREE.BufferGeometry();
    this.pos = new Float32Array(n * 3); this.col = new Float32Array(n * 3);
    this.size = new Float32Array(n); this.alpha = new Float32Array(n);
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    g.setAttribute('aColor', new THREE.BufferAttribute(this.col, 3));
    g.setAttribute('size', new THREE.BufferAttribute(this.size, 1));
    g.setAttribute('alpha', new THREE.BufferAttribute(this.alpha, 1));
    this.g = g; this.points = new THREE.Points(g, pointMat()); this.points.frustumCulled = false; scene.add(this.points);
  }
  set(i, p, c, s, a) { this.pos[i*3]=p.x; this.pos[i*3+1]=p.y; this.pos[i*3+2]=p.z; this.col[i*3]=c.r; this.col[i*3+1]=c.g; this.col[i*3+2]=c.b; this.size[i]=s; this.alpha[i]=a; }
  hide(i) { this.alpha[i] = 0; }
  flush() { for (const k of ['position', 'aColor', 'size', 'alpha']) this.g.attributes[k].needsUpdate = true; }
}

// ---------------------------------------------------------------- services shell (bus synaptic graph)
const R_SHELL = 24, MAX_SVC = 400, SEG = 20;
const services = new Map(); // id -> {id, n, last, pos, rate, phase}
let serviceRatesPrev = null, linkRatesPrev = null, stateNow = 0;
function hash01(s, salt) { let h = 2166136261 ^ salt; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return ((h >>> 0) % 100000) / 100000; }
function shellPos(id) { // stable spot on the shell from the service's own name
  const y = hash01(id, 7) * 1.6 - 0.8, th = hash01(id, 13) * Math.PI * 2, r = Math.sqrt(1 - y * y);
  return new THREE.Vector3(Math.cos(th) * r, y, Math.sin(th) * r).multiplyScalar(R_SHELL);
}
const svcPool = new Pool(MAX_SVC);
const C_LIVE = new THREE.Color('#d6e6ff'), C_DEAD = new THREE.Color('#3a3e52'), C_PULSE = new THREE.Color('#9cc8ff');
let links = []; let linkLines = null;
function rebuildLinks(raw, rates) { // rates: measured msgs/s by link key (missing = not yet measured)
  const keep = new Map(links.map((l) => [l.key, l]));
  links = raw.filter((l) => services.has(l.a) && services.has(l.b)).map((l) => {
    const key = `${l.a}>${l.b}`, old = keep.get(key);
    if (old) { old.n = l.n; old.last = l.last; old.rate = key in rates ? rates[key] : null; return old; }
    const a = services.get(l.a).pos, b = services.get(l.b).pos, curve = [];
    for (let s = 0; s <= SEG; s++) { const t = s / SEG; curve.push(a.clone().normalize().lerp(b.clone().normalize(), t).normalize().multiplyScalar(R_SHELL * (1 + 0.14 * Math.sin(Math.PI * t)))); }
    return { key, a: l.a, b: l.b, n: l.n, last: l.last, curve, rate: key in rates ? rates[key] : null, acc: Math.random() };
  });
  if (linkLines) { scene.remove(linkLines); linkLines.geometry.dispose(); }
  const pos = [], col = [];
  for (const l of links) {
    const w = l.rate == null ? 0.25 : Math.min(1, 0.25 + 0.15 * Math.log2(1 + l.rate));
    const c = new THREE.Color('#345a8c').multiplyScalar(w);
    for (let s = 0; s < SEG; s++) { pos.push(...l.curve[s].toArray(), ...l.curve[s + 1].toArray()); col.push(c.r, c.g, c.b, c.r, c.g, c.b); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  linkLines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false }));
  scene.add(linkLines);
}
const pulsePool = new Pool(700); const pulses = [];

// ---------------------------------------------------------------- experience organs (inner ring)
const R_RING = 9.5; let organs = []; const organBy = new Map(); let organPool = null; let labels = []; let ringLine = null;
function buildOrgans(names) { // (re)builds the ring when the set of organs changes; counts are restored by recount()
  if (organs.length === names.length && names.every((n, i) => organs[i].name === n)) return;
  if (organPool) { scene.remove(organPool.points); organPool.g.dispose(); }
  if (ringLine) { scene.remove(ringLine); ringLine.geometry.dispose(); }
  for (const el of labels) el.remove();
  organBy.clear();
  organs = names.map((name, i) => {
    const th = i / names.length * Math.PI * 2;
    return { name, pos: new THREE.Vector3(Math.cos(th) * R_RING, Math.sin(th * 2) * 0.6, Math.sin(th) * R_RING), color: new THREE.Color(colorOf(name)), recent: [], flash: 0, total: 0, last: '' };
  });
  for (const o of organs) organBy.set(o.name, o);
  organPool = new Pool(organs.length * 2);
  const ring = []; for (let i = 0; i <= 128; i++) { const th = i / 128 * Math.PI * 2; ring.push(new THREE.Vector3(Math.cos(th) * R_RING, Math.sin(th * 2) * 0.6, Math.sin(th) * R_RING)); }
  ringLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(ring), new THREE.LineBasicMaterial({ color: 0x2a3160, transparent: true, opacity: 0.6 }));
  scene.add(ringLine);
  labels = organs.map((o) => { const el = document.createElement('div'); el.className = 'label'; el.textContent = nice(o.name); el.style.color = colorOf(o.name); document.body.appendChild(el); return el; });
}

// ---------------------------------------------------------------- core, comets, binding arcs
const corePool = new Pool(3); let coreBoost = 0;
const C_CORE = new THREE.Color('#fff1c4'), C_HALO = new THREE.Color('#ffd98a');
const TRAIL = 8, MAXC = 450; const cometPool = new Pool(MAXC * TRAIL); const comets = [];
const freeComet = Array.from({ length: MAXC }, (_, i) => MAXC - 1 - i);
function spawnComet(e) {
  const o = organBy.get(e.organ); if (!o || !freeComet.length) return; // at the cap a comet is not drawn; it is still counted
  const lift = new THREE.Vector3((Math.random() - 0.5) * 3, 2 + Math.random() * 3, (Math.random() - 0.5) * 3);
  const sal = typeof e.salience === 'number' ? e.salience : 0;
  comets.push({ slot: freeComet.pop(), p0: o.pos.clone(), p1: o.pos.clone().multiplyScalar(0.5).add(lift), c: o.color, size: 0.55 * (1 + sal), age: 0, life: 2.4 + Math.random() * 0.6 });
}
const bez = (p0, p1, t, out) => out.set((1-t)*(1-t)*p0.x + 2*(1-t)*t*p1.x, (1-t)*(1-t)*p0.y + 2*(1-t)*t*p1.y, (1-t)*(1-t)*p0.z + 2*(1-t)*t*p1.z);
const ARC = 70, MAXA = 8; const arcPool = new Pool(MAXA * ARC); const arcs = [];
const freeArc = Array.from({ length: MAXA }, (_, i) => MAXA - 1 - i);
const C_GOLD = new THREE.Color('#ffc44d'), C_GOLDHI = new THREE.Color('#fff1c4');
function spawnArc(c) {
  const a = organBy.get(c.from), b = organBy.get(c.by); if (!a || !b || !freeArc.length) return;
  const mid = a.pos.clone().add(b.pos).multiplyScalar(0.5).add(new THREE.Vector3(0, 7, 0)); const pts = [];
  for (let i = 0; i < ARC; i++) { const t = i / (ARC - 1); pts.push(new THREE.Vector3((1-t)*(1-t)*a.pos.x + 2*(1-t)*t*mid.x + t*t*b.pos.x, (1-t)*(1-t)*a.pos.y + 2*(1-t)*t*mid.y + t*t*b.pos.y, (1-t)*(1-t)*a.pos.z + 2*(1-t)*t*mid.z + t*t*b.pos.z)); }
  arcs.push({ slot: freeArc.pop(), pts, age: 0, life: 14 });
  a.flash = 2.5; b.flash = 2.5; coreBoost += 0.6;
}

// ---------------------------------------------------------------- feed
const feedList = document.getElementById('feedList');
function feedAdd(html, cls) { const li = document.createElement('li'); if (cls) li.className = cls; li.innerHTML = html; feedList.prepend(li); while (feedList.children.length > 40) feedList.lastChild.remove(); }
function feedEvent(e) { feedAdd(`<span class="dot" style="background:${colorOf(e.organ)}"></span><time>${hhmm(e.t)}</time><span><span class="organ">${esc(nice(e.organ))}</span>${esc(e.label)}</span>`); }
function feedCite(c) { feedAdd(`<span class="dot" style="background:var(--gold)"></span><time>${hhmm(c.t)}</time><span class="what"><b>${esc(nice(c.by))}</b> used a ${esc(nice(c.from))} experience${c.label ? `: ${esc(c.label.slice(0, 90))}` : ''}</span>`, 'bind'); }

// ---------------------------------------------------------------- the record + clock
const events = [], cites = [], seen = new Set(), seenCites = new Set();
let evIdx = 0, ciIdx = 0, simT = 0, mode = 'live', speed = 240, nExp = 0, nBind = 0;
const fieldHeld = document.getElementById('fieldHeld');
const window0 = () => serverNow() - WINDOW_SEC;
function noteEvent(e) { // counts and side text for an event that has now happened on screen (inside the window)
  if (e.t < window0()) return;
  const o = organBy.get(e.organ);
  if (o) { o.recent.push(e.t); o.total++; o.flash = Math.max(o.flash, 1); o.last = e.label; }
  if (e.kind === 'field_dominance_run') fieldHeld.textContent = e.label;
  nExp++;
}
// Counts from the record itself: everything in [window start, simT]. Touches no comet, arc or feed line.
function recount() {
  for (const o of organs) { o.recent = []; o.total = 0; o.last = ''; }
  nExp = 0; nBind = 0; fieldHeld.textContent = '—';
  for (let i = 0; i < evIdx; i++) noteEvent(events[i]);
  for (const o of organs) { o.recent = o.recent.filter((t) => simT - t < 1800); o.flash = 0; }
  const lo = window0();
  for (let i = 0; i < ciIdx; i++) if (cites[i].t >= lo) nBind++;
}
// Drop rows that slid out of the 12 h window (they are all behind the playhead), then recount.
function prune() {
  const lo = window0() - 60;
  let k = 0; while (k < events.length && events[k].t < lo) { seen.delete(events[k].id); k++; }
  if (k) { events.splice(0, k); evIdx = Math.max(0, evIdx - k); }
  let j = 0; while (j < cites.length && cites[j].t < lo) { seenCites.delete(cites[j].id); j++; }
  if (j) { cites.splice(0, j); ciIdx = Math.max(0, ciIdx - j); }
  recount();
}
function seek(t) {
  simT = Math.max(window0(), Math.min(serverNow(), t));
  evIdx = 0; while (evIdx < events.length && events[evIdx].t <= simT) evIdx++;
  ciIdx = 0; while (ciIdx < cites.length && cites[ciIdx].t <= simT) ciIdx++;
  comets.length = 0; arcs.length = 0;
  freeComet.length = 0; for (let i = MAXC - 1; i >= 0; i--) freeComet.push(i);
  freeArc.length = 0; for (let i = MAXA - 1; i >= 0; i--) freeArc.push(i);
  for (let i = 0; i < cometPool.n; i++) cometPool.hide(i);
  for (let i = 0; i < arcPool.n; i++) arcPool.hide(i);
  feedList.innerHTML = '';
  recount();
  for (const e of events.slice(Math.max(0, evIdx - 14), evIdx)) if (e.t >= window0()) feedEvent(e);
  for (let i = 0; i < ciIdx; i++) if (simT - cites[i].t < 3600) feedCite(cites[i]);
}
function setMode(next) {
  mode = next;
  const m = document.getElementById('mode'); m.textContent = mode === 'live' ? 'LIVE' : 'REPLAY'; m.classList.toggle('replay', mode !== 'live');
  document.getElementById('liveBtn').setAttribute('aria-pressed', String(mode === 'live'));
  document.getElementById('replayBtn').setAttribute('aria-pressed', String(mode !== 'live'));
}
function step(dt) {
  if (mode === 'live') simT = serverNow();
  else { simT += dt * speed; if (simT >= serverNow()) { setMode('live'); simT = serverNow(); } }
  while (evIdx < events.length && events[evIdx].t <= simT) { const e = events[evIdx++]; noteEvent(e); spawnComet(e); feedEvent(e); }
  while (ciIdx < cites.length && cites[ciIdx].t <= simT) { const c = cites[ciIdx++]; if (c.t >= window0()) nBind++; spawnArc(c); feedCite(c); }
}

// ---------------------------------------------------------------- polling
const problems = new Map();
function setProblem(key, msg) { if (msg) problems.set(key, msg); else problems.delete(key); const el = document.getElementById('problems'); el.hidden = problems.size === 0; el.textContent = [...problems.values()].join(' · '); }
let cursor = 0, citeCursor = 0, booted = false;
async function pollEvents() {
  try {
    for (let page = 0; page < 10; page++) {
      const res = await fetch(`/api/inner-sky/events?since=${cursor}&cite_since=${citeCursor}`);
      if (!res.ok) throw new Error(`experience feed HTTP ${res.status}`);
      const d = await res.json();
      clockOffset = d.now - Date.now() / 1000;
      const names = [...d.organs];
      for (const e of d.events) if (!names.includes(e.organ)) names.push(e.organ);
      for (const c of d.cites) for (const o of [c.from, c.by]) if (!names.includes(o)) names.push(o);
      const grew = names.length !== organs.length;
      buildOrgans(names);
      if (grew && booted) recount();
      for (const e of d.events) {
        if (seen.has(e.id) || e.t == null) continue; seen.add(e.id);
        const i = core.insertSorted(events, e, (x) => x.t);
        // A late row behind the playhead: count it now; draw it only in Live (in Replay it is in the past).
        if (booted && i < evIdx) { evIdx++; noteEvent(e); if (mode === 'live') { spawnComet(e); feedEvent(e); } }
      }
      for (const c of d.cites) {
        if (seenCites.has(c.id) || c.t == null) continue; seenCites.add(c.id);
        const i = core.insertSorted(cites, c, (x) => x.t);
        if (booted && i < ciIdx) { ciIdx++; if (c.t >= window0()) nBind++; if (mode === 'live') { spawnArc(c); feedCite(c); } }
      }
      cursor = Math.max(cursor, d.cursor); citeCursor = Math.max(citeCursor, d.cite_cursor);
      if (!d.more) break;
    }
    setProblem('events', null);
  } catch (err) { setProblem('events', `Experience feed unavailable: ${err.message}`); }
  if (!booted && organs.length) { booted = true; seek(serverNow()); }
}
async function pollState() {
  try {
    const res = await fetch('/api/inner-sky/state'); if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json(); stateNow = d.now;
    if (d.body) {
      for (const s of d.body.services) {
        if (!services.has(s.id)) { if (services.size >= MAX_SVC) continue; services.set(s.id, { id: s.id, pos: shellPos(s.id), phase: Math.random() * 6.28, rate: null }); }
        Object.assign(services.get(s.id), { n: s.n, last: s.last });
      }
      const sr = core.countRates(serviceRatesPrev, d.body.services, (s) => s.id, d.now); serviceRatesPrev = sr.next;
      for (const [id, r] of Object.entries(sr.rates)) if (services.has(id)) services.get(id).rate = r;
      const lr = core.countRates(linkRatesPrev, d.body.links, (l) => `${l.a}>${l.b}`, d.now); linkRatesPrev = lr.next;
      rebuildLinks(d.body.links, lr.rates);
      const all = [...services.values()];
      document.getElementById('sLive').textContent = `${all.filter((s) => core.isLive(s, d.now, LIVE_SEC)).length}/${all.length}`;
      setProblem('body', null);
    } else setProblem('body', `Service graph unavailable: ${d.body_error || 'no answer'}`);
    if (d.focus) {
      const f = d.focus;
      document.getElementById('attn').textContent = f.attention ? `${f.attention}${f.attention_since ? ` (since ${hhmm(Date.parse(f.attention_since) / 1000)})` : ''}` : 'no open attention arc';
      setProblem('focus', null);
    } else setProblem('focus', `Day frame unavailable: ${d.focus_error || 'no answer'}`);
  } catch (err) { setProblem('body', `Service graph unavailable: ${err.message}`); }
}

// ---------------------------------------------------------------- timeline + controls
const tl = document.getElementById('timeline'), tctx = tl.getContext('2d'), clockEl = document.getElementById('clock');
function drawTimeline() {
  if (EMBED) return;
  const w = tl.clientWidth, h = tl.clientHeight, pr = devicePixelRatio || 1; if (!w) return;
  if (tl.width !== Math.round(w * pr)) { tl.width = Math.round(w * pr); tl.height = Math.round(h * pr); }
  tctx.setTransform(pr, 0, 0, pr, 0, 0); tctx.clearRect(0, 0, w, h);
  const T1 = serverNow(), T0 = T1 - WINDOW_SEC, BINS = 144, bins = core.binCounts(events, T0, T1, BINS), bmax = Math.max(...bins, 1);
  const top = 4, base = h - 16, bw = w / BINS;
  for (let i = 0; i < BINS; i++) { if (!bins[i]) continue; const bh = Math.max(1.5, (base - top) * bins[i] / bmax); tctx.fillStyle = T0 + (i + 1) * WINDOW_SEC / BINS <= simT ? '#6f80c8' : '#2c3566'; tctx.fillRect(i * bw + 0.5, base - bh, Math.max(1, bw - 1), bh); }
  tctx.strokeStyle = '#ffc44d'; tctx.lineWidth = 2;
  for (const c of cites) { if (c.t < T0) continue; const x = (c.t - T0) / WINDOW_SEC * w; tctx.beginPath(); tctx.moveTo(x, top); tctx.lineTo(x, base); tctx.stroke(); }
  tctx.fillStyle = '#8790b8'; tctx.font = '10px "IBM Plex Mono", monospace'; tctx.textBaseline = 'top';
  for (let hr = Math.ceil(T0 / 7200) * 7200; hr < T1; hr += 7200) { const x = (hr - T0) / WINDOW_SEC * w; if (x > 14 && x < w - 30) tctx.fillText(hhmm(hr), x - 14, base + 3); }
  const px = (simT - T0) / WINDOW_SEC * w; tctx.fillStyle = '#ffffff'; tctx.fillRect(px - 1, 0, 2, base);
  clockEl.textContent = hhmm(simT);
}
function seekReplay(t) { if (t >= serverNow() - 30) { setMode('live'); seek(serverNow()); } else { setMode('replay'); seek(t); } }
let dragging = false;
const seekFromPointer = (ev) => { const r = tl.getBoundingClientRect(); seekReplay(serverNow() - WINDOW_SEC + (ev.clientX - r.left) / r.width * WINDOW_SEC); };
tl.addEventListener('pointerdown', (ev) => { dragging = true; tl.setPointerCapture(ev.pointerId); seekFromPointer(ev); });
tl.addEventListener('pointermove', (ev) => { if (dragging) seekFromPointer(ev); });
tl.addEventListener('pointerup', () => { dragging = false; });
tl.addEventListener('keydown', (ev) => { if (ev.key === 'ArrowRight') seekReplay(simT + 900); if (ev.key === 'ArrowLeft') seekReplay(simT - 900); });
document.getElementById('liveBtn').addEventListener('click', () => { setMode('live'); seek(serverNow()); });
document.getElementById('replayBtn').addEventListener('click', () => { setMode('replay'); seek(window0()); });
for (const b of document.querySelectorAll('[data-speed]')) b.addEventListener('click', () => { speed = +b.dataset.speed; for (const o of document.querySelectorAll('[data-speed]')) o.setAttribute('aria-pressed', String(o === b)); });
if (REDUCED) { speed = 60; for (const o of document.querySelectorAll('[data-speed]')) o.setAttribute('aria-pressed', String(o.dataset.speed === '60')); }

// ---------------------------------------------------------------- hover
const tip = document.getElementById('tip'); let mouse = null;
canvas.addEventListener('pointermove', (ev) => { mouse = { x: ev.clientX, y: ev.clientY }; });
canvas.addEventListener('pointerleave', () => { mouse = null; tip.hidden = true; });
const vtmp = new THREE.Vector3();
const toScreen = (p) => { vtmp.copy(p).project(camera); return { x: (vtmp.x + 1) / 2 * innerWidth, y: (1 - vtmp.y) / 2 * innerHeight, z: vtmp.z }; };
function hover() {
  if (!mouse) return;
  let best = null, bd = 18;
  for (const o of organs) { const s = toScreen(o.pos); const d = Math.hypot(s.x - mouse.x, s.y - mouse.y); if (d < bd) { bd = d; best = { o }; } }
  if (!best) for (const s of services.values()) { const p = toScreen(s.pos); const d = Math.hypot(p.x - mouse.x, p.y - mouse.y); if (d < bd) { bd = d; best = { s }; } }
  if (!best) { tip.hidden = true; return; }
  if (best.o) { const o = best.o; tip.innerHTML = `<b style="color:${colorOf(o.name)}">${esc(nice(o.name))}</b><br>${o.total} experiences in the window · ${o.recent.length} in the last 30 min${o.last ? `<br><span style="color:var(--muted)">Latest: ${esc(o.last.slice(0, 140))}</span>` : ''}`; }
  else {
    const s = best.s, live = core.isLive(s, stateNow, LIVE_SEC);
    tip.innerHTML = `<b>${esc(s.id)}</b><br>${s.last == null ? 'Never published' : live ? `Last spoke ${ago(stateNow - s.last)} ago` : `<span style="color:var(--warn)">Silent for ${ago(stateNow - s.last)}</span>`}<br><span style="color:var(--muted)">${s.rate == null ? 'Rate: measuring (two readings needed)' : `${s.rate.toFixed(s.rate < 1 ? 2 : 0)} messages/s now`} · ${(s.n || 0).toLocaleString()} lifetime</span>`;
  }
  tip.hidden = false; tip.style.left = Math.min(mouse.x + 14, innerWidth - 310) + 'px'; tip.style.top = (mouse.y + 14) + 'px';
}

// ---------------------------------------------------------------- frame
const clk = new THREE.Clock(); const tmp = new THREE.Vector3(); const ORIGIN = new THREE.Vector3();
const sExp = document.getElementById('sExp'), sBind = document.getElementById('sBind');
function frame() {
  requestAnimationFrame(frame);
  const dt = Math.min(clk.getDelta(), 0.1), now = clk.elapsedTime;
  if (booted) step(dt);

  let si = 0;
  for (const s of services.values()) {
    const live = core.isLive(s, stateNow, LIVE_SEC);
    const sz = live ? 0.32 + (s.rate ? 0.08 * Math.log2(1 + s.rate) : 0) : 0.28;
    svcPool.set(si++, s.pos, live ? C_LIVE : C_DEAD, Math.min(sz, 0.9), live ? 0.65 + 0.35 * Math.sin(now * 1.3 + s.phase) : 0.55);
  }
  for (; si < svcPool.n; si++) svcPool.hide(si);
  svcPool.flush();

  for (const l of links) { const pps = core.pulsesPerSecond(l.rate || 0); if (!pps) continue; l.acc += dt * pps; if (l.acc >= 1 && pulses.length < pulsePool.n) { l.acc -= 1; pulses.push({ l, t: 0 }); } }
  for (let i = pulses.length - 1; i >= 0; i--) { pulses[i].t += dt / 1.8; if (pulses[i].t >= 1) pulses.splice(i, 1); }
  for (let i = 0; i < pulsePool.n; i++) {
    const p = pulses[i]; if (!p) { pulsePool.hide(i); continue; }
    const f = p.t * SEG, k = Math.floor(f), pts = p.l.curve; tmp.copy(pts[k]).lerp(pts[Math.min(SEG, k + 1)], f - k);
    pulsePool.set(i, tmp, C_PULSE, 0.38, Math.sin(Math.PI * p.t) * 0.9);
  }
  pulsePool.flush();

  if (organPool) {
    organs.forEach((o, i) => {
      while (o.recent.length && simT - o.recent[0] > 1800) o.recent.shift();
      o.flash = Math.max(0, o.flash - dt * 2.2);
      const sz = 0.7 + 0.55 * Math.log2(1 + o.recent.length) + 0.4 * Math.min(o.flash, 1.5);
      organPool.set(i * 2, o.pos, o.color, sz, o.total ? 1 : 0.35);
      organPool.set(i * 2 + 1, o.pos, o.color, sz * 3.2, 0.12 + 0.12 * Math.min(o.flash, 2));
      const s = toScreen(o.pos); labels[i].style.left = s.x + 'px'; labels[i].style.top = s.y + 'px'; labels[i].style.opacity = s.z < 1 ? 0.85 : 0;
    });
    organPool.flush();
  }

  for (let i = comets.length - 1; i >= 0; i--) { const c = comets[i]; c.age += dt; if (c.age >= c.life) { for (let k = 0; k < TRAIL; k++) cometPool.hide(c.slot * TRAIL + k); freeComet.push(c.slot); comets.splice(i, 1); coreBoost += 0.05; } }
  for (const c of comets) {
    const u = c.age / c.life, fade = 1 - u * u * u;
    for (let k = 0; k < TRAIL; k++) { const t = Math.max(0, u - k * 0.03); bez(c.p0, c.p1, t, tmp); cometPool.set(c.slot * TRAIL + k, tmp, c.c, c.size * (1 - k / TRAIL * 0.75), fade * (1 - k / TRAIL) * (t > 0 ? 1 : 0)); }
  }
  cometPool.flush();

  for (let i = arcs.length - 1; i >= 0; i--) { arcs[i].age += dt; if (arcs[i].age >= arcs[i].life) { for (let k = 0; k < ARC; k++) arcPool.hide(arcs[i].slot * ARC + k); freeArc.push(arcs[i].slot); arcs.splice(i, 1); } }
  for (const a of arcs) {
    const head = Math.min(1, a.age / 2.2), fade = a.age < a.life - 4 ? 1 : (a.life - a.age) / 4;
    for (let k = 0; k < ARC; k++) { const t = k / (ARC - 1), nearHead = Math.max(0, 1 - Math.abs(t - head) * 12); arcPool.set(a.slot * ARC + k, a.pts[k], nearHead > 0.3 ? C_GOLDHI : C_GOLD, 0.32 + 0.6 * nearHead, t <= head ? fade * (0.55 + 0.45 * nearHead) : 0); }
  }
  arcPool.flush();

  coreBoost = Math.min(3, coreBoost * Math.pow(0.25, dt));
  corePool.set(0, ORIGIN, C_CORE, 1.6 + 0.5 * coreBoost, 1);
  corePool.set(1, ORIGIN, C_HALO, 5 + 2.5 * coreBoost, 0.22);
  corePool.set(2, ORIGIN, C_HALO, 13 + 3 * coreBoost, 0.06);
  corePool.flush();

  sExp.textContent = nExp.toLocaleString(); sBind.textContent = nBind;
  drawTimeline(); hover();
  controls.update(); composer.render();
}
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); composer.setSize(innerWidth, innerHeight); updateScale(); });

// Each poll schedules the next only after it finishes, so two never run at once.
const loop = (fn, ms) => { const tick = async () => { try { await fn(); } finally { setTimeout(tick, ms); } }; setTimeout(tick, ms); };
await Promise.all([pollEvents(), pollState()]);
loop(pollEvents, EVENTS_POLL_MS);
loop(pollState, STATE_POLL_MS);
setInterval(() => { if (booted) prune(); }, 60000);
frame();
window.__skyStarted = true;
