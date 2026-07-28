import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- STATE ---
// 2026-07-28: replaced OrionTissue's own WS-pushed embedding-drift stats
// (embedding_similarity/novelty_zscore/polarity_diff/mean_abs_activation --
// real, but "no independent theory anchor connecting them to reasoning
// quality, mood, or wellbeing" per this page's own prior disclosed copy)
// with three independently-sourced, independently-verified real signals,
// polled from orion-hub's GET /api/self-brain/frames/tail (backed by
// SubstrateBrainFrameV1, orion-substrate-runtime, persisted every 5s):
//   - confidence: 1 - mean(prediction_error) across 5 Active-Inference
//     domains (honesty_metrics region, unconditional)
//   - coalition: AttentionBroadcastProjectionV1.coalition_stability_score
//     (frame.spotlight, GWT-dispatch lane)
//   - anomaly: mood-arc encoder reconstruction error over live field-channel
//     pressures, calibrated against its own live-observed range
//     (field_anomaly region)
// Each was live-verified individually before being wired in here -- see
// docs/superpowers/specs/2026-07-28-spark-introspector-retirement-and-honest-
// substrate-convergence.md for the candidates that were checked and dropped
// (self_state: dead, zero producer; node_kind/lane max() and min(): both
// ceiling/floor-pinned against real data; raw bus_synaptic gap_zscore:
// not independent of confidence, already one of its 5 domains).
const state = {
    confidence: 0.5, coalition: 0.5, anomaly: 0.0,
    targetConfidence: 0.5, targetCoalition: 0.5, targetAnomaly: 0.0,
    lastUpdate: 0, frameId: null, timestamp: null
};
const INTERPOLATION_SPEED = 0.05;
const POLL_INTERVAL_MS = 5000; // matches BRAIN_FRAME_INTERVAL_SEC default

// --- DOM ---
const elMoodValue = document.getElementById('mood-value');
const elConnStatus = document.getElementById('conn-status');
const elGearBtn = document.getElementById('gear-btn');
const elSettingsMenu = document.getElementById('settings-menu');
const elInfoToggle = document.getElementById('info-toggle-row');
const elExplanation = document.getElementById('explanation-panel');

// --- UI LOGIC ---
elGearBtn.addEventListener('click', (e) => { e.stopPropagation(); elSettingsMenu.classList.toggle('visible'); });
document.addEventListener('click', () => { elSettingsMenu.classList.remove('visible'); });
elInfoToggle.addEventListener('click', () => {
    elExplanation.classList.toggle('visible');
    elInfoToggle.innerText = elExplanation.classList.contains('visible') ? "▲ HIDE INTERNALS ▲" : "▼ SYSTEM INTERNALS EXPLANATION ▼";
});

// --- BRAIN FRAME POLL ---
class BrainFramePoller {
    constructor() { this.timer = null; this.consecutiveFailures = 0; }
    setConnected(isConnected) {
        elConnStatus.innerText = isConnected ? "LIVE" : "STALE";
        elConnStatus.style.color = isConnected ? "#0f0" : "#f00";
    }
    start() {
        this.poll();
        this.timer = setInterval(() => this.poll(), POLL_INTERVAL_MS);
    }
    async poll() {
        try {
            const res = await fetch('/api/self-brain/frames/tail?limit=1');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            const frames = data.frames || [];
            if (!frames.length) {
                this.consecutiveFailures += 1;
                this.setConnected(false);
                return;
            }
            this.consecutiveFailures = 0;
            this.setConnected(true);
            this.handleFrame(frames[frames.length - 1]);
        } catch (e) {
            this.consecutiveFailures += 1;
            this.setConnected(false);
            console.error("Brain frame poll error:", e);
        }
    }
    handleFrame(frame) {
        const regions = frame.regions || [];
        const honesty = regions.find((r) => r.dimension === 'honesty_metrics');
        const anomaly = regions.find((r) => r.dimension === 'field_anomaly');
        const coalitionScore = frame.spotlight ? frame.spotlight.coalition_stability : null;

        if (honesty) state.targetConfidence = honesty.intensity;
        if (anomaly) state.targetAnomaly = anomaly.intensity;
        if (coalitionScore !== null && coalitionScore !== undefined) state.targetCoalition = coalitionScore;

        state.frameId = frame.frame_id;
        state.timestamp = frame.generated_at;
        this.updateDOM();
    }
    updateDOM() {
        document.getElementById('val-id').innerText = (state.frameId || "NULL").substring(0, 12) + '...';
        document.getElementById('val-ts').innerText = state.timestamp || "-";
        document.getElementById('val-confidence').innerText = state.targetConfidence.toFixed(2);
        document.getElementById('val-coalition').innerText = state.targetCoalition.toFixed(2);
        document.getElementById('val-anomaly').innerText = state.targetAnomaly.toFixed(2);
        this.updateMoodText();
        this.updateDiagnosticsTable();
    }
    updateMoodText() {
        const c = state.targetConfidence, s = state.targetCoalition, a = state.targetAnomaly;
        elMoodValue.innerText = `conf ${c.toFixed(2)} · coal ${s.toFixed(2)} · anom ${a.toFixed(2)}`;
        elMoodValue.style.color = "#0ff";
        elMoodValue.style.textShadow = "0 0 10px #0ff";
    }
    updateDiagnosticsTable() {
        document.getElementById('diag-confidence-val').innerText = state.targetConfidence.toFixed(2);
        document.getElementById('diag-coalition-val').innerText = state.targetCoalition.toFixed(2);
        document.getElementById('diag-anomaly-val').innerText = state.targetAnomaly.toFixed(2);
    }
}

// --- VIZ ---
class SynthwaveVisualizer {
    constructor(scene, camera) { this.scene = scene; this.camera = camera; this.runTime = 0; }
    init() {
        this.camera.position.set(0, 3, 25); this.camera.lookAt(0, 4, -50);
        this.scene.fog = new THREE.FogExp2(0x000000, 0.01);
        const geo = new THREE.PlaneGeometry(200, 200, 128, 128);
        const mat = new THREE.MeshBasicMaterial({ color: 0xff00ff, wireframe: true, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending });
        this.plane = new THREE.Mesh(geo, mat);
        this.plane.rotation.x = -Math.PI / 2;
        this.scene.add(this.plane);

        const sunGeo = new THREE.SphereGeometry(30, 32, 32);
        this.sunMat = new THREE.MeshBasicMaterial({ color: 0xffaa00, wireframe: true, transparent: true, opacity: 0.8, side: THREE.DoubleSide });
        this.sun = new THREE.Mesh(sunGeo, this.sunMat);
        this.sun.position.set(0, 10, -85);
        this.scene.add(this.sun);

        const inGeo = new THREE.SphereGeometry(14, 32, 32);
        const inMat = new THREE.MeshBasicMaterial({ color: 0xff4400, transparent: true, opacity: 0.9 });
        this.innerSun = new THREE.Mesh(inGeo, inMat);
        this.sun.add(this.innerSun);
        this.vertexStore = geo.attributes.position.array.slice();
    }
    // Mapping (2026-07-28): only 3 independent real signals exist for 4 visual
    // channels. confidence drives both chaos and hue via different curves --
    // this is the same real number applied twice, not a 4th invented signal.
    //   chaos = 1 - confidence        (uncertain -> more turbulent grid)
    //   amp   = 1 + (1-coalition)*6   (unstable attention coalition -> bigger wave amplitude)
    //   speed = 1 + anomaly*6         (elevated field anomaly -> faster tempo)
    //   hue   = 0.9 + confidence*0.6  (confidence -> color band)
    update(dt) {
        const speed = 1.0 + (state.anomaly * 6.0);
        this.runTime += dt * speed;
        const pos = this.plane.geometry.attributes.position.array;
        const chaos = Math.max(0.1, 1.0 - state.confidence);
        const amp = 1.0 + ((1.0 - state.coalition) * 6.0);
        const hue = 0.9 + (state.confidence * 0.6);

        for (let i = 0; i < pos.length; i += 3) {
            const ox = this.vertexStore[i], oy = this.vertexStore[i+1];
            const sy = oy - this.runTime;
            const wx = Math.sin(sy * 0.1 + this.runTime * 0.5) * 3.0 + Math.cos(sy * 0.05) * 5.0;
            pos[i] = ox + (wx * chaos);

            let z = Math.sin(ox*0.1)*Math.cos(sy*0.1)*4.0 + Math.sin(ox*0.3+sy*0.3)*2.0 + Math.sin(ox*0.8+this.runTime)*Math.cos(sy*0.9)*(chaos*2.0);
            const dist = Math.abs(ox);
            let rf = 0.0;
            if(dist > 8.0) { rf = (dist - 8.0)/15.0; if(rf>1) rf=1; }
            pos[i+2] = z * (0.2 + (0.8*rf)) * amp;
        }
        this.plane.geometry.attributes.position.needsUpdate = true;
        this.plane.material.color.setHSL(hue%1, 1, 0.5);
        this.sunMat.color.setHSL((hue+0.1)%1, 1, 0.6);
        this.innerSun.material.color.setHSL(hue%1, 1, 0.5);
        this.sun.rotation.y += dt*0.1; this.sun.rotation.z -= dt*0.05;
        this.sun.scale.setScalar(1 + Math.sin(this.runTime*2)*0.05*state.anomaly);
    }
}

const poller = new BrainFramePoller(); poller.start();
const ren = new THREE.WebGLRenderer({antialias:true}); ren.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(ren.domElement);
const scn = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
const viz = new SynthwaveVisualizer(scn, cam); viz.init();
const clk = new THREE.Clock();
function anim() {
    requestAnimationFrame(anim);
    const dt = clk.getDelta();
    state.confidence += (state.targetConfidence-state.confidence)*INTERPOLATION_SPEED;
    state.coalition += (state.targetCoalition-state.coalition)*INTERPOLATION_SPEED;
    state.anomaly += (state.targetAnomaly-state.anomaly)*INTERPOLATION_SPEED;
    viz.update(dt);
    ren.render(scn, cam);
}
window.addEventListener('resize', ()=>{ cam.aspect=window.innerWidth/window.innerHeight; cam.updateProjectionMatrix(); ren.setSize(window.innerWidth, window.innerHeight); });
anim();
