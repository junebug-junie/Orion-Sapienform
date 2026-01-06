import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- STATE ---
const state = {
    phi: 0.5, novelty: 0.0, valence: 0.5, arousal: 0.0,
    targetPhi: 0.5, targetNovelty: 0.0, targetValence: 0.5, targetArousal: 0.0,
    lastUpdate: 0, metadata: {}, correlationId: null, timestamp: null
};
const INTERPOLATION_SPEED = 0.05; 

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

// --- WEBSOCKET ---
class WSClient {
    constructor() { this.ws = null; this.reconnectDelay = 1000; }
    setConnected(isConnected) {
        elConnStatus.innerText = isConnected ? "ONLINE" : "OFFLINE";
        elConnStatus.style.color = isConnected ? "#0f0" : "#f00";
    }
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        let path = window.location.pathname;
        if (path.endsWith('/')) path = path.slice(0, -1);
        if (path.endsWith('/ui')) path = path.slice(0, -3);
        if (!path.startsWith('/')) path = '/' + path;
        const wsUrl = `${protocol}//${host}${path}/ws/tissue`;
        
        this.ws = new WebSocket(wsUrl);
        this.ws.onopen = () => { this.setConnected(true); this.reconnectDelay = 1000; };
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === "tissue.update") { this.handleUpdate(data); }
            } catch (e) { console.error("Parse error:", e); }
        };
        this.ws.onclose = () => { this.setConnected(false); setTimeout(() => this.connect(), this.reconnectDelay); };
    }
    handleUpdate(data) {
        if (!data.stats) return;
        state.targetPhi = data.stats.phi;
        state.targetNovelty = data.stats.novelty;
        state.targetValence = data.stats.valence;
        state.targetArousal = data.stats.arousal;
        state.correlationId = data.correlation_id;
        state.timestamp = data.timestamp;
        this.updateDOM();
    }
    updateDOM() {
        document.getElementById('val-id').innerText = (state.correlationId || "NULL").substring(0, 8) + '...';
        document.getElementById('val-ts').innerText = state.timestamp || "-";
        document.getElementById('val-phi').innerText = state.targetPhi.toFixed(2);
        document.getElementById('val-novelty').innerText = state.targetNovelty.toFixed(2);
        document.getElementById('val-valence').innerText = state.targetValence.toFixed(2);
        document.getElementById('val-arousal').innerText = state.targetArousal.toFixed(2);
        this.updateMoodText();
        this.updateDiagnosticsTable();
    }
    updateMoodText() {
        const v = state.targetValence, p = state.targetPhi, a = state.targetArousal, n = state.targetNovelty;
        let mood = "UNKNOWN", color = "#888";
        if (n > 0.8) { mood = "EPIPHANY DETECTED"; color = "#fff"; }
        else if (a > 0.5) {
            if (v > 0.5 && p > 0.5) { mood = "LUCID FLOW"; color = "#0ff"; }
            else if (v > 0.5 && p <= 0.5) { mood = "MANIC CREATIVITY"; color = "#d0f"; }
            else if (v <= 0.5 && p > 0.5) { mood = "CRITICAL FOCUS"; color = "#f80"; }
            else { mood = "COGNITIVE DISSONANCE"; color = "#f00"; }
        } else {
            if (v > 0.5 && p > 0.5) { mood = "DEEP RESONANCE"; color = "#0af"; }
            else if (v > 0.5 && p <= 0.5) { mood = "DAYDREAMING"; color = "#fba"; }
            else if (v <= 0.5 && p > 0.5) { mood = "COLD ANALYSIS"; color = "#88f"; }
            else { mood = "IDLE / STATIC"; color = "#666"; }
        }
        elMoodValue.innerText = `[ ${mood} ]`;
        elMoodValue.style.color = color;
        elMoodValue.style.textShadow = `0 0 15px ${color}`;
    }
    updateDiagnosticsTable() {
        // Helper to update cells
        const updateRow = (id, val, highLow, imp) => {
            document.getElementById(`diag-${id}-val`).innerText = val.toFixed(2);
            document.getElementById(`diag-${id}-state`).innerText = highLow;
            document.getElementById(`diag-${id}-imp`).innerText = imp;
        };
        
        // Logic for diagnostics
        const p = state.targetPhi;
        updateRow('phi', p, p > 0.5 ? "HIGH" : "LOW", p > 0.5 ? "Integrated / Coherent" : "Fragmented / Noisy");
        
        const n = state.targetNovelty;
        updateRow('nov', n, n > 0.5 ? "HIGH" : "LOW", n > 0.5 ? "Surprising / New" : "Routine / Known");
        
        const v = state.targetValence;
        updateRow('val', v, v > 0.5 ? "POS" : "NEG", v > 0.5 ? "Harmony / Flow" : "Stress / Conflict");
        
        const a = state.targetArousal;
        updateRow('aro', a, a > 0.5 ? "HIGH" : "LOW", a > 0.5 ? "Active / Alert" : "Passive / Idle");
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
    update(dt) {
        const speed = 1.0 + (state.arousal * 6.0);
        this.runTime += dt * speed;
        const pos = this.plane.geometry.attributes.position.array;
        const chaos = Math.max(0.1, 1.0 - state.phi);
        const amp = 1.0 + (state.novelty * 6.0);
        const hue = 0.9 + (state.valence * 0.6);
        
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
        this.sun.scale.setScalar(1 + Math.sin(this.runTime*2)*0.05*state.arousal);
    }
}

const ws = new WSClient(); ws.connect();
const ren = new THREE.WebGLRenderer({antialias:true}); ren.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(ren.domElement);
const scn = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
const viz = new SynthwaveVisualizer(scn, cam); viz.init();
const clk = new THREE.Clock();
function anim() {
    requestAnimationFrame(anim);
    const dt = clk.getDelta();
    state.phi += (state.targetPhi-state.phi)*INTERPOLATION_SPEED;
    state.novelty += (state.targetNovelty-state.novelty)*INTERPOLATION_SPEED;
    state.valence += (state.targetValence-state.valence)*INTERPOLATION_SPEED;
    state.arousal += (state.targetArousal-state.arousal)*INTERPOLATION_SPEED;
    viz.update(dt);
    ren.render(scn, cam);
}
window.addEventListener('resize', ()=>{ cam.aspect=window.innerWidth/window.innerHeight; cam.updateProjectionMatrix(); ren.setSize(window.innerWidth, window.innerHeight); });
anim();

// Test Pulse
const btn = document.getElementById('btn-test-signal');
if(btn) btn.addEventListener('click', async () => {
    let p = window.location.pathname; if(p.endsWith('/')) p=p.slice(0,-1); if(p.endsWith('/ui')) p=p.slice(0,-3); if(!p.startsWith('/')) p='/'+p;
    await fetch(p+'/api/test-pulse', {method:"POST"});
});
