"use strict";

const API_BASE = "";
const TAIL_POLL_MS = 3000;
const EKG_WINDOW = 120; // frames kept for realtime EKG

const DIMENSIONS = [
  { key: "node_kind", label: "Node kinds" },
  { key: "lane", label: "Lanes" },
  { key: "self_state", label: "Self-state" },
  { key: "spotlight", label: "Spotlight" },
];

const state = {
  dim: "node_kind",
  live: true,
  frames: [],        // ascending; realtime tail buffer or loaded range
  pollTimer: null,
  window: null,
};

function _get(path) {
  return fetch(API_BASE + path).then((r) => {
    if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
    return r.json();
  });
}

function setStatus(msg) {
  document.getElementById("brainStatus").textContent = msg;
}

function regionsFor(frame, dim) {
  if (!frame) return [];
  if (dim === "spotlight") return [];
  return (frame.regions || []).filter((r) => r.dimension === dim);
}

function stateColor(regionState, intensity) {
  if (regionState === "firing") return `rgba(248,113,113,${0.35 + 0.65 * intensity})`;
  if (regionState === "starving") return `rgba(71,85,105,${0.4 + 0.3 * intensity})`;
  return `rgba(96,165,250,${0.35 + 0.5 * intensity})`;
}

function drawBrain() {
  const canvas = document.getElementById("brainCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const frame = state.frames[state.frames.length - 1];
  if (!frame) { setStatus("no frames"); return; }

  const regions = regionsFor(frame, state.dim);
  // Fixed grid layout = stable, always-labeled anatomical zones.
  const cols = Math.ceil(Math.sqrt(Math.max(1, regions.length)));
  const rows = Math.ceil(regions.length / cols) || 1;
  const cw = canvas.width / cols;
  const chh = canvas.height / rows;

  regions.forEach((r, i) => {
    const cx = (i % cols) * cw + cw / 2;
    const cy = Math.floor(i / cols) * chh + chh / 2;
    const radius = Math.min(cw, chh) * (0.22 + 0.22 * r.intensity);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = stateColor(r.state, r.intensity);
    ctx.fill();
    if (r.stale) {
      ctx.strokeStyle = "rgba(148,163,184,.8)";
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = "#e5e7eb";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(r.label, cx, cy + radius + 12);
    const ageTxt = r.stale ? " (held)" : "";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText(`${(r.intensity * 100) | 0}%${ageTxt}`, cx, cy + 4);
  });

  // Spotlight overlay: dashed hull label if present + spotlight dim selected.
  if (state.dim === "spotlight" && frame.spotlight) {
    ctx.fillStyle = "#f0abfc";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(
      `Spotlight: ${frame.spotlight.attended_node_ids.length} nodes, dwell ${frame.spotlight.dwell_ticks}, stability ${(frame.spotlight.coalition_stability * 100 | 0)}%${frame.spotlight.stale ? " (held)" : ""}`,
      12, 24,
    );
    if (frame.spotlight.description) ctx.fillText(frame.spotlight.description, 12, 44);
  }
  setStatus(`${frame.phase} · tick ${frame.tick_seq} · ${regions.length} regions`);
}

function drawEkg() {
  const canvas = document.getElementById("ekgCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const legend = document.getElementById("ekgLegend");
  legend.innerHTML = "";
  if (state.dim === "spotlight") { legend.textContent = "Select a region dimension for EKG."; return; }

  // Build per-region series over the loaded window (regions are the stable series).
  const ids = new Set();
  state.frames.forEach((f) => regionsFor(f, state.dim).forEach((r) => ids.add(r.region_id)));
  const palette = ["#f87171", "#60a5fa", "#34d399", "#fbbf24", "#c084fc", "#22d3ee", "#f472b6", "#a3e635"];
  const idList = [...ids].slice(0, 8);
  const n = Math.max(1, state.frames.length - 1);

  idList.forEach((id, k) => {
    const color = palette[k % palette.length];
    ctx.beginPath();
    state.frames.forEach((f, xi) => {
      const r = regionsFor(f, state.dim).find((x) => x.region_id === id);
      const v = r ? r.intensity : 0;
      const x = (xi / n) * canvas.width;
      const y = canvas.height - v * (canvas.height - 8) - 4;
      if (xi === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    const label = id.split(":").slice(1).join(":") || id;
    const row = document.createElement("div");
    const swatch = document.createElement("span");
    swatch.style.color = color;
    swatch.textContent = "■ ";
    row.appendChild(swatch);
    row.append(label);
    legend.appendChild(row);
  });
}

function render() { drawBrain(); drawEkg(); }

function pushTailFrames(frames) {
  if (!frames || !frames.length) return;
  const seen = new Set(state.frames.map((f) => f.frame_id));
  frames.forEach((f) => { if (!seen.has(f.frame_id)) state.frames.push(f); });
  if (state.frames.length > EKG_WINDOW) state.frames = state.frames.slice(-EKG_WINDOW);
}

async function pollTail() {
  if (!state.live) return;
  try {
    const data = await _get(`/api/self-brain/frames/tail?limit=30`);
    pushTailFrames(data.frames);
    toggleWarming(data.phase === "warming");
    render();
  } catch (e) { setStatus(`poll error: ${e.message}`); }
}

function toggleWarming(on) {
  document.getElementById("warmingBanner").classList.toggle("hidden", !on);
}

async function loadRange(fromIso, toIso) {
  try {
    const data = await _get(`/api/self-brain/frames/range?from=${encodeURIComponent(fromIso)}&to=${encodeURIComponent(toIso)}&max=240`);
    state.frames = data.frames || [];
    render();
  } catch (e) { setStatus(`range error: ${e.message}`); }
}

function goLive() {
  state.live = true;
  state.frames = [];
  document.getElementById("scrubber").value = 1000;
  document.getElementById("scrubLabel").textContent = "LIVE";
  document.getElementById("liveBtn").classList.add("border-emerald-700", "bg-emerald-900/40");
  pollTail();
}

function onScrub(e) {
  const frac = Number(e.target.value) / 1000;
  if (frac >= 0.999) { goLive(); return; }
  state.live = false;
  document.getElementById("liveBtn").classList.remove("border-emerald-700", "bg-emerald-900/40");
  const w = state.window;
  if (!w || !w.earliest || !w.latest) { setStatus("no window to scrub"); return; }
  const start = new Date(w.earliest).getTime();
  const end = new Date(w.latest).getTime();
  const center = new Date(start + frac * (end - start));
  const half = 5 * 60 * 1000; // 10-minute playback window
  const fromIso = new Date(center.getTime() - half).toISOString();
  const toIso = new Date(center.getTime() + half).toISOString();
  document.getElementById("scrubLabel").textContent = center.toLocaleTimeString();
  loadRange(fromIso, toIso);
}

function buildDimRail() {
  const rail = document.getElementById("dimRail");
  DIMENSIONS.forEach((d) => {
    const btn = document.createElement("button");
    btn.textContent = d.label;
    btn.dataset.key = d.key;
    btn.className = "px-2 py-0.5 rounded border border-gray-700 bg-gray-800 hover:bg-gray-700";
    btn.addEventListener("click", () => {
      state.dim = d.key;
      [...rail.children].forEach((c) => c.classList.toggle("bg-indigo-800", c.dataset.key === d.key));
      render();
    });
    rail.appendChild(btn);
  });
  rail.children[0].classList.add("bg-indigo-800");
}

async function init() {
  buildDimRail();
  document.getElementById("liveBtn").addEventListener("click", goLive);
  document.getElementById("scrubber").addEventListener("input", onScrub);
  try { state.window = await _get("/api/self-brain/window"); } catch (e) { /* empty ok */ }
  await pollTail();
  state.pollTimer = setInterval(pollTail, TAIL_POLL_MS);
}

document.addEventListener("DOMContentLoaded", init);
