"use strict";

const API_BASE = "";
const TAIL_POLL_MS = 3000;
const EKG_WINDOW = 120; // frames kept for realtime EKG

const DIMENSIONS = [
  { key: "node_kind", label: "Node kinds" },
  { key: "lane", label: "Lanes" },
  { key: "self_state", label: "Self-state" },
  { key: "honesty_metrics", label: "Prediction Confidence" },
  // Added 2026-09-04: field_anomaly (mood-arc encoder reconstruction error,
  // orion-field-digester) has been a real region on every brain frame since
  // 2026-07-21 (services/orion-substrate-runtime/app/brain_frame_producer.py
  // ::_field_anomaly_regions), but had no rail entry here -- so the region
  // this whole mood_arc/field-digester consumer was built to feed was never
  // actually selectable in this UI. lattice_layer has the same gap and is
  // left alone here; out of scope for this patch, noted for a follow-up.
  { key: "field_anomaly", label: "Field Anomaly" },
  { key: "spotlight", label: "Spotlight" },
];

const state = {
  dim: "node_kind",
  live: true,
  frames: [],        // ascending; realtime tail buffer or loaded range
  pollTimer: null,
  window: null,
  hitboxes: [],       // [{cx, cy, radius, region}], rebuilt every drawBrain()
  provenance: null,   // {dimension: {producer_service, urn, upstream}}, fetched once
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

let spotlightPulseRAF = null;

function drawSpotlight(ctx, canvas, frame) {
  const sp = frame.spotlight;
  ctx.textAlign = "left";
  if (!sp) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No active spotlight (no coalition selected yet).", canvas.width / 2, canvas.height / 2);
    return;
  }
  const stability = Math.max(0, Math.min(1, sp.coalition_stability));
  const stabPct = (stability * 100) | 0;
  ctx.fillStyle = "#f0abfc";
  ctx.font = "13px sans-serif";
  ctx.fillText(
    `Spotlight · ${sp.attended_node_ids.length} nodes · dwell ${sp.dwell_ticks} · stability ${stabPct}%${sp.stale ? " (held)" : ""}`,
    12, 20,
  );
  if (sp.description) {
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "12px sans-serif";
    ctx.fillText(String(sp.description).slice(0, 96), 12, 38);
  }

  const ids = sp.attended_node_ids || [];
  if (!ids.length) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Coalition present but no attended nodes.", canvas.width / 2, canvas.height / 2);
    return;
  }

  const byId = new Map((frame.nodes || []).map((n) => [n.node_id, n]));
  const top = 54;
  const areaH = canvas.height - top - 8;
  const cols = Math.ceil(Math.sqrt(ids.length));
  const rows = Math.ceil(ids.length / cols) || 1;
  const cw = canvas.width / cols;
  const chh = areaH / rows;
  const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 380); // 0..1

  ids.forEach((id, i) => {
    const cx = (i % cols) * cw + cw / 2;
    const cy = top + Math.floor(i / cols) * chh + chh / 2;
    const node = byId.get(id);
    const act = node ? Math.max(0, Math.min(1, node.activation)) : 0.4;
    const baseR = Math.min(cw, chh) * (0.16 + 0.16 * act);
    // Coalition-stability halo: larger, softer ring when the coalition holds.
    const haloR = baseR + 6 + 12 * stability * (0.6 + 0.4 * pulse);
    ctx.beginPath();
    ctx.arc(cx, cy, haloR, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(240,171,252,${0.05 + 0.15 * stability})`;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx, cy, baseR, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(232,121,249,${0.45 + 0.45 * act})`;
    ctx.fill();
    ctx.strokeStyle = `rgba(240,171,252,${0.45 + 0.5 * pulse})`;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    const label = node && node.label ? node.label : (id.split(":").slice(-1)[0] || id);
    ctx.fillStyle = "#e5e7eb";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(String(label).slice(0, 24), cx, cy + baseR + 12);
  });
}

function startSpotlightPulse() {
  if (spotlightPulseRAF) return;
  const loop = () => {
    if (state.dim !== "spotlight") { spotlightPulseRAF = null; return; }
    const canvas = document.getElementById("brainCanvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const frame = state.frames[state.frames.length - 1];
    if (frame) drawSpotlight(ctx, canvas, frame);
    else setStatus("no frames");
    spotlightPulseRAF = requestAnimationFrame(loop);
  };
  spotlightPulseRAF = requestAnimationFrame(loop);
}

function stopSpotlightPulse() {
  if (spotlightPulseRAF) { cancelAnimationFrame(spotlightPulseRAF); spotlightPulseRAF = null; }
}

function drawBrain() {
  const canvas = document.getElementById("brainCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  state.hitboxes = [];
  const frame = state.frames[state.frames.length - 1];
  if (!frame) { setStatus("no frames"); return; }

  if (state.dim === "spotlight") {
    drawSpotlight(ctx, canvas, frame);
    setStatus(`${frame.phase} · tick ${frame.tick_seq} · spotlight`);
    return;
  }

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
    // Generous click target (>= the label below the circle), not just the
    // drawn circle itself -- a thin ring around a low-intensity region is a
    // frustrating click target otherwise.
    state.hitboxes.push({ cx, cy, radius: Math.max(radius, Math.min(cw, chh) * 0.4), region: r });
  });

  setStatus(`${frame.phase} · tick ${frame.tick_seq} · ${regions.length} regions`);
}

function drawEkg() {
  const canvas = document.getElementById("ekgCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const legend = document.getElementById("ekgLegend");
  legend.innerHTML = "";
  if (state.dim === "spotlight") {
    // Coalition-stability sparkline over the loaded window.
    const pts = state.frames.map((f) => (f.spotlight ? Math.max(0, Math.min(1, f.spotlight.coalition_stability)) : null));
    if (!pts.some((v) => v !== null)) { legend.textContent = "No spotlight history in window."; return; }
    const n2 = Math.max(1, pts.length - 1);
    ctx.beginPath();
    let started = false;
    pts.forEach((v, xi) => {
      if (v === null) return;
      const x = (xi / n2) * canvas.width;
      const y = canvas.height - v * (canvas.height - 8) - 4;
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#f0abfc";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    const row = document.createElement("div");
    const sw = document.createElement("span");
    sw.style.color = "#f0abfc";
    sw.textContent = "■ ";
    row.appendChild(sw);
    row.append("coalition stability");
    legend.appendChild(row);
    return;
  }

  if (state.dim === "honesty_metrics") {
    // Prediction confidence sparkline over the loaded window.
    const pts = state.frames.map((f) => {
      const regions = regionsFor(f, "honesty_metrics");
      return regions.length > 0 ? regions[0].intensity : null;
    });
    if (!pts.some((v) => v !== null)) { legend.textContent = "No prediction confidence in window."; return; }
    const n2 = Math.max(1, pts.length - 1);
    ctx.beginPath();
    let started = false;
    pts.forEach((v, xi) => {
      if (v === null) return;
      const x = (xi / n2) * canvas.width;
      const y = canvas.height - v * (canvas.height - 8) - 4;
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth = 2;
    ctx.stroke();
    const row = document.createElement("div");
    const sw = document.createElement("span");
    sw.style.color = "#34d399";
    sw.textContent = "■ ";
    row.appendChild(sw);
    row.append("Prediction Confidence");
    legend.appendChild(row);
    return;
  }

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

function fmtDetailValue(v) {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  return String(v);
}

function showRegionDetail(region) {
  const panel = document.getElementById("regionDetail");
  document.getElementById("regionDetailTitle").textContent = region.label || region.region_id;
  const body = document.getElementById("regionDetailBody");
  body.innerHTML = "";

  const addRow = (k, v) => {
    const row = document.createElement("div");
    row.innerHTML = `<span class="text-gray-500">${k}:</span> <span class="text-gray-100 font-mono">${v}</span>`;
    body.appendChild(row);
  };

  addRow("region_id", region.region_id);
  addRow("state", region.state);
  addRow("intensity", fmtDetailValue(region.intensity));
  addRow("node_count", region.node_count);
  addRow("as_of", region.as_of || "—");
  addRow("stale", region.stale ? "yes (held)" : "no");

  const detail = region.detail || {};
  const detailKeys = Object.keys(detail);
  if (detailKeys.length) {
    const hdr = document.createElement("div");
    hdr.className = "text-[10px] uppercase tracking-wide text-gray-500 mt-2";
    hdr.textContent = "Detail";
    body.appendChild(hdr);
    detailKeys.forEach((k) => addRow(k, fmtDetailValue(detail[k])));
  }

  const prov = state.provenance ? state.provenance[region.dimension] : null;
  const provHdr = document.createElement("div");
  provHdr.className = "text-[10px] uppercase tracking-wide text-gray-500 mt-2";
  provHdr.textContent = "Provenance";
  body.appendChild(provHdr);
  if (prov) {
    addRow("producer_service", prov.producer_service);
    addRow("urn", prov.urn);
    if (prov.upstream && prov.upstream.length) addRow("upstream", prov.upstream.join(", "));
  } else {
    const row = document.createElement("div");
    row.className = "text-gray-500";
    row.textContent = "unavailable (provenance lookup not loaded)";
    body.appendChild(row);
  }

  panel.classList.remove("hidden");
}

function hideRegionDetail() {
  document.getElementById("regionDetail").classList.add("hidden");
}

// Pure (no DOM): nearest hitbox containing (x, y), or null. Extracted so it
// is unit-testable under `node --test` without a canvas/DOM harness, same
// rationale as this dir's cognitive-loop-card.test.js (see README.md).
function hitTestRegion(hitboxes, x, y) {
  let hit = null;
  let hitDist = Infinity;
  for (const box of hitboxes || []) {
    const d = Math.hypot(x - box.cx, y - box.cy);
    if (d <= box.radius && d < hitDist) { hit = box; hitDist = d; }
  }
  return hit;
}

function onBrainCanvasClick(e) {
  const canvas = document.getElementById("brainCanvas");
  const rect = canvas.getBoundingClientRect();
  // Canvas backing size (720x440) vs. its CSS-scaled displayed size can
  // differ (class="w-full") -- without this scale correction, hit-testing
  // against state.hitboxes (recorded in backing-canvas coordinates) would be
  // wrong on any width other than exactly 720px.
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;

  const hit = hitTestRegion(state.hitboxes, x, y);
  if (hit) showRegionDetail(hit.region); else hideRegionDetail();
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
      if (d.key === "spotlight") startSpotlightPulse(); else stopSpotlightPulse();
      hideRegionDetail();
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
  document.getElementById("brainCanvas").addEventListener("click", onBrainCanvasClick);
  document.getElementById("regionDetailClose").addEventListener("click", hideRegionDetail);
  try { state.window = await _get("/api/self-brain/window"); } catch (e) { /* empty ok */ }
  // Static, one-time fetch (see self_brain_routes.py::region_provenance's own
  // comment) -- a failure here degrades the detail panel's "Provenance"
  // section to "unavailable", not the whole page.
  try { state.provenance = await _get("/api/self-brain/region-provenance"); } catch (e) { /* degrades gracefully */ }
  await pollTail();
  state.pollTimer = setInterval(pollTail, TAIL_POLL_MS);
}

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", init);
}

// node:test entry point only -- this file is otherwise a plain script (no
// module system) loaded directly by self-brain.html, unlike
// workflow-schedule-ui.js's full IIFE+export pattern. Only the two pure
// helpers added 2026-09-04 are exported; everything else here still touches
// `document`/canvas directly and is not meant to run outside a browser.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { hitTestRegion, fmtDetailValue };
}
