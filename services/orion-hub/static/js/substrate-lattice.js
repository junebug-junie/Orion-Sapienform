/**
 * substrate-lattice.js
 * Substrate Lattice Hub Tab — read-only tuning console.
 *
 * Fetches from /api/substrate-lattice/* and renders:
 *   - Producer Lane Rail
 *   - Transport Proof Chain (M3-L11)
 *   - Gate Overlay
 *   - Lattice Values
 *   - Simulate Thresholds (calls /transport/simulate)
 *   - Draft Policy Patch (calls /transport/draft-policy-patch)
 *
 * No mutations. No automatic actions. Simulation is in-memory only.
 */

"use strict";

const API_BASE = "";

let _lastSimThresholds = null;

function _esc(v) {
  if (v === null || v === undefined) return "—";
  return String(v)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function _fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return v.toFixed(3);
  return String(v);
}

function _ts(isoStr) {
  if (!isoStr) return "—";
  try {
    const d = new Date(isoStr);
    const sec = Math.round((Date.now() - d.getTime()) / 1000);
    if (sec < 5) return "just now";
    if (sec < 60) return `${sec}s ago`;
    if (sec < 3600) return `${Math.round(sec / 60)}m ago`;
    return `${Math.round(sec / 3600)}h ago`;
  } catch {
    return isoStr;
  }
}

function _gateColor(state) {
  if (state === "pass") return "text-emerald-400";
  if (state === "quiet") return "text-gray-400";
  if (state === "watch") return "text-amber-400";
  if (state === "blocked") return "text-red-400";
  if (state === "dry_run") return "text-indigo-300";
  return "text-gray-500";
}

function _showError(msg) {
  const el = document.getElementById("latticeError");
  if (!el) return;
  el.textContent = msg;
  el.classList.remove("hidden");
}

function _clearError() {
  const el = document.getElementById("latticeError");
  if (el) el.classList.add("hidden");
}

function _setText(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

async function _get(path) {
  const resp = await fetch(API_BASE + path);
  if (!resp.ok) {
    const detail = await resp.text().catch(() => resp.statusText);
    throw new Error(`GET ${path} → ${resp.status}: ${detail}`);
  }
  return resp.json();
}

async function _post(path, body) {
  const resp = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const detail = await resp.text().catch(() => resp.statusText);
    throw new Error(`POST ${path} → ${resp.status}: ${detail}`);
  }
  return resp.json();
}

function _renderProducerLanes(lanes) {
  const container = document.getElementById("producerLaneList");
  if (!container) return;
  container.innerHTML = lanes
    .map((lane) => {
      const isLive = lane.status === "live";
      const dot = isLive
        ? '<span class="inline-block w-2 h-2 rounded-full bg-emerald-500 mr-1"></span>'
        : '<span class="inline-block w-2 h-2 rounded-full bg-gray-600 mr-1"></span>';
      return `
        <div class="bg-gray-900 border border-gray-800 rounded-lg p-2 flex flex-col gap-0.5">
          <div class="text-xs font-semibold text-gray-200">${dot}${_esc(lane.lane_id)}</div>
          <div class="text-[10px] text-gray-500">${_esc(lane.producer_id)}</div>
          <div class="text-[10px] text-gray-600">${_esc(lane.status)}</div>
        </div>
      `;
    })
    .join("");
}

function _renderProofChain(chain) {
  if (!chain) {
    _setText("m3Body", '<span class="text-red-400">No data</span>');
    return;
  }

  const bus = chain.bus_summary || {};

  _setText("m3Status", _ts(bus.observed_at));
  _setText(
    "m3Body",
    `bus_health: <b>${_fmt(bus.bus_health)}</b> &nbsp;|&nbsp;
     transport_pressure: <b>${_fmt(bus.transport_pressure)}</b> &nbsp;|&nbsp;
     contract_pressure: <b>${_fmt(bus.contract_pressure)}</b><br>
     catalog_drift_pressure: <b>${_fmt(bus.catalog_drift_pressure)}</b> &nbsp;|&nbsp;
     observer_failure_pressure: <b>${_fmt(bus.observer_failure_pressure)}</b><br>
     delivery_confidence: <b>${_fmt(bus.delivery_confidence)}</b>`
  );

  const fv = chain.field_vector || {};
  const fvKeys = Object.keys(fv);
  _setText("m4Status", fvKeys.length ? "present" : "—");
  _setText(
    "m4Body",
    fvKeys.length
      ? fvKeys.map((k) => `${_esc(k)}: <b>${_fmt(fv[k])}</b>`).join(" &nbsp;|&nbsp; ")
      : "capability:transport vector not present in field state"
  );

  const attn = chain.attention || {};
  const capTargets = attn.capability_targets || [];
  const dominated = capTargets.includes("capability:transport");
  _setText("m5Status", _ts(attn.generated_at));
  _setText(
    "m5Body",
    `dominant_targets: ${_esc((attn.dominant_targets || []).join(", ") || "—")}<br>
     capability:transport in bucket: <b class="${dominated ? "text-emerald-400" : "text-gray-400"}">${dominated ? "yes" : "no"}</b>`
  );

  const ss = chain.self_state || {};
  const ti = ss.transport_integrity || {};
  _setText("l6Status", _ts(ss.generated_at));
  _setText(
    "l6Body",
    `condition: <b>${_esc(ss.overall_condition)}</b> &nbsp;|&nbsp;
     transport_integrity score: <b>${_fmt(ti.score)}</b> confidence: <b>${_fmt(ti.confidence)}</b>`
  );

  const props = chain.proposals || {};
  _setText("l7Status", _ts(props.generated_at));
  _setText(
    "l7Body",
    `total candidates: <b>${_fmt(props.count)}</b> &nbsp;|&nbsp; transport candidates: <b>${_fmt(props.transport_count)}</b>`
  );

  const pol = chain.policy || {};
  _setText("l8Status", _ts(pol.generated_at));
  _setText(
    "l8Body",
    `approved: <b>${_fmt(pol.approved_count)}</b> rejected: <b>${_fmt(pol.rejected_count)}</b> mode: <b>${_esc(pol.policy_mode)}</b>`
  );

  const disp = chain.dispatch || {};
  _setText("l9Status", _ts(disp.generated_at));
  _setText(
    "l9Body",
    `dispatch_mode: <b>${_esc(disp.dispatch_mode)}</b> dispatched: <b>${_fmt(disp.dispatch_count)}</b> blocked: <b>${_fmt(disp.blocked_count)}</b>`
  );

  const fb = chain.feedback || {};
  _setText("l10Status", _ts(fb.generated_at));
  _setText(
    "l10Body",
    `outcome_status: <b>${_esc(fb.outcome_status)}</b> feedback_kind: <b>${_esc(fb.feedback_kind)}</b>`
  );

  const motifs = chain.motifs || [];
  _setText("l11Status", `${motifs.length} motif(s)`);
  _setText(
    "l11Body",
    motifs.length
      ? motifs
          .map(
            (m) =>
              `<span class="bg-gray-800 rounded px-1">${_esc(m.label || m.motif_id)}</span> ×${m.recurrence_count || "?"}`
          )
          .join(" ")
      : "—"
  );
}

function _renderLatticeValues(chain) {
  const el = document.getElementById("latticeValueBody");
  if (!el) return;
  if (!chain) {
    el.innerHTML = '<span class="text-red-400">No data</span>';
    return;
  }
  const bus = chain.bus_summary || {};
  const dispatch = chain.dispatch || {};
  const props = chain.proposals || {};
  const pol = chain.policy || {};
  const fb = chain.feedback || {};
  const rows = [
    ["bus_health", bus.bus_health],
    ["transport_pressure", bus.transport_pressure],
    ["contract_pressure", bus.contract_pressure],
    ["catalog_drift_pressure", bus.catalog_drift_pressure],
    ["observer_failure_pressure", bus.observer_failure_pressure],
    ["delivery_confidence", bus.delivery_confidence],
    ["dispatch_mode", dispatch.dispatch_mode],
    ["proposal_count", props.count],
    ["transport_proposals", props.transport_count],
    ["approved_count", pol.approved_count],
    ["feedback_outcome", fb.outcome_status],
  ];
  el.innerHTML = rows
    .map(
      ([k, v]) =>
        `<div class="flex justify-between gap-2"><span class="text-gray-500">${k}</span><span class="font-mono">${_fmt(v)}</span></div>`
    )
    .join("");
}

function _renderGates(gateData) {
  const el = document.getElementById("gateList");
  if (!el) return;
  if (!gateData || !gateData.gates) {
    el.innerHTML = '<span class="text-red-400">No gate data</span>';
    return;
  }
  el.innerHTML = gateData.gates
    .map(
      (g) =>
        `<div class="flex justify-between gap-2">
           <span class="text-gray-500">${_esc(g.gate_id)}</span>
           <span class="${_gateColor(g.state)}" title="${_esc(g.reason)}">${_esc(g.state)}</span>
         </div>`
    )
    .join("");
}

async function _runSimulate() {
  const contractWatchAt = parseFloat(document.getElementById("simContractWatchAt")?.value || "0.50");
  const transportWatchAt = parseFloat(document.getElementById("simTransportWatchAt")?.value || "0.25");
  _lastSimThresholds = {
    contract_pressure_watch_at: contractWatchAt,
    transport_pressure_watch_at: transportWatchAt,
  };

  try {
    const result = await _post("/api/substrate-lattice/transport/simulate", {
      lane_id: "transport",
      thresholds: _lastSimThresholds,
    });
    const el = document.getElementById("simResult");
    if (!el) return;
    el.classList.remove("hidden");
    const changed = result.changed;
    el.innerHTML = `
      <div class="grid grid-cols-2 gap-x-4 gap-y-0.5">
        <span class="text-gray-500">bucket</span>
        <span class="${changed ? "text-amber-300" : "text-gray-300"}">
          ${_esc(result.current?.bucket)} → ${_esc(result.simulated?.bucket)}
        </span>
        <span class="text-gray-500">salience</span>
        <span class="${changed ? "text-amber-300" : "text-gray-300"}">
          ${_fmt(result.current?.salience)} → ${_fmt(result.simulated?.salience)}
        </span>
        <span class="text-gray-500">action ceiling</span>
        <span class="${changed ? "text-amber-300" : "text-gray-300"}">
          ${_esc(result.current?.action_ceiling)} → ${_esc(result.simulated?.action_ceiling)}
        </span>
      </div>
      <div class="text-[10px] mt-1 ${changed ? "text-amber-400" : "text-gray-500"}">
        ${changed ? "⚠ outcome would change" : "✓ no change"}
      </div>
    `;
  } catch (err) {
    const el = document.getElementById("simResult");
    if (el) {
      el.classList.remove("hidden");
      el.innerHTML = `<span class="text-red-400">${_esc(err.message)}</span>`;
    }
  }
}

async function _runDraftPatch() {
  if (!_lastSimThresholds) {
    const el = document.getElementById("draftPatchOutput");
    if (el) {
      el.classList.remove("hidden");
      el.textContent = "Run simulation first to set candidate thresholds.";
    }
    return;
  }
  try {
    const result = await _post("/api/substrate-lattice/transport/draft-policy-patch", {
      lane_id: "transport",
      thresholds: _lastSimThresholds,
    });
    const el = document.getElementById("draftPatchOutput");
    if (!el) return;
    el.classList.remove("hidden");
    el.textContent = result.diff || "(no changes)";
  } catch (err) {
    const el = document.getElementById("draftPatchOutput");
    if (el) {
      el.classList.remove("hidden");
      el.textContent = `Error: ${err.message}`;
    }
  }
}

async function _loadAll() {
  _clearError();
  try {
    const [lanes, chain, gates] = await Promise.all([
      _get("/api/substrate-lattice/lanes"),
      _get("/api/substrate-lattice/transport/latest").catch(() => null),
      _get("/api/substrate-lattice/transport/gates").catch(() => null),
    ]);

    _renderProducerLanes(lanes || []);
    _renderProofChain(chain);
    _renderLatticeValues(chain);
    _renderGates(gates);

    const ts = document.getElementById("latticeLastUpdated");
    if (ts) ts.textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    _showError(`Load error: ${err.message}`);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  _loadAll();
  document.getElementById("latticeRefresh")?.addEventListener("click", _loadAll);
  document.getElementById("simRunBtn")?.addEventListener("click", _runSimulate);
  document.getElementById("draftPatchBtn")?.addEventListener("click", _runDraftPatch);
});
