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

function _statusBadge(status) {
  const map = {
    fresh: "bg-emerald-900/40 text-emerald-300 border-emerald-700",
    stale: "bg-amber-900/40 text-amber-300 border-amber-700",
    missing: "bg-gray-800/40 text-gray-500 border-gray-700",
    inconsistent: "bg-red-900/40 text-red-300 border-red-700",
  };
  const cls = map[status] || map.missing;
  return `<span class="text-[9px] border rounded px-1 py-0.5 ${cls}">${_esc(status || "—")}</span>`;
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

function _renderTargetList(targets, sectionLabel) {
  if (!targets || targets.length === 0) return "";
  const rows = targets
    .map((t) => {
      const isCapTransport = t.target_id === "capability:transport";
      const chs = (t.dominant_channels || []).join(", ") || "—";
      const reasons = (t.reasons || []).join("; ") || "—";
      return `
        <div class="border border-gray-700 rounded p-1.5 flex flex-col gap-0.5 ${isCapTransport ? "border-emerald-700 bg-emerald-950/20" : ""}">
          <div class="flex items-center gap-1">
            <span class="font-mono text-[10px] ${isCapTransport ? "text-emerald-300" : "text-gray-300"}">${_esc(t.target_id)}</span>
            <span class="text-[9px] text-gray-500">${_esc(t.bucket)}</span>
          </div>
          <div class="text-[10px] text-gray-400">
            salience: <b>${t.salience_score !== null && t.salience_score !== undefined ? t.salience_score.toFixed(3) : "—"}</b>
            &nbsp;mode: <b>${_esc(t.suggested_observation_mode)}</b>
          </div>
          <div class="text-[10px] text-gray-500">channels: ${_esc(chs)}</div>
          <div class="text-[10px] text-gray-500">reasons: ${_esc(reasons)}</div>
        </div>`;
    })
    .join("");
  return `<div class="text-[10px] font-semibold text-gray-400 mt-1">${sectionLabel}</div><div class="flex flex-col gap-1">${rows}</div>`;
}

function _renderProofChain(chain) {
  if (!chain) {
    _setText("m3Body", '<span class="text-red-400">No data</span>');
    return;
  }

  // M3
  const m3 = chain.transport?.m3 || {};
  const bus = m3.values || {};
  _setText("m3Status", `${_statusBadge(m3.status)} ${_ts(m3.timestamp)}`);
  _setText(
    "m3Body",
    `bus_health: <b>${_fmt(bus.bus_health)}</b> &nbsp;|&nbsp;
     transport_pressure: <b>${_fmt(bus.transport_pressure)}</b> &nbsp;|&nbsp;
     contract_pressure: <b>${_fmt(bus.contract_pressure)}</b><br>
     catalog_drift_pressure: <b>${_fmt(bus.catalog_drift_pressure)}</b> &nbsp;|&nbsp;
     observer_failure_pressure: <b>${_fmt(bus.observer_failure_pressure)}</b><br>
     delivery_confidence: <b>${_fmt(bus.delivery_confidence)}</b>`
  );

  // M4
  const m4 = chain.transport?.m4 || {};
  const fv = m4.values?.field_vector || {};
  const fvKeys = Object.keys(fv);
  _setText("m4Status", `${_statusBadge(m4.status)} ${_ts(m4.timestamp)}`);
  _setText(
    "m4Body",
    fvKeys.length
      ? fvKeys.map((k) => `${_esc(k)}: <b>${_fmt(fv[k])}</b>`).join(" &nbsp;|&nbsp; ")
      : "capability:transport vector not present in field state"
  );

  // M5
  const m5 = chain.transport?.m5 || {};
  const attn = m5.values || {};
  const capBucket = attn.capability_transport_bucket;
  _setText("m5Status", `${_statusBadge(m5.status)} ${_ts(m5.timestamp)}`);

  const m5TargetsHtml =
    _renderTargetList(attn.dominant_targets || [], "dominant") +
    _renderTargetList(attn.capability_targets || [], "capability") +
    _renderTargetList(attn.suppressed_targets || [], "suppressed");

  const capTransportHtml = capBucket
    ? `<span class="text-emerald-400">capability:transport found in ${_esc(capBucket)}</span>`
    : '<span class="text-gray-500">capability:transport not found in any bucket</span>';

  _setText(
    "m5Body",
    `<div class="mb-1">${capTransportHtml}</div>${m5TargetsHtml || '<span class="text-gray-500">no targets</span>'}`
  );

  // L6
  const l6 = chain.transport?.l6 || {};
  const ss = l6.values || {};
  const ti = ss.transport_integrity || {};
  _setText("l6Status", `${_statusBadge(l6.status)} ${_ts(l6.timestamp)}`);
  _setText(
    "l6Body",
    `condition: <b>${_esc(ss.overall_condition)}</b> &nbsp;|&nbsp; transport_integrity score: <b>${_fmt(ti.score)}</b> confidence: <b>${_fmt(ti.confidence)}</b>`
  );

  // L7
  const l7 = chain.transport?.l7 || {};
  const props = l7.values || {};
  _setText("l7Status", `${_statusBadge(l7.status)} ${_ts(l7.timestamp)}`);
  _setText(
    "l7Body",
    `total candidates: <b>${_fmt(props.count)}</b> &nbsp;|&nbsp; transport candidates: <b>${_fmt(props.transport_count)}</b>`
  );

  // L8
  const l8 = chain.transport?.l8 || {};
  const pol = l8.values || {};
  _setText("l8Status", `${_statusBadge(l8.status)} ${_ts(l8.timestamp)}`);
  _setText(
    "l8Body",
    `approved: <b>${_fmt(pol.approved_count)}</b> rejected: <b>${_fmt(pol.rejected_count)}</b> mode: <b>${_esc(pol.policy_mode)}</b>`
  );

  // L9
  const l9 = chain.transport?.l9 || {};
  const disp = l9.values || {};
  _setText("l9Status", `${_statusBadge(l9.status)} ${_ts(l9.timestamp)}`);
  _setText(
    "l9Body",
    `dispatch_mode: <b>${_esc(disp.dispatch_mode)}</b> dispatched: <b>${_fmt(disp.dispatch_count)}</b> blocked: <b>${_fmt(disp.blocked_count)}</b>`
  );

  // L10
  const l10 = chain.transport?.l10 || {};
  const fb = l10.values || {};
  _setText("l10Status", `${_statusBadge(l10.status)} ${_ts(l10.timestamp)}`);
  _setText(
    "l10Body",
    `outcome_status: <b>${_esc(fb.outcome_status)}</b> feedback_kind: <b>${_esc(fb.feedback_kind)}</b>`
  );

  // L11
  const l11 = chain.transport?.l11 || {};
  const motifs = (l11.values || {}).motifs || [];
  _setText("l11Status", `${_statusBadge(l11.status)} ${_ts(l11.timestamp)}`);
  if (l11.status === "stale") {
    _setText(
      "l11Body",
      `<span class="text-amber-400">stale (${_ts(l11.timestamp)})</span> — ${motifs.length} motif(s) from last consolidation: ${
        motifs
          .map((m) => `<span class="bg-gray-800 rounded px-1">${_esc(m.label || m.motif_id)}</span>`)
          .join(" ")
      }`
    );
  } else if (motifs.length === 0) {
    _setText(
      "l11Body",
      l11.status === "missing"
        ? '<span class="text-gray-500">no consolidation frame found</span>'
        : "— no motifs observed"
    );
  } else {
    _setText(
      "l11Body",
      motifs
        .map((m) => {
          const strength =
            m.strength !== null && m.strength !== undefined
              ? ` strength=${m.strength.toFixed(2)}`
              : "";
          const recurrence = m.recurrence_count ? ` ×${m.recurrence_count}` : "";
          const ts = m.timestamp ? ` (${_ts(m.timestamp)})` : "";
          return `<span class="bg-gray-800 rounded px-1">${_esc(m.label || m.motif_id)}</span>${strength}${recurrence}${ts}`;
        })
        .join(" ")
    );
  }
}

function _renderLatticeValues(chain) {
  const el = document.getElementById("latticeValueBody");
  if (!el) return;
  if (!chain || !chain.transport?.m3) {
    el.innerHTML = '<span class="text-red-400">No data</span>';
    return;
  }
  const bus = chain.transport.m3.values || {};
  const channels = [
    { id: "transport_pressure", label: "transport_pressure", weight: 0.35, watchAt: 0.25, ceiling: "read_only" },
    { id: "contract_pressure", label: "contract_pressure", weight: 0.30, watchAt: 0.50, ceiling: "summarize" },
    { id: "catalog_drift_pressure", label: "catalog_drift_pressure", weight: 0.15, watchAt: 0.50, ceiling: "watch" },
    { id: "observer_failure_pressure", label: "observer_failure_pressure", weight: 0.20, watchAt: 0.25, ceiling: "summarize" },
  ];

  el.innerHTML = channels
    .map((ch) => {
      const val = typeof bus[ch.id] === "number" ? bus[ch.id] : null;
      const passes = val !== null && val >= ch.watchAt;
      const passClass =
        val === null ? "text-gray-500" : passes ? "text-amber-400" : "text-emerald-400";
      const passLabel = val === null ? "?" : passes ? "WATCH" : "quiet";
      const contribution = val !== null && passes ? (val * ch.weight).toFixed(3) : "—";
      return `
        <div class="border border-gray-800 rounded p-2 flex flex-col gap-1" data-channel="${ch.id}">
          <div class="flex items-center justify-between">
            <span class="font-mono text-[10px] text-gray-300">${ch.label}</span>
            <span class="text-[10px] font-semibold ${passClass}">${passLabel}</span>
          </div>
          <div class="grid grid-cols-2 gap-x-2 text-[10px] text-gray-400">
            <span>value: <b class="text-gray-200">${val !== null ? val.toFixed(3) : "—"}</b></span>
            <span>watch_at: <b>${ch.watchAt}</b></span>
            <span>contribution: <b>${contribution}</b></span>
            <span>ceiling: <b>${ch.ceiling}</b></span>
          </div>
          <div class="flex gap-1 mt-1">
            <button class="lattice-judgment text-[9px] rounded px-1.5 py-0.5 bg-gray-800 hover:bg-gray-700 border border-gray-700" data-channel="${ch.id}" data-judgment="too_loud">Too Loud</button>
            <button class="lattice-judgment text-[9px] rounded px-1.5 py-0.5 bg-emerald-900/40 hover:bg-emerald-900/60 border border-emerald-800" data-channel="${ch.id}" data-judgment="right">✓ Right</button>
            <button class="lattice-judgment text-[9px] rounded px-1.5 py-0.5 bg-gray-800 hover:bg-gray-700 border border-gray-700" data-channel="${ch.id}" data-judgment="too_quiet">Too Quiet</button>
            <button class="lattice-judgment text-[9px] rounded px-1.5 py-0.5 bg-amber-900/30 hover:bg-amber-900/50 border border-amber-800" data-channel="${ch.id}" data-judgment="wrong_attribution">Wrong</button>
          </div>
        </div>
      `;
    })
    .join("");

  el.querySelectorAll(".lattice-judgment").forEach((btn) => {
    btn.addEventListener("click", () =>
      _recordJudgment(btn.dataset.channel, btn.dataset.judgment)
    );
  });
}

function _recordJudgment(channelId, judgment) {
  const el = document.getElementById("judgmentFeedback");
  if (el) {
    el.textContent = `Noted: ${channelId} → ${judgment.replace(/_/g, " ")}`;
    el.classList.remove("hidden");
    setTimeout(() => el.classList.add("hidden"), 3000);
  }
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
  const contractWatchAt = parseFloat(
    document.getElementById("simContractWatchAt")?.value || "0.50"
  );
  const transportWatchAt = parseFloat(
    document.getElementById("simTransportWatchAt")?.value || "0.25"
  );
  const catalogWatchAt = parseFloat(
    document.getElementById("simCatalogWatchAt")?.value || "0.50"
  );
  const observerWatchAt = parseFloat(
    document.getElementById("simObserverWatchAt")?.value || "0.25"
  );
  _lastSimThresholds = {
    contract_pressure_watch_at: contractWatchAt,
    transport_pressure_watch_at: transportWatchAt,
    catalog_drift_pressure_watch_at: catalogWatchAt,
    observer_failure_pressure_watch_at: observerWatchAt,
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
      <div class="text-[10px] font-semibold text-gray-400 mb-1">Salience / Bucket / Action Ceiling</div>
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

    const verdictEl = document.getElementById("verdictBanner");
    if (verdictEl && chain) {
      const verdict = chain.verdict || "";
      const isInconsistent = verdict.toLowerCase().includes("inconsistent");
      const isStale = verdict.toLowerCase().includes("stale");
      const bannerClass = isInconsistent
        ? "bg-red-950/40 border-red-700 text-red-200"
        : isStale
        ? "bg-amber-950/40 border-amber-700 text-amber-200"
        : "bg-emerald-950/30 border-emerald-800 text-emerald-200";
      verdictEl.className = `border rounded p-3 text-xs ${bannerClass}`;
      verdictEl.textContent = verdict;
      verdictEl.classList.remove("hidden");
    }

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
