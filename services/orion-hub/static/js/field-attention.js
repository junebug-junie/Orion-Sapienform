// Field Attention operator tab.
//
// Read-only view of FieldAttentionFrameV1 -- the substrate field-attention
// system (Candidate A precision-weighted prediction-error salience +
// Candidate B Global-Workspace novelty salience), built by
// orion/attention/field_attention/ and shipped in PR #1484/#1488. This is a
// DIFFERENT subsystem from attention-organ.js's Attention Organ tab, which
// visualizes orion-heartbeat's N-trajectory dissipation ensemble (AST/HOT) --
// do not conflate the two, and do not reuse "attention-organ" naming here.
//
// Backed entirely by GET /api/substrate/attention/latest
// (services/orion-hub/scripts/substrate_attention_routes.py, already
// registered on the Hub API router -- this file does not add or change any
// route). This module owns rendering and the poll lifecycle only; every
// number it draws comes straight from that one real endpoint, and nothing
// here writes back (no POST anywhere in this file).
//
// Panel show/hide is app.js's setActiveTab; this module exposes
// activate()/deactivate() so polling runs ONLY while the tab is visible --
// same lifecycle contract as attention-organ.js, including its guard against
// a panel hidden by a path other than setActiveTab (self_observability.js
// hides every section[data-panel] directly and never calls deactivate()).
(function () {
  "use strict";

  var LATEST_URL = "/api/substrate/attention/latest";

  // Production ticks every ~2s (verified live 2026-07-30 against persisted
  // substrate_attention_frames rows). 5s is a light multiple of that, the
  // same default attention-organ.js uses for its own poll timer.
  var POLL_MS = 5000;

  // Design-verified 2026-07-30 (see index.html's on-panel note and this
  // file's renderHeader/confidenceBadge): confidence_score takes only 0.0 or
  // 1.0 across 10,747 recent target-observations, so >= 0.5 is a safe binary
  // split rather than a real threshold decision.
  var CONFIDENCE_GROUNDED_THRESHOLD = 0.5;

  var state = {
    active: false,
    timer: null,
    inFlight: false,
    lastFetchedAt: 0,
    lastFrame: null,
  };

  var els = {};

  function $(id) {
    return document.getElementById(id);
  }

  function bindElements() {
    els.panel = $("field-attention");
    els.status = $("fieldAttentionStatus");
    els.header = $("fieldAttentionHeader");
    els.dominant = $("fieldAttentionDominantTargets");
    els.suppressed = $("fieldAttentionSuppressedTargets");
    els.bottom = $("fieldAttentionBottomStrip");
    els.refreshBtn = $("fieldAttentionRefreshBtn");
    return !!els.panel;
  }

  // ---------------------------------------------------------------- helpers

  function num(value, digits) {
    if (value === null || value === undefined || value === "") return "—";
    var parsed = Number(value);
    if (!isFinite(parsed)) return "—";
    return parsed.toFixed(digits === undefined ? 4 : digits);
  }

  function age(seconds) {
    if (seconds === null || seconds === undefined) return "—";
    var s = Number(seconds);
    if (!isFinite(s)) return "—";
    if (s < 0) return s > -2 ? "0.0s" : s.toFixed(1) + "s";
    if (s < 90) return s.toFixed(1) + "s";
    if (s < 5400) return (s / 60).toFixed(1) + "m";
    if (s < 172800) return (s / 3600).toFixed(1) + "h";
    return (s / 86400).toFixed(1) + "d";
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function badge(text, tone, title) {
    var node = el("span", "inline-block px-2 py-0.5 rounded-full text-[10px] font-semibold border " + tone, text);
    if (title) node.title = title;
    return node;
  }

  function note(host, text, tone) {
    host.appendChild(el("p", "text-[10px] leading-relaxed mt-2 " + (tone || "text-gray-500"), text));
  }

  function errorBlock(host, title, message) {
    host.textContent = "";
    host.appendChild(el("h3", "text-sm font-semibold text-gray-100 mb-2", title));
    host.appendChild(
      el(
        "div",
        "text-[11px] font-mono text-red-300 bg-red-950/30 border border-red-900/60 rounded-lg p-2 break-all",
        message || "no data"
      )
    );
  }

  // Candidate A's reason strings start with "precision-weighted
  // prediction-error salience" (orion/attention/field_attention/selectors.py
  // select_node_targets()); Candidate B's start with "Candidate B
  // novelty-only salience" (same file's _novelty_targets()). A target whose
  // reasons match neither (e.g. the system "recent field perturbation
  // count" target) gets no candidate badge rather than a guessed one.
  var CANDIDATE_A_PREFIX = "precision-weighted prediction-error salience";
  var CANDIDATE_B_PREFIX = "Candidate B novelty-only salience";

  function candidateBadgeFor(reasons) {
    var first = Array.isArray(reasons) && reasons.length ? String(reasons[0]) : "";
    if (first.indexOf(CANDIDATE_A_PREFIX) === 0) {
      return badge("A", "border-sky-700 bg-sky-950/50 text-sky-200", "Candidate A -- precision-weighted prediction-error salience");
    }
    if (first.indexOf(CANDIDATE_B_PREFIX) === 0) {
      return badge("B", "border-violet-700 bg-violet-950/50 text-violet-200", "Candidate B -- Global-Workspace novelty salience");
    }
    return badge("—", "border-gray-700 bg-gray-900 text-gray-500", "No candidate reason prefix matched");
  }

  // Rendered as a two-state badge, not a progress bar: confidence_score is
  // binary in real data (verified live 2026-07-30, 10,747 observations, only
  // ever exactly 0.0 or exactly 1.0), so a gradient would imply a precision
  // this signal does not have.
  function confidenceBadge(value) {
    var v = Number(value);
    var grounded = isFinite(v) && v >= CONFIDENCE_GROUNDED_THRESHOLD;
    return badge(
      grounded ? "grounded" : "cold-start",
      grounded
        ? "border-emerald-700 bg-emerald-950/50 text-emerald-200"
        : "border-amber-700 bg-amber-950/50 text-amber-200",
      "confidence_score=" + num(value, 2)
    );
  }

  function observationModeBadge(mode) {
    var tones = {
      inspect: "border-red-700 bg-red-950/50 text-red-200",
      summarize: "border-amber-700 bg-amber-950/50 text-amber-200",
      watch: "border-sky-700 bg-sky-950/50 text-sky-200",
      ignore: "border-gray-700 bg-gray-900 text-gray-500",
    };
    return badge(mode || "watch", tones[mode] || tones.watch);
  }

  // ---------------------------------------------------------------- renderers

  function renderHeader(host, frame) {
    host.textContent = "";
    var generatedAt = frame.generated_at ? Date.parse(frame.generated_at) : NaN;
    var ageSec = isFinite(generatedAt) ? (Date.now() - generatedAt) / 1000 : null;

    var top = el("div", "flex flex-wrap items-baseline justify-between gap-3");
    var left = el("div", "flex flex-wrap items-baseline gap-4");
    left.appendChild(
      el(
        "div",
        "text-[11px] font-mono " + (ageSec !== null && ageSec > 30 ? "text-amber-300" : "text-gray-300"),
        "frame age " + age(ageSec)
      )
    );
    left.appendChild(el("div", "text-[11px] font-mono text-gray-400", "policy " + (frame.attention_policy_id || "—")));
    left.appendChild(el("div", "text-[11px] font-mono text-gray-400", "tick " + (frame.source_field_tick_id || "—")));
    top.appendChild(left);

    var right = el("div", "text-right");
    right.appendChild(el("div", "text-lg font-mono text-gray-200 leading-none", num(frame.overall_salience, 3)));
    right.appendChild(el("div", "text-[9px] uppercase tracking-wide text-gray-500 mt-0.5", "overall_salience (secondary)"));
    top.appendChild(right);
    host.appendChild(top);

    note(
      host,
      "overall_salience is the max of this tick's already per-tick min-max-normalized target scores " +
        "(normalize_across_targets() / build_attention_frame()), so it reads ≈1.0 on almost every tick whenever " +
        "any node target competes -- it is not a “how much is going on” gauge. Read the ranked target table below instead."
    );
  }

  function renderReasonsCell(reasons) {
    var wrap = el("details", "text-[10px]");
    var summary = el(
      "summary",
      "text-gray-400 cursor-pointer",
      (Array.isArray(reasons) ? reasons.length : 0) + " reason(s)"
    );
    wrap.appendChild(summary);
    var body = el("div", "mt-1 text-gray-300 whitespace-pre-wrap break-words max-w-md");
    body.textContent = Array.isArray(reasons) && reasons.length ? reasons.join("\n") : "—";
    wrap.appendChild(body);
    return wrap;
  }

  function renderTargetsTable(host, title, subtitle, targets, opts) {
    opts = opts || {};
    host.textContent = "";
    var head = el("div", "flex items-baseline justify-between gap-2 mb-3");
    head.appendChild(el("h3", "text-sm font-semibold " + (opts.muted ? "text-gray-400" : "text-gray-100"), title));
    if (subtitle) {
      head.appendChild(el("span", "text-[10px] uppercase tracking-wide text-gray-500", subtitle));
    }
    host.appendChild(head);

    if (!targets || !targets.length) {
      note(host, opts.emptyText || "No targets in this list right now.");
      return;
    }

    var table = el("table", "w-full text-[11px] border-collapse" + (opts.muted ? " opacity-70" : ""));
    var thead = el("thead");
    var headRow = el("tr", "text-left text-gray-500 border-b border-gray-800");
    ["target", "kind", "candidate", "magnitude", "confidence", "mode", "reasons"].forEach(function (label) {
      headRow.appendChild(el("th", "font-normal uppercase tracking-wide text-[9px] py-1 pr-3", label));
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = el("tbody");
    targets.forEach(function (target) {
      var row = el("tr", "border-b border-gray-800/60 last:border-b-0 align-top");

      var idCell = el("td", "py-1.5 pr-3 font-mono text-gray-200 break-all", target.target_id);
      row.appendChild(idCell);

      row.appendChild(el("td", "py-1.5 pr-3 text-gray-400", target.target_kind));

      var candCell = el("td", "py-1.5 pr-3");
      candCell.appendChild(candidateBadgeFor(target.reasons));
      row.appendChild(candCell);

      var channels = target.dominant_channels || {};
      var magnitude =
        channels.prediction_error !== undefined && channels.prediction_error !== null
          ? num(channels.prediction_error, 4)
          : "—";
      row.appendChild(el("td", "py-1.5 pr-3 font-mono text-gray-300", magnitude));

      var confCell = el("td", "py-1.5 pr-3");
      confCell.appendChild(confidenceBadge(target.confidence_score));
      row.appendChild(confCell);

      var modeCell = el("td", "py-1.5 pr-3");
      modeCell.appendChild(observationModeBadge(target.suggested_observation_mode));
      row.appendChild(modeCell);

      var reasonsCell = el("td", "py-1.5 pr-3");
      reasonsCell.appendChild(renderReasonsCell(target.reasons));
      row.appendChild(reasonsCell);

      tbody.appendChild(row);
    });
    table.appendChild(tbody);
    host.appendChild(table);
  }

  function renderBottomStrip(host, frame) {
    host.textContent = "";
    host.appendChild(el("h3", "text-sm font-semibold text-gray-100 mb-2", "Recent perturbations & warnings"));

    var perturbations = frame.recent_perturbations || [];
    var warnings = frame.warnings || [];

    var pWrap = el("div", "mb-3");
    pWrap.appendChild(
      el("div", "text-[10px] uppercase tracking-wide text-gray-500 mb-1", "recent_perturbations (" + perturbations.length + ")")
    );
    if (perturbations.length) {
      var pList = el("ul", "text-[11px] text-gray-300 space-y-0.5 list-disc list-inside");
      perturbations.forEach(function (p) {
        pList.appendChild(el("li", "", String(p)));
      });
      pWrap.appendChild(pList);
    } else {
      pWrap.appendChild(el("div", "text-[11px] text-gray-500", "none"));
    }
    host.appendChild(pWrap);

    var wWrap = el("div");
    wWrap.appendChild(
      el("div", "text-[10px] uppercase tracking-wide text-gray-500 mb-1", "warnings (" + warnings.length + ")")
    );
    if (warnings.length) {
      var wList = el("ul", "text-[11px] text-amber-300 space-y-0.5 list-disc list-inside");
      warnings.forEach(function (w) {
        wList.appendChild(el("li", "", String(w)));
      });
      wWrap.appendChild(wList);
    } else {
      wWrap.appendChild(el("div", "text-[11px] text-gray-500", "none"));
    }
    host.appendChild(wWrap);
  }

  function renderFrame(frame) {
    renderHeader(els.header, frame);
    renderTargetsTable(
      els.dominant,
      "Dominant targets",
      "ranked by salience_score, backend-sorted/capped",
      frame.dominant_targets || [],
      { emptyText: "No active targets this tick." }
    );
    renderTargetsTable(
      els.suppressed,
      "Suppressed targets",
      "below the policy's suppress_below threshold -- still real competitors, muted",
      frame.suppressed_targets || [],
      { muted: true, emptyText: "Nothing suppressed this tick." }
    );
    renderBottomStrip(els.bottom, frame);
  }

  // ------------------------------------------------------------ poll cycle

  function setStatus(text, tone) {
    if (!els.status) return;
    els.status.textContent = text;
    els.status.className = "text-xs min-h-[1.25rem] " + (tone || "text-gray-400");
  }

  async function poll() {
    if (state.inFlight) return;
    state.inFlight = true;
    try {
      var resp = await fetch(LATEST_URL, { headers: { Accept: "application/json" } });
      if (resp.status === 404) {
        // Not an error -- the substrate simply has not written a frame yet.
        errorBlock(els.header, "Field Attention", "No FieldAttentionFrameV1 persisted yet.");
        if (els.dominant) els.dominant.textContent = "";
        if (els.suppressed) els.suppressed.textContent = "";
        if (els.bottom) els.bottom.textContent = "";
        setStatus("No frame available yet (HTTP 404) — " + new Date().toLocaleTimeString(), "text-amber-400");
        return;
      }
      if (!resp.ok) {
        throw new Error("HTTP " + resp.status + " from " + LATEST_URL);
      }
      var frame = await resp.json();
      state.lastFrame = frame;
      state.lastFetchedAt = Date.now();
      renderFrame(frame);
      setStatus("Updated " + new Date().toLocaleTimeString() + " · polling every " + POLL_MS / 1000 + "s", "text-gray-400");
    } catch (err) {
      setStatus("Failed to load: " + (err.message || err), "text-red-400");
    } finally {
      state.inFlight = false;
    }
  }

  function stopTimer() {
    if (state.timer !== null) {
      clearInterval(state.timer);
      state.timer = null;
    }
  }

  function startTimer() {
    stopTimer();
    if (!state.active) return;
    state.timer = setInterval(function () {
      // Visibility, not just the router's activate/deactivate calls, gates
      // polling -- self_observability.js hides every section[data-panel]
      // directly and preventDefault()s its own click, so app.js's
      // setActiveTab (the only caller of deactivate()) never runs on that
      // path. Checking the DOM here holds the contract against that case,
      // same guard attention-organ.js uses.
      if (!els.panel || els.panel.classList.contains("hidden")) {
        deactivate();
        return;
      }
      poll();
    }, POLL_MS);
  }

  function activate() {
    if (!els.panel && !bindElements()) return;
    state.active = true;
    poll();
    startTimer();
  }

  function deactivate() {
    state.active = false;
    stopTimer();
    // Rendered DOM is deliberately left in place, same as attention-organ.js
    // -- switching away and back shows the last frame immediately.
  }

  function wireControls() {
    if (els.refreshBtn) {
      els.refreshBtn.addEventListener("click", function () {
        poll();
      });
    }
  }

  function init() {
    if (!bindElements()) return;
    wireControls();
    // Defensive only: both scripts are `defer`, and app.js's DOMContentLoaded
    // handler (which calls setActiveTab) runs after this file finishes
    // loading, so the panel is still hidden here in practice. This branch
    // exists so a future load-order change cannot leave a visible panel
    // unpolled -- same defensive shape as attention-organ.js's init().
    if (!els.panel.classList.contains("hidden")) {
      activate();
    }
  }

  window.OrionFieldAttention = {
    activate: activate,
    deactivate: deactivate,
    refresh: function () {
      poll();
    },
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
