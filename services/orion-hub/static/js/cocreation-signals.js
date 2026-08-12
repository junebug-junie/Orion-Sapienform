// Cocreation Signals operator tab.
//
// Live view of the codebase-mass Predictive Processing domain
// (docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md):
//
//   orion-cocreation-signals (git/PR/graph deltas, real external I/O)
//     -> orion-substrate-runtime's codebase_prediction_error() scorer
//       -> substrate_codebase_mass_baseline (EWMA state) + substrate_codebase_delta_log (raw history)
//
// Backed entirely by /api/cocreation-signals/snapshot + /history. Panel
// show/hide is app.js's setActiveTab; this module exposes
// activate()/deactivate() so polling runs only while the tab is visible,
// following attention-organ.js's lifecycle contract. Rendered state is left
// in the DOM on deactivate.
//
// Ticks here are far slower than Attention Organ's (git ~60s only on a new
// commit, pr_lifecycle ~15min unconditional, graph ~5min) so this polls on a
// slower default cadence -- a 2s poll against data that moves every 15
// minutes would just be repeated identical reads.
(function () {
  "use strict";

  var SNAPSHOT_URL = "/api/cocreation-signals/snapshot";
  var HISTORY_URL = "/api/cocreation-signals/history";
  var DEV_ECONOMICS_URL = "/api/cocreation-signals/dev-economics";
  var SVG_NS = "http://www.w3.org/2000/svg";

  var HISTORY_REFRESH_MS = 30000;

  var DOMAIN_LABELS = {
    git: "git churn",
    pr_lifecycle: "PR lifecycle",
    graph: "graph structure",
  };

  // CodebaseMassBaseline.to_json_dict() (orion/substrate/prediction_error.py)
  // uses the short key "pr", not "pr_lifecycle" -- a real, deliberate
  // difference from CodebaseDeltaV1's wire field name that DOMAIN_LABELS
  // above is keyed on. Kept as its own map (not a rename) so each label set
  // matches the real key space of the table it renders, rather than forcing
  // one spelling and silently mislabeling the other.
  var BASELINE_DOMAIN_LABELS = {
    git: "git churn",
    pr: "PR lifecycle",
    graph: "graph structure",
  };

  var state = {
    active: false,
    timer: null,
    inFlight: false,
    lastSnapshotAt: 0,
    lastHistoryAt: 0,
    historyError: null,
    intervalMs: 15000,
    windowHours: 24,
    live: true,
  };

  var els = {};
  var lastHistory = null;

  function $(id) {
    return document.getElementById(id);
  }

  function bindElements() {
    els.panel = $("cocreation-signals");
    els.status = $("cocreationSignalsStatus");
    els.baseline = $("cocreationSignalsBaseline");
    els.domains = $("cocreationSignalsDomains");
    els.devEconomics = $("cocreationSignalsDevEconomics");
    els.windowSelect = $("cocreationSignalsWindow");
    els.intervalSelect = $("cocreationSignalsInterval");
    els.liveToggle = $("cocreationSignalsLive");
    els.refreshBtn = $("cocreationSignalsRefreshBtn");
    return !!els.panel;
  }

  // ---------------------------------------------------------------- helpers

  function num(value, digits) {
    if (value === null || value === undefined || value === "") return "—";
    var parsed = Number(value);
    if (!isFinite(parsed)) return "—";
    return parsed.toFixed(digits === undefined ? 4 : digits);
  }

  function int(value) {
    if (value === null || value === undefined || value === "") return "—";
    var parsed = Number(value);
    if (!isFinite(parsed)) return "—";
    return parsed.toLocaleString();
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

  function card(host, title, subtitle) {
    host.textContent = "";
    var head = el("div", "flex items-baseline justify-between gap-2 mb-3");
    head.appendChild(el("h3", "text-sm font-semibold text-gray-100", title));
    if (subtitle) {
      head.appendChild(el("span", "text-[10px] uppercase tracking-wide text-gray-500", subtitle));
    }
    host.appendChild(head);
    return host;
  }

  function statTile(label, value, tone) {
    var wrap = el("div", "rounded-lg border border-gray-800 bg-black/30 px-2 py-1.5");
    wrap.appendChild(el("div", "text-[10px] uppercase tracking-wide text-gray-500", label));
    wrap.appendChild(el("div", "text-sm font-mono " + (tone || "text-gray-100"), value));
    return wrap;
  }

  function kvRow(label, value, tone) {
    var row = el("div", "flex items-start justify-between gap-3 py-1 border-b border-gray-800/60 last:border-b-0");
    row.appendChild(el("div", "text-[11px] text-gray-400 shrink-0", label));
    row.appendChild(el("div", "text-[11px] font-mono text-right break-all " + (tone || "text-gray-200"), value));
    return row;
  }

  function badge(text, tone) {
    return el("span", "inline-block px-2 py-0.5 rounded-full text-[10px] font-semibold border " + tone, text);
  }

  function note(host, text, tone) {
    host.appendChild(el("p", "text-[10px] leading-relaxed mt-2 " + (tone || "text-gray-500"), text));
  }

  function errorBlock(host, title, message) {
    card(host, title, "unavailable");
    host.appendChild(
      el(
        "div",
        "text-[11px] font-mono text-red-300 bg-red-950/30 border border-red-900/60 rounded-lg p-2 break-all",
        message || "no data"
      )
    );
  }

  function svg(viewBox, heightClass) {
    var node = document.createElementNS(SVG_NS, "svg");
    node.setAttribute("viewBox", viewBox);
    node.setAttribute("preserveAspectRatio", "none");
    node.setAttribute("class", "w-full " + (heightClass || "h-16"));
    return node;
  }

  function svgEl(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach(function (key) {
      node.setAttribute(key, attrs[key]);
    });
    return node;
  }

  // Score is unbounded (a z-score-derived quantity, not clamped to [0,1] the
  // way ACTIVE_INFERENCE_DOMAINS' prediction_error is) -- render against the
  // window's own observed max rather than a fixed scale, since a fixed 0..1
  // scale would flatten every real point to the same pixel for this domain.
  function renderSeries(host, points, opts) {
    opts = opts || {};
    var withValue = points.filter(function (p) {
      return isFinite(p.score);
    });
    if (!withValue.length) {
      note(host, opts.emptyText || "No samples in this window yet.");
      return;
    }
    var chart = svg("0 0 100 40", opts.heightClass || "h-20");
    var maxScore = withValue.reduce(function (acc, p) {
      return Math.max(acc, p.score);
    }, 0);
    var hi = maxScore > 0 ? maxScore * 1.1 : 1;

    function y(value) {
      return 38 - (Math.max(0, Math.min(hi, value)) / hi) * 36;
    }

    var times = points.map(function (p) {
      var parsed = Date.parse(p.observed_at);
      return Number.isFinite(parsed) ? parsed : NaN;
    });
    var usable = times.filter(Number.isFinite);
    var min = usable.length ? Math.min.apply(null, usable) : 0;
    var max = usable.length ? Math.max.apply(null, usable) : 0;
    var span = max - min;

    function x(index) {
      if (span > 0 && Number.isFinite(times[index])) {
        return ((times[index] - min) / span) * 100;
      }
      return points.length > 1 ? (index / (points.length - 1)) * 100 : 0;
    }

    var circles = [];
    points.forEach(function (p, index) {
      if (!isFinite(p.score)) return;
      circles.push({ cx: x(index), cy: y(p.score), score: p.score });
    });
    if (circles.length > 1) {
      chart.appendChild(
        svgEl("polyline", {
          points: circles.map(function (c) {
            return c.cx + "," + c.cy;
          }).join(" "),
          fill: "none",
          stroke: opts.color || "#818cf8",
          "stroke-width": 0.8,
          "vector-effect": "non-scaling-stroke",
        })
      );
    }
    circles.forEach(function (c) {
      chart.appendChild(svgEl("circle", { cx: c.cx, cy: c.cy, r: 1.2, fill: opts.color || "#818cf8" }));
    });
    host.appendChild(chart);
    host.appendChild(
      el("div", "text-[9px] text-gray-500 font-mono", "n=" + withValue.length + " · max " + num(maxScore, 3))
    );
  }

  // ---------------------------------------------------------------- panels

  function renderBaseline(host, snapshot) {
    var baseline = snapshot.baseline || {};
    if (!baseline.ok || !baseline.model) {
      errorBlock(host, "EWMA baseline", baseline.error || "No substrate_codebase_mass_baseline row found yet.");
      return;
    }
    var model = baseline.model;
    card(host, "EWMA baseline", "substrate_codebase_mass_baseline, live");

    var grid = el("div", "grid grid-cols-3 gap-2");
    KNOWN(model).forEach(function (domain) {
      var sub = model[domain] || {};
      var wrap = el("div", "rounded-lg border border-gray-800 bg-black/30 px-2 py-1.5");
      wrap.appendChild(el("div", "text-[10px] uppercase tracking-wide text-gray-500", BASELINE_DOMAIN_LABELS[domain] || domain));
      wrap.appendChild(el("div", "text-sm font-mono text-gray-100", "ewma " + num(sub.ewma, 2)));
      wrap.appendChild(el("div", "text-[10px] font-mono text-gray-500", "var " + num(sub.variance, 2) + " · n=" + int(sub.n)));
      grid.appendChild(wrap);
    });
    host.appendChild(grid);

    var ageWrap = el("div", "mt-2");
    ageWrap.appendChild(
      kvRow(
        "baseline age",
        age(
          baseline.generated_at
            ? (Date.now() - Date.parse(baseline.generated_at)) / 1000
            : null
        )
      )
    );
    host.appendChild(ageWrap);

    note(
      host,
      "Three independent EWMA sub-baselines, one per domain -- what codebase_prediction_error() currently " +
        "treats as 'normal' git/PR/graph activity. n is real ticks folded in since this baseline first started, " +
        "not a window count."
    );
  }

  function KNOWN(model) {
    return ["git", "pr", "graph"].filter(function (key) {
      return model && Object.prototype.hasOwnProperty.call(model, key);
    });
  }

  function renderDomains(host, snapshot, history) {
    var domains = snapshot.domains || {};
    if (!domains.ok) {
      errorBlock(host, "Domain activity", domains.error);
      return;
    }
    card(host, "Domain activity", (history ? history.window_hours + "h window" : "…"));

    (domains.rows || []).forEach(function (row) {
      var wrap = el("div", "mb-4 last:mb-0");
      var head = el("div", "flex items-baseline justify-between gap-2 mb-1");
      var labelWrap = el("div", "flex items-center gap-1.5");
      labelWrap.appendChild(el("span", "text-[12px] text-gray-200", DOMAIN_LABELS[row.domain] || row.domain));
      if (!row.present) {
        labelWrap.appendChild(badge("no events yet", "border-gray-700 bg-gray-900 text-gray-500"));
      }
      head.appendChild(labelWrap);
      head.appendChild(
        el(
          "span",
          "text-[10px] font-mono text-gray-500",
          "latest " + num(row.latest_score, 3) + " · seen " + age(row.latest_age_sec)
        )
      );
      wrap.appendChild(head);

      var statsGrid = el("div", "grid grid-cols-3 gap-2 mb-2");
      statsGrid.appendChild(statTile("Events (window)", int(row.window_event_count)));
      statsGrid.appendChild(statTile("Avg score", num(row.window_avg_score, 3)));
      statsGrid.appendChild(statTile("Max score", num(row.window_max_score, 3)));
      wrap.appendChild(statsGrid);

      var seriesWrap = el("div");
      var points = (history && history.series && history.series[row.domain]) || [];
      renderSeries(seriesWrap, points, { emptyText: "No " + row.domain + " ticks in this window." });
      wrap.appendChild(seriesWrap);

      host.appendChild(wrap);
    });

    if (history && history.truncated) {
      note(
        host,
        "History truncated at the " + history.row_cap + "-row cap -- widen the window or wait for a shorter one.",
        "text-amber-400/80"
      );
    }
  }

  // dev_economics_ledger_log's first real Hub consumer -- own card, own
  // fetch, own error boundary (see cocreation_signals_routes.py's
  // dev_economics() endpoint docstring for why this isn't folded into
  // renderDomains's score/domain shape).
  function renderDevEconomics(host, data) {
    if (!data || !data.ok) {
      errorBlock(host, "Dev economics ledger", (data && data.error) || "unavailable");
      return;
    }
    card(host, "Dev economics ledger", "dev_economics_ledger_log, live");

    if (!data.present) {
      note(host, "No ticks yet -- producer disabled, or cold-started and waiting on its first real delta.");
      return;
    }

    var statsGrid = el("div", "grid grid-cols-3 gap-2 mb-2");
    statsGrid.appendChild(statTile("Ticks shown", int(data.window_tick_count)));
    statsGrid.appendChild(
      statTile(
        "Total cost",
        data.window_total_cost_usd === null ? "—" : "$" + num(data.window_total_cost_usd, 2)
      )
    );
    statsGrid.appendChild(statTile("Total tokens", int(data.window_total_tokens)));
    host.appendChild(statsGrid);

    host.appendChild(kvRow("latest tick", age(data.latest_age_sec)));
    if (data.latest && data.latest.unpriced_session_count > 0) {
      note(
        host,
        data.latest.unpriced_session_count + " session(s) in the latest tick had no priceable model -- excluded from that tick's cost, not fabricated as $0.",
        "text-amber-400/80"
      );
    }

    var list = el("div", "mt-2 space-y-1 max-h-64 overflow-y-auto");
    (data.recent_ticks || []).slice(0, 20).forEach(function (tick) {
      var row = el("div", "flex items-center justify-between gap-3 text-[11px] font-mono border-b border-gray-800/60 py-1 last:border-b-0");
      row.appendChild(el("span", "text-gray-500", tick.observed_at ? new Date(tick.observed_at).toLocaleTimeString() : "—"));
      row.appendChild(el("span", "text-gray-300", int(tick.session_count) + " sess"));
      row.appendChild(el("span", "text-gray-300", int(tick.total_tokens) + " tok"));
      row.appendChild(
        el(
          "span",
          "text-gray-100",
          tick.total_estimated_cost_usd === null ? "—" : "$" + num(tick.total_estimated_cost_usd, 4)
        )
      );
      list.appendChild(row);
    });
    host.appendChild(list);
  }

  // ------------------------------------------------------------ poll cycle

  function setStatus(text, tone) {
    if (!els.status) return;
    els.status.textContent = text;
    els.status.className = "text-xs min-h-[1.25rem] " + (tone || "text-gray-400");
  }

  function historyAgeBanner(host) {
    if (!state.historyError) return;
    var ageSec = state.lastHistoryAt ? (Date.now() - state.lastHistoryAt) / 1000 : null;
    host.appendChild(
      el(
        "div",
        "mb-2 rounded-lg border border-amber-800 bg-amber-950/30 px-2 py-1 text-[10px] text-amber-200",
        "History refresh failing (" +
          state.historyError +
          ") — showing data from " +
          (ageSec === null ? "an earlier fetch" : age(ageSec) + " ago") +
          ", not now."
      )
    );
  }

  async function fetchJson(url) {
    var resp = await fetch(url, { headers: { Accept: "application/json" } });
    if (!resp.ok) {
      throw new Error("HTTP " + resp.status + " from " + url);
    }
    return resp.json();
  }

  async function poll(opts) {
    opts = opts || {};
    if (state.inFlight) return;
    state.inFlight = true;
    try {
      var snapshot = await fetchJson(SNAPSHOT_URL);
      state.lastSnapshotAt = Date.now();

      var needHistory =
        opts.forceHistory || !lastHistory || Date.now() - state.lastHistoryAt >= HISTORY_REFRESH_MS;
      if (needHistory) {
        try {
          lastHistory = await fetchJson(HISTORY_URL + "?hours=" + state.windowHours);
          state.lastHistoryAt = Date.now();
          state.historyError = null;
        } catch (err) {
          state.historyError = String(err.message || err);
        }
      }

      if (state.historyError) historyAgeBanner(els.domains);
      renderBaseline(els.baseline, snapshot);
      renderDomains(els.domains, snapshot, lastHistory);

      var devEconomicsData = null;
      var devEconomicsError = null;
      try {
        devEconomicsData = await fetchJson(DEV_ECONOMICS_URL);
      } catch (err) {
        devEconomicsError = String(err.message || err);
      }
      if (els.devEconomics) {
        renderDevEconomics(els.devEconomics, devEconomicsData || { ok: false, error: devEconomicsError });
      }

      var problems = [];
      if (!(snapshot.baseline || {}).ok) problems.push("baseline");
      if (!(snapshot.domains || {}).ok) problems.push("domains");
      if (state.historyError) problems.push("history");
      if (devEconomicsError || !(devEconomicsData || {}).ok) problems.push("dev-economics");
      if (problems.length) {
        setStatus(
          "Updated " + new Date().toLocaleTimeString() + " — degraded source(s): " + problems.join(", "),
          "text-amber-400"
        );
      } else {
        setStatus(
          "Updated " +
            new Date().toLocaleTimeString() +
            " · polling every " +
            state.intervalMs / 1000 +
            "s" +
            (state.live ? "" : " (paused)"),
          "text-gray-400"
        );
      }
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
    if (!state.active || !state.live) return;
    state.timer = setInterval(function () {
      if (!els.panel || els.panel.classList.contains("hidden")) {
        deactivate();
        return;
      }
      poll();
    }, state.intervalMs);
  }

  function activate() {
    if (!els.panel && !bindElements()) return;
    state.active = true;
    poll({ forceHistory: true });
    startTimer();
  }

  function deactivate() {
    state.active = false;
    stopTimer();
  }

  function wireControls() {
    if (els.windowSelect) {
      els.windowSelect.addEventListener("change", function () {
        state.windowHours = Number(els.windowSelect.value) || 24;
        poll({ forceHistory: true });
      });
    }
    if (els.intervalSelect) {
      els.intervalSelect.addEventListener("change", function () {
        state.intervalMs = Number(els.intervalSelect.value) || 15000;
        startTimer();
      });
    }
    if (els.liveToggle) {
      els.liveToggle.addEventListener("change", function () {
        state.live = !!els.liveToggle.checked;
        if (state.live) {
          poll();
          startTimer();
        } else {
          stopTimer();
          setStatus("Paused. Press Refresh for a single update.", "text-gray-400");
        }
      });
    }
    if (els.refreshBtn) {
      els.refreshBtn.addEventListener("click", function () {
        poll({ forceHistory: true });
      });
    }
  }

  function init() {
    if (!bindElements()) return;
    state.intervalMs = Number(els.intervalSelect && els.intervalSelect.value) || 15000;
    state.windowHours = Number(els.windowSelect && els.windowSelect.value) || 24;
    state.live = !els.liveToggle || !!els.liveToggle.checked;
    wireControls();
    if (!els.panel.classList.contains("hidden")) {
      activate();
    }
  }

  window.OrionCocreationSignals = {
    activate: activate,
    deactivate: deactivate,
    refresh: function () {
      poll({ forceHistory: true });
    },
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
