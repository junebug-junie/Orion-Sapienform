// Attention Organ operator tab.
//
// Near-realtime view of the precision-weighted attention arc described in
// docs/superpowers/specs/2026-07-28-precision-weighted-attention-organ-and-
// heartbeat-discrimination-design.md:
//
//   orion-heartbeat's N-trajectory dissipation ensemble  (the organ)
//     -> AttentionSelfModelV1's heartbeat_* fields       (the wire)
//       -> AST/HOT's own per-domain prediction error     (the problem)
//
// Backed entirely by /api/attention-organ/snapshot + /history. This file owns
// rendering and the poll lifecycle only; every number it draws comes from a
// real producer via that API — nothing here computes a cognition signal, and
// nothing is written back.
//
// Panel show/hide is app.js's setActiveTab; this module exposes
// activate()/deactivate() so polling runs ONLY while the tab is visible
// (following aitown-panel.js's lifecycle contract). Rendered state is left
// in the DOM on deactivate, so switching away and back shows the last frame
// immediately instead of an empty panel, and the rest of the Hub — chat,
// websocket, every other tab's state — is untouched throughout.
(function () {
  "use strict";

  var SNAPSHOT_URL = "/api/attention-organ/snapshot";
  var HISTORY_URL = "/api/attention-organ/history";
  var SVG_NS = "http://www.w3.org/2000/svg";

  // History is far cheaper to serve stale than the live snapshot and moves on
  // a ~30s tick anyway (substrate-runtime's attention self-model cadence), so
  // it refreshes on its own slower multiple rather than on every poll.
  var HISTORY_REFRESH_MS = 30000;

  // orion-heartbeat's own live incident threshold: absorb_queue_size reaching
  // ~30 under normal traffic was enough to starve the dissipation loop's
  // socket I/O (see services/orion-heartbeat/app/service.py's
  // _absorb_worker_loop docstring). Flagged, not silently rendered.
  var ABSORB_QUEUE_WARN = 30;

  var state = {
    active: false,
    timer: null,
    inFlight: false,
    lastSnapshotAt: 0,
    lastHistoryAt: 0,
    historyError: null,
    intervalMs: 5000,
    windowMinutes: 60,
    live: true,
    // Rolling client-side trace of the live /h1 reading, at poll resolution.
    // Distinct from the /history series (which is the persisted AST/HOT row's
    // COPY of the ratio, at substrate-runtime's ~30s tick) — this one shows
    // the organ's own faster movement between those writes. Bounded so a tab
    // left open for hours cannot grow without limit.
    liveTrace: [],
    liveTraceMax: 240,
  };

  var els = {};

  function $(id) {
    return document.getElementById(id);
  }

  function bindElements() {
    els.panel = $("attention-organ");
    els.status = $("attentionOrganStatus");
    els.ensemble = $("attentionOrganEnsemble");
    els.discrimination = $("attentionOrganDiscrimination");
    els.domains = $("attentionOrganDomains");
    els.dominance = $("attentionOrganDominance");
    els.selfModel = $("attentionOrganSelfModel");
    els.link = $("attentionOrganLink");
    els.intake = $("attentionOrganIntake");
    els.dissipation = $("attentionOrganDissipation");
    els.windowSelect = $("attentionOrganWindow");
    els.intervalSelect = $("attentionOrganInterval");
    els.liveToggle = $("attentionOrganLive");
    els.refreshBtn = $("attentionOrganRefreshBtn");
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
    // A slightly negative age is sub-second clock skew between the producing
    // container and Hub (seen live: -0.4s), not information — render it as
    // "now". A LARGE negative is a genuinely wrong clock somewhere and stays
    // visible as a negative rather than being quietly clamped away.
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

  function verdictTone(verdict) {
    if (verdict === "redundant") return "border-amber-700 bg-amber-950/50 text-amber-200";
    if (verdict === "concentrated") return "border-emerald-700 bg-emerald-950/50 text-emerald-200";
    if (verdict === "mixed") return "border-sky-700 bg-sky-950/50 text-sky-200";
    return "border-gray-700 bg-gray-900 text-gray-400";
  }

  function note(host, text, tone) {
    host.appendChild(
      el("p", "text-[10px] leading-relaxed mt-2 " + (tone || "text-gray-500"), text)
    );
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

  function svg(viewBox, className, heightClass) {
    var node = document.createElementNS(SVG_NS, "svg");
    node.setAttribute("viewBox", viewBox);
    node.setAttribute("preserveAspectRatio", "none");
    node.setAttribute("class", "w-full " + (heightClass || "h-24") + " " + (className || ""));
    return node;
  }

  function svgEl(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach(function (key) {
      node.setAttribute(key, attrs[key]);
    });
    return node;
  }

  // ------------------------------------------------------------- renderers

  // Horizontal 0..1 band gauge with the live verdict thresholds drawn where
  // orion-heartbeat actually classifies them (read from /health's config, not
  // hardcoded here — see attention_organ_routes.py's "deliberately not
  // computed here" note).
  function renderBandGauge(host, meanRatio, stdRatio, thresholds) {
    var low = thresholds && isFinite(thresholds.low_ratio) ? thresholds.low_ratio : null;
    var high = thresholds && isFinite(thresholds.high_ratio) ? thresholds.high_ratio : null;
    var chart = svg("0 0 100 22", "", "h-14");

    function band(x0, x1, fill) {
      chart.appendChild(
        svgEl("rect", { x: x0 * 100, y: 2, width: Math.max(0, (x1 - x0) * 100), height: 8, fill: fill })
      );
    }
    if (low !== null && high !== null) {
      band(0, low, "rgba(16,185,129,0.30)");
      band(low, high, "rgba(56,189,248,0.30)");
      band(high, 1, "rgba(245,158,11,0.30)");
    } else {
      band(0, 1, "rgba(75,85,99,0.35)");
    }

    if (meanRatio !== null && meanRatio !== undefined && isFinite(meanRatio)) {
      var cx = Math.max(0, Math.min(1, Number(meanRatio))) * 100;
      if (stdRatio !== null && stdRatio !== undefined && isFinite(stdRatio)) {
        var lo = Math.max(0, Math.min(1, Number(meanRatio) - Number(stdRatio))) * 100;
        var hi = Math.max(0, Math.min(1, Number(meanRatio) + Number(stdRatio))) * 100;
        chart.appendChild(
          svgEl("rect", {
            x: lo,
            y: 4.5,
            width: Math.max(0.4, hi - lo),
            height: 3,
            fill: "rgba(255,255,255,0.45)",
          })
        );
      }
      chart.appendChild(
        svgEl("rect", { x: Math.max(0, cx - 0.5), y: 0.5, width: 1, height: 11, fill: "#f8fafc" })
      );
    }

    [
      [0, "0"],
      [low, low === null ? null : String(low)],
      [high, high === null ? null : String(high)],
      [1, "1"],
    ].forEach(function (pair) {
      if (pair[0] === null || pair[1] === null) return;
      var tick = svgEl("text", {
        x: Math.max(2, Math.min(96, pair[0] * 100)),
        y: 19,
        fill: "#6b7280",
        "font-size": "5",
        "text-anchor": "middle",
      });
      tick.textContent = pair[1];
      chart.appendChild(tick);
    });
    host.appendChild(chart);

    var legend = el("div", "flex items-center justify-between text-[9px] text-gray-500 -mt-1");
    legend.appendChild(el("span", "", "concentrated"));
    legend.appendChild(el("span", "", "mixed"));
    legend.appendChild(el("span", "", "redundant"));
    host.appendChild(legend);
  }

  // The N individual trajectories behind the mean. At small N the mean+std
  // alone cannot distinguish "all trajectories agree" from "two clusters that
  // average out" — this is the whole point of running an ensemble rather than
  // one trajectory, so it gets drawn rather than summarized away.
  function renderTrajectories(host, ratios, seeds, meanRatio) {
    if (!Array.isArray(ratios) || !ratios.length) {
      note(
        host,
        "Per-trajectory ratios not reported by this orion-heartbeat build (/h1 returns mean + std only). " +
          "Rebuild the heartbeat service to populate them.",
        "text-amber-400/80"
      );
      return;
    }
    var chart = svg("0 0 100 16", "", "h-10");
    chart.appendChild(svgEl("line", { x1: 0, y1: 8, x2: 100, y2: 8, stroke: "#374151", "stroke-width": 0.4 }));
    if (meanRatio !== null && meanRatio !== undefined && isFinite(meanRatio)) {
      var mx = Math.max(0, Math.min(1, Number(meanRatio))) * 100;
      chart.appendChild(
        svgEl("line", { x1: mx, y1: 1, x2: mx, y2: 15, stroke: "#f8fafc", "stroke-width": 0.6 })
      );
    }
    ratios.forEach(function (ratio) {
      if (!isFinite(ratio)) return;
      chart.appendChild(
        svgEl("circle", {
          cx: Math.max(0, Math.min(1, Number(ratio))) * 100,
          cy: 8,
          r: 1.6,
          fill: "rgba(129,140,248,0.85)",
        })
      );
    });
    host.appendChild(chart);

    var caption = "n=" + ratios.length;
    if (Array.isArray(seeds) && seeds.length === ratios.length) {
      caption += " · seeds " + seeds.join(", ");
    }
    host.appendChild(el("div", "text-[9px] text-gray-500 font-mono", caption));
  }

  // Time series with an optional ±spread band. `points` entries are
  // {x, value, spread}; x is already normalized 0..1 by the caller.
  function renderSeries(host, points, opts) {
    opts = opts || {};
    if (!points.length) {
      note(host, opts.emptyText || "No samples in this window yet.");
      return;
    }
    var chart = svg("0 0 100 40", "", opts.heightClass || "h-28");
    var lo = opts.min === undefined ? 0 : opts.min;
    var hi = opts.max === undefined ? 1 : opts.max;
    var span = hi - lo || 1;

    function y(value) {
      return 38 - ((Math.max(lo, Math.min(hi, value)) - lo) / span) * 36;
    }

    (opts.guides || []).forEach(function (guide) {
      if (guide.value === null || guide.value === undefined || !isFinite(guide.value)) return;
      chart.appendChild(
        svgEl("line", {
          x1: 0,
          y1: y(guide.value),
          x2: 100,
          y2: y(guide.value),
          stroke: guide.color || "#4b5563",
          "stroke-width": 0.4,
          "stroke-dasharray": "2 2",
        })
      );
    });

    var withSpread = points.filter(function (p) {
      return isFinite(p.spread) && isFinite(p.value);
    });
    if (withSpread.length > 1) {
      var upper = withSpread.map(function (p) {
        return p.x * 100 + "," + y(p.value + p.spread);
      });
      var lower = withSpread
        .slice()
        .reverse()
        .map(function (p) {
          return p.x * 100 + "," + y(p.value - p.spread);
        });
      chart.appendChild(
        svgEl("polygon", {
          points: upper.concat(lower).join(" "),
          fill: "rgba(129,140,248,0.18)",
        })
      );
    }

    var line = points
      .filter(function (p) {
        return isFinite(p.value);
      })
      .map(function (p) {
        return p.x * 100 + "," + y(p.value);
      });
    if (line.length > 1) {
      chart.appendChild(
        svgEl("polyline", {
          points: line.join(" "),
          fill: "none",
          stroke: opts.color || "#818cf8",
          "stroke-width": 0.8,
          "vector-effect": "non-scaling-stroke",
        })
      );
    } else if (line.length === 1) {
      var only = line[0].split(",");
      chart.appendChild(
        svgEl("circle", { cx: only[0], cy: only[1], r: 1, fill: opts.color || "#818cf8" })
      );
    }
    host.appendChild(chart);
  }

  function renderTallyBars(host, counts, opts) {
    opts = opts || {};
    var keys = Object.keys(counts || {});
    if (!keys.length) {
      note(host, opts.emptyText || "No samples in this window.");
      return;
    }
    var total = keys.reduce(function (acc, key) {
      return acc + counts[key];
    }, 0);
    keys
      .sort(function (a, b) {
        return counts[b] - counts[a];
      })
      .forEach(function (key) {
        var pct = total ? (counts[key] / total) * 100 : 0;
        var row = el("div", "mb-1.5");
        var head = el("div", "flex items-baseline justify-between gap-2");
        head.appendChild(
          el("span", "text-[11px] " + (opts.labelTone ? opts.labelTone(key) : "text-gray-300"), key)
        );
        head.appendChild(
          el("span", "text-[10px] font-mono text-gray-500", counts[key] + " · " + pct.toFixed(1) + "%")
        );
        row.appendChild(head);
        var track = el("div", "h-1.5 rounded-full bg-gray-800 overflow-hidden mt-0.5");
        var fill = el("div", "h-full " + (opts.barTone ? opts.barTone(key) : "bg-indigo-500"));
        fill.style.width = Math.max(1, pct) + "%";
        track.appendChild(fill);
        row.appendChild(track);
        host.appendChild(row);
      });
  }

  // ---------------------------------------------------------------- panels

  // Normalize a series' generated_at timestamps into 0..1 x positions.
  // Returns index-proportional spacing if the timestamps are missing or all
  // identical, so a degenerate window still renders rather than collapsing.
  function timePositions(points) {
    var times = points.map(function (point) {
      var parsed = Date.parse(point.generated_at);
      return Number.isFinite(parsed) ? parsed : NaN;
    });
    var usable = times.filter(function (t) {
      return Number.isFinite(t);
    });
    if (usable.length !== times.length || times.length < 2) {
      return points.map(function (_point, index) {
        return points.length > 1 ? index / (points.length - 1) : 0;
      });
    }
    var min = Math.min.apply(null, usable);
    var max = Math.max.apply(null, usable);
    var span = max - min;
    if (span <= 0) {
      return points.map(function (_point, index) {
        return points.length > 1 ? index / (points.length - 1) : 0;
      });
    }
    return times.map(function (t) {
      return (t - min) / span;
    });
  }

  // Both history-backed panels render `lastHistory`, which is deliberately
  // retained across a failed refresh so a transient error does not blank them.
  // That retention is only honest if the panel says how old it is.
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

  function renderEnsemble(host, snapshot) {
    var hb = snapshot.heartbeat || {};
    if (!hb.h1) {
      errorBlock(
        host,
        "Ensemble H1",
        hb.h1_error || hb.h1_reason || "orion-heartbeat has not computed an H1 tick yet."
      );
      return;
    }
    var h1 = hb.h1;
    var config = (hb.health && hb.health.config) || null;
    card(host, "Ensemble H1", "orion-heartbeat /h1");

    var headline = el("div", "flex items-end justify-between gap-3 mb-2");
    var left = el("div");
    left.appendChild(el("div", "text-3xl font-mono text-white leading-none", num(h1.mean_ratio, 4)));
    left.appendChild(el("div", "text-[10px] uppercase tracking-wide text-gray-500 mt-1", "ensemble mean ratio"));
    headline.appendChild(left);
    var right = el("div", "text-right");
    right.appendChild(badge(h1.verdict || "unknown", verdictTone(h1.verdict)));
    right.appendChild(el("div", "text-[10px] text-gray-500 mt-1 font-mono", "±" + num(h1.std_ratio, 4) + " spread"));
    headline.appendChild(right);
    host.appendChild(headline);

    renderBandGauge(host, h1.mean_ratio, h1.std_ratio, config);
    renderTrajectories(host, h1.ratios, h1.seeds, h1.mean_ratio);

    var grid = el("div", "grid grid-cols-2 gap-2 mt-3");
    grid.appendChild(statTile("Absorbed ticks", int(h1.tick_count)));
    grid.appendChild(statTile("Trajectories", int(h1.seeds ? h1.seeds.length : (config ? config.n_trajectories : null))));
    grid.appendChild(statTile("Max bond", int(hb.health ? hb.health.max_bond : null)));
    grid.appendChild(statTile("Norm", num(hb.health ? hb.health.norm : null, 4)));
    host.appendChild(grid);

    note(
      host,
      "Spread is agreement between independently-seeded trajectories absorbing the same real event stream — " +
        "wide spread means the self-model's state is genuinely undetermined, not noisy measurement."
    );
  }

  function renderDiscrimination(host, snapshot, history) {
    card(host, "Discrimination over window", "acceptance check 1");

    var hb = snapshot.heartbeat || {};
    var config = (hb.health && hb.health.config) || null;

    var liveWrap = el("div", "mb-3");
    liveWrap.appendChild(
      el(
        "div",
        "text-[10px] uppercase tracking-wide text-gray-500 mb-1",
        "Live /h1 trace (" + state.liveTrace.length + " ensemble ticks, ±spread band)"
      )
    );
    var traceCount = state.liveTrace.length;
    renderSeries(
      liveWrap,
      state.liveTrace.map(function (point, index) {
        return {
          x: traceCount > 1 ? index / (traceCount - 1) : 0,
          value: point.mean,
          spread: point.std,
        };
      }),
      {
        guides: config
          ? [
              { value: config.high_ratio, color: "#b45309" },
              { value: config.low_ratio, color: "#047857" },
            ]
          : [],
        emptyText: "Waiting for the first poll.",
        heightClass: "h-24",
      }
    );
    host.appendChild(liveWrap);

    if (!history) {
      note(host, "History not loaded yet.");
      return;
    }
    historyAgeBanner(host);

    var histWrap = el("div", "mb-3");
    histWrap.appendChild(
      el(
        "div",
        "text-[10px] uppercase tracking-wide text-gray-500 mb-1",
        "Persisted AST/HOT copy — " +
          history.window_minutes +
          " min, " +
          history.sample_count +
          " rows" +
          (history.truncated ? " (truncated at row cap)" : "")
      )
    );
    var series = (history.series || []).filter(function (point) {
      return point.heartbeat_mean_ratio !== null && point.heartbeat_mean_ratio !== undefined;
    });
    renderSeries(
      histWrap,
      // Positioned by real elapsed time, not row index: a window containing a
      // substrate-runtime outage must draw as a gap, not as an unbroken
      // evenly-spaced line that hides the outage entirely. Falls back to index
      // spacing only if the timestamps are unusable.
      timePositions(series).map(function (x, index) {
        return {
          x: x,
          value: Number(series[index].heartbeat_mean_ratio),
          spread: NaN,
        };
      }),
      {
        color: "#38bdf8",
        guides: config
          ? [
              { value: config.high_ratio, color: "#b45309" },
              { value: config.low_ratio, color: "#047857" },
            ]
          : [],
        emptyText: "No heartbeat_mean_ratio values persisted in this window.",
        heightClass: "h-24",
      }
    );
    host.appendChild(histWrap);

    var tally = el("div");
    tally.appendChild(
      el("div", "text-[10px] uppercase tracking-wide text-gray-500 mb-1", "Verdict bands fired")
    );
    renderTallyBars(tally, history.verdict_counts || {}, {
      barTone: function (key) {
        if (key === "redundant") return "bg-amber-500";
        if (key === "concentrated") return "bg-emerald-500";
        return "bg-sky-500";
      },
      emptyText: "No heartbeat verdict persisted in this window.",
    });
    if (history.null_verdict_count) {
      note(
        tally,
        history.null_verdict_count +
          " row(s) in this window carried no heartbeat verdict at all — the wire was down for those ticks, " +
          "which is not the same as a band never firing."
      );
    }
    host.appendChild(tally);

    var distinct = history.verdict_distinct_values || 0;
    var checkTone =
      distinct > 1
        ? "border-emerald-700 bg-emerald-950/40 text-emerald-200"
        : "border-amber-700 bg-amber-950/40 text-amber-200";
    var check = el("div", "mt-3 rounded-lg border p-2 text-[10px] leading-relaxed " + checkTone);
    check.appendChild(
      el(
        "div",
        "font-semibold mb-0.5",
        distinct > 1
          ? "Acceptance check 1 met in this window: verdict took " + distinct + " distinct values."
          : "Acceptance check 1 NOT met in this window: verdict took a single value."
      )
    );
    check.appendChild(
      el(
        "div",
        "",
        "The spec records the concentrated band" +
          (config ? " (<=" + config.low_ratio + ")" : "") +
          " as a structural gap, not a waiting game: 4 of 5 organs stay continuously active regardless of chat, " +
          "so real operation may never produce the simultaneous silence it needs. Deferred by explicit decision (2026-07-29)."
      )
    );
    host.appendChild(check);
  }

  function renderDomains(host, snapshot) {
    var domains = snapshot.domains || {};
    if (!domains.ok) {
      errorBlock(host, "Per-domain prediction error", domains.error);
      return;
    }
    card(host, "Per-domain prediction error", "substrate graph, live");

    var rows = domains.rows || [];
    var active = rows.filter(function (row) {
      return row.active;
    });
    var inactive = rows.filter(function (row) {
      return !row.active;
    });

    function drawRow(row) {
      var wrap = el("div", "mb-2");
      var head = el("div", "flex items-baseline justify-between gap-2");
      var labelWrap = el("div", "flex items-center gap-1.5 min-w-0");
      labelWrap.appendChild(
        el("span", "text-[11px] " + (row.active ? "text-gray-200" : "text-gray-500 line-through"), row.domain)
      );
      if (!row.present) {
        labelWrap.appendChild(badge("no node", "border-red-800 bg-red-950/50 text-red-300"));
      } else if (row.at_ceiling) {
        labelWrap.appendChild(badge("at 1.0", "border-red-800 bg-red-950/50 text-red-300"));
      } else if (row.at_floor) {
        labelWrap.appendChild(badge("at 0.0", "border-red-800 bg-red-950/50 text-red-300"));
      }
      head.appendChild(labelWrap);
      head.appendChild(
        el(
          "span",
          "text-[10px] font-mono text-gray-500 shrink-0",
          num(row.prediction_error, 4) + " · seen " + age(row.observed_age_sec)
        )
      );
      wrap.appendChild(head);

      var track = el("div", "h-2 rounded-full bg-gray-800 overflow-hidden mt-0.5");
      var value = row.prediction_error;
      if (value !== null && value !== undefined && isFinite(value)) {
        var fill = el(
          "div",
          "h-full " +
            (row.at_ceiling || row.at_floor
              ? "bg-red-500"
              : row.active
              ? "bg-indigo-500"
              : "bg-gray-600")
        );
        fill.style.width = Math.max(1, Math.max(0, Math.min(1, Number(value))) * 100) + "%";
        track.appendChild(fill);
      }
      wrap.appendChild(track);
      return wrap;
    }

    host.appendChild(
      el(
        "div",
        "text-[10px] uppercase tracking-wide text-gray-500 mb-1",
        "ACTIVE_INFERENCE_DOMAINS (" + active.length + ") — averaged into prediction_error_confidence"
      )
    );
    active.forEach(function (row) {
      host.appendChild(drawRow(row));
    });

    if (inactive.length) {
      host.appendChild(
        el(
          "div",
          "text-[10px] uppercase tracking-wide text-gray-500 mt-3 mb-1",
          "Excluded from the aggregate (" + inactive.length + ") — still ticking, still readable elsewhere"
        )
      );
      inactive.forEach(function (row) {
        host.appendChild(drawRow(row));
      });
    }

    var rec = domains.reconciliation || {};
    var recTone;
    if (rec.verdict === "exact" || rec.verdict === "within_drift") {
      recTone = "border-emerald-800 bg-emerald-950/30 text-emerald-200";
    } else if (rec.verdict === "divergent") {
      recTone = "border-red-800 bg-red-950/30 text-red-200";
    } else {
      recTone = "border-gray-700 bg-gray-900/60 text-gray-400";
    }
    var recBox = el("div", "mt-3 rounded-lg border p-2 text-[10px] leading-relaxed " + recTone);
    var recHead = el("div", "flex items-center justify-between gap-2 mb-1");
    recHead.appendChild(
      el("span", "font-semibold uppercase tracking-wide", rec.verdict || "unknown")
    );
    recHead.appendChild(
      el("span", "font-mono opacity-70", "row " + age(rec.row_age_sec) + " old")
    );
    recBox.appendChild(recHead);
    recBox.appendChild(
      el(
        "div",
        "font-mono",
        "1 - mean(active domains) = " +
          num(rec.recomputed, 4) +
          "   vs persisted prediction_error_confidence = " +
          num(rec.persisted, 4) +
          "   (delta " +
          num(rec.delta, 4) +
          ", n=" +
          (rec.domains_used === null || rec.domains_used === undefined ? "—" : rec.domains_used) +
          ")"
      )
    );
    if (rec.basis) {
      recBox.appendChild(el("div", "opacity-80 mt-1", rec.basis));
    }
    host.appendChild(recBox);
    note(
      host,
      "A domain pinned at exactly 1.0 or decayed to exactly 0.0 is flagged red because both have real incident " +
        "history on these nodes (bus_synaptic frozen at 1.0, PR #1449; route decayed to ~0 by a generic staleness loop). " +
        "A flag is a prompt to check provenance, not proof of a bug."
    );
  }

  function renderDominance(host, history) {
    card(host, "Domain dominance over window", "acceptance check 3");
    if (!history) {
      note(host, "History not loaded yet.");
      return;
    }
    historyAgeBanner(host);
    host.appendChild(
      el(
        "div",
        "text-[10px] uppercase tracking-wide text-gray-500 mb-1",
        "Which domain won predicted_shift — " +
          history.window_minutes +
          " min, " +
          history.sample_count +
          " rows"
      )
    );
    renderTallyBars(host, history.domain_counts || {}, {
      emptyText: "No predicted_shift domain resolved in this window.",
    });
    if (history.null_domain_count) {
      note(
        host,
        history.null_domain_count + " row(s) had no predicted_shift at all (no trend resolved that tick)."
      );
    }
    if (history.malformed_row_count) {
      note(
        host,
        history.malformed_row_count +
          " row(s) in this window could not be parsed at all — the tallies above describe the rest, not the window.",
        "text-amber-400/80"
      );
    }
    note(
      host,
      "The spec's 168h baseline was biometrics=69819, bus_synaptic=57360, execution=27, chat=1, route=1 — " +
        "two of those five were later found to be broken instruments, not quiet domains (PR #1434, PR #1449). " +
        "A flatter distribution here is the evidence a precision-weighting fix would have to produce."
    );
  }

  function renderSelfModel(host, snapshot) {
    var sm = snapshot.self_model || {};
    if (!sm.ok || !sm.model) {
      errorBlock(host, "AST/HOT self-model", sm.error || "No substrate_attention_self_model row found.");
      return;
    }
    var model = sm.model;
    card(host, "AST/HOT self-model", "AttentionSelfModelV1");

    var grid = el("div", "grid grid-cols-1 md:grid-cols-2 gap-x-4");
    var leftCol = el("div");
    leftCol.appendChild(kvRow("attention_reason", model.attention_reason || "—"));
    leftCol.appendChild(kvRow("confidence", num(model.confidence, 4)));
    leftCol.appendChild(kvRow("confidence_basis", model.confidence_basis || "—", "text-gray-400"));
    leftCol.appendChild(kvRow("prediction_error_confidence", num(model.prediction_error_confidence, 4)));
    leftCol.appendChild(kvRow("predicted_shift", model.predicted_shift || "—"));
    leftCol.appendChild(kvRow("predicted_shift_basis", model.predicted_shift_basis || "—", "text-gray-400"));
    grid.appendChild(leftCol);

    var rightCol = el("div");
    rightCol.appendChild(kvRow("heartbeat_mean_ratio", num(model.heartbeat_mean_ratio, 4)));
    rightCol.appendChild(
      kvRow(
        "heartbeat_verdict",
        model.heartbeat_verdict || "—",
        model.heartbeat_verdict ? "text-gray-200" : "text-amber-300"
      )
    );
    rightCol.appendChild(kvRow("field_lane_present", String(model.field_lane_present)));
    rightCol.appendChild(kvRow("field_overall_salience", num(model.field_overall_salience, 4)));
    rightCol.appendChild(
      kvRow(
        "broadcast_lane",
        (model.broadcast_lane_present ? "present" : "absent") +
          (model.broadcast_lane_stale ? " · stale" : " · fresh") +
          " · " +
          age(model.broadcast_lane_age_sec),
        model.broadcast_lane_stale ? "text-amber-300" : "text-gray-200"
      )
    );
    rightCol.appendChild(
      kvRow("voluntary_override", model.voluntary_override ? "set" : "none")
    );
    grid.appendChild(rightCol);
    host.appendChild(grid);

    var targets = model.field_salient_target_ids || [];
    var targetsWrap = el("div", "mt-3");
    targetsWrap.appendChild(
      el("div", "text-[10px] uppercase tracking-wide text-gray-500 mb-1", "Salient target ids (" + targets.length + ")")
    );
    if (targets.length) {
      var chips = el("div", "flex flex-wrap gap-1");
      targets.forEach(function (id) {
        chips.appendChild(badge(id, "border-gray-700 bg-gray-900 text-gray-300"));
      });
      targetsWrap.appendChild(chips);
    } else {
      targetsWrap.appendChild(el("div", "text-[11px] text-gray-500", "none"));
    }
    host.appendChild(targetsWrap);

    if (model.reason_narrative) {
      var narrative = el("div", "mt-3");
      narrative.appendChild(
        el("div", "text-[10px] uppercase tracking-wide text-gray-500 mb-1", "Reason narrative")
      );
      narrative.appendChild(
        el("div", "text-[11px] text-gray-300 leading-relaxed", model.reason_narrative)
      );
      host.appendChild(narrative);
    }
  }

  function renderLink(host, snapshot) {
    card(host, "Organ → AST/HOT wire", "staleness");
    var link = snapshot.link || {};

    var wired = !!link.self_model_has_heartbeat;
    host.appendChild(
      badge(
        wired ? "wire live" : "wire not carrying heartbeat",
        wired
          ? "border-emerald-700 bg-emerald-950/50 text-emerald-200"
          : "border-red-800 bg-red-950/50 text-red-300"
      )
    );

    var rows = el("div", "mt-2");
    rows.appendChild(
      kvRow(
        "row age",
        age(link.self_model_age_sec),
        Number.isFinite(link.self_model_age_sec) && Number(link.self_model_age_sec) > 120
          ? "text-amber-300"
          : "text-gray-200"
      )
    );
    rows.appendChild(kvRow("row's mean_ratio", num(link.self_model_mean_ratio, 4)));
    rows.appendChild(kvRow("organ's mean_ratio now", num(link.live_mean_ratio, 4)));
    rows.appendChild(kvRow("organ tick_count now", int(link.live_tick_count)));

    // The row's own basis string names the tick it was built from, so the gap
    // against the organ's current tick_count is a real staleness measure — a
    // non-null heartbeat field alone would not tell you whether the value is
    // seconds or hours old.
    var basisTick = null;
    var match = /tick_count=(\d+)/.exec(link.self_model_basis || "");
    if (match) basisTick = Number(match[1]);
    rows.appendChild(kvRow("row's basis tick", basisTick === null ? "—" : int(basisTick)));
    // Number.isFinite, NOT the global isFinite: the global coerces, so
    // isFinite(null) is true and an offline organ (live_tick_count === null)
    // would render a confident negative lag — a number no producer produced,
    // shown precisely when the organ is down.
    if (basisTick !== null && Number.isFinite(link.live_tick_count)) {
      var lag = Number(link.live_tick_count) - basisTick;
      // Reported without a health verdict attached: tick_count counts absorbed
      // events, whose rate depends entirely on live organ traffic, so no fixed
      // lag threshold means anything. Row age below is the real staleness read.
      rows.appendChild(kvRow("tick lag", int(lag) + " ticks"));
    }
    host.appendChild(rows);

    if (link.self_model_basis) {
      note(host, link.self_model_basis, "text-gray-500 font-mono");
    }
    note(
      host,
      "AttentionSelfModelV1 still has no bus channel and no downstream consumer — this wire ends here, at " +
        "the persisted row and this panel. Nothing in Orion's decision path reads it yet."
    );
  }

  function renderIntake(host, snapshot) {
    var hb = snapshot.heartbeat || {};
    if (!hb.health) {
      errorBlock(host, "Intake & backpressure", hb.health_error);
      return;
    }
    var health = hb.health;
    card(host, "Intake & backpressure", "orion-heartbeat /health");

    var queueSize = Number(health.absorb_queue_size);
    var queueTone =
      isFinite(queueSize) && queueSize >= ABSORB_QUEUE_WARN ? "text-amber-300" : "text-gray-100";

    var grid = el("div", "grid grid-cols-2 md:grid-cols-4 gap-2");
    grid.appendChild(statTile("Events seen", int(health.events_seen)));
    grid.appendChild(statTile("Queued", int(health.events_queued)));
    grid.appendChild(statTile("Absorbed", int(health.events_absorbed)));
    grid.appendChild(
      statTile(
        "Queue depth",
        int(health.absorb_queue_size) +
          (health.absorb_queue_maxsize ? " / " + int(health.absorb_queue_maxsize) : ""),
        queueTone
      )
    );
    grid.appendChild(
      statTile(
        "Dropped (queue full)",
        int(health.events_dropped_queue_full),
        Number(health.events_dropped_queue_full) > 0 ? "text-red-300" : "text-gray-100"
      )
    );
    grid.appendChild(statTile("Skipped: organ", int(health.events_skipped_organ)));
    grid.appendChild(
      statTile(
        "Skipped: atom type",
        int(health.events_skipped_atom_type),
        Number(health.events_skipped_atom_type) > 0 ? "text-amber-300" : "text-gray-100"
      )
    );
    grid.appendChild(
      statTile(
        "Skipped: malformed",
        int(health.events_skipped_malformed),
        Number(health.events_skipped_malformed) > 0 ? "text-amber-300" : "text-gray-100"
      )
    );
    host.appendChild(grid);

    var organs = health.allowlisted_organs || [];
    if (organs.length) {
      var organWrap = el("div", "mt-3");
      organWrap.appendChild(
        el(
          "div",
          "text-[10px] uppercase tracking-wide text-gray-500 mb-1",
          "Routed organs (" + organs.length + " sites)"
        )
      );
      var chips = el("div", "flex flex-wrap gap-1");
      organs.forEach(function (organ) {
        var site = health.organ_site_map ? health.organ_site_map[organ] : undefined;
        chips.appendChild(
          badge(
            organ + (site === undefined ? "" : " → site " + site),
            "border-gray-700 bg-gray-900 text-gray-300"
          )
        );
      });
      organWrap.appendChild(chips);
      host.appendChild(organWrap);
    }

    if (isFinite(queueSize) && queueSize >= ABSORB_QUEUE_WARN) {
      note(
        host,
        "Queue depth at or above " +
          ABSORB_QUEUE_WARN +
          " — the depth at which a live burst previously starved the dissipation loop's socket I/O and " +
          "produced a spurious FalkorDB timeout. The substrate's freshness lags real events here, not its correctness.",
        "text-amber-400/80"
      );
    }
  }

  function renderDissipation(host, snapshot) {
    var hb = snapshot.heartbeat || {};
    var health = hb.health;
    if (!health) {
      errorBlock(host, "Dissipation drive", hb.health_error);
      return;
    }
    card(host, "Dissipation drive", "decay ⇄ reheat");

    var reheat = health.last_reheat;
    if (reheat) {
      var saturated = Number(reheat.signal) >= 1.0;
      var chain = el(
        "div",
        "rounded-lg border p-2 mb-3 " +
          (saturated ? "border-red-900/60 bg-red-950/20" : "border-gray-800 bg-black/30")
      );
      var chainHead = el("div", "flex items-center justify-between gap-2 mb-1");
      chainHead.appendChild(
        el("div", "text-[10px] uppercase tracking-wide text-gray-500", "Last real reheat tick")
      );
      if (saturated) {
        chainHead.appendChild(badge("saturated", "border-red-800 bg-red-950/50 text-red-300"));
      }
      chain.appendChild(chainHead);
      var flow = el("div", "text-[11px] font-mono text-gray-200 break-all");
      flow.textContent =
        "raw mean|gap_z| " +
        num(reheat.raw_mean_abs_gap_zscore, 4) +
        "  ÷ " +
        num(reheat.zscore_saturation, 1) +
        "  → signal " +
        num(reheat.signal, 4) +
        "  × scale  → reheat_prob " +
        num(reheat.reheat_prob, 5);
      chain.appendChild(flow);
      // service.py only overwrites last_reheat after a SUCCESSFUL tick, so a
      // persistently failing FalkorDB query serves the last good block
      // forever. Age it against the loop's own configured interval rather
      // than rendering the timestamp as inert gray text.
      var reheatAgeSec = null;
      if (reheat.at) {
        var parsedAt = Date.parse(reheat.at);
        if (Number.isFinite(parsedAt)) reheatAgeSec = (Date.now() - parsedAt) / 1000;
      }
      var interval = config && Number(config.decay_reheat_interval_sec);
      var reheatStale =
        reheatAgeSec !== null && Number.isFinite(interval) && interval > 0
          ? reheatAgeSec > interval * 5
          : false;
      chain.appendChild(
        el(
          "div",
          "text-[10px] mt-1 " + (reheatStale ? "text-amber-300" : "text-gray-500"),
          (reheat.at || "") +
            (reheatAgeSec === null ? "" : "  ·  " + age(reheatAgeSec) + " ago") +
            (reheatStale ? "  ·  STALE — dissipation loop may be failing" : "")
        )
      );
      host.appendChild(chain);
      if (saturated) {
        note(
          host,
          "Reheat signal is clipped at its ceiling: raw mean|gap_z| is above the saturation constant, so " +
            "min(1, z/sat) returns 1.0 no matter how bus activity moves. Reheat is running at constant maximum, " +
            "which means this input is not currently gating anything — the design spec's calibration assumed a " +
            "raw value near 1.0–1.1. Worth re-checking the saturation constant against real data before trusting " +
            "reheat as a dynamic driver.",
          "text-red-300/90"
        );
      }
    } else {
      note(
        host,
        "No dissipation tick recorded yet by this orion-heartbeat build. Rebuild the heartbeat service to " +
          "surface the live reheat inputs.",
        "text-amber-400/80"
      );
    }

    var config = health.config;
    if (config) {
      var cfg = el("div");
      cfg.appendChild(
        el("div", "text-[10px] uppercase tracking-wide text-gray-500 mb-1", "Live tuning in force")
      );
      [
        ["n_trajectories", int(config.n_trajectories)],
        ["gamma (relaxation)", num(config.gamma, 3)],
        ["base_decay_prob", num(config.base_decay_prob, 3)],
        ["decay_spread_sensitivity", num(config.decay_spread_sensitivity, 2)],
        ["reheat_strength", num(config.reheat_strength, 3)],
        ["reheat_prob_scale", num(config.reheat_prob_scale, 3)],
        ["decay/reheat interval", num(config.decay_reheat_interval_sec, 1) + "s"],
        ["h1 interval", num(config.h1_interval_sec, 1) + "s"],
        ["verdict bands", "≤" + num(config.low_ratio, 2) + " / ≥" + num(config.high_ratio, 2)],
      ].forEach(function (pair) {
        cfg.appendChild(kvRow(pair[0], pair[1]));
      });
      host.appendChild(cfg);
    } else {
      note(host, "Live tuning not reported by this orion-heartbeat build.", "text-amber-400/80");
    }

    note(
      host,
      "Decay is gated by trajectory spread (disagreement suppresses relaxation); reheat is driven by real " +
        "inter-service bus timing jitter. Without both, the substrate thermalizes and can never read anything but 'redundant'."
    );
  }

  // ------------------------------------------------------------ poll cycle

  function setStatus(text, tone) {
    if (!els.status) return;
    els.status.textContent = text;
    els.status.className = "text-xs min-h-[1.25rem] " + (tone || "text-gray-400");
  }

  function pushLiveTrace(h1) {
    if (!h1 || h1.mean_ratio === null || h1.mean_ratio === undefined) return;
    var last = state.liveTrace[state.liveTrace.length - 1];
    // The organ recomputes H1 on its own ~30s timer, so consecutive polls
    // usually return the SAME tick. Deduplicating on tick_count keeps the
    // trace a record of real ensemble ticks rather than a flat line whose
    // apparent resolution is just the poll rate.
    if (last && last.tick === h1.tick_count) return;
    state.liveTrace.push({
      tick: h1.tick_count,
      mean: Number(h1.mean_ratio),
      std: Number(h1.std_ratio),
      at: h1.generated_at,
    });
    if (state.liveTrace.length > state.liveTraceMax) {
      state.liveTrace.splice(0, state.liveTrace.length - state.liveTraceMax);
    }
  }

  var lastHistory = null;

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
      pushLiveTrace(snapshot.heartbeat && snapshot.heartbeat.h1);
      state.lastSnapshotAt = Date.now();

      var needHistory =
        opts.forceHistory ||
        !lastHistory ||
        Date.now() - state.lastHistoryAt >= HISTORY_REFRESH_MS;
      if (needHistory) {
        try {
          lastHistory = await fetchJson(HISTORY_URL + "?minutes=" + state.windowMinutes);
          state.lastHistoryAt = Date.now();
          state.historyError = null;
        } catch (err) {
          // A history failure must not blank the live panels next to it — but
          // the previous payload then keeps rendering with its own
          // window/sample labels, so the failure has to survive into the
          // status line and into both consuming panels as a real age, or a
          // stale window reads as the current one.
          state.historyError = String(err.message || err);
        }
      }

      renderEnsemble(els.ensemble, snapshot);
      renderDiscrimination(els.discrimination, snapshot, lastHistory);
      renderDomains(els.domains, snapshot);
      renderDominance(els.dominance, lastHistory);
      renderSelfModel(els.selfModel, snapshot);
      renderLink(els.link, snapshot);
      renderIntake(els.intake, snapshot);
      renderDissipation(els.dissipation, snapshot);

      var problems = [];
      if (!(snapshot.heartbeat || {}).ok) problems.push("heartbeat");
      if (!(snapshot.self_model || {}).ok) problems.push("self-model");
      if (!(snapshot.domains || {}).ok) problems.push("domains");
      if (state.historyError) problems.push("history");
      if (problems.length) {
        setStatus(
          "Updated " +
            new Date().toLocaleTimeString() +
            " — degraded source(s): " +
            problems.join(", "),
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
      // Visibility, not just the router's activate/deactivate calls, is what
      // actually gates polling. self_observability.js hides every
      // section[data-panel] directly and preventDefault()s its own click, so
      // app.js's setActiveTab -- the only caller of deactivate() -- never runs
      // on that path, and the timer would otherwise keep firing forever into a
      // hidden panel. Checking the DOM makes the contract hold against any
      // future panel that does the same thing.
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
    // Rendered DOM is deliberately left in place — see this file's header.
  }

  function wireControls() {
    if (els.windowSelect) {
      els.windowSelect.addEventListener("change", function () {
        state.windowMinutes = Number(els.windowSelect.value) || 60;
        poll({ forceHistory: true });
      });
    }
    if (els.intervalSelect) {
      els.intervalSelect.addEventListener("change", function () {
        state.intervalMs = Number(els.intervalSelect.value) || 5000;
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
    state.intervalMs = Number(els.intervalSelect && els.intervalSelect.value) || 5000;
    state.windowMinutes = Number(els.windowSelect && els.windowSelect.value) || 60;
    state.live = !els.liveToggle || !!els.liveToggle.checked;
    wireControls();
    // Defensive only. Both scripts are `defer` and this one is listed before
    // app.js, so in practice app.js's DOMContentLoaded handler (which is what
    // calls setActiveTab) has not run yet and the panel is still hidden here —
    // setActiveTab does the real activation a moment later. This branch exists
    // so a future load-order change cannot leave a visible panel unpolled.
    if (!els.panel.classList.contains("hidden")) {
      activate();
    }
  }

  window.OrionAttentionOrgan = {
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
