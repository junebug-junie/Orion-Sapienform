/**
 * Biometrics view -- drives two related surfaces from one module:
 *
 * 1. The Cognitive EKG card's toggle (landing "hub" tab): swaps between the
 *    /spark/ui Substrate Brain State iframe and a compact Athena+Circe
 *    biometrics preview, in the same card slot. Clicking the preview opens
 *    the full modal.
 * 2. The near-fullscreen Biometrics modal: 4 sub-tabs (Athena / Circe / GPU /
 *    Cabinet). Modal open/close/Escape/backdrop mechanics live in app.js
 *    (openBiometricsModal/closeBiometricsModal, matching every other Hub
 *    modal); this module owns subview switching and data loading only, and
 *    is told about open/close via onModalOpen()/onModalClose().
 *
 * Same activate()/deactivate() + wireOnce() idempotency-guard + loaded.*
 * lazy-load lifecycle contract as reverie-tab.js and cabinet-sensors.js.
 * Backed by services/orion-hub/scripts/biometrics_preview_routes.py
 * (/api/biometrics/preview/{snapshot,history,induction,gpu}).
 */
(function () {
  "use strict";

  var CARD_POLL_MS = 10000; // preview tiles refresh cadence while shown
  var GPU_POLL_MS = 10000; // GPU subview refresh cadence while open

  var cardView = "brain"; // "brain" | "biometrics"
  var modalOpen = false;
  var modalSubview = "athena"; // "athena" | "circe" | "gpu" | "cabinet"
  var gpuNode = "athena"; // "athena" | "circe"

  var loaded = { cardPreview: false, athena: false, circe: false, gpu: { athena: false, circe: false } };
  var cardPollTimer = null;
  var gpuPollTimer = null;

  function el(id) {
    return document.getElementById(id);
  }

  function clear(node) {
    if (!node) return;
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function fmt(value, digits) {
    if (value === null || value === undefined || value === "") return "—";
    var n = Number(value);
    if (Number.isNaN(n)) return String(value);
    return digits === undefined ? String(n) : n.toFixed(digits);
  }

  async function fetchJson(url) {
    var response = await fetch(url);
    return response.json();
  }

  // --- Status color + trend -----------------------------------------------
  //
  // Reuses this file's own established dark-surface tone convention (see
  // cabinet-sensors.js's badge()) rather than inventing a second palette --
  // emerald=good, amber=warning, red=critical, gray=no signal. Status color
  // is reserved meaning: it always ships with the icon+label pair below,
  // never a bare color swatch, so it survives colorblindness and isn't
  // mistaken for a fourth categorical series.
  var TONE = {
    good: { border: "border-emerald-700", bg: "bg-emerald-950/40", text: "text-emerald-200", icon: "●", label: "good" },
    warning: { border: "border-amber-700", bg: "bg-amber-950/40", text: "text-amber-200", icon: "▲", label: "warning" },
    critical: { border: "border-red-700", bg: "bg-red-950/40", text: "text-red-200", icon: "■", label: "critical" },
    neutral: { border: "border-gray-800", bg: "bg-gray-950/60", text: "text-gray-500", icon: "○", label: "no signal" },
  };
  var TONE_RANK = { critical: 0, warning: 1, good: 2, neutral: 3 };

  // 0-1 pressure channels: higher = more loaded. homeostasis/stability are
  // the opposite (higher = healthier), so callers pass invert:true for those.
  function toneForPressure(value, invert) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "neutral";
    var v = Number(value);
    var effective = invert ? 1 - v : v;
    if (effective >= 0.75) return "critical";
    if (effective >= 0.5) return "warning";
    return "good";
  }

  // A node that failed to answer at all is worse news than "no data yet" --
  // it must render critical (red), not neutral (gray, indistinguishable
  // from "hasn't reported"). Review finding: with the prior "unreachable ->
  // neutral" mapping, a fully-down node's status tile was the ONLY tile
  // left visible (every value tile gets filtered out when there's no
  // summary data) and it rendered gray -- exactly invisible to an operator
  // scanning for red tiles, defeating the point of color-coding at all.
  function toneForNodeStatus(payload) {
    if (!payload || payload.ok === false) return "critical";
    var status = String(payload.status || "").toUpperCase();
    if (status === "OK") return "good";
    if (status === "STALE" || status === "DEGRADED") return "warning";
    if (status === "NO_SIGNAL") return "critical";
    return "neutral"; // genuinely no status reported yet, not a known failure
  }

  function trendArrow(trendValue) {
    if (trendValue === null || trendValue === undefined || Number.isNaN(Number(trendValue))) return null;
    var t = Number(trendValue);
    if (Math.abs(t) < 0.02) return { glyph: "→", label: "flat" };
    return t > 0 ? { glyph: "↑", label: "rising " + fmt(Math.abs(t), 2) } : { glyph: "↓", label: "falling " + fmt(Math.abs(t), 2) };
  }

  function tile(label, value, sub, opts) {
    opts = opts || {};
    var tone = TONE[opts.tone] || TONE.neutral;
    var wrap = document.createElement("div");
    wrap.className = "rounded-lg border-l-4 border border-gray-800 bg-gray-950/60 px-3 py-2 " + tone.border;
    wrap.title = tone.label + (opts.trend ? " · " + opts.trend.label : "");

    var head = document.createElement("div");
    head.className = "flex items-center justify-between gap-2";
    var l = document.createElement("div");
    l.className = "text-[10px] uppercase tracking-wide text-gray-500 truncate";
    l.textContent = label;
    head.appendChild(l);
    if (opts.tone && opts.tone !== "neutral") {
      var dot = document.createElement("span");
      dot.className = "shrink-0 text-[10px] " + tone.text;
      dot.textContent = tone.icon;
      head.appendChild(dot);
    }
    wrap.appendChild(head);

    var valueRow = document.createElement("div");
    valueRow.className = "mt-1 flex items-baseline gap-1.5";
    var v = document.createElement("span");
    v.className = "font-mono text-sm " + (opts.tone ? tone.text : "text-gray-200");
    v.textContent = value;
    valueRow.appendChild(v);
    if (opts.trend) {
      var arrow = document.createElement("span");
      arrow.className = "font-mono text-xs text-indigo-300";
      arrow.textContent = opts.trend.glyph;
      arrow.title = opts.trend.label;
      valueRow.appendChild(arrow);
    }
    wrap.appendChild(valueRow);

    if (sub) {
      var s = document.createElement("div");
      s.className = "text-[10px] text-gray-500 mt-0.5";
      s.textContent = sub;
      wrap.appendChild(s);
    }
    return wrap;
  }

  var SVG_NS = "http://www.w3.org/2000/svg";

  function sparkline(points) {
    var wrap = document.createElement("div");
    wrap.className = "rounded-lg border border-gray-800 bg-gray-950/60 p-2";
    var values = (points || [])
      .map(function (p) {
        return Number(p.v !== undefined ? p.v : p.utilization_gpu);
      })
      .filter(function (v) {
        return !Number.isNaN(v);
      });
    if (!values.length) {
      var empty = document.createElement("div");
      empty.className = "text-[10px] text-gray-500";
      empty.textContent = "no data yet";
      wrap.appendChild(empty);
      return wrap;
    }
    var min = Math.min.apply(null, values);
    var max = Math.max.apply(null, values);
    var range = max - min || 1;
    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 100 30");
    svg.setAttribute("preserveAspectRatio", "none");
    svg.setAttribute("class", "w-full h-8");
    var path = values
      .map(function (v, i) {
        var x = values.length > 1 ? (i / (values.length - 1)) * 100 : 0;
        var y = 30 - ((v - min) / range) * 28 - 1;
        return (i === 0 ? "M" : "L") + x.toFixed(1) + "," + y.toFixed(1);
      })
      .join(" ");
    var polyline = document.createElementNS(SVG_NS, "path");
    polyline.setAttribute("d", path);
    polyline.setAttribute("fill", "none");
    polyline.setAttribute("stroke", "#818cf8");
    polyline.setAttribute("stroke-width", "1.5");
    svg.appendChild(polyline);
    wrap.appendChild(svg);
    return wrap;
  }

  // --- Cognitive EKG card toggle ---------------------------------------

  async function loadCardPreview() {
    var status = el("biometricsPreviewStatus");
    var grid = el("biometricsPreviewGrid");
    if (!grid) return;
    if (status) status.textContent = "Loading…";
    clear(grid);
    var nodes = ["athena", "circe"];
    var results = await Promise.all(
      nodes.map(function (n) {
        return Promise.all([
          fetchJson("/api/biometrics/preview/snapshot?node=" + n).catch(function () {
            return { ok: false, node: n };
          }),
          fetchJson("/api/biometrics/preview/induction?node=" + n).catch(function () {
            return { ok: false, metrics: {} };
          }),
        ]);
      })
    );
    results.forEach(function (pair) {
      var payload = pair[0];
      var induction = pair[1];
      var composites = (payload.summary && payload.summary.composites) || {};
      var strain = composites.strain;
      var trendInfo = induction.metrics && induction.metrics.strain;
      var label = (payload.node || "?") + (payload.ok ? "" : " (unreachable)");
      grid.appendChild(
        tile(label, strain !== undefined ? fmt(strain, 2) : "—", "strain · " + (payload.status || "—"), {
          tone: payload.ok && strain !== undefined ? toneForPressure(strain) : toneForNodeStatus(payload),
          trend: trendInfo ? trendArrow(trendInfo.trend) : null,
        })
      );
    });
    if (status) status.textContent = results.every((r) => r[0].ok) ? "live" : "partial";
    loaded.cardPreview = true;
  }

  function showCardView(view) {
    cardView = view;
    var brain = el("stateVisualizerContainer");
    var preview = el("biometricsPreviewContainer");
    if (brain) brain.classList.toggle("hidden", cardView !== "brain");
    if (preview) preview.classList.toggle("hidden", cardView !== "biometrics");
    if (cardView === "biometrics") {
      if (!loaded.cardPreview) loadCardPreview();
      if (!cardPollTimer) {
        cardPollTimer = setInterval(loadCardPreview, CARD_POLL_MS);
      }
    } else if (cardPollTimer) {
      clearInterval(cardPollTimer);
      cardPollTimer = null;
    }
  }

  function toggleCardView() {
    showCardView(cardView === "brain" ? "biometrics" : "brain");
  }

  // --- Modal: Athena / Circe subviews -----------------------------------

  // Full channel set the backend can chart (matches biometrics_preview_routes.py's
  // _CHANNEL_COLUMN exactly) -- every one of these gets both a snapshot tile
  // and a trend chart, not just a hand-picked 4. homeostasis/stability read
  // "higher is healthier" (inverted); everything else is a pressure where
  // higher = more loaded.
  var COMPOSITE_CHANNELS = ["strain", "homeostasis", "stability"];
  var PRESSURE_CHANNELS = [
    "cpu", "gpu_util", "gpu_mem", "mem", "swap", "disk", "net", "thermal", "power", "disk_capacity", "fan",
  ];
  var ALL_CHANNELS = COMPOSITE_CHANNELS.concat(PRESSURE_CHANNELS);
  var INVERTED_CHANNELS = { homeostasis: true, stability: true };

  async function loadNodeDetail(node) {
    var snapEl = el("biometrics" + cap(node) + "Snapshot");
    var histEl = el("biometrics" + cap(node) + "History");
    var indEl = el("biometrics" + cap(node) + "Induction");
    if (snapEl) clear(snapEl);
    if (histEl) clear(histEl);
    if (indEl) clear(indEl);

    // Snapshot, history (one per channel), and induction are three
    // independent reads -- kick all of them off together instead of
    // awaiting each in turn, so total load time is the slowest single leg,
    // not their sum. Snapshot and induction are awaited together below
    // because rendering a snapshot tile needs BOTH the current value (tone)
    // and the induction trend (arrow) -- that's the direct fix for "can't
    // tell what's changing": the arrow lives on the same tile as the value,
    // not buried in a separate section.
    var snapshotPromise = fetchJson("/api/biometrics/preview/snapshot?node=" + node).catch(function () {
      return { ok: false };
    });
    // One request for every channel, not one request PER channel -- the
    // /history endpoint opens its own short-lived Postgres connection per
    // call with no pooling, and this repo has live incident history with
    // connection exhaustion (PR #2010); N concurrent connections on every
    // modal open is a real resource risk, not just N round trips.
    var historiesPromise = fetchJson(
      "/api/biometrics/preview/history_multi?node=" + node + "&channels=" + ALL_CHANNELS.join(",") + "&window=24h"
    ).catch(function () {
      return { ok: false, series: {} };
    });
    var inductionPromise = fetchJson("/api/biometrics/preview/induction?node=" + node).catch(function () {
      return { ok: false, metrics: {} };
    });

    var pair = await Promise.all([snapshotPromise, inductionPromise]);
    var snapshot = pair[0];
    var induction = pair[1];
    var metrics = induction.metrics || {};

    if (snapEl) {
      var composites = (snapshot.summary && snapshot.summary.composites) || {};
      var pressures = (snapshot.summary && snapshot.summary.pressures) || {};
      var rows = ALL_CHANNELS.map(function (ch) {
        var isComposite = COMPOSITE_CHANNELS.indexOf(ch) !== -1;
        var value = isComposite ? composites[ch] : pressures[ch];
        if (value === undefined) return null; // absent channel on this node -- omit, never zero-fill
        // (an unreachable node has no composites/pressures at all, so every
        // row already short-circuits above -- the "node status" tile below
        // is what carries the critical tone for that case.)
        var invert = !!INVERTED_CHANNELS[ch];
        var tone = toneForPressure(value, invert);
        var trendInfo = metrics[ch];
        return { ch: ch, value: value, tone: tone, trend: trendInfo ? trendArrow(trendInfo.trend) : null };
      }).filter(Boolean);
      // Worst-first: the point of color-coding is drawing the eye to what
      // needs attention without the operator scanning every tile.
      rows.sort(function (a, b) {
        return TONE_RANK[a.tone] - TONE_RANK[b.tone];
      });
      if (!rows.length) {
        var noData = document.createElement("div");
        noData.className = "text-[11px] text-gray-500 col-span-full";
        noData.textContent = "no summary data for this node yet";
        snapEl.appendChild(noData);
      }
      rows.forEach(function (row) {
        snapEl.appendChild(tile(row.ch, fmt(row.value, 2), null, { tone: row.tone, trend: row.trend }));
      });
      snapEl.appendChild(
        tile("node status", snapshot.status || (snapshot.ok ? "—" : "unreachable"), "freshness " + fmt(snapshot.freshness_s, 1) + "s", {
          tone: toneForNodeStatus(snapshot),
        })
      );
    }

    var histories = await historiesPromise;
    if (histEl) {
      var seriesByChannel = histories.series || {};
      ALL_CHANNELS.forEach(function (ch) {
        var box = document.createElement("div");
        var label = document.createElement("div");
        label.className = "text-[10px] uppercase tracking-wide text-gray-500 mb-1";
        label.textContent = ch + " (24h)";
        box.appendChild(label);
        box.appendChild(sparkline(seriesByChannel[ch] || []));
        histEl.appendChild(box);
      });
    }

    if (indEl) {
      var keys = Object.keys(metrics);
      if (!keys.length) {
        var none = document.createElement("div");
        none.className = "text-[11px] text-gray-500 col-span-full";
        none.textContent = "no induction row within freshness window";
        indEl.appendChild(none);
      }
      keys.forEach(function (key) {
        var m = metrics[key] || {};
        indEl.appendChild(
          tile(key, "L " + fmt(m.level, 2), "vol " + fmt(m.volatility, 2) + " · spike " + fmt(m.spike_rate, 2), {
            trend: trendArrow(m.trend),
          })
        );
      });
    }
  }

  function cap(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  // --- Modal: GPU subview -------------------------------------------------

  function gpuCard(gpu) {
    var box = document.createElement("div");
    box.className = "rounded-xl border border-gray-800 bg-gray-950/40 p-3 flex flex-col gap-2";

    var head = document.createElement("div");
    head.className = "flex items-center justify-between";
    var title = document.createElement("div");
    title.className = "text-sm font-semibold text-gray-100";
    title.textContent = "#" + gpu.index + " " + (gpu.name || "?");
    var lane = document.createElement("span");
    lane.className =
      "text-[10px] uppercase tracking-wide px-2 py-0.5 rounded-full border " +
      (gpu.lane === "unassigned"
        ? "border-gray-700 bg-gray-900 text-gray-500"
        : "border-indigo-700 bg-indigo-950/60 text-indigo-200");
    lane.textContent = gpu.lane;
    head.appendChild(title);
    head.appendChild(lane);
    box.appendChild(head);

    var memFraction =
      gpu.memory_used_mb !== null && gpu.memory_used_mb !== undefined &&
      gpu.memory_total_mb !== null && gpu.memory_total_mb !== undefined && Number(gpu.memory_total_mb) > 0
        ? Number(gpu.memory_used_mb) / Number(gpu.memory_total_mb)
        : null;
    var utilFraction =
      gpu.utilization_gpu !== null && gpu.utilization_gpu !== undefined ? Number(gpu.utilization_gpu) / 100 : null;

    var grid = document.createElement("div");
    grid.className = "grid grid-cols-3 gap-2";
    grid.appendChild(tile("util", fmt(gpu.utilization_gpu, 0) + "%", null, { tone: toneForPressure(utilFraction) }));
    grid.appendChild(
      tile("mem", fmt(gpu.memory_used_mb, 0) + " / " + fmt(gpu.memory_total_mb, 0) + " MB", null, {
        tone: toneForPressure(memFraction),
      })
    );
    grid.appendChild(tile("power", fmt(gpu.power_draw_watts, 1) + " W"));
    box.appendChild(grid);

    box.appendChild(sparkline(gpu.trend || []));

    var procHeader = document.createElement("div");
    procHeader.className = "text-[10px] uppercase tracking-wide text-gray-500 mt-1";
    procHeader.textContent = "processes";
    box.appendChild(procHeader);
    var procs = gpu.processes || [];
    if (!procs.length) {
      var none = document.createElement("div");
      none.className = "text-[11px] text-gray-500";
      none.textContent = "none reported";
      box.appendChild(none);
    } else {
      procs.forEach(function (p) {
        var row = document.createElement("div");
        row.className = "text-[11px] font-mono text-gray-300 truncate";
        row.textContent = "pid " + p.pid + " · " + p.process_name + " · " + fmt(p.used_memory_mb, 0) + " MB";
        box.appendChild(row);
      });
    }
    return box;
  }

  async function loadGpu(node) {
    var status = el("biometricsGpuStatus");
    var grid = el("biometricsGpuGrid");
    if (!grid) return;
    if (status) status.textContent = "Loading…";
    // limit=40 (endpoint max is 60): the default 5-sample buffer read made
    // the "realtime trend" sparkline look almost flat/empty -- 40 samples at
    // orion-biometrics' collection cadence gives a real trend to look at.
    var payload = await fetchJson("/api/biometrics/preview/gpu?node=" + node + "&limit=40").catch(function () {
      return { ok: false, gpus: [] };
    });
    clear(grid);
    (payload.gpus || []).forEach(function (gpu) {
      grid.appendChild(gpuCard(gpu));
    });
    if (status) {
      status.textContent = payload.ok
        ? (payload.gpus || []).length + " GPU(s) on " + node
        : "GPU data unavailable for " + node;
    }
    loaded.gpu[node] = true;
  }

  function setGpuNode(node) {
    gpuNode = node;
    document.querySelectorAll("[data-biometrics-gpu-node]").forEach(function (btn) {
      var active = btn.getAttribute("data-biometrics-gpu-node") === node;
      btn.classList.toggle("border-indigo-500", active);
      btn.classList.toggle("bg-indigo-950/60", active);
      btn.classList.toggle("text-indigo-200", active);
      btn.classList.toggle("border-gray-700", !active);
      btn.classList.toggle("bg-gray-900", !active);
      btn.classList.toggle("text-gray-400", !active);
    });
    loadGpu(node);
  }

  // --- Modal subview switching --------------------------------------------

  function showModalSubview(name) {
    modalSubview = name;
    var panels = {
      athena: el("biometricsSubviewAthena"),
      circe: el("biometricsSubviewCirce"),
      gpu: el("biometricsSubviewGpu"),
      cabinet: el("cabinet"),
    };
    var buttons = {
      athena: el("biometricsSubtabAthena"),
      circe: el("biometricsSubtabCirce"),
      gpu: el("biometricsSubtabGpu"),
      cabinet: el("biometricsSubtabCabinet"),
    };
    Object.keys(panels).forEach(function (key) {
      if (panels[key]) panels[key].classList.toggle("hidden", key !== name);
      if (buttons[key]) {
        var active = key === name;
        buttons[key].classList.toggle("bg-gray-700", active);
        buttons[key].classList.toggle("text-gray-100", active);
        buttons[key].classList.toggle("bg-gray-800", !active);
        buttons[key].classList.toggle("text-gray-300", !active);
      }
    });

    if (gpuPollTimer) {
      clearInterval(gpuPollTimer);
      gpuPollTimer = null;
    }
    if (window.OrionCabinetSensors && typeof window.OrionCabinetSensors.deactivate === "function") {
      window.OrionCabinetSensors.deactivate();
    }

    if (name === "athena" && !loaded.athena) {
      loaded.athena = true;
      loadNodeDetail("athena");
    } else if (name === "circe" && !loaded.circe) {
      loaded.circe = true;
      loadNodeDetail("circe");
    } else if (name === "gpu") {
      if (!loaded.gpu[gpuNode]) loadGpu(gpuNode);
      gpuPollTimer = setInterval(function () {
        loadGpu(gpuNode);
      }, GPU_POLL_MS);
    } else if (name === "cabinet") {
      if (window.OrionCabinetSensors && typeof window.OrionCabinetSensors.activate === "function") {
        window.OrionCabinetSensors.activate();
      }
    }
  }

  // --- Modal open/close (mechanics live in app.js) ------------------------

  function openModal() {
    if (typeof window.openBiometricsModal === "function") window.openBiometricsModal();
  }

  function closeModal() {
    if (typeof window.closeBiometricsModal === "function") window.closeBiometricsModal();
  }

  function onModalOpen() {
    modalOpen = true;
    showModalSubview(modalSubview);
  }

  function onModalClose() {
    modalOpen = false;
    if (gpuPollTimer) {
      clearInterval(gpuPollTimer);
      gpuPollTimer = null;
    }
    if (window.OrionCabinetSensors && typeof window.OrionCabinetSensors.deactivate === "function") {
      window.OrionCabinetSensors.deactivate();
    }
  }

  // --- Wiring ---------------------------------------------------------

  var wired = false;

  function wireOnce() {
    if (wired) return;
    wired = true;

    var toggle = el("ekgViewToggle");
    if (toggle) toggle.addEventListener("click", toggleCardView);

    var preview = el("biometricsPreviewContainer");
    if (preview) preview.addEventListener("click", openModal);

    [
      ["biometricsSubtabAthena", "athena"],
      ["biometricsSubtabCirce", "circe"],
      ["biometricsSubtabGpu", "gpu"],
      ["biometricsSubtabCabinet", "cabinet"],
    ].forEach(function (pair) {
      var btn = el(pair[0]);
      if (btn) btn.addEventListener("click", () => showModalSubview(pair[1]));
    });

    document.querySelectorAll("[data-biometrics-gpu-node]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        setGpuNode(btn.getAttribute("data-biometrics-gpu-node"));
      });
    });
  }

  function activate() {
    wireOnce();
    // Resume the preview poll if the operator had it toggled on before
    // navigating away (deactivate() only stops the timer, it doesn't
    // reset cardView -- coming back to "brain" silently would be a second,
    // unrelated bug).
    if (cardView === "biometrics") showCardView("biometrics");
  }

  function deactivate() {
    if (cardPollTimer) {
      clearInterval(cardPollTimer);
      cardPollTimer = null;
    }
  }

  document.addEventListener("DOMContentLoaded", wireOnce);

  window.OrionBiometricsView = {
    activate,
    deactivate,
    openModal,
    closeModal,
    onModalOpen,
    onModalClose,
    showModalSubview,
  };
})();
