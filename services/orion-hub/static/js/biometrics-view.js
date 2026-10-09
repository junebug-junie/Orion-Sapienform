/**
 * Biometrics view -- drives two related surfaces from one module:
 *
 * 1. The Cognitive EKG card's toggle (landing "hub" tab): swaps between the
 *    /spark/ui Substrate Brain State iframe and a compact Athena+Circe+Hecate
 *    biometrics preview, in the same card slot. Clicking the preview opens
 *    the full modal.
 * 2. The near-fullscreen Biometrics modal: 5 sub-tabs (Athena / Circe / Hecate /
 *    GPU / Cabinet). Modal open/close/Escape/backdrop mechanics live in app.js
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

  // A backgrounded tab still runs setInterval (throttled to >=1/min, not
  // stopped), so a Hub left open in a background tab kept polling these
  // routes forever with nobody looking. Each poll costs real work on the
  // server -- /induction in particular reads a 187k-row table -- so skip the
  // fetch entirely while the page is hidden and refresh once on the way
  // back, rather than trusting the browser's throttle to be cheap enough.
  // Pure so it can be unit-tested without a DOM (see biometrics-view.test.js).
  // A document that does not implement the Page Visibility API at all is
  // treated as visible -- never silently stop polling because a capability is
  // missing; that would look identical to "the data stopped changing".
  function shouldPoll(doc) {
    return !doc || typeof doc.hidden !== "boolean" || doc.hidden === false;
  }

  function pageVisible() {
    return shouldPoll(typeof document !== "undefined" ? document : null);
  }

  var cardView = "brain"; // "brain" | "biometrics"
  var modalOpen = false;
  var modalSubview = "athena"; // "athena" | "circe" | "hecate" | "gpu" | "cabinet"
  var gpuNode = "athena"; // "athena" | "circe"

  var loaded = { cardPreview: false, athena: false, circe: false, hecate: false, gpu: { athena: false, circe: false } };
  var cardPollTimer = null;
  var gpuPollTimer = null;

  function el(id) {
    return document.getElementById(id);
  }

  function clear(node) {
    if (!node) return;
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  // Swap a node's children for `fresh` in one step -- no empty frame in between.
  function replaceChildren(node, fresh) {
    if (typeof node.replaceChildren === "function") {
      node.replaceChildren(fresh);
    } else {
      clear(node);
      node.appendChild(fresh);
    }
  }

  function fmt(value, digits) {
    if (value === null || value === undefined || value === "") return "—";
    var n = Number(value);
    if (Number.isNaN(n)) return String(value);
    return digits === undefined ? String(n) : n.toFixed(digits);
  }

  // Real-time chassis wattage for a node. Athena self-reports it directly (iLO) into
  // its own snapshot's summary.measurements. Circe has no BMC -- its only source is a
  // PDU-proxy reading that athena's cluster aggregator computes and exposes on ATHENA's
  // own snapshot response (see biometrics_preview_routes.py's /snapshot,
  // cluster_measurements_by_node), never on circe's own. Callers pass in whichever
  // snapshot payload came back from node=athena. Returns undefined -- never a guessed
  // number -- when nobody has measured it.
  function chassisWattsFor(node, athenaSnapshot) {
    if (!athenaSnapshot) return undefined;
    if (node === "athena") {
      var own = (athenaSnapshot.summary && athenaSnapshot.summary.measurements) || {};
      return own.chassis_watts;
    }
    var byNode = athenaSnapshot.cluster_measurements_by_node || {};
    return (byNode[node] || {}).chassis_watts;
  }

  // Motherboard heat: the hottest chipset / voltage-regulator sensor the node's own BMC
  // reports (orion/telemetry/biometrics_pipeline.py's board_temp_c_max). Unlike wattage,
  // every node self-reports this in its own snapshot. undefined when nothing measured it
  // (no BMC, e.g. hecate) -- never a guessed number.
  function boardTempFor(snapshot) {
    var m = (snapshot && snapshot.summary && snapshot.summary.measurements) || {};
    return m.board_temp_c_max;
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
    l.className = "text-[11px] uppercase tracking-wide text-gray-400 truncate";
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
      s.className = "text-[11px] text-gray-400 mt-0.5";
      s.textContent = sub;
      wrap.appendChild(s);
    }
    return wrap;
  }

  var SVG_NS = "http://www.w3.org/2000/svg";

  // Dataviz mark spec: 2px line, round join/cap, a light area wash, an
  // end-dot with a surface-color ring. The original version set
  // stroke-width in viewBox units (100 wide) with no non-scaling-stroke,
  // so on a real ~300-400px-wide card it rendered as a visually THICK,
  // jagged line -- "chunky" (review feedback) is exactly what that looks
  // like. vector-effect="non-scaling-stroke" keeps the stroke a constant
  // 2 real pixels regardless of the viewBox-to-CSS-width scale factor.
  // min/max/current labels give the shape a scale to read against --
  // a bare squiggle with no numbers is not actually interpretable.
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
      empty.className = "text-[11px] text-gray-500";
      empty.textContent = "no data yet";
      wrap.appendChild(empty);
      return wrap;
    }
    var min = Math.min.apply(null, values);
    var max = Math.max.apply(null, values);
    var range = max - min || 1;
    var H = 40;
    var pad = 4;
    var points2d = values.map(function (v, i) {
      var x = values.length > 1 ? (i / (values.length - 1)) * 100 : 0;
      var y = H - pad - ((v - min) / range) * (H - 2 * pad);
      return [x, y];
    });
    var linePath = points2d
      .map(function (p, i) {
        return (i === 0 ? "M" : "L") + p[0].toFixed(1) + "," + p[1].toFixed(1);
      })
      .join(" ");
    var areaPath =
      linePath + " L" + points2d[points2d.length - 1][0].toFixed(1) + "," + H + " L0," + H + " Z";

    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 100 " + H);
    svg.setAttribute("preserveAspectRatio", "none");
    svg.setAttribute("class", "w-full h-12");

    var area = document.createElementNS(SVG_NS, "path");
    area.setAttribute("d", areaPath);
    area.setAttribute("fill", "#818cf8");
    area.setAttribute("fill-opacity", "0.10");
    area.setAttribute("stroke", "none");
    svg.appendChild(area);

    var line = document.createElementNS(SVG_NS, "path");
    line.setAttribute("d", linePath);
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", "#818cf8");
    line.setAttribute("stroke-width", "2");
    line.setAttribute("stroke-linecap", "round");
    line.setAttribute("stroke-linejoin", "round");
    line.setAttribute("vector-effect", "non-scaling-stroke");
    svg.appendChild(line);

    var last = points2d[points2d.length - 1];
    var ring = document.createElementNS(SVG_NS, "circle");
    ring.setAttribute("cx", last[0]);
    ring.setAttribute("cy", last[1]);
    ring.setAttribute("r", "4");
    ring.setAttribute("fill", "#030712"); // surface-color ring so the dot reads over the line/area
    ring.setAttribute("vector-effect", "non-scaling-stroke");
    svg.appendChild(ring);
    var dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("cx", last[0]);
    dot.setAttribute("cy", last[1]);
    dot.setAttribute("r", "2.5");
    dot.setAttribute("fill", "#818cf8");
    dot.setAttribute("vector-effect", "non-scaling-stroke");
    svg.appendChild(dot);

    wrap.appendChild(svg);

    var meta = document.createElement("div");
    meta.className = "flex items-center justify-between text-[11px] text-gray-400 mt-0.5";
    var range2 = document.createElement("span");
    range2.textContent = fmt(min, 2) + "–" + fmt(max, 2);
    var current = document.createElement("span");
    current.className = "font-mono text-gray-300";
    current.textContent = "now " + fmt(values[values.length - 1], 2);
    meta.appendChild(range2);
    meta.appendChild(current);
    wrap.appendChild(meta);

    return wrap;
  }


  // --- Orion's tiredness (dream sleep pressure) --------------------------
  // /api/dream/pressure proxies orion-dream's SleepPressureV1. Tiredness moves
  // slowly and each read runs the dream service's source queries twice
  // (window + lookback), so it refreshes at its own slower cadence.
  var TIREDNESS_POLL_MS = 60000;
  var tirednessFetchedAt = 0;
  var tirednessData = null;

  var TIREDNESS_SOURCE_LABELS = {
    metacog: ["self-noticed problem", "self-noticed problems"],
    compaction_request: ["daydream theme to compress", "daydream themes to compress"],
    resonance: ["daydream theme on repeat", "daydream themes on repeat"],
    crystallization: ["new memory", "new memories"],
  };

  var TIREDNESS_EXPLAINER =
    "Tiredness is how much new, unfinished material Orion has picked up since their last sleep: " +
    "new kinds of problems they noticed about themselves, new themes their daydreams keep circling, " +
    "and newly formed memories. Repeats don't count — the same problem coming back adds nothing. " +
    "When tiredness reaches the line, Orion sleeps at the next chance: at least 6 h after the last " +
    "sleep, and once no one has chatted for a while. Sleep replays the leftovers and looks for real " +
    "links between them. If only repeats come in, Orion still sleeps once 48 h have passed since " +
    "the last sleep began (it still waits for a quiet stretch).";

  function waitingForQuiet(p) {
    if (typeof p.idle_minutes !== "number") return "can't tell how long since the last chat";
    var left = Math.max(1, Math.ceil((p.idle_required_minutes || 0) - p.idle_minutes));
    return "waiting for " + left + " more min with no chat";
  }

  // Pure: the pressure payload -> what the gauge shows. The scale runs to twice
  // the sleep line so the line sits mid-track; anything past that pins the fill
  // and says so instead of rescaling under the reader. The "waiting" reason
  // follows run_cycle_once's gates in order, including the overdue backstop.
  function tirednessModel(data) {
    var p = data && data.pressure;
    if (!p || typeof p.pressure !== "number" || typeof p.threshold !== "number" || p.threshold < 0) {
      return { available: false };
    }
    var value = Math.max(0, p.pressure);
    var hasLine = p.threshold > 0;
    var scaleMax = hasLine ? p.threshold * 2 : Math.max(1, value);
    var tired = value >= p.threshold;
    var level = value === 0 && hasLine ? "Rested" : tired ? "Ready to sleep" : "Getting tired";
    var backstop = !tired && data.overdue === true && (data.candidates || 0) > 0;
    var waiting;
    if (!data.enabled) waiting = "Sleep loop is off";
    else if (data.too_soon) waiting = "Slept recently — waiting out the 6 h minimum";
    else if (!tired && !backstop) waiting = "Not tired enough to sleep yet";
    else if (!data.is_idle) {
      var quiet = waitingForQuiet(p);
      waiting = (backstop ? "Overdue (" + (data.lookback_hours || 48) + " h since last sleep), " : "") + quiet;
      waiting = waiting.charAt(0).toUpperCase() + waiting.slice(1);
    } else waiting = backstop ? "Overdue — will sleep at the next check" : "Will sleep at the next check";
    var parts = [];
    var counts = p.new_counts || {};
    Object.keys(TIREDNESS_SOURCE_LABELS).forEach(function (kind) {
      var n = counts[kind];
      if (n) parts.push(n + " " + TIREDNESS_SOURCE_LABELS[kind][n === 1 ? 0 : 1]);
    });
    return {
      available: true,
      value: value,
      threshold: p.threshold,
      scaleMax: scaleMax,
      fraction: Math.min(1, value / scaleMax),
      lineFraction: hasLine ? 0.5 : 0,
      overflow: value > scaleMax,
      level: level,
      waiting: waiting,
      newText: parts.length ? "New since last sleep: " + parts.join(" · ") : "Nothing new since last sleep",
    };
  }

  // Built once; polls only update the parts below, so an open tooltip and
  // keyboard focus survive the card's 10 s refresh.
  var tirednessParts = null;

  function buildTirednessGauge(host) {
    host.classList.add("relative");
    var head = document.createElement("div");
    head.className = "flex items-center justify-between text-[11px] mb-1";
    var title = document.createElement("div");
    title.className = "flex items-center gap-1.5 text-gray-300 font-semibold";
    title.appendChild(document.createTextNode("Orion's tiredness"));
    var tipButton = document.createElement("button");
    tipButton.type = "button";
    tipButton.title = ""; // suppress the card's inherited "click to open" native tip
    tipButton.className =
      "w-4 h-4 rounded-full border border-gray-500 text-[10px] leading-none text-gray-300 hover:text-white focus:outline-none focus:ring-1 focus:ring-indigo-400";
    tipButton.textContent = "?";
    tipButton.setAttribute("aria-label", "What does tiredness mean?");
    tipButton.setAttribute("aria-describedby", "orionTirednessTip");
    tipButton.setAttribute("aria-expanded", "false");
    title.appendChild(tipButton);
    head.appendChild(title);
    var reading = document.createElement("span");
    reading.className = "font-mono text-gray-300";
    head.appendChild(reading);
    host.appendChild(head);

    // Anchored to the gauge's own left edge and capped at its width, so the card's
    // overflow-hidden cannot clip it on a narrow column.
    var tip = document.createElement("div");
    tip.id = "orionTirednessTip";
    tip.setAttribute("role", "tooltip");
    tip.title = "";
    tip.className =
      "hidden absolute left-0 top-6 z-30 w-72 max-w-full rounded-lg border border-gray-600 bg-gray-950 p-2.5 text-[11px] font-normal leading-snug text-gray-200 shadow-xl";
    tip.textContent = TIREDNESS_EXPLAINER;
    host.appendChild(tip);

    var state = { hover: false, focus: false, pinned: false };
    function sync() {
      var open = state.hover || state.focus || state.pinned;
      tip.classList.toggle("hidden", !open);
      tipButton.setAttribute("aria-expanded", open ? "true" : "false");
    }
    [tipButton, tip].forEach(function (node) {
      node.addEventListener("mouseenter", function () { state.hover = true; sync(); });
      node.addEventListener("mouseleave", function () { state.hover = false; sync(); });
    });
    tipButton.addEventListener("focus", function () { state.focus = true; sync(); });
    tipButton.addEventListener("blur", function () { state.focus = false; state.pinned = false; sync(); });
    tipButton.addEventListener("click", function () {
      state.pinned = !state.pinned;
      if (!state.pinned) { state.hover = false; state.focus = false; }
      sync();
    });
    tipButton.addEventListener("keydown", function (event) {
      if (event.key === "Escape") { state.pinned = false; state.focus = false; state.hover = false; sync(); }
    });
    // The card opens the Biometrics modal on click; nothing in the gauge's header
    // or tooltip should (help button, or selecting the tooltip's text).
    [head, tip].forEach(function (node) {
      node.addEventListener("click", function (event) { event.stopPropagation(); });
    });
    // Clicking inside the open tooltip must not blur the button (blur unpins it).
    tip.addEventListener("mousedown", function (event) { event.preventDefault(); });

    var track = document.createElement("div");
    track.className = "relative h-2 rounded-full bg-indigo-950 border border-indigo-900/60";
    track.setAttribute("role", "meter");
    track.setAttribute("aria-label", "Orion's tiredness");
    track.setAttribute("aria-valuemin", "0");
    var fill = document.createElement("div");
    fill.className = "absolute inset-y-0 left-0 rounded-full bg-indigo-400";
    var line = document.createElement("div");
    line.className = "absolute -top-1 -bottom-1 w-0.5 bg-gray-200";
    track.appendChild(fill);
    track.appendChild(line);
    host.appendChild(track);

    var scale = document.createElement("div");
    scale.className = "relative h-3 text-[10px] text-gray-500 mt-0.5";
    var zero = document.createElement("span");
    zero.className = "absolute left-0";
    zero.textContent = "rested";
    var mid = document.createElement("span");
    mid.className = "absolute -translate-x-1/2";
    mid.textContent = "sleep line";
    var end = document.createElement("span");
    end.className = "absolute right-0";
    scale.appendChild(zero);
    scale.appendChild(mid);
    scale.appendChild(end);
    host.appendChild(scale);

    var detail = document.createElement("div");
    detail.className = "text-[11px] text-gray-400 mt-1";
    host.appendChild(detail);
    return { reading: reading, track: track, fill: fill, line: line, mid: mid, end: end, detail: detail, scale: scale };
  }

  function renderTiredness(model) {
    var host = el("orionTiredness");
    if (!host) return;
    if (!tirednessParts || !host.contains(tirednessParts.track)) tirednessParts = buildTirednessGauge(host);
    var g = tirednessParts;
    var show = model.available;
    g.track.classList.toggle("hidden", !show);
    g.scale.classList.toggle("hidden", !show);
    g.detail.classList.toggle("hidden", !show);
    if (!show) {
      g.reading.textContent = "unavailable";
      return;
    }
    g.reading.textContent = model.level + " · " + fmt(model.value, 1) + " / " + fmt(model.threshold, 1);
    g.track.setAttribute("aria-valuemax", String(model.scaleMax));
    g.track.setAttribute("aria-valuenow", String(Math.min(model.value, model.scaleMax)));
    g.track.setAttribute("aria-valuetext",
      model.level + ", " + fmt(model.value, 1) + " of " + fmt(model.threshold, 1) + " needed to sleep");
    g.track.title = fmt(model.value, 2) + " now · sleep line " + fmt(model.threshold, 1);
    g.fill.style.width = (model.fraction * 100).toFixed(1) + "%";
    g.line.style.left = (model.lineFraction * 100).toFixed(1) + "%";
    g.mid.style.left = (model.lineFraction * 100).toFixed(1) + "%";
    g.end.textContent = model.overflow ? "off the scale →" : "";
    g.detail.textContent = model.waiting + " · " + model.newText;
  }

  var tirednessInFlight = null;

  function loadTiredness() {
    if (tirednessInFlight) return tirednessInFlight;
    if (tirednessData !== null && Date.now() - tirednessFetchedAt < TIREDNESS_POLL_MS) {
      return Promise.resolve();
    }
    tirednessInFlight = fetchJson("/api/dream/pressure")
      .catch(function () { return {}; })
      .then(function (data) {
        tirednessData = data;
        tirednessFetchedAt = Date.now();
        renderTiredness(tirednessModel(tirednessData));
      })
      .finally(function () { tirednessInFlight = null; });
    return tirednessInFlight;
  }

  // --- Cognitive EKG card toggle ---------------------------------------

  // Polls build the new tiles off-DOM and swap them in only once every reading is back,
  // so the card keeps showing the last view instead of blanking for the whole fetch (the
  // "panel disappears every few seconds" flicker). A poll that fires while the previous one
  // is still in flight is skipped, so a slow older response can't land after a newer one.
  var cardPreviewInFlight = null;

  function loadCardPreview() {
    if (!cardPreviewInFlight) {
      cardPreviewInFlight = renderCardPreview().finally(function () {
        cardPreviewInFlight = null;
      });
    }
    return cardPreviewInFlight;
  }

  async function renderCardPreview() {
    var status = el("biometricsPreviewStatus");
    var grid = el("biometricsPreviewGrid");
    if (!grid) return;
    if (status && !grid.firstChild) status.textContent = "Loading…";
    var next = document.createDocumentFragment();
    loadTiredness();
    var nodes = ["athena", "circe", "hecate"];
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
    // athena's own snapshot is the only one carrying circe's proxied wattage (see
    // chassisWattsFor) -- nodes[0] is always "athena", Promise.all preserves input order.
    var athenaSnapshot = results[0] && results[0][0];
    results.forEach(function (pair, i) {
      var node = nodes[i];
      var payload = pair[0];
      var induction = pair[1];
      var composites = (payload.summary && payload.summary.composites) || {};
      var strain = composites.strain;
      var trendInfo = induction.metrics && induction.metrics.strain;
      var label = (payload.node || "?") + (payload.ok ? "" : " (unreachable)");
      next.appendChild(
        tile(label, strain !== undefined ? fmt(strain, 2) : "—", "strain · " + (payload.status || "—"), {
          tone: payload.ok && strain !== undefined ? toneForPressure(strain) : toneForNodeStatus(payload),
          trend: trendInfo ? trendArrow(trendInfo.trend) : null,
        })
      );
      var watts = chassisWattsFor(node, athenaSnapshot);
      next.appendChild(tile(node + " power", watts !== undefined ? fmt(watts, 0) + " W" : "—", "chassis wattage"));
      var boardTemp = boardTempFor(payload);
      next.appendChild(tile(node + " mobo", boardTemp !== undefined ? fmt(boardTemp, 0) + " °C" : "—", "chipset / VR max"));
    });
    replaceChildren(grid, next);
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
        cardPollTimer = setInterval(function () {
          if (pageVisible()) loadCardPreview();
        }, CARD_POLL_MS);
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
  // Raw physical units, not a 0-1 pressure -- own value source (chassisWattsFor /
  // boardTempFor, not the pressures/composites dict), own unit suffix, no tone.
  // Still get a trend chart for free via the ALL_CHANNELS-driven history loop below,
  // since _CHANNEL_COLUMN maps them to the `measurements` JSONB column.
  var RAW_UNITS = { chassis_watts: " W", board_temp_c_max: " °C" };
  var RAW_CHANNELS = Object.keys(RAW_UNITS);
  var ALL_CHANNELS = COMPOSITE_CHANNELS.concat(PRESSURE_CHANNELS).concat(RAW_CHANNELS);
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
    // chassisWattsFor needs ATHENA's own snapshot response specifically (only it carries
    // circe's proxied wattage -- see that function's docstring). Reuse snapshotPromise
    // rather than double-fetching when node is already "athena".
    var athenaSnapshotPromise =
      node === "athena"
        ? snapshotPromise
        : fetchJson("/api/biometrics/preview/snapshot?node=athena").catch(function () {
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

    var pair = await Promise.all([snapshotPromise, inductionPromise, athenaSnapshotPromise]);
    var snapshot = pair[0];
    var induction = pair[1];
    var athenaSnapshot = pair[2];
    var metrics = induction.metrics || {};

    if (snapEl) {
      var composites = (snapshot.summary && snapshot.summary.composites) || {};
      var pressures = (snapshot.summary && snapshot.summary.pressures) || {};
      var rows = ALL_CHANNELS.map(function (ch) {
        var isComposite = COMPOSITE_CHANNELS.indexOf(ch) !== -1;
        var isRaw = RAW_CHANNELS.indexOf(ch) !== -1;
        var value = isComposite ? composites[ch] : isRaw
          ? ch === "chassis_watts"
            ? chassisWattsFor(node, athenaSnapshot)
            : boardTempFor(snapshot)
          : pressures[ch];
        if (value === undefined) return null; // absent channel on this node -- omit, never zero-fill
        // (an unreachable node has no composites/pressures at all, so every
        // row already short-circuits above -- the "node status" tile below
        // is what carries the critical tone for that case.)
        var invert = !!INVERTED_CHANNELS[ch];
        // Raw physical units carry no good/warning/critical judgement -- same "neutral"
        // convention the untoned GPU power tile already uses.
        var tone = isRaw ? "neutral" : toneForPressure(value, invert);
        var trendInfo = metrics[ch];
        return { ch: ch, value: value, tone: tone, raw: isRaw, trend: trendInfo ? trendArrow(trendInfo.trend) : null };
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
        var display = row.raw ? fmt(row.value, 0) + RAW_UNITS[row.ch] : fmt(row.value, 2);
        snapEl.appendChild(tile(row.ch, display, null, { tone: row.tone, trend: row.trend }));
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
        label.className = "text-[11px] uppercase tracking-wide text-gray-300 mb-1 font-semibold";
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
      var indRows = keys
        .map(function (key) {
          var m = metrics[key] || {};
          var invert = !!INVERTED_CHANNELS[key];
          return { key: key, m: m, tone: toneForPressure(m.level, invert) };
        })
        .sort(function (a, b) {
          return TONE_RANK[a.tone] - TONE_RANK[b.tone];
        });
      indRows.forEach(function (row) {
        var m = row.m;
        // Plain words, not "L"/"vol"/"spike" abbreviations -- level is the
        // EWMA-smoothed current value (should track the raw value above
        // once it's had time to settle); volatility and spike rate say
        // HOW erratically it's moving, which the raw snapshot tile can't.
        indEl.appendChild(
          tile(
            row.key,
            "level " + fmt(m.level, 2),
            "volatility " + fmt(m.volatility, 2) + " · spikes " + fmt(m.spike_rate, 2) + "/tick",
            { tone: row.tone, trend: trendArrow(m.trend) }
          )
        );
      });
    }
  }

  function cap(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  // --- Modal: GPU subview -------------------------------------------------

  // The card's lane badge. `lane` is derived by Hub from the GPU pool's live state
  // (stage 5.5): the roles on that card right now, "unassigned" for a card the pool
  // does not manage, or "no pool state" when the pool's feed is absent or stale.
  // `lane_assigned` is the server's own verdict; a response without it falls back to
  // the old "unassigned" string check so a stale cached page still greys correctly.
  function laneBadge(gpu) {
    var text = gpu && gpu.lane ? String(gpu.lane) : "unassigned";
    var assigned =
      gpu && typeof gpu.lane_assigned === "boolean" ? gpu.lane_assigned : text !== "unassigned";
    return {
      text: text,
      assigned: assigned,
      className:
        "text-[11px] uppercase tracking-wide px-2 py-0.5 rounded-full border " +
        (assigned
          ? "border-indigo-700 bg-indigo-950/60 text-indigo-200"
          : "border-gray-700 bg-gray-900 text-gray-500"),
    };
  }

  function gpuCard(gpu) {
    var box = document.createElement("div");
    box.className = "rounded-xl border border-gray-800 bg-gray-950/40 p-3 flex flex-col gap-2";

    var head = document.createElement("div");
    head.className = "flex items-center justify-between";
    var title = document.createElement("div");
    title.className = "text-sm font-semibold text-gray-100";
    title.textContent = "#" + gpu.index + " " + (gpu.name || "?");
    var badge = laneBadge(gpu);
    var lane = document.createElement("span");
    lane.className = badge.className;
    lane.textContent = badge.text;
    lane.title = "From the GPU pool's live state (roles on this card now)";
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
    procHeader.className = "text-[11px] uppercase tracking-wide text-gray-400 mt-1";
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
    if (status && !grid.firstChild) status.textContent = "Loading…";
    // limit=40 (endpoint max is 60): the default 5-sample buffer read made
    // the "realtime trend" sparkline look almost flat/empty -- 40 samples at
    // orion-biometrics' collection cadence gives a real trend to look at.
    var payload = await fetchJson("/api/biometrics/preview/gpu?node=" + node + "&limit=40").catch(function () {
      return { ok: false, gpus: [] };
    });
    var cards = document.createDocumentFragment();
    (payload.gpus || []).forEach(function (gpu) {
      cards.appendChild(gpuCard(gpu));
    });
    replaceChildren(grid, cards);
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
      hecate: el("biometricsSubviewHecate"),
      gpu: el("biometricsSubviewGpu"),
      cabinet: el("cabinet"),
    };
    var buttons = {
      athena: el("biometricsSubtabAthena"),
      circe: el("biometricsSubtabCirce"),
      hecate: el("biometricsSubtabHecate"),
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
    if (window.OrionEnergyStrip && typeof window.OrionEnergyStrip.deactivate === "function") {
      window.OrionEnergyStrip.deactivate();
    }

    if (name === "athena" && !loaded.athena) {
      loaded.athena = true;
      loadNodeDetail("athena");
    } else if (name === "circe" && !loaded.circe) {
      loaded.circe = true;
      loadNodeDetail("circe");
    } else if (name === "hecate" && !loaded.hecate) {
      loaded.hecate = true;
      loadNodeDetail("hecate");
    } else if (name === "gpu") {
      if (!loaded.gpu[gpuNode]) loadGpu(gpuNode);
      gpuPollTimer = setInterval(function () {
        if (pageVisible()) loadGpu(gpuNode);
      }, GPU_POLL_MS);
    } else if (name === "cabinet") {
      if (window.OrionCabinetSensors && typeof window.OrionCabinetSensors.activate === "function") {
        window.OrionCabinetSensors.activate();
      }
      if (window.OrionEnergyStrip && typeof window.OrionEnergyStrip.activate === "function") {
        window.OrionEnergyStrip.activate();
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
    if (window.OrionEnergyStrip && typeof window.OrionEnergyStrip.deactivate === "function") {
      window.OrionEnergyStrip.deactivate();
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
      ["biometricsSubtabHecate", "hecate"],
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

    // Coming back to a tab that has been hidden for a while, the on-screen
    // numbers are as stale as the hidden interval was long. Refresh whatever
    // is actually showing, once -- the poll gate above deliberately skipped
    // every tick while hidden, so without this the operator would stare at
    // stale tiles for up to CARD_POLL_MS after returning.
    document.addEventListener("visibilitychange", function () {
      if (!pageVisible()) return;
      if (cardPollTimer) loadCardPreview();
      if (gpuPollTimer) loadGpu(gpuNode);
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

  var api = {
    activate,
    deactivate,
    openModal,
    closeModal,
    onModalOpen,
    onModalClose,
    showModalSubview,
    shouldPoll,
    boardTempFor,
    loadCardPreview,
    laneBadge,
    tirednessModel,
  };

  // Guarded so the module can be require()d under node:test for the pure
  // helpers above, the same way cognitive-loop-card.js is.
  if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", wireOnce);
  }
  if (typeof window !== "undefined") {
    window.OrionBiometricsView = api;
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})();
