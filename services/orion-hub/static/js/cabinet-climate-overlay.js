// Cabinet / Office climate overlay (Biometrics -> Cabinet / Office subview).
//
// Read-only view of GET /api/cabinet/climate/overlay (cabinet_climate_routes.py):
// office, cabinet and outdoor temperature overlaid on one °C plot, with Orion's total
// wall watts in a panel underneath on the same time axis. One crosshair and tooltip
// span both. Watts get their own y-scale panel rather than a second axis on the
// temperature plot, so no line's slope is an artifact of two scales.
//
// Filter chips above the chart include/exclude each series (and act as the legend).
// Absent is not zero: an empty bucket breaks the line; the tooltip says "no reading".
//
// Polls only while the #cabinet panel is visible.
(function () {
  "use strict";

  var URL = "/api/cabinet/climate/overlay?window=";
  var SVG_NS = "http://www.w3.org/2000/svg";
  var POLL_MS = 60000;
  var TICK_MS = 2000;
  // A spacing wider than this many series grains is an outage: break the line there.
  var GAP_GRAINS = 2.5;

  // Fixed per series, never by rank (dataviz reference palette, dark steps; validated on
  // the Hub surface). Watts uses the Orion-total violet from the Orion energy tab.
  var SERIES = [
    { key: "office_temp_c", label: "office", color: "#3987e5", unit: "°C", digits: 1, panel: "temp" },
    { key: "cabinet_temp_c", label: "cabinet", color: "#d95926", unit: "°C", digits: 1, panel: "temp" },
    { key: "outdoor_temp_c", label: "outdoor", color: "#199e70", unit: "°C", digits: 1, panel: "temp" },
    { key: "orion_watts", label: "Orion power", color: "#9085e9", unit: "W", digits: 0, panel: "watts" },
  ];

  var els = {};
  var state = {
    window: "24h",
    enabled: { office_temp_c: true, cabinet_temp_c: true, outdoor_temp_c: true, orion_watts: true },
    payload: null,
    lastFetchAt: 0,
    request: 0,
    timer: null,
  };

  // ---------------------------------------------------------------- pure helpers

  // [{t, v}] -> runs of [{ms, v, ...}], split wherever spacing exceeds GAP_GRAINS x grain.
  function splitAtGaps(points, grainSec) {
    var limit = (grainSec > 0 ? grainSec : 300) * 1000 * GAP_GRAINS;
    var runs = [];
    var prev = null;
    (points || []).forEach(function (p) {
      var ms = p && p.t ? Date.parse(p.t) : NaN;
      if (!isFinite(ms) || typeof p.v !== "number" || !isFinite(p.v)) return;
      var point = Object.assign({}, p, { ms: ms });
      if (prev === null || ms - prev > limit) runs.push([point]);
      else runs[runs.length - 1].push(point);
      prev = ms;
    });
    return runs;
  }

  // About `count` round tick values covering [min, max], plus the padded domain they span.
  function niceTicks(min, max, count) {
    if (!(max > min)) {
      min -= 1;
      max += 1;
    }
    var raw = (max - min) / Math.max(1, count);
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var step = [1, 2, 2.5, 5, 10].map(function (m) { return m * mag; }).find(function (s) { return s >= raw; });
    var lo = Math.floor(min / step) * step;
    var hi = Math.ceil(max / step) * step;
    var ticks = [];
    for (var v = lo; v <= hi + step / 2; v += step) ticks.push(Math.round(v / step) * step);
    return { min: lo, max: hi, ticks: ticks };
  }

  // Nearest point to `ms` within `toleranceMs`, else null (a gap reads as no reading).
  function valueAt(points, ms, toleranceMs) {
    var best = null;
    var bestDist = Infinity;
    for (var i = 0; i < points.length; i++) {
      var dist = Math.abs(points[i].ms - ms);
      if (dist < bestDist) {
        bestDist = dist;
        best = points[i];
      }
    }
    return best && bestDist <= toleranceMs ? best : null;
  }

  function formatReading(spec, point) {
    if (!point) return "no reading";
    var text = point.v.toFixed(spec.digits) + " " + spec.unit;
    if (typeof point.sources === "number") {
      text += " (" + point.sources + " source" + (point.sources === 1 ? "" : "s") + ")";
    }
    return text;
  }

  function formatTick(ms, windowKey) {
    var d = new Date(ms);
    var time = d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
    if (windowKey === "24h") return time;
    return d.toLocaleDateString(undefined, { weekday: "short" }) + " " +
      d.toLocaleTimeString(undefined, { hour: "numeric" });
  }

  // ---------------------------------------------------------------- rendering

  function svgEl(tag, attrs, text) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function setText(node, text) { if (node) node.textContent = text; }

  function seriesRuns(payload) {
    var grains = (payload && payload.series_grain_sec) || {};
    var out = {};
    SERIES.forEach(function (spec) {
      var pts = (payload && payload.series && payload.series[spec.key]) || [];
      var runs = splitAtGaps(pts, grains[spec.key] || payload.grain_sec);
      out[spec.key] = { runs: runs, flat: [].concat.apply([], runs), grainMs: (grains[spec.key] || 300) * 1000 };
    });
    return out;
  }

  function render() {
    var host = els.chart;
    if (!host) return;
    host.textContent = "";
    var payload = state.payload;
    if (!payload) return;
    var data = seriesRuns(payload);
    var visible = SERIES.filter(function (s) { return state.enabled[s.key] && data[s.key].flat.length; });
    if (!visible.length) {
      var empty = document.createElement("div");
      empty.className = "h-40 flex items-center justify-center text-[11px] text-gray-500";
      empty.textContent = SERIES.some(function (s) { return state.enabled[s.key]; })
        ? "no readings in this window"
        : "every series is filtered out";
      host.appendChild(empty);
      return;
    }

    var now = Date.now();
    var hours = { "24h": 24, "3d": 72, "7d": 168 }[state.window] || 24;
    var tMin = now - hours * 3600 * 1000;
    var tMax = now;

    var W = Math.max(320, Math.round(host.clientWidth || 640));
    var LEFT = 40, RIGHT = 10, TOP = 6, AXIS = 18, GAP = 12;
    var panels = [];
    var temps = visible.filter(function (s) { return s.panel === "temp"; });
    var watts = visible.filter(function (s) { return s.panel === "watts"; });
    if (temps.length) panels.push({ id: "temp", series: temps, height: 170, unit: "°C" });
    if (watts.length) panels.push({ id: "watts", series: watts, height: temps.length ? 80 : 170, unit: "W" });
    var y0 = TOP;
    panels.forEach(function (p) {
      p.top = y0;
      y0 += p.height + GAP;
      var values = [];
      p.series.forEach(function (s) { data[s.key].flat.forEach(function (pt) { values.push(pt.v); }); });
      p.scale = niceTicks(Math.min.apply(null, values), Math.max.apply(null, values), p.id === "temp" ? 4 : 2);
    });
    var H = y0 - GAP + AXIS;
    var plotW = W - LEFT - RIGHT;
    var x = function (ms) { return LEFT + ((ms - tMin) / (tMax - tMin)) * plotW; };
    var yIn = function (p, v) { return p.top + p.height - ((v - p.scale.min) / (p.scale.max - p.scale.min)) * p.height; };

    var svg = svgEl("svg", {
      viewBox: "0 0 " + W + " " + H,
      width: "100%",
      height: H,
      class: "block cursor-crosshair",
      role: "img",
      "aria-label": "Climate overlay: " + visible.map(function (s) { return s.label; }).join(", ") +
        " over the last " + state.window + ".",
    });

    panels.forEach(function (p) {
      // Recessive grid with small y ticks and labels.
      p.scale.ticks.forEach(function (v) {
        var yy = yIn(p, v);
        svg.appendChild(svgEl("line", { x1: LEFT, x2: W - RIGHT, y1: yy, y2: yy, stroke: "#1f2937", "stroke-width": 1 }));
        svg.appendChild(svgEl("line", { x1: LEFT - 4, x2: LEFT, y1: yy, y2: yy, stroke: "#4b5563", "stroke-width": 1 }));
        svg.appendChild(svgEl("text", {
          x: LEFT - 6, y: yy + 3, "text-anchor": "end", "font-size": 9, fill: "#9ca3af", "font-family": "ui-monospace, monospace",
        }, String(+v.toFixed(2))));
      });
      svg.appendChild(svgEl("text", {
        x: 2, y: p.top + 8, "font-size": 9, fill: "#6b7280", "font-family": "ui-monospace, monospace",
      }, p.unit));
      p.series.forEach(function (s) {
        data[s.key].runs.forEach(function (run) {
          if (run.length === 1) {
            svg.appendChild(svgEl("circle", { cx: x(run[0].ms), cy: yIn(p, run[0].v), r: 2, fill: s.color }));
            return;
          }
          var d = run.map(function (pt, i) {
            return (i ? "L" : "M") + x(pt.ms).toFixed(1) + " " + yIn(p, pt.v).toFixed(1);
          }).join(" ");
          svg.appendChild(svgEl("path", { d: d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }));
        });
      });
    });

    // Time axis: small ticks under the last panel.
    var axisY = H - AXIS;
    svg.appendChild(svgEl("line", { x1: LEFT, x2: W - RIGHT, y1: axisY, y2: axisY, stroke: "#374151", "stroke-width": 1 }));
    var nTicks = Math.max(2, Math.min(7, Math.floor(plotW / 100)));
    for (var i = 0; i <= nTicks; i++) {
      var ms = tMin + ((tMax - tMin) * i) / nTicks;
      var tx = x(ms);
      svg.appendChild(svgEl("line", { x1: tx, x2: tx, y1: axisY, y2: axisY + 4, stroke: "#4b5563", "stroke-width": 1 }));
      svg.appendChild(svgEl("text", {
        x: tx, y: axisY + 14, "text-anchor": i === 0 ? "start" : i === nTicks ? "end" : "middle",
        "font-size": 9, fill: "#9ca3af", "font-family": "ui-monospace, monospace",
      }, formatTick(ms, state.window)));
    }

    // Hover layer: one crosshair over every panel, a ringed dot per visible series.
    var guide = svgEl("line", { x1: 0, x2: 0, y1: TOP, y2: axisY, stroke: "#9ca3af", "stroke-width": 1, opacity: 0 });
    svg.appendChild(guide);
    var dots = visible.map(function (s) {
      var panel = panels.filter(function (p) { return p.series.indexOf(s) >= 0; })[0];
      var dot = svgEl("circle", { r: 4, fill: s.color, stroke: "#0b0f19", "stroke-width": 2, opacity: 0 });
      svg.appendChild(dot);
      return { spec: s, panel: panel, node: dot };
    });
    host.appendChild(svg);

    var hover = window.OrionChartHover;
    var tip = hover ? hover.createTooltip(host) : null;

    function show(evt) {
      var rect = svg.getBoundingClientRect();
      if (!rect.width) return;
      var px = ((evt.clientX - rect.left) / rect.width) * W;
      if (px < LEFT || px > W - RIGHT) return hide();
      var ms = tMin + ((px - LEFT) / plotW) * (tMax - tMin);
      guide.setAttribute("x1", px);
      guide.setAttribute("x2", px);
      guide.setAttribute("opacity", 1);
      if (tip) tip.node.textContent = "";
      var head = document.createElement("div");
      head.className = "text-gray-400";
      head.textContent = hover ? hover.formatHoverWhen(new Date(ms).toISOString()) : new Date(ms).toLocaleString();
      if (tip) tip.node.appendChild(head);
      dots.forEach(function (d) {
        var series = data[d.spec.key];
        var pt = valueAt(series.flat, ms, series.grainMs * 0.75);
        if (pt) {
          d.node.setAttribute("cx", x(pt.ms));
          d.node.setAttribute("cy", yIn(d.panel, pt.v));
          d.node.setAttribute("opacity", 1);
        } else {
          d.node.setAttribute("opacity", 0);
        }
        if (!tip) return;
        var row = document.createElement("div");
        row.className = "flex items-center gap-1.5";
        var swatch = document.createElement("span");
        swatch.className = "inline-block w-2 h-2 rounded-full";
        swatch.style.background = d.spec.color;
        var label = document.createElement("span");
        label.className = pt ? "text-gray-200" : "text-gray-500";
        label.textContent = d.spec.label + " " + formatReading(d.spec, pt);
        row.appendChild(swatch);
        row.appendChild(label);
        tip.node.appendChild(row);
      });
      if (tip) tip.place(evt);
    }

    function hide() {
      guide.setAttribute("opacity", 0);
      dots.forEach(function (d) { d.node.setAttribute("opacity", 0); });
      if (tip) tip.hide();
    }

    svg.addEventListener("mousemove", show);
    svg.addEventListener("mouseleave", hide);
  }

  // ---------------------------------------------------------------- wiring

  async function fetchOverlay() {
    var id = ++state.request;
    try {
      var resp = await fetch(URL + encodeURIComponent(state.window), { headers: { Accept: "application/json" } });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      var payload = await resp.json();
      if (id !== state.request) return;
      if (!payload.ok) throw new Error("overlay unavailable");
      state.payload = payload;
      render();
      var failed = Object.keys(payload.errors || {});
      setText(els.status, failed.length
        ? "Updated " + new Date().toLocaleTimeString() + " — unavailable: " + failed.join(", ")
        : "Updated " + new Date().toLocaleTimeString());
    } catch (err) {
      if (id !== state.request) return;
      setText(els.status, "overlay error — keeping last chart (" + (err.message || err) + ")");
    }
  }

  function refreshControls() {
    (els.windowButtons || []).forEach(function (b) {
      var on = b.getAttribute("data-climate-overlay-window") === state.window;
      b.classList.toggle("border-sky-500", on);
      b.classList.toggle("text-sky-200", on);
      b.classList.toggle("border-gray-700", !on);
      b.classList.toggle("text-gray-400", !on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
    (els.seriesButtons || []).forEach(function (b) {
      var on = !!state.enabled[b.getAttribute("data-climate-overlay-series")];
      b.classList.toggle("opacity-40", !on);
      b.classList.toggle("line-through", !on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }

  function panelVisible() {
    return !!(els.panel && !els.panel.classList.contains("hidden"));
  }

  function tick() {
    if (!panelVisible()) { state.lastFetchAt = 0; return; }
    var now = Date.now();
    if (now - state.lastFetchAt >= POLL_MS) {
      state.lastFetchAt = now;
      fetchOverlay();
    }
  }

  function init() {
    els.panel = document.getElementById("cabinet");
    els.chart = document.getElementById("climateOverlayChart");
    els.status = document.getElementById("climateOverlayStatus");
    if (!document.getElementById("climateOverlayStrip")) return;
    els.windowButtons = Array.prototype.slice.call(document.querySelectorAll("[data-climate-overlay-window]"));
    els.seriesButtons = Array.prototype.slice.call(document.querySelectorAll("[data-climate-overlay-series]"));
    els.windowButtons.forEach(function (b) {
      b.addEventListener("click", function () {
        state.window = b.getAttribute("data-climate-overlay-window") || "24h";
        refreshControls();
        fetchOverlay();
      });
    });
    els.seriesButtons.forEach(function (b) {
      b.addEventListener("click", function () {
        var key = b.getAttribute("data-climate-overlay-series");
        state.enabled[key] = !state.enabled[key];
        refreshControls();
        render();
      });
    });
    var resizeTimer = null;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () { if (panelVisible()) render(); }, 150);
    });
    refreshControls();
    state.timer = setInterval(tick, TICK_MS);
    tick();
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { splitAtGaps: splitAtGaps, niceTicks: niceTicks, valueAt: valueAt, formatReading: formatReading, SERIES: SERIES };
  }
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
  }
})();
