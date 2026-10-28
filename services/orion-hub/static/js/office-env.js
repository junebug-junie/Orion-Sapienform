// Office environment strip (Biometrics -> Cabinet / Office subview).
//
// Read-only view of GET /api/office/env/latest and /history (office_env_routes.py),
// fed by orion-office-env (pan-1 SHT40) via Postgres office_env_sample.
//
// Absent-is-not-zero: a missing reading renders "—"/"stale", and a chart breaks its
// line across a gap instead of drawing through it or dropping to 0.
//
// Polls only while the #cabinet panel is visible (it is the Cabinet / Office subview).
(function () {
  "use strict";

  var LATEST_URL = "/api/office/env/latest";
  var HISTORY_URL = "/api/office/env/history?window=";
  var SVG_NS = "http://www.w3.org/2000/svg";
  // Publisher cadence is 30 s; poll at the same grain while visible.
  var POLL_MS = 30000;
  var TICK_MS = 2000;
  // A gap wider than this many median sample spacings breaks the line.
  var GAP_FACTOR = 3;

  var els = {};
  var state = { window: "24h", lastLatestAt: 0, historyRequest: 0, latestInFlight: false, timer: null };

  function num(value, digits) {
    if (typeof value !== "number" || !isFinite(value)) return null;
    return value.toFixed(digits);
  }

  // Split points into runs; a run ends where the time gap exceeds GAP_FACTOR x the
  // median spacing. Pure, so it is unit-tested without a DOM.
  function splitRuns(points, field) {
    var pts = points.filter(function (p) {
      return typeof p[field] === "number" && isFinite(p[field]) && p.t;
    });
    if (pts.length === 0) return [];
    var times = pts.map(function (p) { return Date.parse(p.t); });
    var gaps = [];
    for (var i = 1; i < times.length; i++) gaps.push(times[i] - times[i - 1]);
    var sorted = gaps.slice().sort(function (a, b) { return a - b; });
    var median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
    var limit = median > 0 ? median * GAP_FACTOR : Infinity;
    var runs = [[{ ms: times[0], v: pts[0][field] }]];
    for (var j = 1; j < pts.length; j++) {
      var point = { ms: times[j], v: pts[j][field] };
      if (gaps[j - 1] > limit) runs.push([point]);
      else runs[runs.length - 1].push(point);
    }
    return runs;
  }

  function svgEl(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }

  function renderChart(host, points, field, digits) {
    if (!host) return;
    host.textContent = "";
    var runs = splitRuns(points || [], field);
    if (runs.length === 0) {
      var empty = document.createElement("div");
      empty.className = "text-[11px] text-gray-500";
      empty.textContent = "no readings in this window";
      host.appendChild(empty);
      return;
    }
    var all = [].concat.apply([], runs);
    var tMin = all[0].ms, tMax = all[all.length - 1].ms;
    var vMin = Math.min.apply(null, all.map(function (p) { return p.v; }));
    var vMax = Math.max.apply(null, all.map(function (p) { return p.v; }));
    if (vMax === vMin) { vMax += 0.5; vMin -= 0.5; }
    var W = 400, H = 120, PAD = 4;
    var x = function (ms) { return PAD + (tMax === tMin ? 0 : ((ms - tMin) / (tMax - tMin)) * (W - 2 * PAD)); };
    var y = function (v) { return H - PAD - ((v - vMin) / (vMax - vMin)) * (H - 2 * PAD); };
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, class: "w-full h-32", role: "img" });
    runs.forEach(function (run) {
      if (run.length === 1) {
        svg.appendChild(svgEl("circle", { cx: x(run[0].ms), cy: y(run[0].v), r: 1.5, fill: "currentColor", class: "text-sky-300" }));
        return;
      }
      var d = run.map(function (p, i) { return (i ? "L" : "M") + x(p.ms).toFixed(1) + " " + y(p.v).toFixed(1); }).join(" ");
      svg.appendChild(svgEl("path", { d: d, fill: "none", stroke: "currentColor", "stroke-width": 1.5, class: "text-sky-300" }));
    });
    host.appendChild(svg);
    var legend = document.createElement("div");
    legend.className = "mt-1 flex justify-between font-mono text-[10px] text-gray-500";
    legend.textContent = "min " + num(vMin, digits) + " · max " + num(vMax, digits);
    host.appendChild(legend);
  }

  function setText(node, text) { if (node) node.textContent = text; }

  function renderLatest(payload) {
    var sample = payload && payload.sample;
    if (!sample) {
      setText(els.temp, "—"); setText(els.humidity, "—"); setText(els.age, "—");
      setText(els.live, "no samples yet");
      return;
    }
    var t = num(sample.office_temp_c, 2), h = num(sample.office_humidity_pct, 1);
    setText(els.temp, t === null ? "stale" : t);
    setText(els.humidity, h === null ? "stale" : h);
    setText(els.age, typeof payload.age_sec === "number" ? Math.round(payload.age_sec) + "s" : "—");
    setText(els.live, payload.ok ? "fresh" : (payload.sensor_stale ? "sensor stale" : "no recent publish"));
  }

  async function pollLatest() {
    if (state.latestInFlight) return;
    state.latestInFlight = true;
    try {
      var resp = await fetch(LATEST_URL, { headers: { Accept: "application/json" } });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      renderLatest(await resp.json());
      setText(els.status, "Live updated " + new Date().toLocaleTimeString());
    } catch (err) {
      setText(els.status, "live error — keeping last good values (" + (err.message || err) + ")");
    } finally {
      state.latestInFlight = false;
    }
  }

  async function fetchHistory() {
    var id = ++state.historyRequest;
    try {
      var resp = await fetch(HISTORY_URL + encodeURIComponent(state.window), { headers: { Accept: "application/json" } });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      var payload = await resp.json();
      if (!payload.ok) throw new Error(payload.error || "history unavailable");
      if (id !== state.historyRequest) return;
      renderChart(els.tempChart, payload.points, "office_temp_c", 2);
      renderChart(els.humidityChart, payload.points, "office_humidity_pct", 1);
    } catch (err) {
      if (id !== state.historyRequest) return;
      setText(els.status, "history error (" + (err.message || err) + ")");
    }
  }

  function panelVisible() {
    return !!(els.panel && !els.panel.classList.contains("hidden"));
  }

  function refreshButtons() {
    if (!els.windowButtons) return;
    els.windowButtons.forEach(function (b) {
      var on = b.getAttribute("data-office-env-window") === state.window;
      b.classList.toggle("border-sky-500", on);
      b.classList.toggle("text-sky-200", on);
      b.classList.toggle("border-gray-700", !on);
      b.classList.toggle("text-gray-400", !on);
    });
  }

  function tick() {
    if (!panelVisible()) { state.lastLatestAt = 0; return; }
    var now = Date.now();
    if (now - state.lastLatestAt >= POLL_MS) {
      state.lastLatestAt = now;
      pollLatest();
      fetchHistory();
    }
  }

  function init() {
    els.panel = document.getElementById("cabinet");
    els.status = document.getElementById("officeEnvStatus");
    els.temp = document.getElementById("officeEnvTemp");
    els.humidity = document.getElementById("officeEnvHumidity");
    els.age = document.getElementById("officeEnvAge");
    els.live = document.getElementById("officeEnvLiveStatus");
    els.tempChart = document.getElementById("officeEnvTempChart");
    els.humidityChart = document.getElementById("officeEnvHumidityChart");
    els.windowButtons = document.querySelectorAll("[data-office-env-window]");
    if (!document.getElementById("officeEnvStrip")) return;
    els.windowButtons.forEach(function (b) {
      b.addEventListener("click", function () {
        state.window = b.getAttribute("data-office-env-window") || "24h";
        refreshButtons();
        fetchHistory();
      });
    });
    refreshButtons();
    state.timer = setInterval(tick, TICK_MS);
    tick();
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { splitRuns: splitRuns, num: num };
  }
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
  }
})();
