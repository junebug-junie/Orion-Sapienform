// Office and server-closet environment strips (Biometrics -> Cabinet / Office subview).
//
// Read-only views of GET /api/office/env/latest + /history (office_env_routes.py), fed by
// orion-office-env (pan-1 SHT40 + CCS811 + PMSA003I + TSL2591 + SEN0232 sound meter) via Postgres
// office_env_sample, and GET /api/closet/env/latest + /history (closet_env_routes.py),
// fed by the closet Nano's SCD40 via closet_env_sample. One strip factory drives both.
//
// Absent-is-not-zero: a missing reading renders "—"/"stale", and a chart breaks its
// line across a gap instead of drawing through it or dropping to 0. A saturated light
// sensor reads "too bright to measure", never 0 lux.
//
// Polls only while the #cabinet panel is visible (it is the Cabinet / Office subview).
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  // chart-hover.js loads first (index.html); absent under node:test, where only pure helpers run.
  var chartHover = typeof window !== "undefined" ? window.OrionChartHover : null;
  // Publisher cadence is 30 s; poll at the same grain while visible.
  var POLL_MS = 30000;
  var TICK_MS = 2000;
  // A gap wider than this many median sample spacings breaks the line.
  var GAP_FACTOR = 3;

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
      // `gap` is set server-side where the raw series had an outage inside a wide bucket.
      // `gap_fields`: this one field had an outage the row spacing does not show.
      var fieldGap = Array.isArray(pts[j].gap_fields) && pts[j].gap_fields.indexOf(field) >= 0;
      if (pts[j].gap === true || fieldGap || gaps[j - 1] > limit) runs.push([point]);
      else runs[runs.length - 1].push(point);
    }
    return runs;
  }

  function svgEl(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }

  function renderChart(host, points, field, digits, label) {
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
    // preserveAspectRatio=none so the hover's cursor-to-viewBox mapping is linear.
    var svg = svgEl("svg", {
      viewBox: "0 0 " + W + " " + H,
      preserveAspectRatio: "none",
      class: "w-full h-32 cursor-crosshair",
      role: "img",
    });
    runs.forEach(function (run) {
      if (run.length === 1) {
        svg.appendChild(svgEl("circle", { cx: x(run[0].ms), cy: y(run[0].v), r: 1.5, fill: "currentColor", class: "text-sky-300" }));
        return;
      }
      var d = run.map(function (p, i) { return (i ? "L" : "M") + x(p.ms).toFixed(1) + " " + y(p.v).toFixed(1); }).join(" ");
      svg.appendChild(svgEl("path", { d: d, fill: "none", stroke: "currentColor", "stroke-width": 1.5, "vector-effect": "non-scaling-stroke", class: "text-sky-300" }));
    });
    host.appendChild(svg);
    if (chartHover) {
      var hoverPoints = all.map(function (p) { return { t: new Date(p.ms).toISOString(), value: p.v, ms: p.ms }; });
      chartHover.attachSeriesHover(host, svg, hoverPoints, function (p) { return x(p.ms); }, y, {
        label: label,
        digits: digits,
        color: "#7dd3fc",
        dotRadius: 3,
      });
    }
    var legend = document.createElement("div");
    legend.className = "mt-1 flex justify-between font-mono text-[10px] text-gray-500";
    legend.textContent = "min " + num(vMin, digits) + " · max " + num(vMax, digits);
    host.appendChild(legend);
  }

  function setText(node, text) { if (node) node.textContent = text; }

  // Tile text for one reading. Pure, so it is unit-tested without a DOM. Never shows a
  // number the API did not vouch for as current.
  function readingText(payload, field, digits) {
    var sample = payload && payload.sample;
    if (!sample) return "—";
    if (!payload.ok) return "stale";
    var v = num(sample[field], digits);
    return v === null ? "—" : v;
  }

  // Air tile text. A warming CCS811 has a register value but no valid reading: it says
  // so instead of showing a number.
  function airText(payload, field) {
    var text = readingText(payload, field, 0);
    if (text === "—" && payload && payload.ok && payload.sample && payload.sample.air_warming === true) {
      return "warming up";
    }
    return text;
  }

  function luxDigits(v) {
    return v < 10 ? 2 : (v < 100 ? 1 : 0);
  }

  // Light tile text. Saturated means brighter than the TSL2591 can measure (lux absent,
  // not 0): it says so in words.
  function lightText(payload) {
    var sample = payload && payload.sample;
    var lux = sample && sample.office_lux;
    var text = readingText(payload, "office_lux", typeof lux === "number" ? luxDigits(lux) : 0);
    if (text === "—" && payload && payload.ok && sample && sample.light_saturated === true) {
      return "too bright to measure";
    }
    return text;
  }

  // Office sound summary for one history window. Pure, so it is unit-tested without a
  // DOM. Typical / floor / loud share come from the charted points (bucket averages on
  // long windows); loudest prefers the server's raw max (`stats.sound_max`), which a
  // bucket average would hide. Every value is null when the window has no readings.
  var LOUD_DBA = 60;
  function soundSummary(payload) {
    var points = (payload && Array.isArray(payload.points)) ? payload.points : [];
    var values = points
      .map(function (p) { return p && p.office_sound_dba; })
      .filter(function (v) { return typeof v === "number" && isFinite(v); })
      .sort(function (a, b) { return a - b; });
    if (values.length === 0) return { typical: null, floor: null, loudest: null, loudShare: null };
    function pct(q) {
      var pos = (values.length - 1) * q;
      var lo = Math.floor(pos), hi = Math.ceil(pos);
      return values[lo] + (values[hi] - values[lo]) * (pos - lo);
    }
    var rawMax = payload && payload.stats && payload.stats.sound_max;
    var loud = values.filter(function (v) { return v >= LOUD_DBA; }).length;
    return {
      typical: pct(0.5),
      floor: pct(0.1),
      loudest: typeof rawMax === "number" && isFinite(rawMax) ? rawMax : values[values.length - 1],
      loudShare: loud / values.length,
    };
  }

  function summaryText(payload, key) {
    var v = soundSummary(payload)[key];
    if (v === null) return "—";
    return key === "loudShare" ? Math.round(v * 100) + "%" : num(v, 1);
  }

  // tiles: [{ id, text(payload) }]; charts: [{ id, field, digits, label }];
  // historyTiles: [{ id, text(historyPayload) }], set from the history response.
  // ownWindowButtons: false when another module already styles the window buttons.
  function createStrip(config) {
    var els = {};
    var state = { window: "24h", lastLatestAt: 0, historyRequest: 0, latestInFlight: false };

    function renderLatest(payload) {
      config.tiles.forEach(function (tile) { setText(els[tile.id], tile.text(payload)); });
      var sample = payload && payload.sample;
      if (!sample) {
        setText(els.age, "—");
        setText(els.live, "no samples yet");
        return;
      }
      setText(els.age, typeof payload.age_sec === "number" ? Math.round(payload.age_sec) + "s" : "—");
      setText(els.live, payload.ok ? "fresh" : (payload.publisher_stale ? "no recent publish" : "sensor stale"));
    }

    async function pollLatest() {
      if (state.latestInFlight) return;
      state.latestInFlight = true;
      try {
        var resp = await fetch(config.latestUrl, { headers: { Accept: "application/json" } });
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        var payload = await resp.json();
        if (payload.error) throw new Error(payload.error);
        renderLatest(payload);
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
        var resp = await fetch(config.historyUrl + encodeURIComponent(state.window), { headers: { Accept: "application/json" } });
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        var payload = await resp.json();
        if (!payload.ok) throw new Error(payload.error || "history unavailable");
        if (id !== state.historyRequest) return;
        config.charts.forEach(function (c) {
          renderChart(els[c.id], payload.points, c.field, c.digits, c.label);
        });
        (config.historyTiles || []).forEach(function (t) { setText(els[t.id], t.text(payload)); });
      } catch (err) {
        if (id !== state.historyRequest) return;
        setText(els.status, "history error (" + (err.message || err) + ")");
        // Do not leave the previous window's summary under the newly selected window.
        (config.historyTiles || []).forEach(function (t) { setText(els[t.id], "—"); });
      }
    }

    function refreshButtons() {
      if (config.ownWindowButtons === false) return;
      els.windowButtons.forEach(function (b) {
        var on = b.getAttribute(config.windowAttr) === state.window;
        b.classList.toggle("border-sky-500", on);
        b.classList.toggle("text-sky-200", on);
        b.classList.toggle("border-gray-700", !on);
        b.classList.toggle("text-gray-400", !on);
      });
    }

    function tick(panelVisible) {
      if (!panelVisible) { state.lastLatestAt = 0; return; }
      var now = Date.now();
      if (now - state.lastLatestAt >= POLL_MS) {
        state.lastLatestAt = now;
        pollLatest();
        fetchHistory();
      }
    }

    // Returns false when the strip is not in this page (nothing to drive).
    function init() {
      if (!document.getElementById(config.stripId)) return false;
      els.status = document.getElementById(config.statusId);
      els.age = document.getElementById(config.ageId);
      els.live = document.getElementById(config.liveId);
      config.tiles.forEach(function (t) { els[t.id] = document.getElementById(t.id); });
      config.charts.forEach(function (c) { els[c.id] = document.getElementById(c.id); });
      (config.historyTiles || []).forEach(function (t) { els[t.id] = document.getElementById(t.id); });
      els.windowButtons = document.querySelectorAll("[" + config.windowAttr + "]");
      els.windowButtons.forEach(function (b) {
        b.addEventListener("click", function () {
          var next = b.getAttribute(config.windowAttr) || "24h";
          // Shared buttons: the owning module ignores a click on the active window; match it.
          if (config.ownWindowButtons === false && next === state.window) return;
          state.window = next;
          refreshButtons();
          fetchHistory();
        });
      });
      refreshButtons();
      return true;
    }

    return { init: init, tick: tick, renderLatest: renderLatest };
  }

  var OFFICE = {
    stripId: "officeEnvStrip",
    statusId: "officeEnvStatus",
    ageId: "officeEnvAge",
    liveId: "officeEnvLiveStatus",
    windowAttr: "data-office-env-window",
    latestUrl: "/api/office/env/latest",
    historyUrl: "/api/office/env/history?window=",
    tiles: [
      { id: "officeEnvTemp", text: function (p) { var t = readingText(p, "office_temp_c", 2); return t === "—" && p && p.sample ? "stale" : t; } },
      { id: "officeEnvHumidity", text: function (p) { var t = readingText(p, "office_humidity_pct", 1); return t === "—" && p && p.sample ? "stale" : t; } },
      { id: "officeEnvEco2", text: function (p) { return airText(p, "office_eco2_ppm"); } },
      { id: "officeEnvTvoc", text: function (p) { return airText(p, "office_tvoc_ppb"); } },
      { id: "officeEnvPm1", text: function (p) { return readingText(p, "office_pm1_ug_m3", 0); } },
      { id: "officeEnvPm25", text: function (p) { return readingText(p, "office_pm25_ug_m3", 0); } },
      { id: "officeEnvPm10", text: function (p) { return readingText(p, "office_pm10_ug_m3", 0); } },
      { id: "officeEnvLux", text: lightText },
      { id: "officeEnvSound", text: function (p) { return readingText(p, "office_sound_dba", 1); } },
    ],
    charts: [
      { id: "officeEnvTempChart", field: "office_temp_c", digits: 2, label: "temp °C" },
      { id: "officeEnvHumidityChart", field: "office_humidity_pct", digits: 1, label: "humidity %" },
      { id: "officeEnvEco2Chart", field: "office_eco2_ppm", digits: 0, label: "eCO2 ppm" },
      { id: "officeEnvTvocChart", field: "office_tvoc_ppb", digits: 0, label: "TVOC ppb" },
      { id: "officeEnvPm1Chart", field: "office_pm1_ug_m3", digits: 0, label: "PM1 µg/m³" },
      { id: "officeEnvPm25Chart", field: "office_pm25_ug_m3", digits: 0, label: "PM2.5 µg/m³" },
      { id: "officeEnvPm10Chart", field: "office_pm10_ug_m3", digits: 0, label: "PM10 µg/m³" },
      { id: "officeEnvLuxChart", field: "office_lux", digits: 1, label: "light lux" },
      { id: "officeEnvSoundChart", field: "office_sound_dba", digits: 1, label: "sound dBA" },
    ],
  };

  // Office sound meter inside the Cabinet "Ambient audio" card. Same office API as the
  // strip above; the card's 24h/3d/7d buttons (styled by cabinet-sensors.js) set its window.
  var OFFICE_SOUND = {
    stripId: "officeSoundStrip",
    statusId: "officeSoundStatus",
    ageId: "officeSoundAge",
    liveId: "officeSoundLiveStatus",
    windowAttr: "data-cabinet-ambient-window",
    ownWindowButtons: false,
    latestUrl: "/api/office/env/latest",
    historyUrl: "/api/office/env/history?window=",
    tiles: [
      { id: "officeSoundNow", text: function (p) { return readingText(p, "office_sound_dba", 1); } },
    ],
    historyTiles: [
      { id: "officeSoundTypical", text: function (h) { return summaryText(h, "typical"); } },
      { id: "officeSoundFloor", text: function (h) { return summaryText(h, "floor"); } },
      { id: "officeSoundLoudest", text: function (h) { return summaryText(h, "loudest"); } },
      { id: "officeSoundLoudShare", text: function (h) { return summaryText(h, "loudShare"); } },
    ],
    charts: [
      { id: "officeSoundChart", field: "office_sound_dba", digits: 1, label: "office sound dBA" },
    ],
  };

  var CLOSET = {
    stripId: "closetEnvStrip",
    statusId: "closetEnvStatus",
    ageId: "closetEnvAge",
    liveId: "closetEnvLiveStatus",
    windowAttr: "data-closet-env-window",
    latestUrl: "/api/closet/env/latest",
    historyUrl: "/api/closet/env/history?window=",
    tiles: [
      { id: "closetEnvCo2", text: function (p) { return readingText(p, "closet_co2_ppm", 0); } },
      { id: "closetEnvTemp", text: function (p) { return readingText(p, "closet_temp_c", 1); } },
      { id: "closetEnvHumidity", text: function (p) { return readingText(p, "closet_humidity_pct", 1); } },
    ],
    charts: [
      { id: "closetEnvCo2Chart", field: "closet_co2_ppm", digits: 0, label: "CO2 ppm" },
      { id: "closetEnvTempChart", field: "closet_temp_c", digits: 1, label: "temp °C" },
      { id: "closetEnvHumidityChart", field: "closet_humidity_pct", digits: 1, label: "humidity %" },
    ],
  };

  function init() {
    var panel = document.getElementById("cabinet");
    var strips = [createStrip(OFFICE), createStrip(OFFICE_SOUND), createStrip(CLOSET)].filter(function (s) { return s.init(); });
    if (strips.length === 0) return;
    function tickAll() {
      var visible = !!(panel && !panel.classList.contains("hidden"));
      strips.forEach(function (s) { s.tick(visible); });
    }
    setInterval(tickAll, TICK_MS);
    tickAll();
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      splitRuns: splitRuns,
      num: num,
      airText: airText,
      readingText: readingText,
      lightText: lightText,
      soundSummary: soundSummary,
      summaryText: summaryText,
      OFFICE_SOUND: OFFICE_SOUND,
      createStrip: createStrip,
      OFFICE: OFFICE,
      CLOSET: CLOSET,
    };
  }
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
  }
})();
