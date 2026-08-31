// Cabinet sensors operator tab.
//
// Read-only view of Athena Nano host snapshots from
// GET /api/cabinet/sensors/latest (cabinet_sensors_routes.py). Shows raw
// frame channels (environment / uv / magnetic / particulate / lidar / imu)
// plus Hub-local derived pressures labeled "activity (Hub)".
//
// Absent-is-not-zero (absent is not zero): a missing frame sub-object renders
// as "absent", never as 0. Same invariant as orion.sensor_frame.v1 and
// cabinet_sensors.py.
//
// Panel show/hide is app.js's setActiveTab; this module exposes
// activate()/deactivate() so polling runs ONLY while the tab is visible --
// same lifecycle contract as field-attention.js, including its guard against
// a panel hidden by a path other than setActiveTab.
(function () {
  "use strict";

  var LATEST_URL = "/api/cabinet/sensors/latest";
  var SENSOR_HISTORY_URL = "/api/cabinet/sensors/history?window=";
  var AMBIENT_LATEST_URL = "/api/cabinet/ambient/latest";
  var AMBIENT_HISTORY_URL = "/api/cabinet/ambient/history?window=";
  var AMBIENT_SPIKES_URL = "/api/cabinet/ambient/spikes?window=";
  var SVG_NS = "http://www.w3.org/2000/svg";

  // Nano reader ticks ~1 Hz; poll at the same cadence while the tab is visible.
  var POLL_MS = 1000;

  var SENSOR_TILES = [
    {
      key: "environment",
      title: "Environment (BME680)",
      fields: [
        { path: "temp_c", label: "temp_c", digits: 2 },
        { path: "humidity_pct", label: "humidity_pct", digits: 1 },
        { path: "pressure_hpa", label: "pressure_hpa", digits: 1 },
        { path: "gas_resistance_ohm", label: "gas_ohm", digits: 0 },
      ],
    },
    {
      key: "uv",
      title: "UV / ALS (LTR390)",
      fields: [
        { path: "raw", label: "uv_raw", digits: 0 },
        { path: "als_raw", label: "als_raw", digits: 0 },
      ],
    },
    {
      key: "magnetic",
      title: "Magnetic (MMC5603)",
      fields: [
        { path: "magnitude_ut", label: "mag_uT", digits: 2 },
        { path: "x_ut", label: "x_uT", digits: 2 },
        { path: "y_ut", label: "y_uT", digits: 2 },
        { path: "z_ut", label: "z_uT", digits: 2 },
      ],
    },
    {
      key: "particulate",
      title: "Particulate (PMSA003I)",
      fields: [
        { path: "pm1_ug_m3", label: "pm1", digits: 1 },
        { path: "pm25_ug_m3", label: "pm2.5", digits: 1 },
        { path: "pm10_ug_m3", label: "pm10", digits: 1 },
      ],
    },
    {
      key: "lidar",
      title: "Lidar (VL53L1X)",
      fields: [
        { path: "distance_mm", label: "distance_mm", digits: 0 },
        { path: "status", label: "status", digits: 0 },
      ],
    },
    {
      key: "imu",
      title: "IMU (BNO085)",
      fields: [
        { path: "accel_x", label: "ax", digits: 3 },
        { path: "accel_y", label: "ay", digits: 3 },
        { path: "accel_z", label: "az", digits: 3 },
        { path: "yaw_deg", label: "yaw", digits: 1 },
        { path: "pitch_deg", label: "pitch", digits: 1 },
        { path: "roll_deg", label: "roll", digits: 1 },
      ],
    },
  ];

  var PRESSURE_KEYS = [
    "cabinet_climate_activity",
    "cabinet_particulate_activity",
    "cabinet_em_activity",
    "cabinet_uv_activity",
    "cabinet_vibration_activity",
    "cabinet_proximity_activity",
    "cabinet_sensor_staleness",
  ];

  var SENSOR_HISTORY_CHARTS = [
    { key: "temp_c", hostId: "cabinetSensorChartTempC", title: "Temperature (°C)", label: "Temperature", digits: 2, color: "#fb923c" },
    { key: "humidity_pct", hostId: "cabinetSensorChartHumidity", title: "Humidity (%)", label: "Humidity", digits: 1, color: "#38bdf8" },
    { key: "pressure_hpa", hostId: "cabinetSensorChartPressure", title: "Pressure (hPa)", label: "Pressure", digits: 1, color: "#60a5fa" },
    { key: "gas_resistance_ohm", hostId: "cabinetSensorChartGas", title: "Gas resistance (Ω)", label: "Gas resistance", digits: 0, color: "#c084fc" },
    { key: "lidar_mm", hostId: "cabinetSensorChartLidar", title: "Lidar distance (mm)", label: "Lidar distance", digits: 0, color: "#a78bfa" },
    { key: "als_raw", hostId: "cabinetSensorChartAls", title: "ALS raw (LTR390)", label: "ALS raw", digits: 0, color: "#facc15" },
    { key: "uv_raw", hostId: "cabinetSensorChartUvRaw", title: "UV raw (LTR390)", label: "UV raw", digits: 0, color: "#fde047" },
    { key: "magnetic_ut", hostId: "cabinetSensorChartMagnetic", title: "Magnetic (µT)", label: "Magnetic", digits: 2, color: "#f472b6" },
    { key: "vibration_g", hostId: "cabinetSensorChartVibration", title: "Vibration |g−1|", label: "Vibration", digits: 4, color: "#e879f9" },
    { key: "imu_yaw_deg", hostId: "cabinetSensorChartYaw", title: "IMU yaw (°)", label: "IMU yaw", digits: 1, color: "#818cf8" },
    { key: "imu_pitch_deg", hostId: "cabinetSensorChartPitch", title: "IMU pitch (°)", label: "IMU pitch", digits: 1, color: "#6366f1" },
    { key: "imu_roll_deg", hostId: "cabinetSensorChartRoll", title: "IMU roll (°)", label: "IMU roll", digits: 1, color: "#4f46e5" },
    { key: "climate_activity", hostId: "cabinetSensorChartClimateActivity", title: "climate activity (0–1)", label: "Climate activity", digits: 3, min: 0, max: 1, fixedScale: true, color: "#34d399" },
    { key: "proximity_activity", hostId: "cabinetSensorChartProximityActivity", title: "proximity activity (0–1)", label: "Proximity activity", digits: 3, min: 0, max: 1, fixedScale: true, color: "#fb7185" },
    { key: "em_activity", hostId: "cabinetSensorChartEmActivity", title: "EM activity (0–1)", label: "EM activity", digits: 3, min: 0, max: 1, fixedScale: true, color: "#f97316" },
    { key: "vibration_activity", hostId: "cabinetSensorChartVibrationActivity", title: "vibration activity (0–1)", label: "Vibration activity", digits: 3, min: 0, max: 1, fixedScale: true, color: "#a855f7" },
    { key: "uv_activity", hostId: "cabinetSensorChartUvActivity", title: "UV activity (0–1)", label: "UV activity", digits: 3, min: 0, max: 1, fixedScale: true, color: "#22d3ee" },
  ];

  var state = {
    active: false,
    timer: null,
    inFlight: false,
    lastFetchedAt: 0,
    lastPayload: null,
    ambientLatestInFlight: false,
    ambientLatest: null,
    ambientHistory: null,
    ambientSpikes: null,
    ambientHistoryRequest: 0,
    ambientWindow: "24h",
    ambientLatestNote: "Live values not loaded.",
    ambientHistoryNote: "History not loaded.",
    sensorHistory: null,
    sensorHistoryRequest: 0,
    sensorWindow: "24h",
    sensorHistoryNote: "History not loaded.",
  };

  var els = {};

  function $(id) {
    return document.getElementById(id);
  }

  function bindElements() {
    els.panel = $("cabinet");
    els.status = $("cabinetStatus");
    els.sources = $("cabinetSources");
    els.grid = $("cabinetSensorGrid");
    els.pressures = $("cabinetPressureStrip");
    els.refreshBtn = $("cabinetRefreshBtn");
    els.ambientStatus = $("cabinetAmbientStatus");
    els.ambientRms = $("cabinetAmbientRms");
    els.ambientPeak = $("cabinetAmbientPeak");
    els.ambientAge = $("cabinetAmbientAge");
    els.ambientLiveStatus = $("cabinetAmbientLiveStatus");
    els.ambientRmsChart = $("cabinetAmbientRmsChart");
    els.ambientActivityChart = $("cabinetAmbientActivityChart");
    els.ambientWindowButtons = els.panel
      ? els.panel.querySelectorAll("[data-cabinet-ambient-window]")
      : [];
    els.sensorHistoryStatus = $("cabinetSensorHistoryStatus");
    els.sensorHistoryGrid = $("cabinetSensorHistoryGrid");
    els.sensorWindowButtons = els.panel
      ? els.panel.querySelectorAll("[data-cabinet-sensor-window]")
      : [];
    mountSensorHistoryCharts();
    els.sensorChartHosts = {};
    SENSOR_HISTORY_CHARTS.forEach(function (chart) {
      els.sensorChartHosts[chart.key] = $(chart.hostId);
    });
    return !!els.panel;
  }

  function mountSensorHistoryCharts() {
    var grid = els.sensorHistoryGrid || $("cabinetSensorHistoryGrid");
    if (!grid || grid.childElementCount > 0) {
      return;
    }
    SENSOR_HISTORY_CHARTS.forEach(function (chart) {
      var card = el("div", "rounded-lg border border-gray-800 bg-gray-950/60 p-3");
      card.appendChild(
        el("div", "text-xs font-semibold text-gray-300 mb-2", chart.title || chart.label)
      );
      var host = el("div", "min-h-[8rem]");
      host.id = chart.hostId;
      card.appendChild(host);
      grid.appendChild(card);
    });
  }

  // ---------------------------------------------------------------- helpers

  function num(value, digits) {
    if (value === null || value === undefined || value === "") return null;
    var parsed = Number(value);
    if (!isFinite(parsed)) return null;
    return parsed.toFixed(digits === undefined ? 2 : digits);
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
    var node = el(
      "span",
      "inline-block px-2 py-0.5 rounded-full text-[10px] font-semibold border " + tone,
      text
    );
    if (title) node.title = title;
    return node;
  }

  function svg(viewBox, heightClass) {
    var node = document.createElementNS(SVG_NS, "svg");
    node.setAttribute("viewBox", viewBox);
    node.setAttribute("preserveAspectRatio", "none");
    node.setAttribute("class", "w-full " + (heightClass || "h-28"));
    node.setAttribute("aria-hidden", "true");
    return node;
  }

  function svgEl(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach(function (key) {
      node.setAttribute(key, attrs[key]);
    });
    return node;
  }

  // Format a present numeric field, or return null so callers can show
  // "absent" instead of inventing a zero. Absent-is-not-zero contract.
  function formatAbsentOrValue(value, digits) {
    var formatted = num(value, digits);
    return formatted === null ? null : formatted;
  }

  // ---------------------------------------------------------------- renderers

  function renderStatus(host, payload) {
    host.textContent = "";
    var snapshot = payload && payload.snapshot;
    var boot = payload && payload.boot;
    var frame = snapshot && typeof snapshot.frame === "object" && snapshot.frame ? snapshot.frame : null;

    var row = el("div", "flex flex-wrap items-center gap-2");

    var statusText = snapshot ? String(snapshot.status || "unknown") : "missing";
    var ok = !!(payload && payload.ok);
    var tone = ok
      ? "border-emerald-700 bg-emerald-950/50 text-emerald-200"
      : statusText === "missing" || !snapshot
        ? "border-red-700 bg-red-950/50 text-red-200"
        : "border-amber-700 bg-amber-950/50 text-amber-200";
    row.appendChild(badge(statusText, tone, "reader status"));

    if (!ok && snapshot) {
      row.appendChild(
        badge("stale", "border-amber-700 bg-amber-950/50 text-amber-200", "snapshot present but ok=false")
      );
    }

    if (snapshot && snapshot.device) {
      row.appendChild(el("span", "text-[11px] font-mono text-gray-400 break-all", String(snapshot.device)));
    }

    if (frame) {
      row.appendChild(
        el(
          "span",
          "text-[11px] font-mono text-gray-300",
          "seq " + (frame.seq !== undefined && frame.seq !== null ? frame.seq : "—")
        )
      );
      row.appendChild(
        el(
          "span",
          "text-[11px] font-mono text-gray-400",
          "uptime " +
            (frame.uptime_ms !== undefined && frame.uptime_ms !== null
              ? Number(frame.uptime_ms).toLocaleString() + " ms"
              : "—")
        )
      );
    }

    row.appendChild(
      el("span", "text-[11px] font-mono text-gray-400", "age " + age(payload ? payload.age_sec : null))
    );

    if (boot && boot.i2c && Array.isArray(boot.i2c.addresses) && boot.i2c.addresses.length) {
      row.appendChild(
        el(
          "span",
          "text-[11px] font-mono text-gray-500",
          "i2c " + boot.i2c.addresses.join(", ")
        )
      );
    }

    host.appendChild(row);
  }

  function bootOkSensors(boot) {
    var sensors = boot && boot.sensors && typeof boot.sensors === "object" ? boot.sensors : null;
    if (!sensors) return [];
    return Object.keys(sensors).filter(function (name) {
      var meta = sensors[name];
      return meta && typeof meta === "object" && meta.ok === true;
    });
  }

  function renderSources(host, payload) {
    if (!host) return;
    host.textContent = "";
    var sources = payload && payload.sources && typeof payload.sources === "object" ? payload.sources : null;
    if (!sources) return;

    var labels = { a: "Nano A (env / UV / lidar)", b: "Nano B (Hub B — mag + IMU)" };
    var wrap = el("div", "flex flex-col gap-2");
    Object.keys(sources).sort().forEach(function (key) {
      var entry = sources[key];
      if (!entry || typeof entry !== "object") return;
      var snap = entry.snapshot;
      var boot = entry.boot;
      var row = el("div", "rounded-lg border border-gray-800 bg-gray-950/50 px-3 py-2");
      var title = el("div", "text-[11px] font-semibold text-gray-300", labels[key] || ("Nano " + key.toUpperCase()));
      row.appendChild(title);
      var meta = el("div", "mt-1 flex flex-wrap items-center gap-2 text-[11px] font-mono text-gray-400");
      var statusText = snap ? String(snap.status || "unknown") : "missing";
      meta.appendChild(
        badge(
          statusText,
          statusText === "ok"
            ? "border-emerald-800 bg-emerald-950/40 text-emerald-200"
            : "border-amber-800 bg-amber-950/40 text-amber-200",
          "reader status"
        )
      );
      var addrs =
        boot && boot.i2c && Array.isArray(boot.i2c.addresses) ? boot.i2c.addresses.join(", ") : "—";
      meta.appendChild(el("span", "", "i2c " + addrs));
      var okNames = bootOkSensors(boot);
      if (okNames.length) {
        meta.appendChild(el("span", "text-gray-500", "·"));
        meta.appendChild(el("span", "text-emerald-300/90", okNames.join(", ")));
      }
      row.appendChild(meta);
      wrap.appendChild(row);
    });
    host.appendChild(wrap);
  }

  function renderTile(tile, frame) {
    var card = el("div", "rounded-xl border border-gray-800 bg-gray-950/40 p-3 flex flex-col gap-2 min-h-[7rem]");
    card.appendChild(el("h3", "text-xs font-semibold text-gray-200", tile.title));

    // Absent-is-not-zero: missing frame sub-object → whole tile is "absent".
    var block = frame && typeof frame[tile.key] === "object" && frame[tile.key] ? frame[tile.key] : null;
    if (!block) {
      card.appendChild(el("div", "text-sm font-mono text-gray-500", "absent"));
      return card;
    }

    var list = el("dl", "grid grid-cols-2 gap-x-2 gap-y-1 text-[11px]");
    tile.fields.forEach(function (field) {
      list.appendChild(el("dt", "text-gray-500", field.label));
      var formatted = formatAbsentOrValue(block[field.path], field.digits);
      list.appendChild(
        el(
          "dd",
          "font-mono text-gray-200 text-right",
          formatted === null ? "absent" : formatted
        )
      );
    });
    card.appendChild(list);
    return card;
  }

  function renderSensorGrid(host, payload) {
    host.textContent = "";
    var snapshot = payload && payload.snapshot;
    if (!snapshot) {
      host.appendChild(
        el(
          "div",
          "rounded-xl border border-red-900/60 bg-red-950/20 p-4 text-sm text-red-200",
          "No snapshot — is orion-cabinet-sensors.service running?"
        )
      );
      return;
    }

    var frame =
      snapshot.frame && typeof snapshot.frame === "object" ? snapshot.frame : null;
    var grid = el("div", "grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3");
    SENSOR_TILES.forEach(function (tile) {
      grid.appendChild(renderTile(tile, frame));
    });
    host.appendChild(grid);
  }

  function renderPressureStrip(host, payload) {
    host.textContent = "";
    host.appendChild(
      el("h3", "text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2", "activity (Hub)")
    );

    var pressures = (payload && payload.pressures) || {};
    var present = PRESSURE_KEYS.filter(function (key) {
      return pressures[key] !== undefined && pressures[key] !== null;
    });

    if (!present.length) {
      host.appendChild(el("div", "text-[11px] text-gray-500", "absent"));
      return;
    }

    var row = el("div", "flex flex-wrap gap-2");
    present.forEach(function (key) {
      var chip = el(
        "div",
        "rounded-lg border border-gray-800 bg-gray-950/60 px-2 py-1 text-[11px] font-mono text-gray-300"
      );
      chip.appendChild(el("span", "text-gray-500 mr-1", key.replace(/^cabinet_/, "")));
      chip.appendChild(el("span", "text-gray-200", num(pressures[key], 3) || "—"));
      row.appendChild(chip);
    });
    host.appendChild(row);

    host.appendChild(
      el(
        "p",
        "text-[10px] text-gray-500 mt-2",
        "Hub-local baselines — operator-debug approximations, not the live biometrics field values."
      )
    );
  }

  function renderAmbientStatus() {
    if (!els.ambientStatus) return;
    els.ambientStatus.textContent = state.ambientLatestNote + " · " + state.ambientHistoryNote;
    els.ambientStatus.className =
      "text-[11px] min-h-[1rem] " +
      (state.ambientLatestNote.indexOf("error") !== -1 ||
      state.ambientHistoryNote.indexOf("error") !== -1
        ? "text-amber-400"
        : "text-gray-500");
  }

  function renderAmbientLatest(payload) {
    var snapshot = payload && payload.snapshot;
    if (!snapshot) return;
    if (els.ambientRms) els.ambientRms.textContent = num(snapshot.rms, 1) || "—";
    if (els.ambientPeak) els.ambientPeak.textContent = num(snapshot.peak, 0) || "—";
    if (els.ambientAge) els.ambientAge.textContent = age(payload.age_sec);
    if (els.ambientLiveStatus) {
      var statusText = String(snapshot.status || (payload.ok ? "ok" : "stale"));
      els.ambientLiveStatus.textContent = payload.ok ? statusText : statusText + " / stale";
      els.ambientLiveStatus.className =
        "mt-1 font-mono text-sm " + (payload.ok ? "text-emerald-300" : "text-amber-300");
    }
  }

  function renderAmbientSeries(host, points, key, options) {
    if (!host) return;
    host.textContent = "";
    options = options || {};
    var samples = (Array.isArray(points) ? points : []).map(function (point, index) {
        var rawValue = point && point[key];
        var value =
          rawValue === null || rawValue === undefined || rawValue === ""
            ? null
            : Number(rawValue);
        var timestamp = point && point.t ? Date.parse(point.t) : NaN;
        return {
          t: point && point.t,
          value: value !== null && isFinite(value) ? value : null,
          timestamp: isFinite(timestamp) ? timestamp : null,
          index: index,
        };
      });
    var validSamples = samples.filter(function (point) {
      return point.value !== null;
    });

    if (!validSamples.length) {
      host.setAttribute("role", "img");
      host.setAttribute("aria-label", (options.label || key) + " chart: no samples in this window.");
      host.appendChild(
        el("div", "h-28 flex items-center justify-center text-[11px] text-gray-500", "No samples in this window.")
      );
      return;
    }

    var values = validSamples.map(function (point) {
      return point.value;
    });
    var minValue = options.min !== undefined ? options.min : Math.min.apply(null, values);
    var maxValue = options.max !== undefined ? options.max : Math.max.apply(null, values);
    if (maxValue <= minValue) maxValue = minValue + 1;
    var pad = options.fixedScale ? 0 : (maxValue - minValue) * 0.08;
    minValue = options.fixedScale ? minValue : Math.max(0, minValue - pad);
    maxValue = options.fixedScale ? maxValue : maxValue + pad;

    var timedSamples = samples.filter(function (point) {
      return point.timestamp !== null;
    });
    var minTime = timedSamples.length ? timedSamples[0].timestamp : null;
    var maxTime = timedSamples.length ? timedSamples[timedSamples.length - 1].timestamp : null;
    function x(point) {
      if (point.timestamp !== null && maxTime !== null && maxTime > minTime) {
        return ((point.timestamp - minTime) / (maxTime - minTime)) * 100;
      }
      return samples.length === 1 ? 50 : (point.index / (samples.length - 1)) * 100;
    }
    function y(value) {
      return 38 - ((value - minValue) / (maxValue - minValue)) * 36;
    }

    var chart = svg("0 0 100 40", options.heightClass || "h-28");
    [0, 20, 40].forEach(function (y) {
      chart.appendChild(
        svgEl("line", {
          x1: 0,
          y1: y,
          x2: 100,
          y2: y,
          stroke: "#1f2937",
          "stroke-width": 0.5,
          "vector-effect": "non-scaling-stroke",
        })
      );
    });
    var segments = [];
    var currentSegment = [];
    samples.forEach(function (point) {
      if (point.value === null) {
        if (currentSegment.length) segments = segments.concat([currentSegment]);
        currentSegment = [];
        return;
      }
      currentSegment = currentSegment.concat([x(point) + "," + y(point.value)]);
    });
    if (currentSegment.length) segments = segments.concat([currentSegment]);
    segments.forEach(function (segment) {
      if (segment.length > 1) {
        chart.appendChild(
          svgEl("polyline", {
            points: segment.join(" "),
            fill: "none",
            stroke: options.color || "#818cf8",
            "stroke-width": 1,
            "vector-effect": "non-scaling-stroke",
          })
        );
      } else {
        var only = segment[0].split(",");
        chart.appendChild(
          svgEl("circle", {
            cx: only[0],
            cy: only[1],
            r: 1.4,
            fill: options.color || "#818cf8",
          })
        );
      }
    });
    var markers = Array.isArray(options.markers) ? options.markers : [];
    if (markers.length && maxTime !== null && maxTime > minTime) {
      markers.forEach(function (marker) {
        var markerTime = marker && marker.t ? Date.parse(marker.t) : NaN;
        if (!isFinite(markerTime)) return;
        var mx = ((markerTime - minTime) / (maxTime - minTime)) * 100;
        chart.appendChild(
          svgEl("line", {
            x1: mx,
            y1: 0,
            x2: mx,
            y2: 40,
            stroke: options.markerColor || "#f87171",
            "stroke-width": 0.9,
            "stroke-dasharray": "2 1.5",
            "vector-effect": "non-scaling-stroke",
            opacity: 0.9,
          })
        );
      });
    }
    host.appendChild(chart);

    var firstTime = validSamples[0].t ? new Date(validSamples[0].t).toLocaleString() : "—";
    var lastTime = validSamples[validSamples.length - 1].t
      ? new Date(validSamples[validSamples.length - 1].t).toLocaleString()
      : "—";
    host.setAttribute("role", "img");
    host.setAttribute(
      "aria-label",
      (options.label || key) +
        " chart. " +
        validSamples.length +
        " samples from " +
        firstTime +
        " to " +
        lastTime +
        " range " +
        Math.min.apply(null, values).toFixed(options.digits || 2) +
        " to " +
        Math.max.apply(null, values).toFixed(options.digits || 2) +
        "."
    );
    host.appendChild(
      el(
        "div",
        "mt-1 flex justify-between gap-2 text-[9px] font-mono text-gray-600",
        firstTime + "  →  " + lastTime + " · n=" + validSamples.length
      )
    );
  }

  function renderAmbientHistory(payload, spikesPayload) {
    var points = payload && Array.isArray(payload.points) ? payload.points : [];
    var spikes = spikesPayload && Array.isArray(spikesPayload.spikes) ? spikesPayload.spikes : [];
    renderAmbientSeries(els.ambientRmsChart, points, "rms", {
      label: "Ambient RMS",
      digits: 1,
      color: "#818cf8",
      heightClass: "h-32",
      markers: spikes,
      markerColor: "#f87171",
    });
    renderAmbientSeries(els.ambientActivityChart, points, "activity", {
      label: "Ambient activity",
      digits: 3,
      min: 0,
      max: 1,
      fixedScale: true,
      color: "#34d399",
      heightClass: "h-32",
      markers: spikes,
      markerColor: "#f87171",
    });
  }

  function renderAmbientWindowButtons() {
    Array.prototype.forEach.call(els.ambientWindowButtons || [], function (button) {
      var selected = button.getAttribute("data-cabinet-ambient-window") === state.ambientWindow;
      button.className = selected
        ? "px-2 py-1 rounded border border-indigo-500 bg-indigo-950/60 text-[11px] font-mono text-indigo-200"
        : "px-2 py-1 rounded border border-gray-700 bg-gray-900 text-[11px] font-mono text-gray-400 hover:text-gray-200";
      button.setAttribute("aria-pressed", selected ? "true" : "false");
    });
  }

  function renderSensorHistoryStatus() {
    if (!els.sensorHistoryStatus) return;
    els.sensorHistoryStatus.textContent = state.sensorHistoryNote;
    els.sensorHistoryStatus.className =
      "text-[11px] min-h-[1rem] " +
      (state.sensorHistoryNote.indexOf("error") !== -1 ? "text-amber-400" : "text-gray-500");
  }

  function renderSensorHistory(payload) {
    var series = payload && payload.series ? payload.series : {};
    SENSOR_HISTORY_CHARTS.forEach(function (chart) {
      var host = els.sensorChartHosts[chart.key];
      var points = Array.isArray(series[chart.key]) ? series[chart.key] : [];
      renderAmbientSeries(host, points, "v", {
        label: chart.label,
        digits: chart.digits,
        color: chart.color,
        min: chart.min,
        max: chart.max,
        fixedScale: chart.fixedScale,
        heightClass: "h-32",
      });
    });
  }

  function renderSensorWindowButtons() {
    Array.prototype.forEach.call(els.sensorWindowButtons || [], function (button) {
      var selected = button.getAttribute("data-cabinet-sensor-window") === state.sensorWindow;
      button.className = selected
        ? "px-2 py-1 rounded border border-teal-500 bg-teal-950/60 text-[11px] font-mono text-teal-200"
        : "px-2 py-1 rounded border border-gray-700 bg-gray-900 text-[11px] font-mono text-gray-400 hover:text-gray-200";
      button.setAttribute("aria-pressed", selected ? "true" : "false");
    });
  }

  function renderPayload(payload) {
    if (els.status) renderStatus(els.status, payload);
    if (els.sources) renderSources(els.sources, payload);
    if (els.grid) renderSensorGrid(els.grid, payload);
    if (els.pressures) renderPressureStrip(els.pressures, payload);
  }

  // ------------------------------------------------------------ poll cycle

  function setStatusLine(text, tone) {
    // Status mount is also used for the structured status strip after a
    // successful render. On poll-error we keep last good render and append a
    // badge via a lightweight overlay line on the panel title area by
    // updating a data attribute + optional trailing note on the strip.
    if (!els.panel) return;
    var existing = els.panel.querySelector("[data-cabinet-poll-badge]");
    if (existing) existing.remove();
    if (!text) return;
    var note = el(
      "div",
      "text-xs min-h-[1.25rem] " + (tone || "text-gray-400"),
      text
    );
    note.setAttribute("data-cabinet-poll-badge", "1");
    if (els.status && els.status.parentNode) {
      els.status.parentNode.insertBefore(note, els.status.nextSibling);
    } else {
      els.panel.insertBefore(note, els.panel.firstChild);
    }
  }

  async function poll() {
    if (state.inFlight) return;
    state.inFlight = true;
    try {
      var resp = await fetch(LATEST_URL, { headers: { Accept: "application/json" } });
      if (!resp.ok) {
        throw new Error("HTTP " + resp.status + " from " + LATEST_URL);
      }
      var payload = await resp.json();
      state.lastPayload = payload;
      state.lastFetchedAt = Date.now();
      renderPayload(payload);
      setStatusLine(
        "Updated " + new Date().toLocaleTimeString() + " · polling every " + POLL_MS / 1000 + "s",
        "text-gray-500"
      );
    } catch (err) {
      // Keep last good render on transient fetch error; show poll-error badge.
      if (state.lastPayload) {
        setStatusLine(
          "poll error — keeping last good render (" + (err.message || err) + ")",
          "text-amber-400"
        );
      } else {
        setStatusLine("Failed to load: " + (err.message || err), "text-red-400");
        if (els.grid) {
          els.grid.textContent = "";
          els.grid.appendChild(
            el(
              "div",
              "rounded-xl border border-red-900/60 bg-red-950/20 p-4 text-sm text-red-200",
              "No snapshot — is orion-cabinet-sensors.service running?"
            )
          );
        }
      }
    } finally {
      state.inFlight = false;
    }
  }

  async function pollAmbientLatest() {
    if (state.ambientLatestInFlight) return;
    state.ambientLatestInFlight = true;
    try {
      var resp = await fetch(AMBIENT_LATEST_URL, { headers: { Accept: "application/json" } });
      if (!resp.ok) {
        throw new Error("HTTP " + resp.status + " from " + AMBIENT_LATEST_URL);
      }
      var payload = await resp.json();
      if (!payload.snapshot) {
        throw new Error(payload.error || "ambient snapshot missing");
      }
      state.ambientLatest = payload;
      renderAmbientLatest(payload);
      state.ambientLatestNote =
        "Live updated " + new Date().toLocaleTimeString() + (payload.ok ? "" : " (stale)");
    } catch (err) {
      state.ambientLatestNote =
        "live error — " +
        (state.ambientLatest
          ? "keeping last good live values"
          : "no live values yet") +
        " (" +
        (err.message || err) +
        ")";
    } finally {
      state.ambientLatestInFlight = false;
      renderAmbientStatus();
    }
  }

  async function fetchAmbientHistory() {
    var requestId = ++state.ambientHistoryRequest;
    var requestedWindow = state.ambientWindow;
    state.ambientHistoryNote = "Loading " + requestedWindow + " history…";
    renderAmbientStatus();
    try {
      var url = AMBIENT_HISTORY_URL + encodeURIComponent(requestedWindow);
      var spikesUrl = AMBIENT_SPIKES_URL + encodeURIComponent(requestedWindow);
      var results = await Promise.all([
        fetch(url, { headers: { Accept: "application/json" } }),
        fetch(spikesUrl, { headers: { Accept: "application/json" } }),
      ]);
      var resp = results[0];
      var spikesResp = results[1];
      if (!resp.ok) {
        throw new Error("HTTP " + resp.status + " from ambient history");
      }
      var payload = await resp.json();
      if (!payload.ok) {
        throw new Error(payload.error || "ambient history unavailable");
      }
      var spikesPayload = { spikes: [] };
      if (spikesResp.ok) {
        var parsedSpikes = await spikesResp.json();
        if (parsedSpikes && parsedSpikes.ok) spikesPayload = parsedSpikes;
      }
      if (requestId !== state.ambientHistoryRequest) return;
      state.ambientHistory = payload;
      state.ambientSpikes = spikesPayload;
      renderAmbientHistory(payload, spikesPayload);
      var spikeCount = Array.isArray(spikesPayload.spikes) ? spikesPayload.spikes.length : 0;
      state.ambientHistoryNote =
        requestedWindow +
        " history · n=" +
        (Array.isArray(payload.points) ? payload.points.length : 0) +
        (spikeCount ? " · spikes=" + spikeCount : "");
    } catch (err) {
      if (requestId !== state.ambientHistoryRequest) return;
      state.ambientHistoryNote =
        "history error — " +
        (state.ambientHistory ? "keeping last good charts" : "no chart data yet") +
        " (" +
        (err.message || err) +
        ")";
    } finally {
      if (requestId === state.ambientHistoryRequest) renderAmbientStatus();
    }
  }

  async function fetchSensorHistory() {
    var requestId = ++state.sensorHistoryRequest;
    var requestedWindow = state.sensorWindow;
    state.sensorHistoryNote = "Loading " + requestedWindow + " sensor history…";
    renderSensorHistoryStatus();
    try {
      var url = SENSOR_HISTORY_URL + encodeURIComponent(requestedWindow);
      var resp = await fetch(url, { headers: { Accept: "application/json" } });
      if (!resp.ok) {
        throw new Error("HTTP " + resp.status + " from sensor history");
      }
      var payload = await resp.json();
      if (!payload.ok) {
        throw new Error(payload.error || "sensor history unavailable");
      }
      if (requestId !== state.sensorHistoryRequest) return;
      state.sensorHistory = payload;
      renderSensorHistory(payload);
      var stats = payload.stats || {};
      var tempStats = stats.temp_c || {};
      var note = requestedWindow + " history";
      if (tempStats.n_raw) note += " · temp n=" + tempStats.n_raw;
      if (tempStats.min !== undefined && tempStats.max !== undefined) {
        note +=
          " · temp " +
          Number(tempStats.min).toFixed(1) +
          "–" +
          Number(tempStats.max).toFixed(1) +
          "°C";
      }
      state.sensorHistoryNote = note;
    } catch (err) {
      if (requestId !== state.sensorHistoryRequest) return;
      state.sensorHistoryNote =
        "history error — " +
        (state.sensorHistory ? "keeping last good charts" : "no chart data yet") +
        " (" +
        (err.message || err) +
        ")";
    } finally {
      if (requestId === state.sensorHistoryRequest) renderSensorHistoryStatus();
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
      // same guard field-attention.js uses.
      if (!els.panel || els.panel.classList.contains("hidden")) {
        deactivate();
        return;
      }
      poll();
      pollAmbientLatest();
    }, POLL_MS);
  }

  function activate() {
    if (!els.panel && !bindElements()) return;
    state.active = true;
    poll();
    pollAmbientLatest();
    fetchAmbientHistory();
    fetchSensorHistory();
    startTimer();
  }

  function deactivate() {
    state.active = false;
    stopTimer();
    // Rendered DOM is deliberately left in place -- switching away and back
    // shows the last payload immediately.
  }

  function wireControls() {
    if (els.refreshBtn) {
      els.refreshBtn.addEventListener("click", function () {
        poll();
        pollAmbientLatest();
        fetchAmbientHistory();
        fetchSensorHistory();
      });
    }
    Array.prototype.forEach.call(els.ambientWindowButtons || [], function (button) {
      button.addEventListener("click", function () {
        var nextWindow = button.getAttribute("data-cabinet-ambient-window");
        if (!nextWindow || nextWindow === state.ambientWindow) return;
        state.ambientWindow = nextWindow;
        renderAmbientWindowButtons();
        fetchAmbientHistory();
      });
    });
    Array.prototype.forEach.call(els.sensorWindowButtons || [], function (button) {
      button.addEventListener("click", function () {
        var nextWindow = button.getAttribute("data-cabinet-sensor-window");
        if (!nextWindow || nextWindow === state.sensorWindow) return;
        state.sensorWindow = nextWindow;
        renderSensorWindowButtons();
        fetchSensorHistory();
      });
    });
    renderAmbientWindowButtons();
    renderSensorWindowButtons();
  }

  function init() {
    if (!bindElements()) return;
    wireControls();
    if (!els.panel.classList.contains("hidden")) {
      activate();
    }
  }

  window.OrionCabinetSensors = {
    activate: activate,
    deactivate: deactivate,
    refresh: function () {
      poll();
      pollAmbientLatest();
      fetchAmbientHistory();
      fetchSensorHistory();
    },
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
