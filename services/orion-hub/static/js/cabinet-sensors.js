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

  var state = {
    active: false,
    timer: null,
    inFlight: false,
    lastFetchedAt: 0,
    lastPayload: null,
  };

  var els = {};

  function $(id) {
    return document.getElementById(id);
  }

  function bindElements() {
    els.panel = $("cabinet");
    els.status = $("cabinetStatus");
    els.grid = $("cabinetSensorGrid");
    els.pressures = $("cabinetPressureStrip");
    els.refreshBtn = $("cabinetRefreshBtn");
    return !!els.panel;
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

  function renderPayload(payload) {
    if (els.status) renderStatus(els.status, payload);
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
    // Rendered DOM is deliberately left in place -- switching away and back
    // shows the last payload immediately.
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
    if (!els.panel.classList.contains("hidden")) {
      activate();
    }
  }

  window.OrionCabinetSensors = {
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
