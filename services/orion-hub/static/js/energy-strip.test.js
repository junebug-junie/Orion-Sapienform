const test = require("node:test");
const assert = require("node:assert/strict");
const strip = require("./energy-strip.js");

const { fmtUsd, fmtRate, pressureLabel, pressureLine, importerLabel, reconcileLine, barHeights } = strip;

test("unknown money renders as unknown, never $0.00", () => {
  assert.equal(fmtUsd(null), "unknown");
  assert.equal(fmtUsd(undefined), "unknown");
  assert.equal(fmtUsd(0), "$0.00");
  assert.equal(fmtUsd(-2), "-$2.00");
  assert.equal(fmtUsd(88.4), "$88.40");
});

test("rate and labels", () => {
  assert.equal(fmtRate(0.12), "$0.1200/kWh");
  assert.equal(fmtRate(null), "unknown");
  assert.equal(pressureLabel("over_forecast"), "over RMP forecast");
  assert.equal(pressureLabel("near_forecast"), "near RMP forecast");
  assert.equal(pressureLabel("normal"), "under RMP forecast");
  assert.equal(pressureLabel("bogus"), "unknown");
  assert.equal(importerLabel(null), "importer: no status yet");
  assert.equal(importerLabel({ state: "reauth_required", reason: "session_expired" }), "importer: reauth_required (session_expired)");
  assert.equal(importerLabel({ state: "healthy", reason: null }), "importer: healthy");
});

test("unknown pressure says why", () => {
  assert.equal(pressureLine({ pressure: "unknown", pressure_reason: "coverage_lag_hours=30.0" }), "unknown (coverage_lag_hours=30.0)");
  assert.equal(pressureLine({ pressure: "unknown", pressure_reason: "importer_stale" }), "unknown (importer_stale)");
  assert.equal(pressureLine({ pressure: "over_forecast", pressure_reason: "ratio=1.105" }), "over RMP forecast (ratio=1.105)");
  assert.equal(pressureLine({}), "unknown");
  assert.equal(pressureLine(null), "unknown");
});

test("reconcile lines say gaps plainly", () => {
  assert.equal(reconcileLine("actual", null), "No closed bill reconciled yet.");
  assert.equal(reconcileLine("forecast", null), "No RMP forecast yet.");
  assert.equal(
    reconcileLine("actual", { billing_period_start: "2026-08-12", reconcile_gap: "usage_incomplete" }),
    "Last bill 2026-08-12: Orion can't price this period yet (usage_incomplete)",
  );
  assert.equal(
    reconcileLine("forecast", { billing_period_start: "2026-09-11", orion_total_usd: 88.4, utility_total_usd: 80, delta_usd: 8.4 }),
    "RMP forecast 2026-09-11: Orion $88.40 vs RMP $80.00 (diff $8.40)",
  );
  assert.equal(
    reconcileLine("forecast", { billing_period_start: "2026-09-11", orion_total_usd: 88.4, utility_total_usd: null, delta_usd: null }),
    "RMP forecast 2026-09-11: Orion $88.40 vs RMP unknown (diff unknown)",
  );
});

test("bar heights scale to the max day", () => {
  assert.deepEqual(barHeights([{ kwh: 10 }, { kwh: 20 }, { kwh: 0 }], 60), [30, 60, 0]);
  assert.deepEqual(barHeights([], 60), []);
  assert.deepEqual(barHeights([{ kwh: 0 }], 60), [0]);
});

function fakeDocument(ids) {
  const els = {};
  ids.forEach(function (id) {
    els[id] = {
      textContent: "",
      children: [],
      replaceChildren() { this.children = []; },
      appendChild(child) { this.children.push(child); },
    };
  });
  return {
    els: els,
    getElementById(id) { return els[id] || null; },
    createElement() { return { style: {}, title: "" }; },
  };
}

const IDS = [
  "energyImporterState", "energyCycleToDate", "energyProjected", "energyForecast", "energyMarginal",
  "energyPressure", "energyReconcileActual", "energyReconcileForecast", "energyDailyBars",
];

test("renderLatest writes unknown for null values and real numbers otherwise", () => {
  const doc = fakeDocument(IDS);
  globalThis.document = doc;
  try {
    strip.renderLatest({
      ok: true,
      stakes: {
        cycle_to_date_total_usd: 17.2, orion_projected_total_usd: null, forecast_total_usd: null,
        marginal_usd_per_kwh: 0.1123, pressure: "unknown", pressure_reason: "forecast_not_current",
      },
      importer: { state: "stale", reason: "usage_lag" },
      reconcile: { actual: { billing_period_start: "2026-08-12", orion_total_usd: 99, utility_total_usd: 97.13, delta_usd: 1.87 } },
    });
    assert.equal(doc.els.energyCycleToDate.textContent, "$17.20");
    assert.equal(doc.els.energyProjected.textContent, "unknown");
    assert.equal(doc.els.energyForecast.textContent, "unknown");
    assert.equal(doc.els.energyMarginal.textContent, "$0.1123/kWh");
    assert.equal(doc.els.energyPressure.textContent, "unknown (forecast_not_current)");
    assert.equal(doc.els.energyImporterState.textContent, "importer: stale (usage_lag)");
    assert.equal(doc.els.energyReconcileActual.textContent, "Last bill 2026-08-12: Orion $99.00 vs RMP $97.13 (diff $1.87)");
    assert.equal(doc.els.energyReconcileForecast.textContent, "No RMP forecast yet.");
  } finally {
    delete globalThis.document;
  }
});

test("renderLatest reports an unavailable API instead of zeros", () => {
  const doc = fakeDocument(IDS);
  globalThis.document = doc;
  try {
    strip.renderLatest({ ok: false, error: "energy_unavailable", stakes: null, importer: null, reconcile: {} });
    assert.equal(doc.els.energyImporterState.textContent, "energy data unavailable (energy_unavailable)");
    assert.equal(doc.els.energyCycleToDate.textContent, "unknown");
    assert.equal(doc.els.energyPressure.textContent, "unknown");
  } finally {
    delete globalThis.document;
  }
});

test("renderDaily draws one bar per day with a readable title", () => {
  const doc = fakeDocument(IDS);
  globalThis.document = doc;
  try {
    strip.renderDaily({ ok: true, points: [{ day: "2026-09-25", kwh: 12, hours: 24 }, { day: "2026-09-26", kwh: 24, hours: 23 }] });
    const bars = doc.els.energyDailyBars.children;
    assert.equal(bars.length, 2);
    assert.equal(bars[0].style.height, "30px");
    assert.equal(bars[1].style.height, "60px");
    assert.equal(bars[1].title, "2026-09-26: 24.0 kWh (23 h)");
    strip.renderDaily({ ok: false, points: [] });
    assert.equal(doc.els.energyDailyBars.children.length, 0);
  } finally {
    delete globalThis.document;
  }
});

test("activate polls both endpoints once and deactivate stops the timer", async () => {
  const calls = [];
  const intervals = [];
  const cleared = [];
  const origSet = globalThis.setInterval;
  const origClear = globalThis.clearInterval;
  globalThis.fetch = async function (url) {
    calls.push(url);
    return { json: async () => (url.indexOf("daily") >= 0 ? { ok: true, points: [] } : { ok: false, reconcile: {} }) };
  };
  globalThis.setInterval = function (fn, ms) { intervals.push(ms); return 7; };
  globalThis.clearInterval = function (id) { cleared.push(id); };
  try {
    strip.activate();
    strip.activate();
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(calls.sort(), ["/api/energy/latest", "/api/energy/usage/daily?days=14"]);
    assert.deepEqual(intervals, [60000]);
    strip.deactivate();
    strip.deactivate();
    assert.deepEqual(cleared, [7]);
  } finally {
    globalThis.setInterval = origSet;
    globalThis.clearInterval = origClear;
    delete globalThis.fetch;
  }
});
