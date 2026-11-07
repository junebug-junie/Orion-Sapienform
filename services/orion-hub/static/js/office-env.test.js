const test = require("node:test");
const assert = require("node:assert/strict");
const { splitRuns, num, airText, readingText, lightText, createStrip, OFFICE, CLOSET } = require("./office-env.js");

const pt = (min, v) => ({ t: new Date(Date.UTC(2026, 9, 27, 12, min)).toISOString(), office_temp_c: v });

test("num returns null for absent values and formats present ones", () => {
  assert.equal(num(undefined, 2), null);
  assert.equal(num(null, 2), null);
  assert.equal(num(NaN, 2), null);
  assert.equal(num(21.456, 2), "21.46");
  assert.equal(num(0, 1), "0.0"); // a real zero is still a reading
});

test("splitRuns keeps contiguous samples in one run", () => {
  const runs = splitRuns([pt(0, 20), pt(1, 21), pt(2, 22)], "office_temp_c");
  assert.equal(runs.length, 1);
  assert.equal(runs[0].length, 3);
});

test("splitRuns breaks the line across a sensor outage instead of bridging it", () => {
  const runs = splitRuns([pt(0, 20), pt(1, 21), pt(2, 22), pt(40, 19), pt(41, 19.5)], "office_temp_c");
  assert.equal(runs.length, 2);
  assert.equal(runs[1][0].v, 19);
});

test("splitRuns skips points that lack the field (absent is not zero)", () => {
  const runs = splitRuns([pt(0, 20), { t: pt(1, 0).t }, pt(2, 22)], "office_temp_c");
  assert.deepEqual(runs.flat().map((p) => p.v), [20, 22]);
});

test("splitRuns on empty input yields no runs", () => {
  assert.deepEqual(splitRuns([], "office_temp_c"), []);
});

test("splitRuns breaks at a server-flagged gap even when spacing looks normal", () => {
  const a = pt(0, 20), b = pt(10, 21), c = Object.assign(pt(20, 30), { gap: true }), d = pt(30, 31);
  const runs = splitRuns([a, b, c, d], "office_temp_c");
  assert.equal(runs.length, 2);
  assert.equal(runs[1][0].v, 30);
});

test("airText shows warming instead of a number during CCS811 conditioning", () => {
  const sample = { air_warming: true };
  assert.equal(airText({ ok: true, sample }, "office_eco2_ppm"), "warming up");
  assert.equal(airText({ ok: true, sample: { office_eco2_ppm: 612 } }, "office_eco2_ppm"), "612");
  assert.equal(airText({ ok: true, sample: { office_tvoc_ppb: 0 } }, "office_tvoc_ppb"), "0"); // real zero TVOC
  assert.equal(airText({ ok: true, sample: {} }, "office_eco2_ppm"), "—");
  assert.equal(airText({ ok: false, sample: { office_eco2_ppm: 612 } }, "office_eco2_ppm"), "stale");
  assert.equal(airText({ ok: true, sample: null }, "office_eco2_ppm"), "—");
});

test("splitRuns breaks only the field named in gap_fields", () => {
  const p = (min, extra) => Object.assign({ t: new Date(Date.UTC(2026, 9, 27, 12, min)).toISOString(), office_temp_c: 21, office_eco2_ppm: 600 }, extra || {});
  const pts = [p(0), p(1), p(2, { gap_fields: ["office_eco2_ppm"] }), p(3)];
  assert.equal(splitRuns(pts, "office_eco2_ppm").length, 2);
  assert.equal(splitRuns(pts, "office_temp_c").length, 1);
});

test("lightText says too bright to measure when saturated, never 0", () => {
  assert.equal(lightText({ ok: true, sample: { light_saturated: true } }), "too bright to measure");
  assert.equal(lightText({ ok: true, sample: { light_saturated: false } }), "—"); // absent, not 0
  assert.equal(lightText({ ok: true, sample: {} }), "—");
  assert.equal(lightText({ ok: true, sample: { office_lux: 0 } }), "0.00"); // real dark is a reading
  assert.equal(lightText({ ok: true, sample: { office_lux: 3.456 } }), "3.46");
  assert.equal(lightText({ ok: true, sample: { office_lux: 42.24 } }), "42.2");
  assert.equal(lightText({ ok: true, sample: { office_lux: 12345.6 } }), "12346");
  assert.equal(lightText({ ok: false, sample: { light_saturated: true } }), "stale");
  assert.equal(lightText({ ok: true, sample: null }), "—");
});

test("readingText: PM tiles never borrow the CCS811 warm-up text", () => {
  const warming = { ok: true, sample: { air_warming: true } };
  assert.equal(readingText(warming, "office_pm25_ug_m3", 0), "—");
  assert.equal(readingText({ ok: true, sample: { office_pm25_ug_m3: 0 } }, "office_pm25_ug_m3", 0), "0");
  assert.equal(readingText({ ok: false, sample: { office_pm25_ug_m3: 9 } }, "office_pm25_ug_m3", 0), "stale");
});

// Minimal element stand-ins: renderLatest only sets textContent.
function renderTiles(config, payload) {
  const els = {};
  const realDocument = global.document;
  global.document = {
    getElementById: (id) => (els[id] = els[id] || { id, textContent: "" }),
    querySelectorAll: () => [],
  };
  try {
    const strip = createStrip(config);
    assert.equal(strip.init(), true);
    strip.renderLatest(payload);
  } finally {
    global.document = realDocument;
  }
  return Object.fromEntries(Object.entries(els).map(([k, v]) => [k, v.textContent]));
}

test("office strip renders PM and light tiles, saturated light in words", () => {
  const out = renderTiles(OFFICE, {
    ok: true,
    age_sec: 4.2,
    sample: { office_temp_c: 21.4, office_humidity_pct: 33, office_pm1_ug_m3: 2, office_pm25_ug_m3: 5, office_pm10_ug_m3: 7, light_saturated: true, office_sound_dba: 41.46 },
  });
  assert.equal(out.officeEnvSound, "41.5");
  assert.equal(out.officeEnvPm1, "2");
  assert.equal(out.officeEnvPm25, "5");
  assert.equal(out.officeEnvPm10, "7");
  assert.equal(out.officeEnvLux, "too bright to measure");
  assert.equal(out.officeEnvTemp, "21.40");
  assert.equal(out.officeEnvAge, "4s");
  assert.equal(out.officeEnvLiveStatus, "fresh");
});

test("office strip: rows from before the new sensors show dashes, not zeros", () => {
  const out = renderTiles(OFFICE, { ok: true, age_sec: 1, sample: { office_temp_c: 21.4 } });
  for (const id of ["officeEnvPm1", "officeEnvPm25", "officeEnvPm10", "officeEnvLux", "officeEnvSound"]) assert.equal(out[id], "—", id);
});

test("closet strip renders CO2/temp/humidity and handles no samples", () => {
  const live = renderTiles(CLOSET, { ok: true, age_sec: 3, sample: { closet_co2_ppm: 812, closet_temp_c: 27.31, closet_humidity_pct: 40.04 } });
  assert.equal(live.closetEnvCo2, "812");
  assert.equal(live.closetEnvTemp, "27.3");
  assert.equal(live.closetEnvHumidity, "40.0");
  const none = renderTiles(CLOSET, { ok: false, age_sec: null, sample: null });
  assert.equal(none.closetEnvCo2, "—");
  assert.equal(none.closetEnvLiveStatus, "no samples yet");
  const stale = renderTiles(CLOSET, { ok: false, age_sec: 200, publisher_stale: true, sample: { sensor_online: false } });
  assert.equal(stale.closetEnvCo2, "stale");
  assert.equal(stale.closetEnvLiveStatus, "no recent publish");
});
