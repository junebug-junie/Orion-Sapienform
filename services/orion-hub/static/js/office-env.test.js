const test = require("node:test");
const assert = require("node:assert/strict");
const { splitRuns, num, airText } = require("./office-env.js");

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
