const test = require("node:test");
const assert = require("node:assert/strict");
const { splitAtGaps, niceTicks, valueAt, formatReading, SERIES } = require("./cabinet-climate-overlay.js");

const at = (min, v, extra) => Object.assign({ t: new Date(Date.UTC(2026, 9, 28, 12, min)).toISOString(), v }, extra);

test("splitAtGaps keeps buckets one grain apart in one run", () => {
  const runs = splitAtGaps([at(0, 1), at(5, 2), at(10, 3)], 300);
  assert.equal(runs.length, 1);
  assert.equal(runs[0].length, 3);
});

test("splitAtGaps breaks the line across a missing stretch instead of bridging it", () => {
  const runs = splitAtGaps([at(0, 1), at(5, 2), at(60, 3), at(65, 4)], 300);
  assert.deepEqual(runs.map((r) => r.map((p) => p.v)), [[1, 2], [3, 4]]);
});

test("splitAtGaps uses the series' own grain (hourly weather is not a gap)", () => {
  assert.equal(splitAtGaps([at(0, 10), at(60, 11), at(120, 12)], 3600).length, 1);
});

test("splitAtGaps drops absent values rather than zeroing them", () => {
  const runs = splitAtGaps([at(0, 1), { t: at(5, 0).t, v: null }, at(10, 3)], 300);
  assert.deepEqual(runs.flat().map((p) => p.v), [1, 3]);
});

test("niceTicks covers the data with round steps", () => {
  const s = niceTicks(14.3, 28.9, 4);
  assert.ok(s.min <= 14.3 && s.max >= 28.9);
  assert.deepEqual(s.ticks, [10, 15, 20, 25, 30]);
});

test("niceTicks widens a flat series instead of dividing by zero", () => {
  const s = niceTicks(1500, 1500, 2);
  assert.ok(s.max > s.min);
});

test("valueAt returns null across a gap (no reading, not the far neighbor)", () => {
  const pts = splitAtGaps([at(0, 1), at(60, 2)], 300).flat();
  assert.equal(valueAt(pts, Date.parse(at(30, 0).t), 225000), null);
  assert.equal(valueAt(pts, Date.parse(at(1, 0).t), 225000).v, 1);
});

test("formatReading shows units, source counts, and 'no reading' for absent", () => {
  const watts = SERIES.find((s) => s.key === "orion_watts");
  const office = SERIES.find((s) => s.key === "office_temp_c");
  assert.equal(formatReading(watts, { v: 1569.6, sources: 4 }), "1570 W (4 sources)");
  assert.equal(formatReading(office, { v: 26.08 }), "26.1 °C");
  assert.equal(formatReading(office, null), "no reading");
});
