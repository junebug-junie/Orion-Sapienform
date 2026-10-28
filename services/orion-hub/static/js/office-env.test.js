const test = require("node:test");
const assert = require("node:assert/strict");
const { splitRuns, num } = require("./office-env.js");

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
