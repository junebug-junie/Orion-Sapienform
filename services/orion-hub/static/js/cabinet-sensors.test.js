const test = require("node:test");
const assert = require("node:assert/strict");
const cabinetSensors = require("./cabinet-sensors.js");

const { nearestSeriesPoint, formatHoverTooltipText } = cabinetSensors;

function xFn(point) {
  return point.x;
}

test("nearestSeriesPoint returns the exact match when the target lands on a point", () => {
  const points = [
    { x: 0, value: 1 },
    { x: 25, value: 2 },
    { x: 50, value: 3 },
    { x: 100, value: 4 },
  ];
  assert.equal(nearestSeriesPoint(points, xFn, 50), points[2]);
});

test("nearestSeriesPoint picks whichever neighbor is closer", () => {
  const points = [
    { x: 0, value: 1 },
    { x: 10, value: 2 },
    { x: 90, value: 3 },
  ];
  assert.equal(nearestSeriesPoint(points, xFn, 12), points[1]);
  assert.equal(nearestSeriesPoint(points, xFn, 60), points[2]);
});

// A chart with a single sample still has to resolve every mouse position to
// that one point, not throw.
test("nearestSeriesPoint handles a single-point series", () => {
  const points = [{ x: 42, value: 7 }];
  assert.equal(nearestSeriesPoint(points, xFn, 0), points[0]);
  assert.equal(nearestSeriesPoint(points, xFn, 100), points[0]);
});

test("formatHoverTooltipText rounds the reading to the requested digits and includes the label", () => {
  const point = { t: "2026-09-15T12:00:00Z", value: 21.456 };
  assert.match(formatHoverTooltipText(point, { label: "temp_c", digits: 1 }), /temp_c 21\.5$/);
});

test("formatHoverTooltipText defaults to 2 digits when none are given", () => {
  const point = { t: "2026-09-15T12:00:00Z", value: 21.456 };
  assert.match(formatHoverTooltipText(point, { label: "temp_c" }), /temp_c 21\.46$/);
});

// A sample with no timestamp (index-only chart, or an edge-of-window point
// that failed to parse) must render em dashes, never "Invalid Date".
test("formatHoverTooltipText falls back to em dashes when the sample has no timestamp", () => {
  assert.equal(formatHoverTooltipText({ t: null, value: 5 }, { label: "rms", digits: 0 }), "— —  ·  rms 5");
});
