const test = require("node:test");
const assert = require("node:assert/strict");
const biometricsView = require("./biometrics-view.js");

const { shouldPoll, laneBadge } = biometricsView;

// A backgrounded tab keeps running setInterval (throttled, not stopped), so
// without this gate a Hub left open in a background tab polled
// /api/biometrics/preview/{snapshot,induction,gpu} forever with nobody
// looking -- and /induction is the route that reads a 187k-row table.
test("polls while the page is visible", () => {
  assert.equal(shouldPoll({ hidden: false }), true);
});

test("does not poll while the page is hidden", () => {
  assert.equal(shouldPoll({ hidden: true }), false);
});

// Failing open matters: a document without the Page Visibility API that we
// treated as hidden would stop refreshing forever, and stale-but-rendered
// tiles are indistinguishable from calm live ones.
test("polls when the document does not implement the Page Visibility API", () => {
  assert.equal(shouldPoll({}), true);
  assert.equal(shouldPoll(null), true);
});

// `hidden` is specified as a boolean. Anything else is not a hidden signal,
// and must not be coerced into one -- a truthy non-boolean silently pausing
// every poll is exactly the failure this guard exists to avoid.
test("only a literal boolean true counts as hidden", () => {
  assert.equal(shouldPoll({ hidden: "true" }), true);
  assert.equal(shouldPoll({ hidden: 1 }), true);
  assert.equal(shouldPoll({ hidden: undefined }), true);
});

// Stage 5.5: the GPU lane badge shows what the pool says is on the card, and
// greys out only when the server says the card is not labelled (not by string
// match -- "no pool state" is unknown, and must not render as a live lane).
test("a pool-derived lane renders as an assigned badge", () => {
  const b = laneBadge({ lane: "world, diffusion", lane_assigned: true });
  assert.equal(b.text, "world, diffusion");
  assert.equal(b.assigned, true);
  assert.match(b.className, /indigo/);
});

test("no pool state renders greyed, not as a live lane", () => {
  const b = laneBadge({ lane: "no pool state", lane_assigned: false });
  assert.equal(b.text, "no pool state");
  assert.equal(b.assigned, false);
  assert.match(b.className, /gray/);
});

test("a response without lane_assigned falls back to the unassigned string", () => {
  assert.equal(laneBadge({ lane: "unassigned" }).assigned, false);
  assert.equal(laneBadge({ lane: "chat" }).assigned, true);
  assert.equal(laneBadge({}).text, "unassigned");
});

// Motherboard heat tile: read from the node's own snapshot measurements, and absent
// (not 0) when the node has no BMC board sensor -- the tile then renders "—".
test("boardTempFor reads board_temp_c_max from the node's own snapshot", () => {
  const snap = { summary: { measurements: { board_temp_c_max: 45.0, temp_c_max: 63.0 } } };
  assert.equal(biometricsView.boardTempFor(snap), 45.0);
});

test("boardTempFor is undefined when nothing measured it", () => {
  assert.equal(biometricsView.boardTempFor({ ok: false }), undefined);
  assert.equal(biometricsView.boardTempFor({ summary: { measurements: {} } }), undefined);
  assert.equal(biometricsView.boardTempFor(null), undefined);
});

// Orion's tiredness gauge (dream sleep pressure). Live 2026-10-09: 11.06 / 3.0.
const { tirednessModel } = biometricsView;
const reading = (pressure, extra = {}) => ({
  enabled: true, ready: false, too_soon: false, is_idle: true,
  pressure: { pressure, threshold: 3, idle_required_minutes: 45, new_counts: {}, ...extra.p },
  ...extra.flags,
});

test("tiredness puts the sleep line mid-track and fills proportionally below it", () => {
  const m = tirednessModel(reading(1.5));
  assert.equal(m.available, true);
  assert.equal(m.lineFraction, 0.5);
  assert.equal(m.fraction, 0.25);
  assert.equal(m.level, "Getting tired");
  assert.equal(m.waiting, "Not tired enough to sleep yet");
});

test("tiredness past twice the line pins the fill and says it is off the scale", () => {
  const m = tirednessModel(reading(11.06, { flags: { too_soon: true } }));
  assert.equal(m.fraction, 1);
  assert.equal(m.overflow, true);
  assert.equal(m.level, "Ready to sleep");
  assert.match(m.waiting, /6 h minimum/);
});

test("zero tiredness reads rested, and the reason it is not asleep follows the gates", () => {
  assert.equal(tirednessModel(reading(0)).level, "Rested");
  assert.match(tirednessModel(reading(4, { flags: { is_idle: false } })).waiting, /45 min with no chat/);
  assert.equal(tirednessModel(reading(4, { flags: { ready: true } })).waiting, "Will sleep at the next check");
  assert.equal(tirednessModel(reading(4, { flags: { enabled: false } })).waiting, "Sleep loop is off");
});

test("new material is named in plain words, in a fixed order", () => {
  const m = tirednessModel(reading(9, { p: { new_counts: { crystallization: 1, metacog: 14 } } }));
  assert.equal(m.newText, "New since last sleep: 14 self-noticed problems · 1 new memory");
  assert.equal(tirednessModel(reading(0)).newText, "Nothing new since last sleep");
});

// An unavailable dream service must not render as a calm, rested gauge.
test("a missing or malformed reading is unavailable, never rested", () => {
  assert.equal(tirednessModel({}).available, false);
  assert.equal(tirednessModel({ pressure: { pressure: 0 } }).available, false);
  assert.equal(tirednessModel({ pressure: { pressure: 1, threshold: 0 } }).available, false);
});
