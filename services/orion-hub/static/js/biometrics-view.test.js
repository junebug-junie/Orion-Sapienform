const test = require("node:test");
const assert = require("node:assert/strict");
const biometricsView = require("./biometrics-view.js");

const { shouldPoll } = biometricsView;

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
