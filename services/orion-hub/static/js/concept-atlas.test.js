const test = require("node:test");
const assert = require("node:assert/strict");
const { shouldDeclutterLabels, LABEL_DECLUTTER_MIN_NODES } = require("./concept-atlas.js");

// The bug this locks down (confirmed live 2026-08-28): label decluttering was
// gated on whether ANY god node existed, which is always true -- canonical
// seed concepts are god nodes unconditionally, regardless of degree. So a
// 24-node graph hid 19 of its labels and rendered them as unlabeled dots.
// That is not decluttering, it is loss.

test("a small graph shows every label even though god nodes exist", () => {
  assert.equal(shouldDeclutterLabels(24, true), false);
  assert.equal(shouldDeclutterLabels(62, true), true);
});

test("the threshold boundary is inclusive", () => {
  assert.equal(shouldDeclutterLabels(LABEL_DECLUTTER_MIN_NODES - 1, true), false);
  assert.equal(shouldDeclutterLabels(LABEL_DECLUTTER_MIN_NODES, true), true);
});

test("no god node means nothing to orient on, so never hide labels", () => {
  assert.equal(shouldDeclutterLabels(500, false), false);
});

test("degrades to showing labels on malformed counts rather than hiding them", () => {
  for (const bad of [undefined, null, NaN, "not a number"]) {
    assert.equal(shouldDeclutterLabels(bad, true), false);
  }
});
