const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {
  labelMinZoomedFontSize,
  edgeLabelMinZoomedFontSize,
  LABEL_MIN_ZOOMED_FONT_PX,
  EDGE_LABEL_MIN_ZOOMED_FONT_PX,
  GOD_NODE_FONT_PX,
  NODE_FONT_PX,
} = require("./concept-atlas.js");

// The bug this locks down (confirmed live 2026-08-29, from a screenshot):
// label decluttering was all-or-nothing -- ">= 60 nodes and any god node
// exists" hid every non-god label. At 136 nodes that hid 131 of 136, so the
// atlas rendered as a field of unlabeled dots with five purple exceptions.
// The same class of failure was fixed at 24 nodes on 2026-08-28; the gate
// just moved the cliff rather than removing it.

test("no graph size can make a node permanently unlabeled", () => {
  // The whole point: the threshold is a ZOOM level, never a node count.
  // labelMinZoomedFontSize takes no node count at all, so there is no value
  // any graph could have that switches a label off for good.
  assert.equal(labelMinZoomedFontSize.length, 2);
  for (const godNode of [true, false]) {
    assert.ok(Number.isFinite(labelMinZoomedFontSize(false, godNode)));
  }
});

test("god node labels survive to a lower zoom than ordinary ones", () => {
  // The useful half of the old gate -- orientation from the god nodes --
  // without discarding the other 131 labels to get it.
  assert.ok(labelMinZoomedFontSize(false, true) > labelMinZoomedFontSize(false, false));
});

test("god node labels are rendered larger, which is what buys them that", () => {
  // min-zoomed-font-size compares against the ON-SCREEN font size, so a god
  // node only survives further out because its font is bigger. If these two
  // ever invert, the threshold above silently does the opposite of intended.
  assert.ok(GOD_NODE_FONT_PX > NODE_FONT_PX);
  assert.equal(labelMinZoomedFontSize(false, true), GOD_NODE_FONT_PX);
  assert.equal(labelMinZoomedFontSize(false, false), LABEL_MIN_ZOOMED_FONT_PX);
});

test("show-all disables the threshold entirely rather than merely lowering it", () => {
  // A small-but-nonzero value would still drop labels when zoomed far out,
  // which is exactly what a user checking "show all labels" is complaining
  // about. Only 0 turns the culling off in cytoscape.
  assert.equal(labelMinZoomedFontSize(true, false), 0);
  assert.equal(labelMinZoomedFontSize(true, true), 0);
  assert.equal(edgeLabelMinZoomedFontSize(true), 0);
});

test("edge labels need more zoom than node labels", () => {
  // 461 edges vs 136 nodes live, and far less information per pixel.
  assert.ok(EDGE_LABEL_MIN_ZOOMED_FONT_PX > LABEL_MIN_ZOOMED_FONT_PX);
  assert.equal(edgeLabelMinZoomedFontSize(false), EDGE_LABEL_MIN_ZOOMED_FONT_PX);
});

// --- wiring: the pure helpers are worthless if the stylesheet ignores them.
// cytoscape can't be instantiated headlessly here, so read the source, the
// same way the Hub scheduler policy test pins its own wiring.

const SOURCE = fs.readFileSync(path.join(__dirname, "concept-atlas.js"), "utf8");

test("the node stylesheet labels every node unconditionally", () => {
  assert.match(SOURCE, /label:\s*\(ele\)\s*=>\s*ele\.data\("label"\)/);
  // And nothing may reintroduce a count-based label gate.
  assert.doesNotMatch(SOURCE, /shouldDeclutterLabels/);
  assert.doesNotMatch(SOURCE, /LABEL_DECLUTTER_MIN_NODES/);
});

test("both stylesheets actually call the helpers", () => {
  assert.match(SOURCE, /"min-zoomed-font-size":\s*\(ele\)\s*=>\s*\n?\s*labelMinZoomedFontSize\(/);
  assert.match(SOURCE, /"min-zoomed-font-size":\s*\(\)\s*=>\s*edgeLabelMinZoomedFontSize\(/);
});

test("labels get a background plate so they stay readable over edges", () => {
  // Without this, a label drawn on top of the dense edge mesh in the live
  // screenshot is technically rendered and still unreadable.
  assert.match(SOURCE, /"text-background-opacity":/);
});

test("the page copy no longer promises god-node-only labels", () => {
  const template = fs.readFileSync(
    path.join(__dirname, "..", "..", "templates", "concept_atlas.html"),
    "utf8",
  );
  assert.doesNotMatch(template, /Only god-node labels show by default/);
  assert.match(template, /Every node is labelled/);
});
