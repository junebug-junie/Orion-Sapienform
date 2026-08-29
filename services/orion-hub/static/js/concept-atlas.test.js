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
  collapseEvidenceNodes,
  nodeDisplayLabel,
  structureDiagnosis,
  componentShapeLine,
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


// --- evidence collapse ------------------------------------------------------
//
// Live shape being defended (orion_substrate, 2026-08-29): 136 nodes, of which
// 80 are Evidence nodes with no label of their own. The route names each one
// "Evidence for <concept>", so 59% of the rendered canvas repeated its
// neighbour's name. Folding them into a count takes the default view to 56.

function evidenceFixture() {
  // 2 concepts, 3 evidence nodes: c1 has 2, c2 has 1.
  return {
    nodes: [
      { id: "c1", node_kind: "concept", label: "Home lab infrastructure" },
      { id: "c2", node_kind: "concept", label: "Light folding concept" },
      { id: "e1", node_kind: "evidence", label: "Evidence for Home lab infrastructure" },
      { id: "e2", node_kind: "evidence", label: "Evidence for Home lab infrastructure" },
      { id: "e3", node_kind: "evidence", label: "Evidence for Light folding concept" },
    ],
    edges: [
      { source: "e1", target: "c1", predicate: "supports" },
      { source: "e2", target: "c1", predicate: "supports" },
      { source: "e3", target: "c2", predicate: "supports" },
      { source: "c1", target: "c2", predicate: "co_occurs_with" },
    ],
  };
}

test("collapse folds evidence nodes into a count on the concept they support", () => {
  const { nodes, edges } = evidenceFixture();
  const out = collapseEvidenceNodes(nodes, edges, true);
  assert.equal(out.nodes.length, 2);
  assert.equal(out.collapsedCount, 3);
  assert.equal(out.nodes.find((n) => n.id === "c1").evidence_count, 2);
  assert.equal(out.nodes.find((n) => n.id === "c2").evidence_count, 1);
});

test("collapse drops the edges to folded nodes but keeps concept-concept edges", () => {
  const { nodes, edges } = evidenceFixture();
  const out = collapseEvidenceNodes(nodes, edges, true);
  assert.equal(out.edges.length, 1);
  assert.equal(out.edges[0].predicate, "co_occurs_with");
});

test("collapse disabled is a pass-through that adds no count", () => {
  const { nodes, edges } = evidenceFixture();
  const out = collapseEvidenceNodes(nodes, edges, false);
  assert.equal(out.nodes.length, 5);
  assert.equal(out.edges.length, 4);
  assert.equal(out.collapsedCount, 0);
  assert.equal(out.nodes.every((n) => n.evidence_count === undefined), true);
});

test("collapse does not mutate the caller's nodes", () => {
  // The load path reuses the payload across renders; a mutating collapse
  // would make the count compound on every toggle.
  const { nodes, edges } = evidenceFixture();
  collapseEvidenceNodes(nodes, edges, true);
  assert.equal(nodes.find((n) => n.id === "c1").evidence_count, undefined);
  assert.equal(nodes.length, 5);
});

test("a graph with no evidence nodes is untouched and reports nothing folded", () => {
  const nodes = [{ id: "c1", node_kind: "concept", label: "A" }];
  const edges = [];
  const out = collapseEvidenceNodes(nodes, edges, true);
  assert.equal(out.collapsedCount, 0);
  assert.equal(out.nodes, nodes, "should return the same array, not a copy");
});

test("only `supports` edges from evidence contribute to the count", () => {
  // An entity -> concept `supports` edge must not inflate the evidence count:
  // entities are hydrated into this same node list and are not evidence.
  const nodes = [
    { id: "c1", node_kind: "concept", label: "A" },
    { id: "n1", node_kind: "entity", label: "Juniper" },
    { id: "e1", node_kind: "evidence" },
  ];
  const edges = [
    { source: "n1", target: "c1", predicate: "supports" },
    { source: "e1", target: "c1", predicate: "supports" },
    { source: "e1", target: "c1", predicate: "co_occurs_with" },
  ];
  const out = collapseEvidenceNodes(nodes, edges, true);
  assert.equal(out.nodes.find((n) => n.id === "c1").evidence_count, 1);
  assert.equal(out.nodes.length, 2, "the entity node survives the collapse");
});

test("evidence with no supports edge is still removed, counted nowhere", () => {
  const nodes = [
    { id: "c1", node_kind: "concept", label: "A" },
    { id: "e1", node_kind: "evidence" },
  ];
  const out = collapseEvidenceNodes(nodes, [], true);
  assert.equal(out.nodes.length, 1);
  assert.equal(out.collapsedCount, 1);
  assert.equal(out.nodes[0].evidence_count, undefined);
});

test("label appends the evidence count without replacing the concept name", () => {
  assert.equal(
    nodeDisplayLabel({ id: "c1", label: "Home lab infrastructure", evidence_count: 3 }),
    "Home lab infrastructure (3 evidence)"
  );
  assert.equal(nodeDisplayLabel({ id: "c1", label: "Home lab infrastructure" }), "Home lab infrastructure");
  assert.equal(nodeDisplayLabel({ id: "c1", label: "A", evidence_count: 0 }), "A");
});

test("label keeps the unlabeled-topic marker and stacks the count after it", () => {
  assert.equal(
    nodeDisplayLabel({ id: "x", label: "topic_7", synthetic_label: true, evidence_count: 2 }),
    "topic_7 (unlabeled topic) (2 evidence)"
  );
});

test("label falls back to the node id when there is no label at all", () => {
  assert.equal(nodeDisplayLabel({ id: "sub-evidence-abc" }), "sub-evidence-abc");
});


// --- structural diagnosis ---------------------------------------------------

function livePayload(overrides) {
  // orion_substrate as measured 2026-08-29.
  return Object.assign(
    {
      available: true,
      node_count: 136,
      concept_count: 56,
      edge_count: 461,
      edge_type_counts: { co_occurs_with: 307, supports: 80, associated_with: 74 },
      dominant_edge_type: "co_occurs_with",
      dominant_edge_saturation: 0.1994,
      component_count: 12,
      largest_component_size: 116,
      singleton_count: 10,
    },
    overrides || {}
  );
}

test("diagnosis states the measured numbers, not a severity band", () => {
  const note = structureDiagnosis(livePayload());
  assert.match(note, /307 of 461 edges \(67%\)/);
  assert.match(note, /19\.9% of every possible pair/);
  assert.match(note, /56 concepts/);
  assert.match(note, /co_occurs_with/);
});

test("diagnosis stays silent when one edge type does not dominate", () => {
  // Same saturation, but the dominant type is under half the edges.
  const note = structureDiagnosis(
    livePayload({ edge_type_counts: { co_occurs_with: 200, supports: 261 }, edge_count: 461 })
  );
  assert.equal(note, null);
});

test("diagnosis stays silent on a sparse graph even when one type dominates", () => {
  const note = structureDiagnosis(livePayload({ dominant_edge_saturation: 0.02 }));
  assert.equal(note, null);
});

test("diagnosis stays silent when the payload cannot support the claim", () => {
  assert.equal(structureDiagnosis(null), null);
  assert.equal(structureDiagnosis({ available: false }), null);
  assert.equal(structureDiagnosis(livePayload({ dominant_edge_type: null })), null);
  assert.equal(structureDiagnosis(livePayload({ dominant_edge_saturation: null })), null);
  assert.equal(structureDiagnosis(livePayload({ dominant_edge_saturation: undefined })), null);
});

test("diagnosis does not divide by zero on an edgeless graph", () => {
  // orion_worldview live 2026-08-29: 48 nodes, 0 edges of any type.
  const note = structureDiagnosis(
    livePayload({ edge_count: 0, edge_type_counts: {}, dominant_edge_type: null, dominant_edge_saturation: null })
  );
  assert.equal(note, null);
});

// --- component shape line ---------------------------------------------------

test("component line splits the blob, the middle, and the singletons", () => {
  // "12 components" alone says nothing; this is the read.
  assert.equal(componentShapeLine(livePayload()), "12 components: 1 of 116 + 1 smaller + 10 singletons");
});

test("component line handles an all-singletons graph", () => {
  // orion_worldview: 48 nodes, 0 edges -> 48 components of 1.
  //
  // assert.equal, NOT assert.match. This test originally used two match()
  // calls for /48 components/ and /48 singletons/, and passed on the wrong
  // string: the largest component IS a singleton here, so it was counted
  // twice and the line read "48 components: 1 of 1 + 48 singletons" -- 49
  // components across the parts. Both regexes matched that happily. A whole-
  // string assertion is what catches an extra clause.
  assert.equal(
    componentShapeLine(
      livePayload({ component_count: 48, largest_component_size: 1, singleton_count: 48 })
    ),
    "48 components: 48 singletons"
  );
});

test("component line does not invent a blob for a two-singleton graph", () => {
  assert.equal(
    componentShapeLine(livePayload({ component_count: 2, largest_component_size: 1, singleton_count: 2 })),
    "2 components: 2 singletons"
  );
});

test("component line handles a single fully-connected graph", () => {
  // orion_bus_synapse: 313 nodes, 1 component, no singletons.
  assert.equal(
    componentShapeLine(livePayload({ component_count: 1, largest_component_size: 313, singleton_count: 0 })),
    "1 component: 1 of 313"
  );
});

test("component line says so rather than rendering an empty split", () => {
  assert.equal(componentShapeLine(livePayload({ component_count: 0 })), "no components");
  assert.equal(componentShapeLine({ available: false }), "");
  assert.equal(componentShapeLine(null), "");
});

test("component line uses the singular for exactly one singleton", () => {
  const line = componentShapeLine(
    livePayload({ component_count: 2, largest_component_size: 5, singleton_count: 1 })
  );
  assert.match(line, /1 singleton$/);
});

test("diagnosis never renders NaN from an internally inconsistent payload", () => {
  // Unreachable from the current route (edge_count is the sum of
  // edge_type_counts, so 0 edges implies dominant_edge_type is null), but
  // reachable from a stale cached bundle or a future route that reports the
  // two fields independently. Without the `edges > 0` guard the share is
  // NaN, every comparison against it is false, and the card renders the
  // sentence with "NaN%" in it -- a confidently wrong number rather than
  // silence.
  const note = structureDiagnosis(
    livePayload({ edge_count: 0, dominant_edge_type: "co_occurs_with", edge_type_counts: {} })
  );
  assert.equal(note, null);
});


// --- folded vs dropped ------------------------------------------------------

test("evidence whose concept is not in the view is dropped, not counted as folded", () => {
  // The promotion_state filter can remove a concept while keeping the evidence
  // that supports it. That evidence contributes to no count anywhere -- it just
  // disappears -- so reporting it as "folded in" would claim its information
  // survived when it did not.
  const nodes = [
    { id: "c1", node_kind: "concept", label: "kept" },
    { id: "e1", node_kind: "evidence" },
    { id: "e2", node_kind: "evidence" },
  ];
  const edges = [
    { source: "e1", target: "c1", predicate: "supports" },
    { source: "e2", target: "c_absent", predicate: "supports" },
  ];
  const out = collapseEvidenceNodes(nodes, edges, true);
  assert.equal(out.collapsedCount, 2, "both evidence nodes left the canvas");
  assert.equal(out.foldedCount, 1, "only one landed on a surviving concept");
  assert.equal(out.droppedCount, 1);
  assert.equal(out.nodes.find((n) => n.id === "c1").evidence_count, 1);
});

test("folded and dropped sum to the number of evidence nodes removed", () => {
  const { nodes, edges } = evidenceFixture();
  const out = collapseEvidenceNodes(nodes, edges, true);
  assert.equal(out.foldedCount + out.droppedCount, out.collapsedCount);
  assert.equal(out.droppedCount, 0, "every evidence node in this fixture has a surviving concept");
  assert.equal(out.foldedCount, 3);
});

test("the disabled and no-evidence paths report the same shape", () => {
  // The status line reads these fields unconditionally; an early return that
  // omits them renders "undefined evidence node(s) folded in".
  const { nodes, edges } = evidenceFixture();
  for (const out of [
    collapseEvidenceNodes(nodes, edges, false),
    collapseEvidenceNodes([{ id: "c1", node_kind: "concept" }], [], true),
  ]) {
    assert.equal(out.foldedCount, 0);
    assert.equal(out.droppedCount, 0);
    assert.equal(out.collapsedCount, 0);
  }
});
