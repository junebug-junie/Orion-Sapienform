"use strict";

/**
 * Concept Atlas — standalone iframe-embedded page (Phase 8 of the concept-graph
 * pipeline design). Structurally mirrors substrate-atlas.js: own IIFE module,
 * Cytoscape.js via CDN, exposes window.OrionConceptAtlas = { activate, ... }
 * for the parent Hub page to ping on tab-show (see app.js's
 * conceptAtlasPanelFrame block, parallel to substrateAtlasPanelFrame).
 *
 * Lifecycle note: unlike Substrate Atlas, this page has no notion of a "live"
 * trace, so there is no setInterval-based auto-refresh to leak in the first
 * place — activate() just does an on-demand fetch of all four cards each
 * time the tab is shown. destroy()/deactivate() exists for symmetry and to
 * abort any in-flight fetches so a rapid tab-away doesn't race a stale
 * response into the DOM after the panel is hidden again.
 */

async function apiFetch(path, opts) {
  const r = await fetch(path, { headers: { Accept: "application/json" }, ...(opts || {}) });
  const text = await r.text();

  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }

  if (!r.ok) {
    const detail =
      payload && payload.detail
        ? payload.detail
        : payload
          ? JSON.stringify(payload)
          : text || r.statusText;
    throw new Error(`${r.status}: ${detail}`);
  }

  return payload || {};
}

// Labels are ALWAYS attached to every node. What varies is the zoom at which
// the renderer is allowed to draw them, via cytoscape's own
// `min-zoomed-font-size`: a label whose on-screen text would render smaller
// than this many pixels is skipped entirely for that frame.
//
// This replaces an all-or-nothing declutter gate (">= 60 nodes and any god
// node exists -> hide every non-god label"). Confirmed live 2026-08-29: at
// 136 nodes that hid 131 of 136 labels, so the whole graph rendered as
// unlabeled dots with five purple exceptions. Hiding 96% of the labels is not
// decluttering, it is blanking -- the same failure the 24-node case was fixed
// for on 2026-08-28, just at a threshold nobody had crossed yet.
//
// A zoom threshold has no cliff: zoomed out you get shape and structure with
// no stacked label soup, and zoomed in on any region every label in it
// appears, at any graph size. Nothing to tune per node count.
const LABEL_MIN_ZOOMED_FONT_PX = 9;

// Edge labels (`supports`, `co_occurs_with`) are far denser than node labels
// -- 461 edges vs 136 nodes in the live graph -- and far less informative per
// pixel, so they need to be zoomed in further before they earn the space.
const EDGE_LABEL_MIN_ZOOMED_FONT_PX = 11;

// God nodes carry the orientation signal, so their labels get a larger font
// and therefore survive to a lower zoom level than everything else -- the
// useful half of what the old gate was reaching for, without discarding the
// other 131 labels to get it.
const GOD_NODE_FONT_PX = 14;
const NODE_FONT_PX = 9;

// Pure, and defined at file scope so it is testable without a DOM or
// cytoscape. `showAll` is the checkbox: 0 disables the threshold entirely, so
// labels render at any zoom no matter how they stack.
function labelMinZoomedFontSize(showAll, godNode) {
  if (showAll) return 0;
  return godNode ? GOD_NODE_FONT_PX : LABEL_MIN_ZOOMED_FONT_PX;
}

function edgeLabelMinZoomedFontSize(showAll) {
  return showAll ? 0 : EDGE_LABEL_MIN_ZOOMED_FONT_PX;
}

// 80 of orion_substrate's 136 nodes are Evidence nodes (verified live
// 2026-08-29). They carry no label of their own -- EvidenceNodeV1 has no
// label field -- so concept_atlas_routes.py::_display_labels names each one
// "Evidence for <the concept it supports>". The result is that 59% of the
// canvas is scaffolding repeating its neighbour's name, which is the single
// biggest reason the picture reads as an unreadable hairball.
//
// Collapsing folds each evidence node into a count on the concept it
// supports, so the same information survives (n pieces of evidence back this
// concept) without n extra dots and n extra edges. On the live graph this
// takes the default view from 136 nodes to 56.
//
// Kept as a pure function next to applyClientPromotionStateFilter rather
// than a cytoscape style rule, because a hidden-but-mounted node still
// participates in the cose layout and would keep pushing real concepts
// apart -- the clutter would move, not go away.
function collapseEvidenceNodes(nodes, edges, collapse) {
  if (!collapse) return { nodes, edges, collapsedCount: 0, foldedCount: 0, droppedCount: 0 };
  const isEvidence = (n) => n.node_kind === "evidence";
  const evidenceIds = new Set(nodes.filter(isEvidence).map((n) => n.id));
  if (!evidenceIds.size) return { nodes, edges, collapsedCount: 0, foldedCount: 0, droppedCount: 0 };

  // `supports` runs evidence -> concept (see the topic_foundry adapter), so
  // the target is the concept that gains the count. Counted from edges
  // rather than from a node field because no such field exists in the
  // payload -- deriving it here keeps the route unchanged.
  const counts = new Map();
  edges.forEach((e) => {
    if (e.predicate !== "supports") return;
    if (!evidenceIds.has(e.source)) return;
    counts.set(e.target, (counts.get(e.target) || 0) + 1);
  });

  const kept = nodes
    .filter((n) => !isEvidence(n))
    .map((n) => (counts.has(n.id) ? Object.assign({}, n, { evidence_count: counts.get(n.id) }) : n));
  const keptIds = new Set(kept.map((n) => n.id));
  const keptEdges = edges.filter((e) => keptIds.has(e.source) && keptIds.has(e.target));

  // FOLDED and DROPPED are different things and the status line must not
  // conflate them. An evidence node whose concept is absent from this view --
  // the promotion_state filter removed it, or it was never in the slice --
  // contributes to no count anywhere; it just disappears. Reporting it as
  // "folded in" would claim its information survived when it did not.
  // `foldedCount` sums the counts actually attached to a surviving concept.
  let foldedCount = 0;
  counts.forEach((n, conceptId) => {
    if (keptIds.has(conceptId)) foldedCount += n;
  });
  return {
    nodes: kept,
    edges: keptEdges,
    collapsedCount: evidenceIds.size,
    foldedCount,
    droppedCount: evidenceIds.size - foldedCount,
  };
}

// synthetic_label (see concept_atlas_routes.py's node_payload):
// topic-foundry's adapter falls back to a bare "topic_<id>" placeholder when
// a clustering run produced neither a real topic label nor keywords --
// non-blank, but not a human label. Rendered with an explicit suffix instead
// of masquerading as a real concept name so it reads as "unlabeled," not
// "broken."
//
// The evidence count is appended rather than replacing anything, so a
// collapsed concept still reads as itself first: "Home lab infrastructure
// (3 evidence)".
function nodeDisplayLabel(n) {
  const base = n.synthetic_label ? `${n.label || n.id} (unlabeled topic)` : n.label || n.id;
  const count = n.evidence_count || 0;
  return count > 0 ? `${base} (${count} evidence)` : base;
}

// --- structural diagnosis ---------------------------------------------------
//
// These two numbers decide only whether to SHOW the interpretive line; the
// line's content is the measured numbers themselves, so nothing here is a
// tuned score. A graph below either cutoff simply gets the raw stats with no
// commentary, which is the honest default -- a hint that fires on every graph
// would be noise, and one that invents a severity band would be a knob
// pretending to be a finding.
const DOMINANT_EDGE_SHARE_HINT = 0.5; // one edge type is more than half of all edges
const SATURATION_HINT = 0.1; // ...and links >10% of every possible pair

// Reads the /structure payload and says what shape the graph is in, in plain
// words, with the measured numbers inline.
//
// The finding it exists to state (live orion_substrate, 2026-08-29): 307 of
// 461 edges are `co_occurs_with`, a same-day co-occurrence proxy. Over 56
// concepts that links 19.9% of every possible pair, and label propagation
// returns exactly one community as a result. No centrality measure or layout
// change fixes that -- the graph is a hairball because its dominant edge
// carries almost no information, which is an upstream producer problem.
function structureDiagnosis(payload) {
  if (!payload || !payload.available) return null;
  const edges = payload.edge_count || 0;
  const dominant = payload.dominant_edge_type;
  const dominantCount = dominant ? (payload.edge_type_counts || {})[dominant] || 0 : 0;
  const share = edges > 0 ? dominantCount / edges : 0;
  const saturation = payload.dominant_edge_saturation;
  if (!dominant || saturation === null || saturation === undefined) return null;
  if (share < DOMINANT_EDGE_SHARE_HINT || saturation < SATURATION_HINT) return null;
  return (
    `${dominantCount} of ${edges} edges (${Math.round(share * 100)}%) are \u2018${dominant}\u2019, ` +
    `linking ${(saturation * 100).toFixed(1)}% of every possible pair among ` +
    `${payload.concept_count} concepts. At that density there is no community ` +
    `structure left to find \u2014 the density is an edge-semantics problem upstream, ` +
    `not a layout problem here.`
  );
}

// "12 components" on its own is not a fact about structure -- on the live
// graph it is one blob of 116, one island of 10, and 10 singletons of retired
// telemetry. Spelling that out is the difference between a number and a read.
function componentShapeLine(payload) {
  if (!payload || !payload.available) return "";
  const total = payload.component_count || 0;
  const largest = payload.largest_component_size || 0;
  const singletons = payload.singleton_count || 0;
  if (!total) return "no components";
  // `largest > 1`, not `> 0`: when every component is a singleton the largest
  // IS a singleton, so counting it separately reported it twice --
  // {components:48, largest:1, singletons:48} rendered "48 components: 1 of 1
  // + 48 singletons", 49 components across the parts plus a meaningless
  // "1 of 1" blob. That is exactly orion_worldview's live shape (48 nodes, 0
  // edges), i.e. the graph this line most needs to describe correctly.
  const hasBlob = largest > 1;
  const middle = total - singletons - (hasBlob ? 1 : 0);
  const parts = [];
  if (hasBlob) parts.push(`1 of ${largest}`);
  if (middle > 0) parts.push(`${middle} smaller`);
  if (singletons > 0) parts.push(`${singletons} singleton${singletons === 1 ? "" : "s"}`);
  return `${total} component${total === 1 ? "" : "s"}: ${parts.join(" + ")}`;
}

// --- coverage and drill-down formatting -------------------------------------
//
// The canvas is a WINDOW on the graph, not the graph. `/network` fetches
// query_concept_region(limit_nodes=300, limit_edges=600); measured live
// 2026-08-30 the binding cap is the EDGE one -- 600 of 1464 edges come back,
// and the 464 entity nodes hang off associated_with edges outside that window,
// so the canvas renders 102 of 671 nodes and zero entities. The payload has
// carried `truncated: true` the whole time; nothing showed it. A view silently
// displaying 15% of the graph is the same class of dishonesty as an unlabeled
// node claiming to be a concept.
function coverageLine(networkPayload, structurePayload) {
  if (!networkPayload || !networkPayload.available) return "";
  const shownNodes = (networkPayload.nodes || []).length;
  const shownEdges = (networkPayload.edges || []).length;
  // Whole-graph totals come from /structure, which is unfiltered by
  // construction. Without it we can still say the view was cut, just not by
  // how much -- which is better than implying the canvas is everything.
  const totalNodes = structurePayload && structurePayload.available ? structurePayload.node_count : null;
  const totalEdges = structurePayload && structurePayload.available ? structurePayload.edge_count : null;
  if (!networkPayload.truncated && !networkPayload.hydration_truncated) return "";
  if (totalNodes && totalEdges) {
    const pct = Math.round((shownNodes / totalNodes) * 100);
    return `view truncated — showing ${shownNodes} of ${totalNodes} nodes (${pct}%) and ${shownEdges} of ${totalEdges} edges`;
  }
  return `view truncated — showing ${shownNodes} node(s), ${shownEdges} edge(s)`;
}

// Labels are not unique: "Hospital" prefix-matches three distinct concepts
// live. The route refuses to guess and returns candidates; this renders them
// rather than silently drilling into whichever came back first.
function candidateHint(payload) {
  if (!payload || payload.available) return "";
  const reason = payload.reason || "";
  const cands = payload.candidates || payload.from_candidates || payload.to_candidates || [];
  if (reason === "node_not_found") return "no node matches that name";
  if (cands.length) {
    return `ambiguous — did you mean: ${cands.map((c) => c.label || c.node_id).join(" · ")}`;
  }
  if (reason === "node_ambiguous" || reason === "endpoint_not_resolved") return "ambiguous name";
  if (reason === "invalid_rel_types") return "invalid predicate filter";
  return reason || "unavailable";
}

// "A -> B -> C". Empty means no path WITHIN the searched depth, which is not
// the same as disconnected -- the caller is told which.
function formatPath(payload) {
  if (!payload || !payload.available) return "";
  const hops = payload.hops || [];
  if (!hops.length) {
    return `no path within ${payload.searched_to_depth} hop(s) — they may still be connected further out`;
  }
  return hops.map((h) => h.label || h.node_id).join("  →  ");
}

if (typeof document !== "undefined") {
(function () {
  const STATUS = document.getElementById("caStatus");
  const REFRESH_BTN = document.getElementById("caRefreshBtn");

  const FILTER_SCOPE = document.getElementById("caFilterScope");
  const FILTER_PROMOTION_STATE = document.getElementById("caFilterPromotionState");
  const FILTER_FOCUS = document.getElementById("caFilterFocus");
  const APPLY_FILTERS_BTN = document.getElementById("caApplyFiltersBtn");
  const CLEAR_FILTERS_BTN = document.getElementById("caClearFiltersBtn");

  const SUMMARY_STATUS = document.getElementById("caSummaryStatus");
  const SUMMARY_STATS = document.getElementById("caSummaryStats");
  const PROMOTION_STATE_BREAKDOWN = document.getElementById("caPromotionStateBreakdown");
  const ANCHOR_SCOPE_BREAKDOWN = document.getElementById("caAnchorScopeBreakdown");
  const PREDICATE_BREAKDOWN = document.getElementById("caPredicateBreakdown");
  const AT_RISK_LIST = document.getElementById("caAtRiskList");

  const NETWORK_STATUS = document.getElementById("caNetworkStatus");
  const NETWORK_CY_HOST = document.getElementById("caNetworkCy");
  const NETWORK_INSPECTOR = document.getElementById("caNetworkInspector");
  const SHOW_ALL_LABELS = document.getElementById("caShowAllLabels");
  const COLLAPSE_EVIDENCE = document.getElementById("caCollapseEvidence");

  const STRUCTURE_STATUS = document.getElementById("caStructureStatus");
  const STRUCTURE_STATS = document.getElementById("caStructureStats");
  const STRUCTURE_COMPONENTS = document.getElementById("caStructureComponents");
  const STRUCTURE_INFLUENCE = document.getElementById("caStructureInfluence");
  const STRUCTURE_BRIDGES = document.getElementById("caStructureBridges");
  const STRUCTURE_NOTE = document.getElementById("caStructureNote");
  const STRUCTURE_COMMUNITIES = document.getElementById("caStructureCommunities");

  const NETWORK_COVERAGE = document.getElementById("caNetworkCoverage");
  const DRILL_STATUS = document.getElementById("caDrilldownStatus");
  const NEIGHBOR_NODE = document.getElementById("caNeighborNode");
  const NEIGHBOR_DEPTH = document.getElementById("caNeighborDepth");
  const NEIGHBOR_BTN = document.getElementById("caNeighborBtn");
  const NEIGHBOR_RESULT = document.getElementById("caNeighborResult");
  const PATH_FROM = document.getElementById("caPathFrom");
  const PATH_TO = document.getElementById("caPathTo");
  const PATH_BTN = document.getElementById("caPathBtn");
  const PATH_RESULT = document.getElementById("caPathResult");

  const CLUSTERING_BODY = document.getElementById("caClusteringBody");

  const PREDICATE_COLORS = {
    contradicts: "#ef4444",
    co_occurs_with: "#64748b",
    supports: "#22c55e",
    refines: "#0ea5e9",
  };
  const DEFAULT_EDGE_COLOR = "#22c55e";

  let cy = null;
  let activated = false;
  let fetchGeneration = 0;
  // Default label visibility to god-nodes-only (readability gap named in the
  // 2026-08-18 design spec) -- at 300 nodes, unconditional labels stack
  // illegibly even with componentSpacing/nodeDimensionsIncludeLabels. God
  // nodes are the top-degree handful, so keeping their labels on by default
  // gives orientation without the clutter; the checkbox opts back into the
  // old always-on behavior for anyone who wants to read every label.
  let showAllLabels = false;
  // Default ON: the uncollapsed view is 136 nodes of which 80 say nothing
  // but "Evidence for <neighbour>". Opting IN to that is the right default,
  // not opting out of it.
  let collapseEvidence = true;
  // Last /network payload, so the evidence toggle re-renders instead of refetching.
  let lastNetworkPayload = null;
  let lastStructurePayload = null;
  let lastPromotionState = "";

  // Deterministic hash -> hue so a given topic_id always renders the same
  // color across refreshes/filters, without needing a server-assigned color
  // table. Not cryptographic -- collisions between distinct topic_ids are
  // possible but cosmetic (two clusters sharing a hue), not a correctness
  // issue for a read-only interpretability view.
  function topicColor(topicId) {
    if (topicId === null || topicId === undefined || topicId === "") return null;
    const s = String(topicId);
    let hash = 0;
    for (let i = 0; i < s.length; i += 1) {
      hash = (hash * 31 + s.charCodeAt(i)) >>> 0;
    }
    return `hsl(${hash % 360}, 60%, 55%)`;
  }

  function setStatus(msg, isErr) {
    if (!STATUS) return;
    STATUS.textContent = msg;
    STATUS.classList.toggle("text-red-400", !!isErr);
    STATUS.classList.toggle("text-gray-400", !isErr);
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function currentFilters() {
    return {
      scope: (FILTER_SCOPE && FILTER_SCOPE.value) || "",
      promotionState: (FILTER_PROMOTION_STATE && FILTER_PROMOTION_STATE.value) || "",
      focus: (FILTER_FOCUS && FILTER_FOCUS.value.trim()) || "",
    };
  }

  // --- Summary card -------------------------------------------------------

  function statTile(label, value) {
    const div = document.createElement("div");
    div.className = "rounded-lg border border-gray-800 bg-gray-900/60 px-3 py-2";
    div.innerHTML = `<div class="text-[10px] uppercase tracking-wide text-gray-500">${escapeHtml(label)}</div><div class="text-lg font-semibold text-white">${escapeHtml(value)}</div>`;
    return div;
  }

  function renderBreakdown(host, counts, activeKey) {
    if (!host) return;
    host.innerHTML = "";
    const entries = Object.entries(counts || {});
    if (!entries.length) {
      host.innerHTML = '<p class="text-gray-600">none</p>';
      return;
    }
    entries.forEach(([key, count]) => {
      const row = document.createElement("div");
      const isActive = activeKey && key === activeKey;
      row.className = "flex items-center justify-between " + (isActive ? "text-indigo-300 font-semibold" : "text-gray-300");
      row.innerHTML = `<span>${escapeHtml(key)}</span><span>${escapeHtml(count)}</span>`;
      host.appendChild(row);
    });
  }

  function renderAtRisk(payload) {
    if (!AT_RISK_LIST) return;
    AT_RISK_LIST.innerHTML = "";
    const rows = payload.at_risk || [];
    if (!rows.length) {
      const note = payload.at_risk_note || "no at-risk concepts";
      AT_RISK_LIST.innerHTML = `<p class="text-gray-600">${escapeHtml(note)}</p>`;
      return;
    }
    rows.forEach((row) => {
      const div = document.createElement("div");
      div.className = "flex items-center justify-between text-gray-300";
      div.innerHTML = `<span>${escapeHtml(row.label || row.node_id)}</span><span class="text-amber-400">activation ${Number(row.activation).toFixed(2)} (floor ${Number(row.decay_floor).toFixed(2)})</span>`;
      AT_RISK_LIST.appendChild(div);
    });
  }

  async function fetchSummary() {
    if (SUMMARY_STATUS) SUMMARY_STATUS.textContent = "Loading…";
    try {
      const payload = await apiFetch("/api/substrate/concepts/summary");
      if (!payload.available) {
        if (SUMMARY_STATUS) SUMMARY_STATUS.textContent = `unavailable (${payload.reason || "unknown"})`;
        if (SUMMARY_STATS) SUMMARY_STATS.innerHTML = "";
        renderBreakdown(PROMOTION_STATE_BREAKDOWN, {});
        renderBreakdown(ANCHOR_SCOPE_BREAKDOWN, {});
        renderBreakdown(PREDICATE_BREAKDOWN, {});
        renderAtRisk({ at_risk: [], at_risk_note: payload.reason ? `unavailable: ${payload.reason}` : null });
        return;
      }
      if (SUMMARY_STATS) {
        SUMMARY_STATS.innerHTML = "";
        SUMMARY_STATS.appendChild(statTile("Total concepts", payload.total_concepts || 0));
        SUMMARY_STATS.appendChild(statTile("Promotion states", Object.keys(payload.by_promotion_state || {}).length));
        SUMMARY_STATS.appendChild(statTile("Anchor scopes", Object.keys(payload.by_anchor_scope || {}).length));
        SUMMARY_STATS.appendChild(statTile("At risk", (payload.at_risk || []).length));
      }
      const { scope, promotionState } = currentFilters();
      renderBreakdown(PROMOTION_STATE_BREAKDOWN, payload.by_promotion_state || {}, promotionState || null);
      renderBreakdown(ANCHOR_SCOPE_BREAKDOWN, payload.by_anchor_scope || {}, scope || null);
      renderBreakdown(PREDICATE_BREAKDOWN, payload.edge_counts_by_predicate || {});
      renderAtRisk(payload);
      if (SUMMARY_STATUS) SUMMARY_STATUS.textContent = "";
    } catch (e) {
      if (SUMMARY_STATUS) SUMMARY_STATUS.textContent = `error: ${e.message || e}`;
    }
  }

  // --- Network card --------------------------------------------------------

  function destroyCy() {
    if (cy) {
      try {
        cy.destroy();
      } catch {
        /* ignore */
      }
      cy = null;
    }
    if (NETWORK_CY_HOST) NETWORK_CY_HOST.textContent = "";
  }

  function edgeColor(predicate) {
    return PREDICATE_COLORS[predicate] || DEFAULT_EDGE_COLOR;
  }

  function applyClientPromotionStateFilter(nodes, edges, promotionState) {
    if (!promotionState) return { nodes, edges };
    const kept = nodes.filter((n) => n.promotion_state === promotionState);
    const keptIds = new Set(kept.map((n) => n.id));
    const keptEdges = edges.filter((e) => keptIds.has(e.source) && keptIds.has(e.target));
    return { nodes: kept, edges: keptEdges };
  }

  function graphToElements(nodes, edges) {
    const cyNodes = nodes.map((n) => ({
      data: {
        id: n.id,
        label: nodeDisplayLabel(n),
        nodeKind: n.node_kind,
        anchorScope: n.anchor_scope,
        promotionState: n.promotion_state,
        activation: n.activation,
        salience: n.salience,
        confidence: n.confidence,
        degree: n.degree,
        godNode: !!n.god_node,
        componentId: n.component_id,
        topicId: n.topic_id === undefined ? null : n.topic_id,
        origin: n.origin || "concept",
        syntheticLabel: !!n.synthetic_label,
        evidenceCount: n.evidence_count || 0,
      },
    }));
    const cyEdges = edges.map((e) => ({
      data: {
        id: e.id,
        source: e.source,
        target: e.target,
        label: e.predicate,
        predicate: e.predicate,
      },
    }));
    return [...cyNodes, ...cyEdges];
  }

  function renderInspector(nodeData) {
    if (!NETWORK_INSPECTOR) return;
    if (!nodeData) {
      NETWORK_INSPECTOR.innerHTML = '<p class="text-gray-500">Select a node to inspect fields.</p>';
      return;
    }
    const fields = [
      ["id", nodeData.id],
      ["label", nodeData.label],
      ["node_kind", nodeData.nodeKind],
      ["evidence", nodeData.evidenceCount ? `${nodeData.evidenceCount} node(s) folded in` : "—"],
      ["anchor_scope", nodeData.anchorScope],
      ["promotion_state", nodeData.promotionState],
      ["activation", nodeData.activation],
      ["salience", nodeData.salience],
      ["confidence", nodeData.confidence],
      ["degree", nodeData.degree],
      ["god_node", nodeData.godNode],
      ["component_id", nodeData.componentId],
      ["topic_id", nodeData.topicId],
      ["origin", nodeData.origin],
      ["synthetic_label", nodeData.syntheticLabel],
    ];
    let html = '<dl class="grid grid-cols-2 gap-x-3 gap-y-1">';
    fields.forEach(([k, v]) => {
      html += `<dt class="text-gray-500">${escapeHtml(k)}</dt><dd class="text-gray-200 break-all">${escapeHtml(v)}</dd>`;
    });
    html += "</dl>";
    NETWORK_INSPECTOR.innerHTML = html;
  }

  function mountCytoscape(elements) {
    destroyCy();
    if (!NETWORK_CY_HOST || typeof window.cytoscape !== "function") {
      if (NETWORK_CY_HOST) NETWORK_CY_HOST.textContent = "Cytoscape failed to load.";
      return;
    }
    NETWORK_CY_HOST.textContent = "";
    if (!elements.length) {
      NETWORK_CY_HOST.textContent = "No concept nodes match the current filters.";
      return;
    }
    cy = window.cytoscape({
      container: NETWORK_CY_HOST,
      elements,
      style: [
        {
          selector: "node",
          style: {
            // Every node always carries its label. Whether it is DRAWN is
            // decided per frame by min-zoomed-font-size below, so there is no
            // graph size at which a node becomes a permanently anonymous dot.
            // The checkbox flips the threshold without a remount, since
            // cy.style().update() re-evaluates mapper functions in place.
            label: (ele) => ele.data("label"),
            "font-size": (ele) => (ele.data("godNode") ? GOD_NODE_FONT_PX : NODE_FONT_PX),
            "min-zoomed-font-size": (ele) =>
              labelMinZoomedFontSize(showAllLabels, ele.data("godNode")),
            "text-background-color": "#0b1220",
            "text-background-opacity": 0.72,
            "text-background-padding": 2,
            "text-background-shape": "roundrectangle",
            color: "#e2e8f0",
            "text-valign": "bottom",
            "text-margin-y": 4,
            // God-node purple stays the priority signal (top-degree is the
            // rarer, more load-bearing fact); muted slate marks a node whose
            // only available label is topic-foundry's synthetic "topic_<id>"
            // fallback -- deliberately never the same color as a real named
            // concept, so it can't be mistaken for one at a glance; community
            // coloring from topic-foundry's HDBSCAN cluster id (when the node
            // carries one) fills in for everyone else, default blue when none
            // of the above applies.
            "background-color": (ele) => {
              if (ele.data("godNode")) return "#a855f7";
              if (ele.data("syntheticLabel")) return "#64748b";
              return topicColor(ele.data("topicId")) || "#0ea5e9";
            },
            "border-style": (ele) => (ele.data("syntheticLabel") ? "dashed" : "solid"),
            width: (ele) => (ele.data("godNode") ? 42 : 24),
            height: (ele) => (ele.data("godNode") ? 42 : 24),
            "border-width": 2,
            "border-color": "#1e293b",
          },
        },
        {
          selector: "node:selected",
          style: { "border-color": "#818cf8", "border-width": 3 },
        },
        {
          selector: "edge",
          style: {
            width: 1.5,
            "line-color": (ele) => edgeColor(ele.data("predicate")),
            "target-arrow-color": (ele) => edgeColor(ele.data("predicate")),
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            label: "data(label)",
            "font-size": 9,
            "min-zoomed-font-size": () => edgeLabelMinZoomedFontSize(showAllLabels),
            color: "#94a3b8",
          },
        },
      ],
      layout: {
        name: "cose",
        animate: false,
        padding: 24,
        // Without this, cose's collision physics only avoid overlapping
        // node circles, not their labels -- with font-size 9 labels
        // rendered below/beside each node, that let unrelated nodes'
        // labels stack directly on top of each other once the graph got
        // dense (see the 2026-08-18 design spec's screenshot). Node size
        // is now (radius + label bounding box), so cose's normal
        // node-overlap avoidance keeps labels apart too.
        nodeDimensionsIncludeLabels: true,
        // Denser graphs need more room between disconnected clusters than
        // cose's default, or unrelated components visually collide into
        // one blob -- see the same spec's "connected components" gap this
        // doesn't fix on its own, just makes less bad in the interim.
        componentSpacing: 80,
      },
      wheelSensitivity: 0.3,
    });
    cy.on("tap", "node", (evt) => {
      const data = evt.target.data();
      renderInspector(data);
      // Seed the drill-down with the node id, not the rendered label: the
      // label may carry an appended "(N evidence)" or "(unlabeled topic)"
      // suffix, and an evidence node's label is borrowed from the concept it
      // supports -- either would resolve to the wrong node or to nothing.
      // The id is exact, and the route matches it first.
      if (NEIGHBOR_NODE) NEIGHBOR_NODE.value = data.id;
      if (PATH_FROM && !PATH_FROM.value.trim()) PATH_FROM.value = data.id;
      else if (PATH_TO) PATH_TO.value = data.id;
    });
    try {
      cy.resize();
      cy.fit(undefined, 32);
    } catch {
      /* ignore */
    }
  }

  async function fetchNetwork() {
    if (NETWORK_STATUS) NETWORK_STATUS.textContent = "Loading…";
    const { scope, promotionState, focus } = currentFilters();
    const params = new URLSearchParams();
    if (scope) params.set("scope", scope);
    if (focus) params.set("focus", focus);
    try {
      const payload = await apiFetch(`/api/substrate/concepts/network?${params.toString()}`);
      if (!payload.available) {
        destroyCy();
        if (NETWORK_CY_HOST) NETWORK_CY_HOST.textContent = `Network unavailable (${payload.reason || "unknown"}).`;
        if (NETWORK_STATUS) NETWORK_STATUS.textContent = payload.reason || "unavailable";
        return;
      }
      lastNetworkPayload = payload;
      lastPromotionState = promotionState;
      renderNetworkPayload(payload, promotionState);
    } catch (e) {
      destroyCy();
      if (NETWORK_CY_HOST) NETWORK_CY_HOST.textContent = `Network error: ${e.message || e}`;
      if (NETWORK_STATUS) NETWORK_STATUS.textContent = "error";
    }
  }

  // Split out of fetchNetwork so the evidence toggle can re-render the graph
  // it already has. Collapsing is a pure transform of the /network payload, so
  // re-fetching for it cost four endpoint round trips and three whole-graph
  // centrality runs (/structure re-runs pageRank, betweenness and harmonic) to
  // tick a checkbox.
  function renderNetworkPayload(payload, promotionState) {
      const promotionFiltered = applyClientPromotionStateFilter(payload.nodes || [], payload.edges || [], promotionState);
      const collapsed = collapseEvidenceNodes(promotionFiltered.nodes, promotionFiltered.edges, collapseEvidence);
      const filtered = { nodes: collapsed.nodes, edges: collapsed.edges };
      mountCytoscape(graphToElements(filtered.nodes, filtered.edges));
      renderInspector(null);
      if (NETWORK_STATUS) {
        // component_id is assigned server-side against the pre-client-filter
        // graph, so counting distinct ids among the still-shown nodes is an
        // approximation after a promotion_state filter removes interior
        // nodes (a component could technically fragment further) -- treated
        // as good enough for an informational status line, not re-derived
        // client-side with a second union-find pass.
        // Filter out missing ids (e.g. a stale cached bundle/backend response
        // predating this field) instead of letting a Set of all-undefined
        // collapse to size 1 -- a specific, confidently wrong number is
        // worse than an honest 0 here.
        const shownComponents = new Set(
          filtered.nodes.map((n) => n.component_id).filter((id) => id !== undefined && id !== null)
        );
        // Naming the collapsed count rather than silently showing a smaller
        // graph: a node count that shrank with no explanation reads as data
        // loss, which is exactly the complaint this view already earns.
        const collapsedNote = collapsed.collapsedCount
          ? `, ${collapsed.foldedCount} evidence node(s) folded in` +
            (collapsed.droppedCount
              ? `, ${collapsed.droppedCount} dropped (supported concept not in view)`
              : "")
          : "";
        const base = `${filtered.nodes.length} node(s), ${filtered.edges.length} edge(s), ${payload.god_node_count || 0} god node(s), ${shownComponents.size} component(s)${collapsedNote}`;
        // Surfaced when a non-default store backend (e.g. graphdb) fell back
        // to a stale snapshot after an upstream query failure -- see
        // concept_atlas_routes.py's "degraded" comment. The default
        // in-memory store never sets this.
        NETWORK_STATUS.textContent = payload.degraded ? `${base} — DEGRADED: ${payload.degraded_error || "stale data"}` : base;
        NETWORK_STATUS.classList.toggle("text-amber-400", !!payload.degraded);
      }
      if (NETWORK_COVERAGE) {
        NETWORK_COVERAGE.textContent = coverageLine(payload, lastStructurePayload);
      }
  }


  // --- Graph structure card (whole-graph, engine-computed) -----------------

  function renderRankedList(host, rows, emptyText) {
    if (!host) return;
    host.innerHTML = "";
    if (!rows || !rows.length) {
      host.innerHTML = `<p class="text-gray-600">${escapeHtml(emptyText)}</p>`;
      return;
    }
    rows.slice(0, 6).forEach((r) => {
      const row = document.createElement("div");
      row.className = "flex items-center justify-between gap-2";
      // A node with no label of its own falls back to its id rather than
      // rendering blank -- an empty row reads as a bug, not as a nameless node.
      row.innerHTML =
        `<span class="truncate">${escapeHtml(r.label || r.node_id || "(unnamed)")}</span>` +
        `<span class="text-gray-500 shrink-0">${escapeHtml(Number(r.score).toFixed(3))}</span>`;
      host.appendChild(row);
    });
  }

  async function fetchStructure() {
    if (!STRUCTURE_STATS) return;
    try {
      const payload = await apiFetch("/api/substrate/concepts/structure");
      if (!payload.available) {
        if (STRUCTURE_STATUS) STRUCTURE_STATUS.textContent = payload.reason || "unavailable";
        STRUCTURE_STATS.innerHTML = "";
        if (STRUCTURE_NOTE) STRUCTURE_NOTE.textContent = "";
        return;
      }
      if (STRUCTURE_STATUS) STRUCTURE_STATUS.textContent = payload.graph || "";

      STRUCTURE_STATS.innerHTML = "";
      STRUCTURE_STATS.appendChild(statTile("nodes", payload.node_count));
      STRUCTURE_STATS.appendChild(statTile("concepts", payload.concept_count));
      STRUCTURE_STATS.appendChild(statTile("edges", payload.edge_count));
      STRUCTURE_STATS.appendChild(
        statTile(
          "pair saturation",
          payload.dominant_edge_saturation === null || payload.dominant_edge_saturation === undefined
            ? "—"
            : `${(payload.dominant_edge_saturation * 100).toFixed(1)}%`
        )
      );

      if (STRUCTURE_COMPONENTS) {
        STRUCTURE_COMPONENTS.innerHTML = "";
        const shape = document.createElement("div");
        shape.className = "text-gray-300 mb-1";
        shape.textContent = componentShapeLine(payload);
        STRUCTURE_COMPONENTS.appendChild(shape);
        // Singletons are named, not just counted: on the live graph all ten are
        // retired telemetry nodes, which is only actionable if you can see
        // which ones they are.
        (payload.components || [])
          .filter((c) => c.is_singleton)
          .slice(0, 12)
          .forEach((c) => {
            const row = document.createElement("div");
            row.className = "text-gray-500 truncate";
            row.textContent = `· ${(c.sample_labels || [])[0] || c.component_id}`;
            STRUCTURE_COMPONENTS.appendChild(row);
          });
      }

      lastStructurePayload = payload;
      if (STRUCTURE_COMMUNITIES) {
        STRUCTURE_COMMUNITIES.innerHTML = "";
        const comms = payload.communities || [];
        if (!comms.length) {
          STRUCTURE_COMMUNITIES.innerHTML = '<p class="text-gray-600">none</p>';
        } else {
          comms.slice(0, 8).forEach((c) => {
            const row = document.createElement("div");
            row.className = "truncate " + (c.size <= 3 ? "text-amber-300/80" : "text-gray-300");
            // Small communities are the interesting ones: live 2026-08-30 they
            // were near-duplicate concepts ("Rest and support" / "Rest and
            // recovery"), so they get the highlight rather than the giant blob.
            row.textContent = `${c.size}: ${(c.sample_labels || []).slice(0, 3).join(", ")}`;
            row.title = (c.sample_labels || []).join(", ");
            STRUCTURE_COMMUNITIES.appendChild(row);
          });
        }
      }
      renderRankedList(STRUCTURE_INFLUENCE, (payload.rankings || {}).pagerank, "none");
      renderRankedList(STRUCTURE_BRIDGES, payload.bridges, "no bridges distinct from the influence ranking");

      if (STRUCTURE_NOTE) {
        const note = structureDiagnosis(payload);
        STRUCTURE_NOTE.textContent = note || "";
        STRUCTURE_NOTE.className = note ? "text-[11px] leading-relaxed text-amber-300/90" : "text-[11px]";
      }
    } catch (e) {
      if (STRUCTURE_STATUS) STRUCTURE_STATUS.textContent = "error";
      if (STRUCTURE_NOTE) STRUCTURE_NOTE.textContent = `Structure error: ${e.message || e}`;
    }
  }


  // --- Drill-down: engine-side, whole-graph ---------------------------------

  function setDrillStatus(text, isErr) {
    if (!DRILL_STATUS) return;
    DRILL_STATUS.textContent = text || "";
    DRILL_STATUS.classList.toggle("text-red-400", !!isErr);
  }

  async function runNeighborhood() {
    if (!NEIGHBOR_RESULT) return;
    const node = (NEIGHBOR_NODE && NEIGHBOR_NODE.value || "").trim();
    if (!node) {
      NEIGHBOR_RESULT.innerHTML = '<p class="text-gray-600">enter a node name</p>';
      return;
    }
    const depth = (NEIGHBOR_DEPTH && NEIGHBOR_DEPTH.value) || "1";
    NEIGHBOR_RESULT.innerHTML = '<p class="text-gray-500">querying…</p>';
    setDrillStatus("");
    try {
      const params = new URLSearchParams({ node, depth });
      const payload = await apiFetch(`/api/substrate/concepts/neighborhood?${params.toString()}`);
      if (!payload.available) {
        NEIGHBOR_RESULT.innerHTML = `<p class="text-amber-400">${escapeHtml(candidateHint(payload))}</p>`;
        return;
      }
      NEIGHBOR_RESULT.innerHTML = "";
      const head = document.createElement("div");
      head.className = "text-gray-500 mb-1";
      head.textContent =
        `${payload.nodes.length} node(s) within ${payload.depth} hop(s)` +
        (payload.truncated ? " (truncated at the limit)" : "");
      NEIGHBOR_RESULT.appendChild(head);
      payload.nodes.forEach((n) => {
        const row = document.createElement("div");
        row.className = "flex items-center justify-between gap-2";
        row.innerHTML =
          `<span class="truncate">${escapeHtml(n.label || n.node_id)}</span>` +
          `<span class="text-gray-500 shrink-0">${escapeHtml(n.hops)} hop</span>`;
        NEIGHBOR_RESULT.appendChild(row);
      });
    } catch (e) {
      NEIGHBOR_RESULT.innerHTML = "";
      setDrillStatus(`error: ${e.message || e}`, true);
    }
  }

  async function runPath() {
    if (!PATH_RESULT) return;
    const from = (PATH_FROM && PATH_FROM.value || "").trim();
    const to = (PATH_TO && PATH_TO.value || "").trim();
    if (!from || !to) {
      PATH_RESULT.innerHTML = '<p class="text-gray-600">enter both endpoints</p>';
      return;
    }
    PATH_RESULT.innerHTML = '<p class="text-gray-500">searching…</p>';
    setDrillStatus("");
    try {
      const params = new URLSearchParams({ from, to });
      const payload = await apiFetch(`/api/substrate/concepts/path?${params.toString()}`);
      if (!payload.available) {
        PATH_RESULT.innerHTML = `<p class="text-amber-400">${escapeHtml(candidateHint(payload))}</p>`;
        return;
      }
      const text = formatPath(payload);
      PATH_RESULT.innerHTML = `<p class="${payload.found ? "text-gray-200" : "text-amber-400"}">${escapeHtml(text)}</p>`;
    } catch (e) {
      PATH_RESULT.innerHTML = "";
      setDrillStatus(`error: ${e.message || e}`, true);
    }
  }

  // --- Clustering card (read-only topic-foundry summary) -------------------

  function renderClusteringPlaceholder(message) {
    if (!CLUSTERING_BODY) return;
    CLUSTERING_BODY.innerHTML = `<p class="text-gray-500">${escapeHtml(message)}</p>`;
  }

  async function fetchClustering() {
    if (!CLUSTERING_BODY) return;
    CLUSTERING_BODY.innerHTML = '<p class="text-gray-500">Loading latest topic-foundry run…</p>';
    try {
      const payload = await apiFetch("/api/topic-foundry/runs");
      const runs = Array.isArray(payload) ? payload : payload.items || payload.runs || [];
      if (!runs.length) {
        renderClusteringPlaceholder("not yet connected — no topic-foundry runs found");
        return;
      }
      const latest = runs[0];
      const fields = [
        ["run_id", latest.run_id || latest.id],
        ["status", latest.status || latest.state],
        ["dataset", latest.dataset_id || latest.dataset],
        ["created_at", latest.created_at || latest.started_at],
      ];
      let html = '<dl class="grid grid-cols-2 gap-x-3 gap-y-1">';
      fields.forEach(([k, v]) => {
        if (v == null) return;
        html += `<dt class="text-gray-500">${escapeHtml(k)}</dt><dd class="text-gray-200 break-all">${escapeHtml(v)}</dd>`;
      });
      html += "</dl>";
      CLUSTERING_BODY.innerHTML = html;
    } catch (e) {
      renderClusteringPlaceholder(`not yet connected — ${e.message || e}`);
    }
  }

  // --- Orchestration ---------------------------------------------------------

  async function refreshAll() {
    const myGeneration = ++fetchGeneration;
    setStatus("Loading…");
    // Independent try/catch per card so one failing endpoint never blanks
    // the others (Phase 8 acceptance check).
    const results = await Promise.allSettled([fetchSummary(), fetchStructure(), fetchNetwork(), fetchClustering()]);
    if (myGeneration !== fetchGeneration) return; // superseded by a newer refresh
    const failed = results.filter((r) => r.status === "rejected").length;
    setStatus(failed ? `Loaded with ${failed} card error(s)` : "Loaded", failed > 0);
  }

  function activate() {
    activated = true;
    refreshAll();
  }

  function deactivate() {
    // No recurring timers exist today (see module header), so this only
    // bumps the fetch generation so any in-flight refreshAll() from before
    // the tab was hidden no-ops instead of writing into a hidden DOM.
    fetchGeneration += 1;
  }

  function init() {
    if (REFRESH_BTN) REFRESH_BTN.addEventListener("click", refreshAll);
    if (APPLY_FILTERS_BTN) APPLY_FILTERS_BTN.addEventListener("click", refreshAll);
    if (SHOW_ALL_LABELS) {
      SHOW_ALL_LABELS.addEventListener("change", () => {
        showAllLabels = !!SHOW_ALL_LABELS.checked;
        // Mapper-function style values (label above) don't auto-recompute on
        // plain data changes -- style().update() forces re-evaluation
        // without destroying/remounting the whole graph.
        if (cy) cy.style().update();
      });
    }
    if (NEIGHBOR_BTN) NEIGHBOR_BTN.addEventListener("click", runNeighborhood);
    if (PATH_BTN) PATH_BTN.addEventListener("click", runPath);
    [NEIGHBOR_NODE, PATH_FROM, PATH_TO].forEach((el) => {
      if (!el) return;
      el.addEventListener("keydown", (ev) => {
        if (ev.key !== "Enter") return;
        if (el === NEIGHBOR_NODE) runNeighborhood();
        else runPath();
      });
    });
    if (COLLAPSE_EVIDENCE) {
      COLLAPSE_EVIDENCE.checked = collapseEvidence;
      COLLAPSE_EVIDENCE.addEventListener("change", () => {
        collapseEvidence = !!COLLAPSE_EVIDENCE.checked;
        // Unlike the label toggle above, this changes the node SET, so a
        // style().update() is not enough -- the graph has to be rebuilt and
        // the layout re-run over the new element list. It does NOT need new
        // data: collapsing is a pure transform of the payload already held.
        if (lastNetworkPayload) {
          renderNetworkPayload(lastNetworkPayload, lastPromotionState);
        } else {
          refreshAll();
        }
      });
    }
    if (CLEAR_FILTERS_BTN) {
      CLEAR_FILTERS_BTN.addEventListener("click", () => {
        if (FILTER_SCOPE) FILTER_SCOPE.value = "";
        if (FILTER_PROMOTION_STATE) FILTER_PROMOTION_STATE.value = "";
        if (FILTER_FOCUS) FILTER_FOCUS.value = "";
        refreshAll();
      });
    }
    window.addEventListener("beforeunload", deactivate);
    window.OrionConceptAtlas = { activate, deactivate, destroy: deactivate, refresh: refreshAll };
    if (!activated) {
      // Load once eagerly too, in case the page is opened standalone
      // (not just via the Hub iframe, which pings activate() itself).
      activate();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
}

if (typeof module !== "undefined" && module.exports) {
  // All three are plain file-scope declarations above the `typeof document`
  // guard, so they exist whether or not the browser IIFE ran -- which is what
  // makes concept-atlas.test.js able to import them under node with no DOM.
  module.exports = {
    apiFetch,
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
    coverageLine,
    candidateHint,
    formatPath,
    DOMINANT_EDGE_SHARE_HINT,
    SATURATION_HINT,
  };
}
