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

// Below this many nodes every label fits, so show them all.
//
// The declutter used to be gated on whether ANY god node existed, which is
// always true -- canonical seed concepts are god nodes unconditionally,
// regardless of degree (see concept_atlas_routes.py's canonical_ids
// handling). Confirmed live 2026-08-28: a 24-node graph rendered 19 of its
// nodes as unlabeled dots, which is pure loss rather than decluttering.
const LABEL_DECLUTTER_MIN_NODES = 60;

// Pure, and defined at file scope so it is testable without a DOM or
// cytoscape. Both conditions matter: below the threshold there is no clutter
// to trade a label away for, and with no god node there is nothing left to
// orient on once everything else is hidden. A non-numeric count degrades to
// showing labels rather than hiding them.
function shouldDeclutterLabels(nodeCount, hasGodNodes) {
  return Number(nodeCount) >= LABEL_DECLUTTER_MIN_NODES && Boolean(hasGodNodes);
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
        // synthetic_label (see concept_atlas_routes.py's node_payload):
        // topic-foundry's adapter falls back to a bare "topic_<id>"
        // placeholder when a clustering run produced neither a real topic
        // label nor keywords -- non-blank, but not a human label. Rendered
        // with an explicit suffix instead of masquerading as a real concept
        // name so it reads as "unlabeled," not "broken."
        label: n.synthetic_label ? `${n.label || n.id} (unlabeled topic)` : n.label || n.id,
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
    // God-nodes-only labels are meant to declutter a dense, mostly-default
    // view -- they're useless as a default on a filtered/sparse subgraph
    // that happens to have zero god nodes (e.g. every node degree-0 after an
    // anchor_scope filter), where hiding every label leaves unlabeled dots
    // with nothing to orient on. Fall back to showing everything in that
    // case; the checkbox still overrides either way.
    const nodeCount = elements.filter((el) => el.data && !el.data.source).length;
    const hasGodNodes = elements.some((el) => el.data && el.data.godNode);
    const declutterLabels = shouldDeclutterLabels(nodeCount, hasGodNodes);
    cy = window.cytoscape({
      container: NETWORK_CY_HOST,
      elements,
      style: [
        {
          selector: "node",
          style: {
            // God-nodes-only once the graph is dense enough to need it (see
            // LABEL_DECLUTTER_MIN_NODES above); the checkbox flips this
            // without needing a full remount, since cy.style().update()
            // re-evaluates mapper functions in place.
            label: (ele) =>
              showAllLabels || !declutterLabels || ele.data("godNode") ? ele.data("label") : "",
            "font-size": 9,
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
            "font-size": 7,
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
      renderInspector(evt.target.data());
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
      const filtered = applyClientPromotionStateFilter(payload.nodes || [], payload.edges || [], promotionState);
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
        const base = `${filtered.nodes.length} node(s), ${filtered.edges.length} edge(s), ${payload.god_node_count || 0} god node(s), ${shownComponents.size} component(s)`;
        // Surfaced when a non-default store backend (e.g. graphdb) fell back
        // to a stale snapshot after an upstream query failure -- see
        // concept_atlas_routes.py's "degraded" comment. The default
        // in-memory store never sets this.
        NETWORK_STATUS.textContent = payload.degraded ? `${base} — DEGRADED: ${payload.degraded_error || "stale data"}` : base;
        NETWORK_STATUS.classList.toggle("text-amber-400", !!payload.degraded);
      }
    } catch (e) {
      destroyCy();
      if (NETWORK_CY_HOST) NETWORK_CY_HOST.textContent = `Network error: ${e.message || e}`;
      if (NETWORK_STATUS) NETWORK_STATUS.textContent = "error";
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
    const results = await Promise.allSettled([fetchSummary(), fetchNetwork(), fetchClustering()]);
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
  // Assign, do not replace: the IIFE above already attached
  // shouldDeclutterLabels/LABEL_DECLUTTER_MIN_NODES to this object.
  module.exports = { apiFetch, shouldDeclutterLabels, LABEL_DECLUTTER_MIN_NODES };
}
