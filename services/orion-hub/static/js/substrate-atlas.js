"use strict";

async function apiFetch(path) {
  const r = await fetch(path, { headers: { Accept: "application/json" } });
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

if (typeof document !== "undefined") {
(function () {
  const TRACE_LIST = document.getElementById("atlasTraceList");
  const LAYER_RAIL = document.getElementById("atlasLayerRail");
  const DIMENSION_BAR = document.getElementById("atlasDimensionBar");
  const CY_HOST = document.getElementById("grammarAtlasCy");
  const INSPECTOR = document.getElementById("atlasAtomInspector");
  const TIMELINE = document.getElementById("atlasTimeline");
  const STATUS = document.getElementById("atlasStatus");
  const REFRESH_BTN = document.getElementById("atlasRefreshBtn");

  const pollAttr = document.body && document.body.getAttribute("data-atlas-poll-ms");
  const POLL_MS = Math.max(500, parseInt(pollAttr || "3000", 10) || 3000);
  const LAYER_COLORS = [
    "#64748b",
    "#0ea5e9",
    "#22c55e",
    "#a855f7",
    "#f59e0b",
    "#ec4899",
    "#14b8a6",
    "#6366f1",
    "#ef4444",
    "#84cc16",
    "#06b6d4",
  ];

  let cy = null;
  let selectedTraceId = null;
  let selectedLayer = null;
  let activeDimensions = new Set();
  let pollTimer = null;
  let lastGraphPayload = null;
  let lastGraphSignature = null;
  let lastTraceMeta = null;
  let nodeById = new Map();
  let atlasUserHasPanned = false;
  let atlasInitialFitDone = false;
  let atlasFitTimer = null;

  function setStatus(msg, isErr) {
    if (!STATUS) return;
    STATUS.textContent = msg;
    STATUS.classList.toggle("text-red-400", !!isErr);
    STATUS.classList.toggle("text-gray-400", !isErr);
  }

  function layerColor(layer) {
    let h = 0;
    const s = String(layer || "");
    for (let i = 0; i < s.length; i += 1) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return LAYER_COLORS[h % LAYER_COLORS.length];
  }

  function destroyCy() {
    if (cy) {
      try {
        cy.destroy();
      } catch {
        /* ignore */
      }
      cy = null;
    }
    if (CY_HOST) CY_HOST.textContent = "";
  }

  function graphPayloadSignature(payload) {
    const nodes = (payload && payload.nodes) || [];
    const edges = (payload && payload.edges) || [];
    const nodeIds = nodes.map((n) => n.id).sort().join(",");
    const edgeIds = edges.map((e) => e.id).sort().join(",");
    return `${nodes.length}|${edges.length}|${nodeIds}|${edgeIds}`;
  }

  function fitAtlasGraph(force) {
    if (!cy) return;
    if (!force && atlasUserHasPanned) return;
    try {
      cy.resize();
      cy.fit(undefined, 32);
      atlasInitialFitDone = true;
    } catch {
      /* ignore */
    }
  }

  function scheduleAtlasFit(force) {
    if (atlasFitTimer) clearTimeout(atlasFitTimer);
    atlasFitTimer = setTimeout(() => fitAtlasGraph(force), force ? 0 : 80);
  }

  function traceIsLive(meta) {
    return meta && String(meta.status || "").toLowerCase() === "open";
  }

  function activateAtlasPanel() {
    if (selectedTraceId) {
      if (!cy) {
        loadGraph(selectedTraceId, { silent: true });
        return;
      }
      if (!atlasUserHasPanned) scheduleAtlasFit(false);
      return;
    }
    const firstBtn = TRACE_LIST && TRACE_LIST.querySelector("button[data-trace-id]");
    if (firstBtn && firstBtn.__traceRow) {
      selectTrace(firstBtn.__traceRow);
    }
  }

  function stopPoll() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function startPoll(traceMeta) {
    stopPoll();
    if (!selectedTraceId || !traceIsLive(traceMeta || lastTraceMeta)) return;
    pollTimer = setInterval(() => {
      if (selectedTraceId && traceIsLive(lastTraceMeta)) {
        loadGraph(selectedTraceId, { silent: true });
      } else {
        stopPoll();
      }
    }, POLL_MS);
  }

  function formatTs(value) {
    if (!value) return "—";
    try {
      return new Date(value).toLocaleString();
    } catch {
      return String(value);
    }
  }

  function renderTimeline(traceMeta, graphPayload) {
    if (!TIMELINE) return;
    const parts = [];
    if (traceMeta) {
      parts.push(
        `trace ${traceMeta.trace_id || selectedTraceId} · ${traceMeta.status || "?"} · ${formatTs(traceMeta.started_at)} → ${formatTs(traceMeta.ended_at)}`,
      );
    }
    const nodes = (graphPayload && graphPayload.nodes) || [];
    const withTime = nodes.filter((n) => n.time_start || n.time_end);
    if (withTime.length) {
      parts.push(
        `atoms: ${withTime
          .slice(0, 8)
          .map((n) => `${n.id?.slice(0, 12) || "?"}@${formatTs(n.time_start)}`)
          .join(" · ")}${withTime.length > 8 ? " …" : ""}`,
      );
    }
    TIMELINE.textContent = parts.length ? parts.join(" | ") : "No timeline data for this trace.";
  }

  function renderLayerRail(groups) {
    if (!LAYER_RAIL) return;
    LAYER_RAIL.innerHTML = "";
    const layers = (groups && groups.layers) || [];
    if (!layers.length) {
      LAYER_RAIL.textContent = "No layers in graph.";
      return;
    }
    const allBtn = document.createElement("button");
    allBtn.type = "button";
    allBtn.dataset.layer = "";
    allBtn.className =
      "px-2 py-1 rounded-full text-[10px] border " +
      (selectedLayer ? "border-gray-700 bg-gray-900 text-gray-400" : "border-indigo-500 bg-indigo-900/50 text-indigo-100");
    allBtn.textContent = "All layers";
    allBtn.addEventListener("click", () => {
      selectedLayer = null;
      renderLayerRail(groups);
      applyGraphFilters();
    });
    LAYER_RAIL.appendChild(allBtn);

    layers.forEach((entry) => {
      const layer = entry.layer || entry;
      const count = entry.atom_count != null ? entry.atom_count : "";
      const btn = document.createElement("button");
      btn.type = "button";
      btn.dataset.layer = layer;
      const active = selectedLayer === layer;
      btn.className =
        "px-2 py-1 rounded-full text-[10px] border flex items-center gap-1 " +
        (active ? "border-indigo-500 bg-indigo-900/50 text-indigo-100" : "border-gray-700 bg-gray-900 text-gray-300 hover:bg-gray-800");
      const dot = document.createElement("span");
      dot.className = "inline-block w-2 h-2 rounded-full";
      dot.style.backgroundColor = layerColor(layer);
      btn.appendChild(dot);
      btn.appendChild(document.createTextNode(`${layer}${count !== "" ? ` (${count})` : ""}`));
      btn.addEventListener("click", () => {
        selectedLayer = layer;
        renderLayerRail(groups);
        applyGraphFilters();
      });
      LAYER_RAIL.appendChild(btn);
    });
  }

  function renderDimensionBar(groups) {
    if (!DIMENSION_BAR) return;
    DIMENSION_BAR.innerHTML = "";
    const dims = (groups && groups.dimensions) || [];
    if (!dims.length) {
      DIMENSION_BAR.textContent = "No dimensions in graph.";
      return;
    }
    dims.forEach((entry) => {
      const dim = entry.dimension || entry;
      const count = entry.atom_count != null ? entry.atom_count : "";
      const chip = document.createElement("button");
      chip.type = "button";
      chip.dataset.dimension = dim;
      const on = activeDimensions.has(dim);
      chip.className =
        "px-2 py-1 rounded-full text-[10px] border " +
        (on ? "border-emerald-500 bg-emerald-900/40 text-emerald-100" : "border-gray-700 bg-gray-900 text-gray-400 hover:bg-gray-800");
      chip.textContent = `${dim}${count !== "" ? ` (${count})` : ""}`;
      chip.addEventListener("click", () => {
        if (activeDimensions.has(dim)) activeDimensions.delete(dim);
        else activeDimensions.add(dim);
        renderDimensionBar(groups);
        applyGraphFilters();
      });
      DIMENSION_BAR.appendChild(chip);
    });
  }

  function nodeMatchesFilters(nodeData) {
    if (selectedLayer && nodeData.layer !== selectedLayer) return false;
    if (activeDimensions.size > 0) {
      const dims = nodeData.dimensions || [];
      const hit = dims.some((d) => activeDimensions.has(d));
      if (!hit) return false;
    }
    return true;
  }

  function applyGraphFilters() {
    if (!cy) return;
    cy.nodes().forEach((ele) => {
      const match = nodeMatchesFilters(ele.data());
      ele.removeClass("atlas-dimmed atlas-highlight");
      if (selectedLayer || activeDimensions.size > 0) {
        ele.addClass(match ? "atlas-highlight" : "atlas-dimmed");
      }
    });
    cy.edges().forEach((ele) => {
      const src = ele.source();
      const tgt = ele.target();
      const visible = src.hasClass("atlas-highlight") && tgt.hasClass("atlas-highlight");
      const dimmed = src.hasClass("atlas-dimmed") || tgt.hasClass("atlas-dimmed");
      ele.removeClass("atlas-dimmed atlas-highlight");
      if (selectedLayer || activeDimensions.size > 0) {
        ele.addClass(visible ? "atlas-highlight" : dimmed ? "atlas-dimmed" : "");
      }
    });
  }

  function graphToElements(payload) {
    const nodes = (payload && payload.nodes) || [];
    const edges = (payload && payload.edges) || [];
    nodeById = new Map();
    const layerCols = new Map();
    const cyNodes = nodes.map((n) => {
      const id = String(n.id);
      nodeById.set(id, n);
      const layer = n.layer || "";
      const col = layerCols.get(layer) || 0;
      layerCols.set(layer, col + 1);
      const position =
        n.x != null && n.y != null
          ? { x: Number(n.x), y: Number(n.y) }
          : n.y != null
            ? { x: col * 100, y: Number(n.y) }
            : undefined;
      return {
        data: {
          id,
          label: n.label || n.type || id,
          layer,
          dimensions: n.dimensions || [],
          atomType: n.type || "",
          confidence: n.confidence,
          salience: n.salience,
          semantic_role: n.semantic_role,
        },
        position,
      };
    });
    const cyEdges = edges.map((e) => ({
      data: {
        id: String(e.id),
        source: String(e.source),
        target: String(e.target),
        label: e.type || "",
        edgeType: e.type || "",
      },
    }));
    return [...cyNodes, ...cyEdges];
  }

  function renderInspector(atomId, provenance) {
    if (!INSPECTOR) return;
    const node = nodeById.get(atomId) || {};
    const atom = (provenance && provenance.atom) || {};
    const fields = [
      ["atom_id", atom.atom_id || atomId],
      ["type", atom.atom_type || node.type],
      ["layer", atom.layer || node.layer],
      ["role", atom.semantic_role || node.semantic_role],
      ["summary", atom.summary || node.label],
      ["confidence", atom.confidence ?? node.confidence],
      ["salience", atom.salience ?? node.salience],
      ["dimensions", (atom.dimensions || node.dimensions || []).join(", ")],
    ];
    let html = '<div class="space-y-2"><h3 class="text-sm font-semibold text-white">Atom</h3><dl class="space-y-1">';
    fields.forEach(([k, v]) => {
      html += `<div><dt class="text-gray-500">${k}</dt><dd class="text-gray-200 break-all">${escapeHtml(String(v ?? "—"))}</dd></div>`;
    });
    html += "</dl>";
    if (provenance) {
      html += '<h3 class="text-sm font-semibold text-white mt-3">Provenance</h3>';
      html += `<p class="text-gray-400">parents: ${(provenance.parent_atom_ids || []).join(", ") || "—"}</p>`;
      html += `<p class="text-gray-400">children: ${(provenance.child_atom_ids || []).join(", ") || "—"}</p>`;
      if (provenance.source_event) {
        html += `<pre class="mt-2 p-2 bg-gray-950 border border-gray-800 rounded text-[10px] overflow-auto max-h-32">${escapeHtml(
          JSON.stringify(provenance.source_event, null, 2),
        )}</pre>`;
      }
      const inE = provenance.incoming_edges || [];
      const outE = provenance.outgoing_edges || [];
      if (inE.length || outE.length) {
        html += `<p class="text-gray-500 mt-2">edges in ${inE.length} / out ${outE.length}</p>`;
      }
    }
    html += "</div>";
    INSPECTOR.innerHTML = html;
  }

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function loadProvenance(atomId) {
    try {
      const payload = await apiFetch(`/api/substrate/atlas/atoms/${encodeURIComponent(atomId)}/provenance`);
      renderInspector(atomId, payload);
    } catch (e) {
      renderInspector(atomId, null);
      if (INSPECTOR) {
        INSPECTOR.innerHTML += `<p class="text-red-400 mt-2">${escapeHtml(e.message || String(e))}</p>`;
      }
    }
  }

  function mountCytoscape(elements, opts) {
    const allowAutoFit = !(opts && opts.skipAutoFit);
    if (!CY_HOST || typeof window.cytoscape !== "function") {
      setStatus("Cytoscape failed to load.", true);
      return;
    }
    destroyCy();
    atlasUserHasPanned = false;
    atlasInitialFitDone = false;
    CY_HOST.textContent = "";
    if (!elements.length) {
      CY_HOST.textContent = "No nodes in trace graph.";
      return;
    }
    const hasPreset = elements.some((el) => el.position && el.position.y != null);
    cy = window.cytoscape({
      container: CY_HOST,
      elements,
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "font-size": 8,
            color: "#e2e8f0",
            "text-valign": "bottom",
            "text-margin-y": 4,
            "background-color": (ele) => layerColor(ele.data("layer")),
            width: 28,
            height: 28,
            "border-width": 2,
            "border-color": "#1e293b",
          },
        },
        {
          selector: "node:selected",
          style: { "border-color": "#818cf8", "border-width": 3 },
        },
        {
          selector: "node.atlas-dimmed",
          style: { opacity: 0.2 },
        },
        {
          selector: "node.atlas-highlight",
          style: { opacity: 1 },
        },
        {
          selector: "edge",
          style: {
            width: 1.5,
            "line-color": "#64748b",
            "target-arrow-color": "#64748b",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            label: "data(label)",
            "font-size": 6,
            color: "#94a3b8",
          },
        },
        {
          selector: "edge.atlas-dimmed",
          style: { opacity: 0.15 },
        },
      ],
      layout: hasPreset
        ? { name: "preset", fit: false, padding: 24 }
        : { name: "breadthfirst", directed: true, padding: 24, animate: false },
      wheelSensitivity: 0.3,
      autoungrabify: false,
    });
    cy.on("zoom pan", () => {
      atlasUserHasPanned = true;
    });
    cy.on("tap", "node", (evt) => {
      const id = evt.target.id();
      loadProvenance(id);
    });
    applyGraphFilters();
    if (allowAutoFit) {
      scheduleAtlasFit(true);
    }
  }

  async function loadGraph(traceId, opts) {
    const silent = opts && opts.silent;
    if (!silent) setStatus(`Loading graph for ${traceId}…`);
    try {
      const payload = await apiFetch(
        `/api/substrate/atlas/traces/${encodeURIComponent(traceId)}/graph?layout=layered&depth=4`,
      );
      const signature = graphPayloadSignature(payload);
      if (silent && cy && signature === lastGraphSignature) {
        return;
      }
      lastGraphSignature = signature;
      lastGraphPayload = payload;
      const elements = graphToElements(payload);
      const preserveView = silent && cy && atlasUserHasPanned;
      if (preserveView) {
        const zoom = cy.zoom();
        const pan = cy.pan();
        mountCytoscape(elements, { skipAutoFit: true });
        try {
          cy.zoom(zoom);
          cy.pan(pan);
        } catch {
          /* ignore */
        }
        atlasUserHasPanned = true;
      } else {
        mountCytoscape(elements);
      }
      renderLayerRail((payload.groups) || {});
      renderDimensionBar((payload.groups) || {});
      renderTimeline(lastTraceMeta, payload);
      if (!silent) setStatus(`Graph: ${(payload.nodes || []).length} nodes, ${(payload.edges || []).length} edges`);
    } catch (e) {
      destroyCy();
      if (!silent) setStatus(`Graph error: ${e.message || e}`, true);
    }
  }

  function renderTraceList(items) {
    if (!TRACE_LIST) return;
    TRACE_LIST.innerHTML = "";
    if (!items.length) {
      const li = document.createElement("li");
      li.className = "px-3 py-4 text-gray-500";
      li.textContent = "No traces found.";
      TRACE_LIST.appendChild(li);
      return;
    }
    items.forEach((t) => {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className =
        "w-full text-left px-3 py-2 hover:bg-gray-900/80 " +
        (selectedTraceId === t.trace_id ? "bg-indigo-950/50 border-l-2 border-indigo-500" : "");
      btn.dataset.traceId = t.trace_id || "";
      btn.__traceRow = t;
      const selected = selectedTraceId === t.trace_id;
      btn.innerHTML = `<div class="font-medium text-gray-200 truncate">${escapeHtml(t.trace_id || "")}</div>
        <div class="text-[10px] text-gray-500">${escapeHtml(t.trace_type || "")} · ${t.atom_count ?? 0} atoms · ${t.edge_count ?? 0} edges</div>
        <div class="text-[10px] text-gray-600">${formatTs(t.started_at)}</div>
        <div class="text-[10px] mt-1 ${selected ? "text-indigo-300" : "text-gray-500"}">${selected ? "Graph loaded" : "Click to load graph"}</div>`;
      btn.addEventListener("click", () => selectTrace(t));
      li.appendChild(btn);
      TRACE_LIST.appendChild(li);
    });
  }

  async function selectTrace(traceRow) {
    selectedTraceId = traceRow.trace_id;
    lastTraceMeta = traceRow;
    selectedLayer = null;
    activeDimensions = new Set();
    await fetchTraces();
    await loadGraph(selectedTraceId);
    startPoll(lastTraceMeta);
    try {
      const detail = await apiFetch(`/api/substrate/atlas/traces/${encodeURIComponent(selectedTraceId)}`);
      lastTraceMeta = detail.trace || traceRow;
      if (detail.temporal_hops && TIMELINE) {
        const hops = detail.temporal_hops.slice(0, 12);
        const hopLine = hops.map((h) => `${h.from_atom_id}→${h.to_atom_id} (${h.hop_type})`).join(" · ");
        if (hopLine) TIMELINE.textContent = `${TIMELINE.textContent} | hops: ${hopLine}`;
      }
    } catch {
      /* trace detail optional */
    }
  }

  async function fetchTraces() {
    setStatus("Loading traces…");
    try {
      const payload = await apiFetch("/api/substrate/atlas/traces?limit=50");
      const items = payload.items || [];
      renderTraceList(items);
      if (items.length && !selectedTraceId) {
        await selectTrace(items[0]);
      } else if (items.length) {
        setStatus(`${items.length} trace(s) · ${selectedTraceId}`);
      } else {
        setStatus(`${items.length} trace(s)`);
      }
      return items;
    } catch (e) {
      renderTraceList([]);
      setStatus(`Atlas unavailable: ${e.message || e}`, true);
      return [];
    }
  }

  async function init() {
    if (REFRESH_BTN) {
      REFRESH_BTN.addEventListener("click", async () => {
        await fetchTraces();
        if (selectedTraceId) await loadGraph(selectedTraceId);
      });
    }
    window.addEventListener("beforeunload", stopPoll);
    window.addEventListener("resize", () => {
      if (!atlasUserHasPanned) scheduleAtlasFit(false);
    });
    if (CY_HOST && typeof IntersectionObserver === "function") {
      const obs = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting) && !atlasUserHasPanned && !atlasInitialFitDone) {
          scheduleAtlasFit(false);
        }
      });
      obs.observe(CY_HOST);
    }
    window.OrionSubstrateAtlas = { activate: activateAtlasPanel, fit: fitAtlasGraph };
    await fetchTraces();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { apiFetch };
}
