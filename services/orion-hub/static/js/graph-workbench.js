"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const source = $("source");
  const focus = $("seed");
  const query = $("query");
  const depth = $("depth");
  const limit = $("limit");
  const lineage = $("lineage");
  const frame = $("graph-frame");
  const status = $("status");
  const focusStatus = $("focus-status");
  const download = $("download");
  const refresh = $("refresh");
  if (!source || !focus || !query || !depth || !limit || !lineage || !frame || !status || !focusStatus || !download || !refresh) return;

  const descriptions = {
    crystallizations: {
      overview: "Overview — recent memories",
      help: "Recent memory crystallizations and the evidence connected to them.",
      placeholder: "Type words from a memory…",
      loading: "memory crystallizations",
    },
    worldview: {
      overview: "Overview — current worldview",
      help: "Orion’s beliefs and claims, with the evidence and revisions connecting them.",
      placeholder: "Type a belief, claim, or concept…",
      loading: "the worldview",
    },
    substrate: {
      overview: "Overview — most active concept",
      help: "Orion’s active concepts and the relationships between them.",
      placeholder: "Type a concept or relationship…",
      loading: "the substrate",
    },
  };
  let searchVersion = 0;
  let searchTimer;
  let loadVersion = 0;
  let snapshotUrl;
  let pendingImport;

  const errorText = async (response) => {
    try {
      const data = await response.json();
      return typeof data.detail === "string" ? data.detail : `Request failed (${response.status}).`;
    } catch (_) {
      return `Request failed (${response.status}).`;
    }
  };

  const exportUrl = () => {
    const params = new URLSearchParams({
      seed: focus.value,
      depth: depth.value,
      limit: limit.value,
      lineage: lineage.checked,
    });
    return `/api/graph-workbench/export/${source.value}.gexf?${params}`;
  };

  const nativeCounts = () => {
    try {
      const text = frame.contentDocument?.body?.innerText || "";
      const nodeMatch = text.match(/Nodes\n([\d,]+)/);
      const edgeMatch = text.match(/Edges\n([\d,]+)/);
      return {
        nodes: nodeMatch ? Number(nodeMatch[1].replaceAll(",", "")) : 0,
        edges: edgeMatch ? Number(edgeMatch[1].replaceAll(",", "")) : -1,
      };
    } catch (_) {
      return {nodes: 0, edges: -1};
    }
  };

  const confirmImport = async (version, expectedNodes, expectedEdges, summary) => {
    for (let attempt = 0; attempt < 80 && version === loadVersion; attempt += 1) {
      if (frame.contentWindow.__orionGraphImportVersion !== version) return;
      const current = nativeCounts();
      if (current.nodes === expectedNodes && current.edges === expectedEdges) {
        frame.dataset.confirmedVersion = String(version);
        frame.dataset.confirmedDocumentVersion = String(frame.contentWindow.__orionGraphImportVersion);
        pendingImport = undefined;
        status.textContent = `${summary} Ready. Use Gephi’s Data, Layout, Filters, and Metrics below.`;
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    if (version === loadVersion) {
      pendingImport = undefined;
      status.textContent = `${summary} Gephi opened, but import was not confirmed. Try Refresh graph.`;
    }
  };

  const loadGraph = async () => {
    const version = ++loadVersion;
    const graphUrl = new URL(exportUrl(), window.location.origin);
    const selectedSource = source.value;
    status.textContent = `Preparing ${descriptions[selectedSource].loading}…`;
    refresh.disabled = true;
    try {
      const response = await fetch(graphUrl, {cache: "no-store"});
      if (!response.ok) throw new Error(await errorText(response));
      const blob = await response.blob();
      if (version !== loadVersion) return;
      if (snapshotUrl) URL.revokeObjectURL(snapshotUrl);
      snapshotUrl = URL.createObjectURL(blob);
      const nodes = Number(response.headers.get("X-Graph-Nodes"));
      const edges = Number(response.headers.get("X-Graph-Edges"));
      const truncated = response.headers.get("X-Graph-Truncated") === "true";
      const summary = `${nodes} nodes · ${edges} edges${truncated ? " · limited slice" : ""}.`;
      download.href = snapshotUrl;
      download.download = `${selectedSource}.gexf`;
      frame.dataset.source = selectedSource;
      frame.dataset.seed = focus.value;
      frame.dataset.requestVersion = String(version);
      frame.dataset.confirmedVersion = "";
      frame.dataset.confirmedDocumentVersion = "";
      const frameSrc = new URL(`/gephi-lite/?file=${encodeURIComponent(snapshotUrl)}`, window.location.origin).href;
      pendingImport = {version, nodes, edges, summary, frameSrc};
      frame.src = frameSrc;
      status.textContent = `${summary} Opening Gephi…`;
    } catch (error) {
      if (version === loadVersion) status.textContent = `Graph unavailable: ${error.message}`;
    } finally {
      if (version === loadVersion) refresh.disabled = false;
    }
  };

  const refreshFocus = async (text = "") => {
    const version = ++searchVersion;
    const selectedSource = source.value;
    const description = descriptions[selectedSource];
    const activeValue = focus.value;
    const activeLabel = (focus.selectedOptions[0]?.textContent || "Current focus").replace(/ — currently shown$/, "");
    focus.disabled = true;
    focusStatus.textContent = text ? "Filtering…" : "Loading named choices…";
    try {
      const response = await fetch(`/api/graph-workbench/search/${selectedSource}?q=${encodeURIComponent(text)}`, {cache: "no-store"});
      if (!response.ok) throw new Error(await errorText(response));
      const {matches} = await response.json();
      if (version !== searchVersion || selectedSource !== source.value) return;
      const choices = [new Option(description.overview, "")];
      if (activeValue && !matches.some((node) => String(node.id) === activeValue)) {
        choices.push(new Option(`${activeLabel} — currently shown`, activeValue));
      }
      choices.push(...matches.map((node) => new Option(`${node.label} — ${node.kind || "node"}`, node.id)));
      focus.replaceChildren(...choices);
      if (choices.some((option) => option.value === activeValue)) focus.value = activeValue;
      focusStatus.textContent = text ? `${matches.length} matching choices` : `${matches.length} named choices`;
    } catch (error) {
      if (version === searchVersion) focusStatus.textContent = `Choices unavailable: ${error.message}`;
    } finally {
      if (version === searchVersion) focus.disabled = false;
    }
  };

  const applySource = () => {
    const description = descriptions[source.value];
    $("source-help").textContent = description.help;
    query.value = "";
    query.placeholder = description.placeholder;
    focus.replaceChildren(new Option(description.overview, ""));
    lineage.disabled = source.value !== "crystallizations";
    refreshFocus();
    loadGraph();
  };

  source.addEventListener("change", applySource);
  focus.addEventListener("change", loadGraph);
  query.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => refreshFocus(query.value.trim()), 250);
  });
  depth.addEventListener("change", loadGraph);
  limit.addEventListener("change", loadGraph);
  lineage.addEventListener("change", loadGraph);
  refresh.addEventListener("click", loadGraph);
  frame.addEventListener("load", () => {
    const pending = pendingImport;
    if (!pending || pending.version !== loadVersion) return;
    try {
      if (frame.contentWindow.location.href !== pending.frameSrc) return;
      frame.contentWindow.__orionGraphImportVersion = pending.version;
    } catch (_) {
      return;
    }
    confirmImport(pending.version, pending.nodes, pending.edges, pending.summary);
  });
  window.addEventListener("beforeunload", () => { if (snapshotUrl) URL.revokeObjectURL(snapshotUrl); });

  applySource();
})();
