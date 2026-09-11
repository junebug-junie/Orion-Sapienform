"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const help = {
    worldview: "Default view: up to the node limit in worldview. Pick a starting node to explore its neighborhood.",
    substrate: "Default view: a neighborhood around the most activated concept. Search to choose your own starting node.",
    crystallizations: "Default view: the 50 most recently updated active crystallizations, plus lineage within the node limit. Search to focus on one memory.",
  };
  let downloadUrl;
  const clearDownload = () => {
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    downloadUrl = null;
    $("download").hidden = true;
  };
  const errorText = async (response) => {
    try { const data = await response.json(); return typeof data.detail === "string" ? data.detail : "Check the node id and limits."; }
    catch (_) { return `Request failed (${response.status}).`; }
  };
  $("source").addEventListener("change", () => {
    $("seed").replaceChildren(new Option("Default view", ""));
    $("source-help").textContent = help[$("source").value];
    $("lineage").disabled = $("source").value !== "crystallizations";
    $("status").textContent = "";
    clearDownload();
  });
  $("source").dispatchEvent(new Event("change"));
  $("search").addEventListener("click", async () => {
    const source = $("source").value;
    $("search").disabled = true;
    $("status").textContent = "Searching…";
    try {
      const response = await fetch(`/api/graph-workbench/search/${source}?q=${encodeURIComponent($("query").value)}`, {cache:"no-store"});
      if (!response.ok) throw new Error(await errorText(response));
      const {matches} = await response.json();
      if (source !== $("source").value) return;
      $("seed").replaceChildren(new Option("Default view", ""), ...matches.map(n => new Option(`${n.label} · ${n.kind || "node"}`, n.id)));
      if (matches.length) $("seed").value = matches[0].id;
      $("status").textContent = `${matches.length} matching nodes. Select one, then open the workbench.`;
    } catch (error) { $("status").textContent = error.message; }
    finally { $("search").disabled = false; }
  });
  $("query").addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); $("search").click(); }
  });
  $("workbench-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    // Open immediately on the gesture to avoid popup blockers after the fetch.
    const tab = window.open("about:blank", "_blank");
    if (!tab) { $("status").textContent = "Allow popups for Hub to open Gephi Lite."; return; }
    tab.opener = null;
    const source = $("source").value;
    const params = new URLSearchParams({seed:$("seed").value, depth:$("depth").value, limit:$("limit").value, lineage:$("lineage").checked});
    $("open").disabled = true;
    $("status").textContent = "Preparing graph…";
    clearDownload();
    try {
      const response = await fetch(`/api/graph-workbench/export/${source}.gexf?${params}`, {cache:"no-store"});
      if (!response.ok) throw new Error(await errorText(response));
      // Keep the checked export available for download. Gephi loads a fresh
      // snapshot from the same authenticated endpoint, so its URL is reloadable.
      const blob = await response.blob();
      downloadUrl = URL.createObjectURL(blob);
      $("download").href = downloadUrl;
      $("download").download = `${source}.gexf`;
      $("download").hidden = false;
      const file = new URL(`/api/graph-workbench/export/${source}.gexf?${params}`, location.origin);
      tab.location = `/gephi-lite/?file=${encodeURIComponent(file.href)}`;
      const truncated = response.headers.get("X-Graph-Truncated") === "true";
      $("status").textContent = `${response.headers.get("X-Graph-Nodes")} nodes · ${response.headers.get("X-Graph-Edges")} edges.${truncated ? " Limited view: choose a smaller neighborhood or raise the node limit." : ""}`;
    } catch (error) { tab.close(); $("status").textContent = error.message; }
    finally { $("open").disabled = false; }
  });
})();
