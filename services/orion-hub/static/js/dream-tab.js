/* Read-only operator view. No requests to the waking hypothesis claim path. */
(() => {
  "use strict";
  const el = id => document.getElementById(id);
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
  const time = value => value ? new Date(value).toLocaleString() : "Unknown";
  const num = value => typeof value === "number" ? value.toFixed(1) : "Unknown";
  const rate = value => typeof value === "number" ? `${(100 * value).toFixed(1)}%` : "Not measured";
  let active = false, controller = null, detailController = null, wired = false;
  let cursors = [null], pageIndex = 0, nextCursor = null, selected = null;

  async function read(path, signal) {
    const response = await fetch(`/api/dream/${path}`, {signal, cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }
  function unavailable(id, label, error) {
    if (error.name !== "AbortError") el(id).textContent = `${label} unavailable (${error.message}). Refresh to retry; this is not an empty result.`;
  }
  function renderPressure(data) {
    const p = data.pressure;
    const gates = [
      ["Sleep loop", data.enabled ? "Enabled" : "Disabled"],
      ["Sleep pressure", `${num(p.pressure)} / ${num(p.threshold)} needed`],
      ["Time without chat", `${num(p.idle_minutes)} minutes / ${num(p.idle_required_minutes)} needed`],
      ["Minimum interval", data.too_soon ? "Still waiting" : "Clear"],
      ["Waking offers", data.offer_enabled ? "Enabled" : "Disabled"],
    ];
    el("dreamPressure").innerHTML = `<h3 class="font-semibold text-lg ${data.ready ? "text-emerald-300" : "text-gray-200"}">${data.ready ? "Ready for the next sleep check" : data.enabled ? "Not ready · waiting for sleep gates" : "Sleep loop is disabled"}</h3>
      <div class="grid grid-cols-2 lg:grid-cols-5 gap-4 my-3">${gates.map(([name,value]) => `<div><div class="text-xs text-gray-400">${esc(name)}</div><div class="text-sm mt-1">${esc(value)}</div></div>`).join("")}</div>
      <p class="text-xs text-gray-400">Read ${esc(time(p.computed_at))} · Material since ${esc(time(p.since))}</p>
      <p class="text-xs text-gray-400 mt-1">Candidate counts: ${Object.entries(p.counts || {}).map(([kind,count]) => `${esc(kind.replaceAll("_", " "))}: ${esc(count)}`).join(" · ") || "None reported"}</p>`;
  }
  function renderScore(data) {
    el("dreamScore").innerHTML = `<p class="text-sm text-indigo-200 mb-3">${esc(data.verdict)}</p>
      <div class="overflow-x-auto"><table class="w-full text-sm text-left"><thead class="text-gray-400"><tr><th>Pairs</th><th>Offered</th><th>Adopted</th><th>Tested</th><th>Adoption</th><th>Supported or revised / tested</th></tr></thead><tbody>
      ${["dream", "control"].map(arm => { const s = data.arms[arm]; return `<tr class="border-t border-gray-800"><th class="py-2">${arm === "dream" ? "Dream" : "Random control"}</th><td>${esc(s.offered)}</td><td>${esc(s.adopted)}</td><td>${esc(s.tested)}</td><td>${rate(s.adoption_rate)}</td><td>${rate(s.support_rate)}</td></tr>`; }).join("")}</tbody></table></div>
      <p class="text-xs text-gray-400 mt-2">At least 20 dream and 5 control offers before an adoption verdict. This is a descriptive comparison, not proof of significance. Unmatched prior references: ${esc(data.unmatched_priors)}.</p>`;
  }
  async function showCycle(id) {
    detailController?.abort();
    detailController = new AbortController();
    selected = id;
    el("dreamDetail").textContent = "Loading sleep…";
    try {
      const data = await read(`cycles/${encodeURIComponent(id)}`, detailController.signal);
      if (!active || selected !== id) return;
      const c = data.cycle;
      el("dreamDetail").innerHTML = `<h3 class="font-semibold text-lg">${esc(time(c.started_at))} · ${esc(c.status)}</h3>
        <p class="text-xs text-gray-400 mt-1">${esc(c.cycle_id)} · ${esc(c.trigger)} · ended ${esc(time(c.ended_at))}</p>
        <div class="grid grid-cols-3 gap-3 my-4 text-sm"><div>No link: ${esc(c.no_link_count)}</div><div>Garbled: ${esc(c.unparseable_count)}</div><div>LLM failures: ${esc(c.llm_failures)}</div></div>
        ${c.status === "failed" ? '<p class="text-amber-300 text-sm mb-3">Failed sleep. Its backlog is retained for a later attempt.</p>' : ""}
        <p class="text-sm text-gray-400 mb-3">REM compaction: ${c.compaction_delta_id ? `staged receipt ${esc(c.compaction_delta_id)}; nothing applied` : "no staged receipt recorded"}.</p>
        ${c.note ? `<p class="text-xs text-gray-400 mb-3">${esc(c.note)}</p>` : ""}
        <h4 class="font-semibold">Replayed memories</h4><div class="flex flex-col gap-2 my-3">${(c.replay || []).map(item => `<article class="rounded-lg bg-gray-950/60 p-3"><div class="text-xs text-indigo-300">${esc(item.source_kind)} · weight ${num(item.weight)}</div><p class="text-sm my-1 whitespace-pre-wrap">${esc(item.text)}</p><p class="text-xs text-gray-400">${esc(item.reason)}</p><p class="text-xs text-gray-500 break-all">${esc(item.ref_id)}</p></article>`).join("") || '<p class="text-gray-400 text-sm">No replay items recorded.</p>'}</div>
        <h4 class="font-semibold">Proposed links</h4><div class="flex flex-col gap-3 mt-3">${data.hypotheses.map(h => `<article class="rounded-lg border border-gray-700 p-3"><div class="text-xs text-indigo-300">${h.arm === "control" ? "Random control" : "Dream"} · ${h.offered_at ? `offered ${esc(time(h.offered_at))}` : h.expired ? "expired without offer" : "waiting for waking offer"}</div><p class="text-sm my-2 whitespace-pre-wrap">${esc(h.claim)}</p><p class="text-xs text-gray-400 whitespace-pre-wrap">${esc(h.why)}</p><p class="text-xs text-gray-500 mt-2 break-all">${esc(h.hypothesis_id)}<br>${esc(h.ref_a)} ↔ ${esc(h.ref_b)}</p>${h.offered_run_id ? `<p class="text-xs text-gray-400 mt-1 break-all">Curiosity run: ${esc(h.offered_run_id)}</p>` : ""}</article>`).join("") || '<p class="text-sm text-gray-400">No hypotheses recorded. See the pair counts and outcomes above.</p>'}</div>`;
    } catch (error) { unavailable("dreamDetail", "Sleep details", error); }
  }
  async function loadCycles(signal) {
    const cursor = cursors[pageIndex];
    const query = cursor ? `?${new URLSearchParams(cursor)}` : "";
    selected = null;
    el("dreamDetail").textContent = "Loading sleep history…";
    el("dreamCycles").textContent = "Loading sleeps…";
    el("dreamNewer").disabled = true;
    el("dreamOlder").disabled = true;
    try {
      const data = await read(`cycles${query}`, signal);
      nextCursor = data.has_more ? data.next_cursor : null;
      el("dreamCycles").innerHTML = data.cycles.map(c => `<button type="button" data-cycle="${esc(c.cycle_id)}" class="text-left rounded-lg border border-gray-700 hover:border-indigo-400 p-3"><span class="block text-sm">${esc(time(c.started_at))}</span><span class="text-xs text-gray-400">${esc(c.status)} · ${esc(c.replay_count)} replayed · ${esc(c.hypothesis_count)} links</span></button>`).join("") || '<p class="text-sm text-gray-400">No sleep cycles recorded yet.</p>';
      el("dreamNewer").disabled = pageIndex === 0;
      el("dreamOlder").disabled = !nextCursor;
      if (data.cycles.length) await showCycle(data.cycles[0].cycle_id);
      else el("dreamDetail").textContent = "No sleep to inspect yet.";
    } catch (error) {
      unavailable("dreamCycles", "Sleep history", error);
      if (error.name !== "AbortError") el("dreamDetail").textContent = "Sleep history unavailable. Refresh to retry.";
    }
  }
  async function refresh() {
    if (!active) return;
    controller?.abort(); detailController?.abort();
    controller = new AbortController();
    const signal = controller.signal;
    cursors = [null]; pageIndex = 0; nextCursor = null;
    el("dreamPressure").textContent = "Reading sleep readiness…";
    el("dreamScore").textContent = "Reading both groups…";
    el("dreamDetail").textContent = "Loading sleep history…";
    await Promise.all([
      read("pressure", signal).then(renderPressure).catch(e => unavailable("dreamPressure", "Readiness", e)),
      read("scorecard", signal).then(renderScore).catch(e => unavailable("dreamScore", "Scorecard", e)),
      loadCycles(signal),
    ]);
  }
  function activate() {
    active = true;
    if (!wired) {
      wired = true;
      el("dreamRefresh").addEventListener("click", refresh);
      el("dreamCycles").addEventListener("click", event => {
        const button = event.target.closest("[data-cycle]");
        if (button) showCycle(button.dataset.cycle);
      });
      el("dreamOlder").addEventListener("click", () => {
        if (!nextCursor) return;
        cursors[++pageIndex] = nextCursor;
        detailController?.abort();
        loadCycles(controller.signal);
      });
      el("dreamNewer").addEventListener("click", () => {
        if (pageIndex === 0) return;
        pageIndex--;
        detailController?.abort();
        loadCycles(controller.signal);
      });
    }
    refresh();
  }
  function deactivate() {
    active = false;
    controller?.abort(); detailController?.abort();
  }
  window.OrionDream = {activate, deactivate, refresh};
})();
