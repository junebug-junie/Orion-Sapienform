/* Orion Hub — Memory crystallization observatory (inbox + detail + projection health) */
(function () {
  const pathSegments = window.location.pathname.split("/").filter((p) => p.length > 0);
  const URL_PREFIX = pathSegments.length > 0 ? `/${pathSegments[0]}` : "";
  const API_BASE = window.location.origin + URL_PREFIX;

  function sessionHeader() {
    const sid = localStorage.getItem("orion_sid");
    return sid ? { "X-Orion-Session-Id": sid } : {};
  }

  async function apiFetch(path, opts) {
    const res = await fetch(`${API_BASE}${path}`, {
      ...opts,
      headers: {
        "Content-Type": "application/json",
        ...sessionHeader(),
        ...(opts && opts.headers ? opts.headers : {}),
      },
    });
    const text = await res.text();
    let body = null;
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      body = { raw: text };
    }
    if (!res.ok) {
      const err = new Error(`HTTP ${res.status}`);
      err.status = res.status;
      err.body = body;
      throw err;
    }
    return body;
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function setStatus(el, msg, isErr) {
    if (!el) return;
    el.textContent = msg || "";
    el.classList.toggle("text-red-400", !!isErr);
    el.classList.toggle("text-gray-400", !isErr);
  }

  function chatTurnCount(item) {
    const evidence = Array.isArray(item && item.evidence) ? item.evidence : [];
    const ids = new Set(
      evidence.filter((e) => e && e.source_kind === "chat_turn" && e.source_id).map((e) => e.source_id),
    );
    return ids.size;
  }

  function splitExcerpt(excerpt) {
    const text = String(excerpt || "");
    const idx = text.indexOf("\n");
    if (idx < 0) return { prompt: text, response: "" };
    return { prompt: text.slice(0, idx), response: text.slice(idx + 1) };
  }

  function renderProvenance(provenance) {
    const p = provenance && typeof provenance === "object" ? provenance : {};
    const rows = [];
    if (Array.isArray(p.gate_reasons) && p.gate_reasons.length) {
      rows.push(`<div><span class="text-gray-500">Gate:</span> ${escapeHtml(p.gate_reasons.join(", "))}</div>`);
    }
    if (p.dominant_shift) {
      rows.push(`<div><span class="text-gray-500">Dominant shift:</span> ${escapeHtml(p.dominant_shift)}</div>`);
    }
    if (p.window_novelty_max != null) {
      rows.push(`<div><span class="text-gray-500">Window novelty max:</span> ${escapeHtml(Number(p.window_novelty_max).toFixed(3))}</div>`);
    }
    if (p.window_significance_max != null) {
      rows.push(`<div><span class="text-gray-500">Window significance max:</span> ${escapeHtml(Number(p.window_significance_max).toFixed(3))}</div>`);
    }
    if (p.memory_window_id) {
      rows.push(`<div><span class="text-gray-500">Window:</span> <code class="text-[10px]">${escapeHtml(p.memory_window_id)}</code></div>`);
    }
    if (p.formation_policy) {
      rows.push(`<div><span class="text-gray-500">Formation:</span> ${escapeHtml(p.formation_policy)}</div>`);
    }
    if (p.formation_policy_downgrade) {
      rows.push(`<div class="text-amber-300"><span class="text-gray-500">Downgrade:</span> ${escapeHtml(p.formation_policy_downgrade)}</div>`);
    }
    if (!rows.length) {
      return `<div class="text-gray-500">No gate provenance stored (older proposals pre-schema).</div>`;
    }
    return rows.join("");
  }

  function renderEvidence(evidence) {
    const items = Array.isArray(evidence) ? evidence : [];
    const chatTurns = items.filter((e) => e && e.source_kind === "chat_turn");
    const grammar = items.filter((e) => e && e.source_kind === "grammar_event");
    if (!chatTurns.length && !grammar.length) {
      return `<div class="text-gray-500">No source evidence attached.</div>`;
    }
    const parts = [];
    if (chatTurns.length) {
      parts.push(`<div class="font-medium text-gray-300 mt-2">Chat turns (${chatTurns.length})</div>`);
      chatTurns.forEach((ev, idx) => {
        const { prompt, response } = splitExcerpt(ev.excerpt);
        parts.push(`<div class="border border-gray-800 rounded p-2 mt-1 bg-gray-950/40">
          <div class="text-[10px] text-gray-500">Turn ${idx + 1} · <code>${escapeHtml(ev.source_id || "")}</code></div>
          ${ev.note ? `<div class="text-[10px] text-indigo-300/90 mt-1">${escapeHtml(ev.note)}</div>` : ""}
          <div class="mt-1"><span class="text-gray-500">User:</span> ${escapeHtml(prompt)}</div>
          <div class="mt-1"><span class="text-gray-500">Orion:</span> ${escapeHtml(response)}</div>
        </div>`);
      });
    }
    if (grammar.length) {
      parts.push(`<div class="text-gray-500 mt-2">${grammar.length} grammar event ref(s)</div>`);
    }
    return parts.join("");
  }

  function renderDetail(row, links, health) {
    const dyn = row.dynamics && typeof row.dynamics === "object" ? row.dynamics : {};
    const planning = Array.isArray(row.planning_effects) ? row.planning_effects : [];
    const retrieval = Array.isArray(row.retrieval_affordances) ? row.retrieval_affordances : [];
    const turnCount = chatTurnCount(row);
    return `<div class="space-y-2">
      <div><strong>${escapeHtml(row.subject)}</strong> <span class="text-gray-500">[${escapeHtml(row.kind)}]</span></div>
      <div>${escapeHtml(row.summary)}</div>
      <div class="text-gray-500">Status: ${escapeHtml(row.status)} · Confidence: ${escapeHtml(row.confidence)} · Salience: ${escapeHtml(String(row.salience ?? ""))}</div>
      <div class="text-gray-500">Activation: ${escapeHtml(String(dyn.activation ?? "0"))} · Reinforcements: ${escapeHtml(String(dyn.reinforcement_count ?? "0"))}</div>
      <div class="text-gray-500">Source turns in window: ${turnCount}</div>
      <div class="border border-gray-800 rounded p-2">${renderProvenance(row.provenance)}</div>
      ${planning.length ? `<div><span class="text-gray-500">Planning:</span><ul class="list-disc ml-4">${planning.map((p) => `<li>${escapeHtml(p)}</li>`).join("")}</ul></div>` : ""}
      ${retrieval.length ? `<div><span class="text-gray-500">Retrieval:</span> ${escapeHtml(retrieval.join(", "))}</div>` : ""}
      <div class="border border-gray-800 rounded p-2 max-h-48 overflow-y-auto">${renderEvidence(row.evidence)}</div>
      <div class="text-gray-500">Projection refs: cards=${(row.projection_refs && row.projection_refs.memory_card_ids || []).length}, chroma=${(row.projection_refs && row.projection_refs.chroma_doc_ids || []).length}, graphiti_eps=${((row.projection_refs && row.projection_refs.graphiti_episode_ids) || []).length}, graphiti_edges=${((row.projection_refs && row.projection_refs.graphiti_edge_ids) || []).length}</div>
      <div class="text-gray-500">Links: ${(links.items || []).length}</div>
      <div class="text-gray-500">Health: chroma=${escapeHtml(health.chroma_collection || "")}, graphiti=${health.graphiti_enabled ? "on" : "off"}</div>
      <div class="flex gap-2 mt-2">
        <button type="button" data-act="approve" class="px-2 py-1 rounded border border-emerald-700 text-emerald-200">Approve</button>
        <button type="button" data-act="reject" class="px-2 py-1 rounded border border-red-800 text-red-200">Reject</button>
        <button type="button" data-act="validate" class="px-2 py-1 rounded border border-gray-600 text-gray-200">Validate</button>
        <button type="button" data-act="sync-graphiti" class="px-2 py-1 rounded border border-sky-700 text-sky-200">Sync Graphiti</button>
      </div>
    </div>`;
  }

  function renderRow(item, onOpen) {
    const row = document.createElement("div");
    row.className = "flex justify-between gap-2 border border-gray-800 rounded px-2 py-1 bg-gray-900/60";
    const turns = chatTurnCount(item);
    row.innerHTML = `<div><div class="font-medium text-gray-100">${escapeHtml(item.subject || "")}</div>
      <div class="text-[10px] text-gray-500">${escapeHtml(item.kind || "")} · ${escapeHtml(item.status || "")} · salience ${escapeHtml(String(item.salience ?? ""))}${turns ? ` · ${turns} turn(s)` : ""}</div></div>`;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "text-indigo-300 text-xs";
    btn.textContent = "Open";
    btn.addEventListener("click", () => onOpen(item));
    row.appendChild(btn);
    return row;
  }

  async function loadInbox(listEl, statusEl, detailEl) {
    setStatus(statusEl, "Loading proposals…", false);
    listEl.innerHTML = "";
    try {
      const data = await apiFetch("/api/memory/crystallizations/proposals");
      const items = (data && data.items) || [];
      if (!items.length) {
        setStatus(statusEl, "No proposals in inbox.", false);
        return;
      }
      setStatus(statusEl, `${items.length} proposal(s)`, false);
      items.forEach((item) => {
        listEl.appendChild(
          renderRow(item, async (row) => {
            detailEl.classList.remove("hidden");
            setStatus(statusEl, "Loading proposal detail…", false);
            const full = await apiFetch(`/api/memory/crystallizations/proposals/${encodeURIComponent(row.crystallization_id)}`);
            const links = await apiFetch(`/api/memory/crystallizations/${row.crystallization_id}/links`).catch(() => ({ items: [] }));
            const health = await apiFetch("/api/memory/crystallizations/projection/health").catch(() => ({}));
            detailEl.innerHTML = renderDetail(full, links, health);
            setStatus(statusEl, `${items.length} proposal(s)`, false);
            detailEl.querySelectorAll("button[data-act]").forEach((btn) => {
              btn.addEventListener("click", async () => {
                const act = btn.getAttribute("data-act");
                try {
                  if (act === "sync-graphiti") {
                    await apiFetch(`/api/memory/graphiti/sync/${row.crystallization_id}`, { method: "POST", body: "{}" });
                  } else {
                    await apiFetch(`/api/memory/crystallizations/proposals/${row.crystallization_id}/${act}`, { method: "POST", body: act === "validate" ? undefined : "{}" });
                  }
                  setStatus(statusEl, `${act} ok`, false);
                  await loadInbox(listEl, statusEl, detailEl);
                } catch (e) {
                  setStatus(statusEl, e.message || String(e), true);
                }
              });
            });
          }),
        );
      });
    } catch (e) {
      setStatus(statusEl, e.message || String(e), true);
    }
  }

  async function activate() {
    const listEl = document.getElementById("memoryCrystallizationList");
    const statusEl = document.getElementById("memoryCrystallizationStatus");
    const detailEl = document.getElementById("memoryCrystallizationDetail");
    const healthEl = document.getElementById("memoryCrystallizationHealth");
    const panel = document.getElementById("memoryCrystallizationPanel");
    if (!listEl || !statusEl || !detailEl) return;
    if (detailEl) detailEl.classList.add("hidden");
    await loadInbox(listEl, statusEl, detailEl);
    if (healthEl) {
      try {
        const h = await apiFetch("/api/memory/crystallizations/projection/health");
        healthEl.textContent = `Chroma: ${h.chroma_collection || "—"} · Graphiti: ${h.graphiti_enabled ? "enabled" : "disabled"} · RDF: ${h.rdf_memory_graph || "unchanged"}`;
      } catch (e) {
        healthEl.textContent = e.message || "health unavailable";
      }
    }
    if (panel && typeof panel.scrollIntoView === "function") {
      panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  window.OrionMemoryCrystallizationUI = { activate };
})();
