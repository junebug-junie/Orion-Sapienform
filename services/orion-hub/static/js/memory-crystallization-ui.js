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

  function renderRow(item, onOpen) {
    const row = document.createElement("div");
    row.className = "flex justify-between gap-2 border border-gray-800 rounded px-2 py-1 bg-gray-900/60";
    row.innerHTML = `<div><div class="font-medium text-gray-100">${escapeHtml(item.subject || "")}</div>
      <div class="text-[10px] text-gray-500">${escapeHtml(item.kind || "")} · ${escapeHtml(item.status || "")} · salience ${escapeHtml(String(item.salience ?? ""))}</div></div>`;
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
            const links = await apiFetch(`/api/memory/crystallizations/${row.crystallization_id}/links`).catch(() => ({ items: [] }));
            const health = await apiFetch("/api/memory/crystallizations/projection/health").catch(() => ({}));
            detailEl.innerHTML = `<div class="space-y-2">
              <div><strong>${escapeHtml(row.subject)}</strong> <span class="text-gray-500">[${escapeHtml(row.kind)}]</span></div>
              <div>${escapeHtml(row.summary)}</div>
              <div class="text-gray-500">Status: ${escapeHtml(row.status)} · Confidence: ${escapeHtml(row.confidence)}</div>
              <div class="text-gray-500">Projection refs: cards=${(row.projection_refs && row.projection_refs.memory_card_ids || []).length}, chroma=${(row.projection_refs && row.projection_refs.chroma_doc_ids || []).length}</div>
              <div class="text-gray-500">Links: ${(links.items || []).length}</div>
              <div class="text-gray-500">Health: chroma=${escapeHtml(health.chroma_collection || "")}, graphiti=${health.graphiti_enabled ? "on" : "off"}</div>
              <div class="flex gap-2 mt-2">
                <button type="button" data-act="approve" class="px-2 py-1 rounded border border-emerald-700 text-emerald-200">Approve</button>
                <button type="button" data-act="reject" class="px-2 py-1 rounded border border-red-800 text-red-200">Reject</button>
                <button type="button" data-act="validate" class="px-2 py-1 rounded border border-gray-600 text-gray-200">Validate</button>
              </div>
            </div>`;
            detailEl.querySelectorAll("button[data-act]").forEach((btn) => {
              btn.addEventListener("click", async () => {
                const act = btn.getAttribute("data-act");
                try {
                  await apiFetch(`/api/memory/crystallizations/proposals/${row.crystallization_id}/${act}`, { method: "POST", body: act === "validate" ? undefined : "{}" });
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
