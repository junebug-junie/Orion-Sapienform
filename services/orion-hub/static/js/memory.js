/* Orion Hub — Memory cards operator UI (Phase 4) */
(function () {
  let memoryCyInstance = null;
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

  function setStatus(el, msg, isErr) {
    if (!el) return;
    el.textContent = msg || "";
    el.classList.toggle("text-red-400", !!isErr);
    el.classList.toggle("text-gray-400", !isErr);
  }

  function formatMemoryApiError(err) {
    const detail = err && err.body && err.body.detail;
    if (detail === "memory_schema_missing" || (err && err.status === 503 && detail === "memory_schema_missing")) {
      return "Memory tables are missing or DDL failed. Restart Hub after Postgres is up (Hub applies schema on startup), or check DB permissions (e.g. pgcrypto).";
    }
    if (detail === "memory_store_unavailable" || (err && err.status === 503 && detail === "memory_store_unavailable")) {
      return "Memory store is unavailable. Configure RECALL_PG_DSN on Hub and restart (see the notice under “Memory cards”).";
    }
    if (typeof detail === "string") return detail;
    return (err && err.message) || "Request failed";
  }

  function showSubview(review, all, log, key) {
    [review, all, log].forEach((p) => p && p.classList.add("hidden"));
    const map = { review, all: all, log };
    const target = map[key];
    if (target) target.classList.remove("hidden");
  }

  function renderCardRow(card, onOpen) {
    const row = document.createElement("div");
    row.className = "flex justify-between gap-2 border border-gray-800 rounded px-2 py-1 bg-gray-900/60";
    row.innerHTML = `<div><div class="font-medium text-gray-100">${escapeHtml(card.title || "")}</div>
      <div class="text-[10px] text-gray-500">${escapeHtml(card.slug || "")} · ${escapeHtml(card.status || "")}</div></div>`;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "text-[10px] text-indigo-300 hover:text-indigo-200";
    btn.textContent = "Open";
    btn.addEventListener("click", () => onOpen(card));
    row.appendChild(btn);
    return row;
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function loadReview(reviewPanel, statusEl, detailEl, cyHost) {
    reviewPanel.innerHTML = "";
    setStatus(statusEl, "Loading review queue…", false);
    try {
      const data = await apiFetch("/api/memory/cards?status=pending_review&limit=100");
      const items = data.items || [];
      if (!items.length) {
        reviewPanel.textContent = "No cards in pending_review.";
      }
      items.forEach((c) => {
        reviewPanel.appendChild(
          renderCardRow(c, async (card) => {
            detailEl.classList.remove("hidden");
            detailEl.innerHTML = `<div class="space-y-2"><div class="font-semibold">${escapeHtml(card.title)}</div>
              <pre class="whitespace-pre-wrap text-[10px]">${escapeHtml(JSON.stringify(card, null, 2))}</pre>
              <div class="flex gap-2"><button type="button" data-act="approve" class="px-2 py-1 bg-emerald-700 rounded text-[10px]">Approve</button>
              <button type="button" data-act="reject" class="px-2 py-1 bg-red-800 rounded text-[10px]">Reject</button></div></div>`;
            const approve = detailEl.querySelector('[data-act="approve"]');
            const reject = detailEl.querySelector('[data-act="reject"]');
            approve.addEventListener("click", async () => {
              await apiFetch(`/api/memory/cards/${encodeURIComponent(card.card_id)}/status`, {
                method: "POST",
                body: JSON.stringify({ status: "active" }),
              });
              loadReview(reviewPanel, statusEl, detailEl, cyHost);
            });
            reject.addEventListener("click", async () => {
              await apiFetch(`/api/memory/cards/${encodeURIComponent(card.card_id)}/status`, {
                method: "POST",
                body: JSON.stringify({ status: "rejected" }),
              });
              loadReview(reviewPanel, statusEl, detailEl, cyHost);
            });
            await loadNeighborhood(card.card_id, cyHost);
          })
        );
      });
      setStatus(statusEl, `Loaded ${items.length} card(s).`, false);
    } catch (e) {
      setStatus(statusEl, formatMemoryApiError(e), true);
    }
  }

  async function loadAll(allPanel, statusEl, detailEl, cyHost) {
    allPanel.innerHTML = "";
    const controls = document.createElement("div");
    controls.className = "flex flex-wrap gap-2 items-center text-[11px]";
    controls.innerHTML = `<label class="text-gray-400">Status <select id="memFilterStatus" class="bg-gray-800 border border-gray-600 rounded px-1">
        <option value="">any</option><option value="pending_review">pending_review</option><option value="active">active</option>
        <option value="rejected">rejected</option></select></label>
        <button type="button" id="memReloadAll" class="px-2 py-1 bg-gray-700 rounded border border-gray-600">Reload</button>`;
    allPanel.appendChild(controls);
    const list = document.createElement("div");
    list.className = "flex flex-col gap-1 mt-2";
    allPanel.appendChild(list);

    async function reload() {
      list.innerHTML = "";
      const st = allPanel.querySelector("#memFilterStatus")?.value || "";
      const qs = new URLSearchParams();
      if (st) qs.set("status", st);
      qs.set("limit", "100");
      setStatus(statusEl, "Loading…", false);
      try {
        const data = await apiFetch(`/api/memory/cards?${qs.toString()}`);
        const items = data.items || [];
        items.forEach((c) => list.appendChild(renderCardRow(c, async (card) => {
          detailEl.classList.remove("hidden");
          detailEl.innerHTML = `<pre class="whitespace-pre-wrap text-[10px]">${escapeHtml(JSON.stringify(card, null, 2))}</pre>`;
          await loadNeighborhood(card.card_id, cyHost);
        })));
        setStatus(statusEl, `${items.length} card(s)`, false);
      } catch (e) {
        setStatus(statusEl, formatMemoryApiError(e), true);
      }
    }
    controls.querySelector("#memReloadAll").addEventListener("click", reload);
    await reload();
  }

  async function loadLog(logPanel, statusEl) {
    logPanel.innerHTML = "";
    setStatus(statusEl, "Select a card in Review or All, then use API for filtered history — or paste card_id:", false);
    const row = document.createElement("div");
    row.className = "flex gap-2 items-center";
    row.innerHTML = `<input id="memHistCardId" type="text" class="flex-1 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs" placeholder="card UUID" />
      <button type="button" id="memHistLoad" class="px-2 py-1 bg-gray-700 rounded border border-gray-600 text-xs">Load history</button>`;
    logPanel.appendChild(row);
    const out = document.createElement("div");
    out.className = "mt-2 space-y-1 text-[11px]";
    logPanel.appendChild(out);
    row.querySelector("#memHistLoad").addEventListener("click", async () => {
      const cid = row.querySelector("#memHistCardId").value.trim();
      if (!cid) return;
      out.innerHTML = "";
      try {
        const data = await apiFetch(`/api/memory/history?card_id=${encodeURIComponent(cid)}&limit=100`);
        const items = data.items || [];
        items.forEach((h) => {
          const line = document.createElement("div");
          line.className = "border-b border-gray-800 pb-1";
          const canReverse = ["update", "status_change", "edge_add"].includes(h.op);
          line.innerHTML = `<div class="text-gray-300">${escapeHtml(h.op)} · ${escapeHtml(h.actor)} · ${escapeHtml(
            h.history_id
          )}</div>`;
          if (canReverse) {
            const b = document.createElement("button");
            b.type = "button";
            b.className = "text-[10px] text-amber-300 mt-1";
            b.textContent = "Reverse this";
            b.addEventListener("click", async () => {
              try {
                await apiFetch(`/api/memory/history/${encodeURIComponent(h.history_id)}/reverse`, {
                  method: "POST",
                  body: JSON.stringify({}),
                });
                row.querySelector("#memHistLoad").click();
              } catch (err) {
                setStatus(statusEl, formatMemoryApiError(err), true);
              }
            });
            line.appendChild(b);
          }
          out.appendChild(line);
        });
      } catch (e) {
        setStatus(statusEl, formatMemoryApiError(e), true);
      }
    });
  }

  async function loadNeighborhood(cardId, cyHost) {
    if (!window.cytoscape || !cyHost) return;
    try {
      const nb = await apiFetch(`/api/memory/cards/${encodeURIComponent(cardId)}/neighborhood?hops=1`);
      cyHost.classList.remove("hidden");
      if (memoryCyInstance) {
        try {
          memoryCyInstance.destroy();
        } catch {
          /* ignore */
        }
        memoryCyInstance = null;
      }
      cyHost.innerHTML = "";
      const nodes = [{ data: { id: String(cardId), label: "self" } }];
      const edges = [];
      const by = nb.by_edge_type || {};
      Object.keys(by).forEach((et) => {
        (by[et] || []).forEach((n, i) => {
          const nid = n.card_id;
          nodes.push({ data: { id: nid, label: n.title || nid } });
          edges.push({ data: { id: `${et}-${nid}-${i}`, source: String(cardId), target: nid, label: et } });
        });
      });
      memoryCyInstance = window.cytoscape({
        container: cyHost,
        elements: { nodes, edges },
        style: [
          { selector: "node", style: { label: "data(label)", "font-size": 8, color: "#cbd5e1", "background-color": "#334155" } },
          { selector: "edge", style: { width: 1, "line-color": "#64748b", "target-arrow-color": "#64748b", "target-arrow-shape": "triangle", label: "data(label)", "font-size": 6, color: "#94a3b8" } },
        ],
        layout: { name: "cose" },
      });
    } catch {
      cyHost.classList.add("hidden");
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const statusEl = document.getElementById("memoryStatus");
    const review = document.getElementById("memoryReviewPanel");
    const all = document.getElementById("memoryAllPanel");
    const log = document.getElementById("memoryLogPanel");
    const detail = document.getElementById("memoryDetail");
    const cyHost = document.getElementById("memoryCytoscapeHost");
    const btnR = document.getElementById("memorySubviewReview");
    const btnA = document.getElementById("memorySubviewAll");
    const btnL = document.getElementById("memorySubviewLog");
    if (!review || !all || !log) return;

    btnR.addEventListener("click", () => {
      showSubview(review, all, log, "review");
      loadReview(review, statusEl, detail, cyHost);
    });
    btnA.addEventListener("click", () => {
      showSubview(review, all, log, "all");
      loadAll(all, statusEl, detail, cyHost);
    });
    btnL.addEventListener("click", () => {
      showSubview(review, all, log, "log");
      loadLog(log, statusEl);
    });

    const draftTa = document.getElementById("memoryGraphDraftJson");
    const graphOut = document.getElementById("memoryGraphAnnotatorOut");
    const vBtn = document.getElementById("memoryGraphValidateBtn");
    const aBtn = document.getElementById("memoryGraphApproveBtn");
    const sBtn = document.getElementById("memoryGraphSuggestBtn");
    function graphSetOut(obj, isErr) {
      if (!graphOut) return;
      graphOut.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
      graphOut.classList.toggle("text-red-400", !!isErr);
      graphOut.classList.toggle("text-gray-400", !isErr);
    }
    if (vBtn && draftTa) {
      vBtn.addEventListener("click", async () => {
        graphSetOut("…", false);
        try {
          const raw = draftTa.value.trim();
          const body = JSON.parse(raw);
          const data = await apiFetch("/api/memory/graph/validate", { method: "POST", body: JSON.stringify(body) });
          graphSetOut(data, !data.ok);
        } catch (e) {
          graphSetOut(e.body || e.message || String(e), true);
        }
      });
    }
    if (aBtn && draftTa) {
      aBtn.addEventListener("click", async () => {
        graphSetOut("…", false);
        try {
          const raw = draftTa.value.trim();
          const body = JSON.parse(raw);
          const data = await apiFetch("/api/memory/graph/approve", { method: "POST", body: JSON.stringify(body) });
          graphSetOut(data, !data.ok);
        } catch (e) {
          graphSetOut(e.body || e.message || String(e), true);
        }
      });
    }
    if (sBtn && draftTa) {
      sBtn.addEventListener("click", async () => {
        graphSetOut("…", false);
        try {
          const raw =
            draftTa.value.trim() ||
            "Draft a structured memory-graph JSON object for this session (ontology_version, entities, situations, edges). Output only JSON.";
          const headers = { "Content-Type": "application/json", ...sessionHeader() };
          const payload = {
            mode: "brain",
            verbs: ["memory_graph_suggest"],
            messages: [{ role: "user", content: raw }],
            use_recall: false,
            no_write: true,
          };
          const res = await fetch(`${API_BASE}/api/chat`, { method: "POST", headers, body: JSON.stringify(payload) });
          const text = await res.text();
          let data = null;
          try {
            data = text ? JSON.parse(text) : null;
          } catch {
            data = { raw: text };
          }
          if (!res.ok) {
            graphSetOut(data || text, true);
            return;
          }
          const t = (data && (data.text || (data.raw && data.raw.final_text))) || text;
          draftTa.value = typeof t === "string" ? t : JSON.stringify(t, null, 2);
          graphSetOut(
            { ok: true, note: "Replaced the box with the model reply. If prose wrapped the JSON, delete the wrapper lines and use Validate." },
            false
          );
        } catch (e) {
          graphSetOut(e.message || String(e), true);
        }
      });
    }

    function refreshVisibleMemorySubview() {
      if (review && !review.classList.contains("hidden")) {
        loadReview(review, statusEl, detail, cyHost);
      } else if (all && !all.classList.contains("hidden")) {
        loadAll(all, statusEl, detail, cyHost);
      } else if (log && !log.classList.contains("hidden")) {
        loadLog(log, statusEl);
      }
    }
    window.addEventListener("orion-hub-memory-tab-activated", refreshVisibleMemorySubview);
    window.addEventListener("orion-hub-memory-graph-draft-import", () => {
      const v = sessionStorage.getItem("orion_memory_graph_draft_import");
      if (v && draftTa) draftTa.value = v;
      sessionStorage.removeItem("orion_memory_graph_draft_import");
      graphSetOut(
        { ok: true, note: "Draft loaded from chat bridge. Review JSON then Validate." },
        false
      );
    });
  });
})();
