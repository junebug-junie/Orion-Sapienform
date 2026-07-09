/* Orion Hub — Memory cards operator UI (Phase 4) */
(function () {
  let memoryCyInstance = null;
  const pathSegments = window.location.pathname.split("/").filter((p) => p.length > 0);
  const URL_PREFIX = pathSegments.length > 0 ? `/${pathSegments[0]}` : "";
  const API_BASE = window.location.origin + URL_PREFIX;
  function memoryGraphSuggestFetchTimeoutMs() {
    const ui = window.OrionMemoryGraphDraftUI || {};
    if (typeof ui.resolveMemoryGraphSuggestFetchTimeoutMs === "function") {
      return ui.resolveMemoryGraphSuggestFetchTimeoutMs();
    }
    const raw = Number((window.__HUB_CFG__ || {}).memoryGraphSuggestFetchTimeoutMs);
    return Number.isFinite(raw) && raw > 0 ? raw : 205000;
  }
  const MEMORY_GRAPH_SUGGEST_INPUT_TOTAL_CHARS = 12000;

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

  function boundedSuggestPrompt(raw) {
    const text = String(raw || "");
    if (text.length <= MEMORY_GRAPH_SUGGEST_INPUT_TOTAL_CHARS) {
      return { value: text, clipped: false };
    }
    return {
      value: `${text.slice(0, MEMORY_GRAPH_SUGGEST_INPUT_TOTAL_CHARS)}…`,
      clipped: true,
    };
  }

  let lastApprovedCardIds = [];

  function styleSubviewButtons(btnReview, btnConsolidationDrafts, btnAll, btnLog, btnCrystallizations, activeKey) {
    const pairs = [
      ["review", btnReview],
      ["consolidation_drafts", btnConsolidationDrafts],
      ["all", btnAll],
      ["log", btnLog],
      ["crystallizations", btnCrystallizations],
    ];
    pairs.forEach(([key, btn]) => {
      if (!btn) return;
      const active = key === activeKey;
      btn.classList.toggle("border-indigo-500", active);
      btn.classList.toggle("bg-indigo-900/40", active);
      btn.classList.toggle("text-indigo-100", active);
      btn.classList.toggle("border-gray-600", !active);
      btn.classList.toggle("bg-gray-800", !active);
      btn.classList.toggle("text-gray-200", !active);
    });
  }

  function showSubview(review, consolidationDrafts, all, log, crystallizations, key) {
    const panels = [
      ["review", review],
      ["consolidation_drafts", consolidationDrafts],
      ["all", all],
      ["log", log],
      ["crystallizations", crystallizations],
    ];
    panels.forEach(([panelKey, panel]) => {
      if (!panel) return;
      const active = panelKey === key;
      panel.classList.toggle("hidden", !active);
      panel.classList.toggle("flex", active);
    });
    const target = { review, consolidation_drafts: consolidationDrafts, all, log, crystallizations }[key];
    if (target && typeof target.scrollIntoView === "function") {
      target.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function renderCardRow(card, onOpen) {
    const row = document.createElement("div");
    row.className = "flex justify-between gap-2 border border-gray-800 rounded px-2 py-1 bg-gray-900/60";
    row.innerHTML = `<div><div class="font-medium text-gray-100">${escapeHtml(card.title || "")}</div>
      <div class="text-[10px] text-gray-500">${escapeHtml(card.slug || "")} · ${escapeHtml(card.status || "")} · ${escapeHtml(card.priority || "episodic_detail")}</div></div>`;
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

  const MEMORY_CONFIDENCE = ["certain", "likely", "possible", "uncertain"];
  const MEMORY_SENSITIVITY = ["public", "private", "intimate"];
  const MEMORY_PRIORITY = ["always_inject", "high_recall", "episodic_detail", "archival"];
  const MEMORY_PROVENANCE = ["operator_highlight", "operator_distiller", "auto_extractor", "imported", "repo_compactor"];
  const MEMORY_VISIBILITY = ["chat", "social", "intimate", "all"];
  const MEMORY_TIME_KIND = ["timeless", "era_bound", "current", "expiring"];

  function selectField(label, id, options, value) {
    const opts = options
      .map((o) => `<option value="${escapeHtml(o)}"${o === value ? " selected" : ""}>${escapeHtml(o)}</option>`)
      .join("");
    return `<label class="block text-[10px] text-gray-500">${escapeHtml(label)}
      <select data-field="${escapeHtml(id)}" class="mt-0.5 w-full bg-gray-900 border border-gray-700 rounded px-1 py-0.5 text-[11px] text-gray-200">${opts}</select></label>`;
  }

  function textField(label, id, value, placeholder) {
    return `<label class="block text-[10px] text-gray-500">${escapeHtml(label)}
      <input data-field="${escapeHtml(id)}" type="text" value="${escapeHtml(value || "")}" placeholder="${escapeHtml(placeholder || "")}"
        class="mt-0.5 w-full bg-gray-900 border border-gray-700 rounded px-1 py-0.5 text-[11px] text-gray-200" /></label>`;
  }

  function textareaField(label, id, value, rows, placeholder) {
    return `<label class="block text-[10px] text-gray-500">${escapeHtml(label)}
      <textarea data-field="${escapeHtml(id)}" rows="${rows || 2}" placeholder="${escapeHtml(placeholder || "")}"
        class="mt-0.5 w-full bg-gray-900 border border-gray-700 rounded px-1 py-0.5 text-[11px] text-gray-200 font-mono">${escapeHtml(value || "")}</textarea></label>`;
  }

  function visibilityField(scopes) {
    const selected = new Set(Array.isArray(scopes) ? scopes : ["chat"]);
    const chips = MEMORY_VISIBILITY.map(
      (v) =>
        `<label class="inline-flex items-center gap-1 mr-2 text-[10px] text-gray-300"><input type="checkbox" data-vis="${escapeHtml(v)}"${
          selected.has(v) ? " checked" : ""
        } /> ${escapeHtml(v)}</label>`
    ).join("");
    return `<div class="block text-[10px] text-gray-500">Visibility scope<div class="mt-1 flex flex-wrap gap-1">${chips}</div></div>`;
  }

  function timeHorizonFields(th) {
    const obj = th && typeof th === "object" ? th : {};
    return `<div class="grid grid-cols-2 gap-2 border border-gray-800 rounded p-2">
      ${selectField("Time kind", "time_horizon.kind", MEMORY_TIME_KIND, obj.kind || "timeless")}
      ${textField("Start", "time_horizon.start", obj.start || "", "YYYY-MM-DD")}
      ${textField("End", "time_horizon.end", obj.end || "", "YYYY-MM-DD")}
      ${textField("As of", "time_horizon.as_of", obj.as_of || "", "YYYY-MM-DD")}
    </div>`;
  }

  function renderCardDetailShell(card, { showActions }) {
    const stillTrue = Array.isArray(card.still_true) ? card.still_true.join("\n") : "";
    const evidenceJson = JSON.stringify(Array.isArray(card.evidence) ? card.evidence : [], null, 2);
    const actions = showActions
      ? `<div class="flex flex-wrap gap-2 pt-1">
          <button type="button" data-act="save" class="px-2 py-1 bg-indigo-800 rounded text-[10px]">Save metadata</button>
          <button type="button" data-act="approve" class="px-2 py-1 bg-emerald-700 rounded text-[10px]">Approve</button>
          <button type="button" data-act="reject" class="px-2 py-1 bg-red-800 rounded text-[10px]">Reject</button>
        </div>`
      : `<div class="flex flex-wrap gap-2 pt-1">
          <button type="button" data-act="save" class="px-2 py-1 bg-indigo-800 rounded text-[10px]">Save metadata</button>
        </div>`;
    return `<div class="space-y-2" data-card-id="${escapeHtml(card.card_id)}">
      <div class="font-semibold text-gray-100">${escapeHtml(card.title || "")}</div>
      <div class="text-[10px] text-gray-500">${escapeHtml(card.slug || "")} · ${escapeHtml(card.status || "")} · ${escapeHtml(card.priority || "")}</div>
      <div class="grid grid-cols-2 gap-2">
        ${selectField("Confidence", "confidence", MEMORY_CONFIDENCE, card.confidence || "likely")}
        ${selectField("Sensitivity", "sensitivity", MEMORY_SENSITIVITY, card.sensitivity || "private")}
        ${selectField("Priority", "priority", MEMORY_PRIORITY, card.priority || "episodic_detail")}
        ${selectField("Provenance", "provenance", MEMORY_PROVENANCE, card.provenance || "operator_highlight")}
      </div>
      ${visibilityField(card.visibility_scope)}
      ${textareaField("Summary", "summary", card.summary || "", 3, "Card summary")}
      ${textareaField("Still true (one per line)", "still_true", stillTrue, 2, "bullet facts that remain true")}
      ${textareaField("Evidence (JSON array)", "evidence", evidenceJson, 4, '[{"source":"…","excerpt":"…"}]')}
      <div class="text-[10px] text-gray-500">Time horizon</div>
      ${timeHorizonFields(card.time_horizon)}
      ${actions}
    </div>`;
  }

  function readVisibilityScope(root) {
    const out = [];
    root.querySelectorAll("input[data-vis]").forEach((el) => {
      if (el.checked) out.push(el.getAttribute("data-vis"));
    });
    return out.length ? out : ["chat"];
  }

  function readPatchPayload(root) {
    const get = (name) => {
      const el = root.querySelector(`[data-field="${name}"]`);
      return el ? String(el.value || "").trim() : "";
    };
    let evidence = [];
    const evidenceRaw = get("evidence");
    if (evidenceRaw) {
      try {
        const parsed = JSON.parse(evidenceRaw);
        if (Array.isArray(parsed)) evidence = parsed;
      } catch (_) {
        throw new Error("Evidence must be valid JSON array");
      }
    }
    const stillLines = get("still_true")
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    const thKind = get("time_horizon.kind") || "timeless";
    const timeHorizon = {
      kind: thKind,
      start: get("time_horizon.start") || null,
      end: get("time_horizon.end") || null,
      as_of: get("time_horizon.as_of") || null,
    };
    return {
      confidence: get("confidence") || undefined,
      sensitivity: get("sensitivity") || undefined,
      priority: get("priority") || undefined,
      provenance: get("provenance") || undefined,
      visibility_scope: readVisibilityScope(root),
      summary: get("summary") || undefined,
      still_true: stillLines.length ? stillLines : [],
      evidence,
      time_horizon: timeHorizon,
    };
  }

  async function wireCardDetailPanel(detailEl, card, handlers) {
    detailEl.classList.remove("hidden");
    detailEl.innerHTML = renderCardDetailShell(card, { showActions: !!handlers.showReviewActions });
    const root = detailEl.firstElementChild;
    if (!root) return;

    async function saveMetadata() {
      const patch = readPatchPayload(root);
      const updated = await apiFetch(`/api/memory/cards/${encodeURIComponent(card.card_id)}`, {
        method: "PATCH",
        body: JSON.stringify(patch),
      });
      Object.assign(card, updated);
      if (handlers.onSaved) await handlers.onSaved(updated);
      return updated;
    }

    const saveBtn = root.querySelector('[data-act="save"]');
    if (saveBtn) {
      saveBtn.addEventListener("click", async () => {
        try {
          await saveMetadata();
          if (handlers.statusEl) setStatus(handlers.statusEl, "Saved card metadata.", false);
        } catch (e) {
          if (handlers.statusEl) setStatus(handlers.statusEl, e.message || formatMemoryApiError(e), true);
        }
      });
    }
    const approveBtn = root.querySelector('[data-act="approve"]');
    if (approveBtn) {
      approveBtn.addEventListener("click", async () => {
        try {
          await saveMetadata();
          await apiFetch(`/api/memory/cards/${encodeURIComponent(card.card_id)}/status`, {
            method: "POST",
            body: JSON.stringify({ status: "active" }),
          });
          if (handlers.onApprove) await handlers.onApprove();
        } catch (e) {
          if (handlers.statusEl) setStatus(handlers.statusEl, formatMemoryApiError(e), true);
        }
      });
    }
    const rejectBtn = root.querySelector('[data-act="reject"]');
    if (rejectBtn) {
      rejectBtn.addEventListener("click", async () => {
        try {
          await apiFetch(`/api/memory/cards/${encodeURIComponent(card.card_id)}/status`, {
            method: "POST",
            body: JSON.stringify({ status: "rejected" }),
          });
          if (handlers.onReject) await handlers.onReject();
        } catch (e) {
          if (handlers.statusEl) setStatus(handlers.statusEl, formatMemoryApiError(e), true);
        }
      });
    }
    if (handlers.cyHost) await loadNeighborhood(card.card_id, handlers.cyHost, card.title);
  }

  async function loadConsolidationDrafts(listEl, panelStatusEl, statusEl, draftTa, memoryDraftViz, memoryDraftForm, graphSetOut, onLoaded, onRejected) {
    if (!listEl) return;
    listEl.innerHTML = "";
    setStatus(panelStatusEl, "Loading consolidation drafts…", false);
    try {
      const data = await apiFetch("/api/memory/consolidation/drafts?status=pending_review&limit=50");
      const items = data.items || [];
      if (!items.length) {
        listEl.textContent =
          "No automated graph drafts awaiting review. Consolidation creates drafts when conversation windows close.";
      }
      items.forEach((item) => {
        const row = document.createElement("div");
        row.className = "flex flex-wrap justify-between gap-2 border border-gray-800 rounded px-2 py-2 bg-gray-900/60";
        const summary = item.summary || {};
        const created = item.created_at ? String(item.created_at).replace("T", " ").slice(0, 19) : "";
        const meta = document.createElement("div");
        meta.className = "min-w-0 flex-1 text-[11px]";
        meta.innerHTML = `<div class="font-medium text-gray-100">${escapeHtml(String(item.draft_id || "").slice(0, 8))}…</div>
          <div class="text-gray-500">${escapeHtml(created)} · ${Number(item.turn_count || 0)} turn(s) · e=${Number(summary.entities || 0)} s=${Number(summary.situations || 0)} ed=${Number(summary.edges || 0)}</div>`;
        const actions = document.createElement("div");
        actions.className = "flex flex-wrap gap-1 items-start";
        const loadBtn = document.createElement("button");
        loadBtn.type = "button";
        loadBtn.className = "px-2 py-0.5 rounded border border-indigo-700 bg-indigo-900/30 text-indigo-100 text-[10px]";
        loadBtn.textContent = "Load in editor";
        loadBtn.addEventListener("click", async () => {
          try {
            setStatus(panelStatusEl, "Loading draft…", false);
            const detail = await apiFetch(`/api/memory/consolidation/drafts/${encodeURIComponent(item.draft_id)}`);
            if (!draftTa) return;
            draftTa.value = JSON.stringify(detail.draft || {}, null, 2);
            if (memoryDraftViz && memoryDraftViz.refresh) memoryDraftViz.refresh();
            if (memoryDraftForm && memoryDraftForm.refresh) memoryDraftForm.refresh();
            if (typeof onLoaded === "function") onLoaded(String(item.draft_id || ""));
            graphSetOut({ ok: true, note: "Loaded consolidation draft into editor.", draft_id: item.draft_id }, false);
            setStatus(statusEl, `Loaded graph draft ${String(item.draft_id || "").slice(0, 8)}… — Validate / Approve above.`, false);
            setStatus(panelStatusEl, "Draft loaded in editor.", false);
            const annotator = document.getElementById("memoryGraphAnnotator");
            if (annotator && typeof annotator.scrollIntoView === "function") {
              annotator.scrollIntoView({ behavior: "smooth", block: "start" });
            }
          } catch (e) {
            setStatus(panelStatusEl, formatMemoryApiError(e), true);
          }
        });
        const rejectBtn = document.createElement("button");
        rejectBtn.type = "button";
        rejectBtn.className = "px-2 py-0.5 rounded border border-red-900 bg-red-950/40 text-red-200 text-[10px]";
        rejectBtn.textContent = "Reject";
        rejectBtn.addEventListener("click", async () => {
          try {
            await apiFetch(`/api/memory/consolidation/drafts/${encodeURIComponent(item.draft_id)}/status`, {
              method: "POST",
              body: JSON.stringify({ status: "rejected" }),
            });
            if (typeof onRejected === "function") onRejected(String(item.draft_id || ""));
            await loadConsolidationDrafts(
              listEl,
              panelStatusEl,
              statusEl,
              draftTa,
              memoryDraftViz,
              memoryDraftForm,
              graphSetOut,
              onLoaded,
              onRejected,
            );
          } catch (e) {
            setStatus(panelStatusEl, formatMemoryApiError(e), true);
          }
        });
        actions.appendChild(loadBtn);
        actions.appendChild(rejectBtn);
        row.appendChild(meta);
        row.appendChild(actions);
        listEl.appendChild(row);
      });
      setStatus(panelStatusEl, `Loaded ${items.length} graph draft(s).`, false);
    } catch (e) {
      setStatus(panelStatusEl, formatMemoryApiError(e), true);
    }
  }

  async function loadReview(reviewPanel, statusEl, detailEl, cyHost) {
    reviewPanel.innerHTML = "";
    setStatus(statusEl, "Loading review queue…", false);
    try {
      const data = await apiFetch("/api/memory/cards?status=pending_review&limit=100");
      const items = data.items || [];
      if (!items.length) {
        reviewPanel.textContent =
          "No cards awaiting manual review. Graph approve creates active situation cards directly — use Graph drafts for automated consolidation drafts, or All Cards to browse.";
      }
      items.forEach((c) => {
        reviewPanel.appendChild(
          renderCardRow(c, async (card) => {
            await wireCardDetailPanel(detailEl, card, {
              showReviewActions: true,
              statusEl,
              cyHost,
              onApprove: () => {
                detailEl.classList.add("hidden");
                detailEl.innerHTML = "";
                loadReview(reviewPanel, statusEl, detailEl, cyHost);
              },
              onReject: () => {
                detailEl.classList.add("hidden");
                detailEl.innerHTML = "";
                loadReview(reviewPanel, statusEl, detailEl, cyHost);
              },
            });
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
        <option value="">any</option><option value="pending_review">pending_review</option><option value="active" selected>active</option>
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
          await wireCardDetailPanel(detailEl, card, {
            showReviewActions: false,
            statusEl,
            cyHost,
          });
        })));
        setStatus(statusEl, `${items.length} card(s)`, false);
      } catch (e) {
        setStatus(statusEl, formatMemoryApiError(e), true);
      }
    }
    controls.querySelector("#memReloadAll").addEventListener("click", reload);
    await reload();
  }

  function formatHistoryWhen(raw) {
    const text = String(raw || "").trim();
    if (!text) return "";
    const d = new Date(text);
    if (Number.isNaN(d.getTime())) return text;
    return d.toLocaleString();
  }

  async function renderHistoryItems(items, out, statusEl, reloadFn) {
    out.innerHTML = "";
    if (!items.length) {
      out.textContent = "No activity yet. Approve a memory graph draft or create a card to populate history.";
      setStatus(statusEl, "No activity entries.", false);
      return;
    }
    items.forEach((h) => {
      const line = document.createElement("div");
      line.className = "border-b border-gray-800 pb-1";
      const canReverse = ["update", "status_change", "edge_add"].includes(h.op);
      const cardRef = h.card_id ? String(h.card_id).slice(0, 8) + "…" : "—";
      const when = formatHistoryWhen(h.created_at);
      line.innerHTML = `<div class="text-gray-300">${escapeHtml(h.op)} · ${escapeHtml(h.actor)} · card ${escapeHtml(
        cardRef
      )} · ${escapeHtml(when)}</div>
        <div class="text-[10px] text-gray-500">${escapeHtml(h.history_id)}</div>`;
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
            if (typeof reloadFn === "function") await reloadFn();
          } catch (err) {
            setStatus(statusEl, formatMemoryApiError(err), true);
          }
        });
        line.appendChild(b);
      }
      out.appendChild(line);
    });
    setStatus(statusEl, `${items.length} history entr${items.length === 1 ? "y" : "ies"}.`, false);
  }

  async function loadLog(logPanel, statusEl, options) {
    const opts = options && typeof options === "object" ? options : {};
    const prefilledIds = Array.isArray(opts.cardIds)
      ? opts.cardIds.map((id) => String(id || "").trim()).filter(Boolean)
      : [];
    logPanel.innerHTML = "";
    setStatus(
      statusEl,
      prefilledIds.length
        ? `Loaded ${prefilledIds.length} approved card id(s). History shown below — paste another card UUID to switch.`
        : "Paste a card UUID below, or approve a memory graph draft to auto-load history here.",
      false
    );
    const row = document.createElement("div");
    row.className = "flex gap-2 items-center";
    row.innerHTML = `<input id="memHistCardId" type="text" class="flex-1 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs" placeholder="card UUID" />
      <button type="button" id="memHistLoad" class="px-2 py-1 bg-gray-700 rounded border border-gray-600 text-xs">Load history</button>`;
    logPanel.appendChild(row);
    const out = document.createElement("div");
    out.className = "mt-2 space-y-1 text-[11px]";
    logPanel.appendChild(out);

    async function loadRecentHistory() {
      out.innerHTML = "";
      setStatus(statusEl, "Loading recent activity…", false);
      try {
        const data = await apiFetch("/api/memory/history?limit=100");
        await renderHistoryItems(data.items || [], out, statusEl, loadRecentHistory);
      } catch (e) {
        setStatus(statusEl, formatMemoryApiError(e), true);
      }
    }

    async function loadHistoryForCard(cid) {
      const cardId = String(cid || "").trim();
      if (!cardId) {
        await loadRecentHistory();
        return;
      }
      out.innerHTML = "";
      setStatus(statusEl, "Loading history…", false);
      try {
        const data = await apiFetch(`/api/memory/history?card_id=${encodeURIComponent(cardId)}&limit=100`);
        await renderHistoryItems(data.items || [], out, statusEl, () => loadHistoryForCard(cardId));
      } catch (e) {
        setStatus(statusEl, formatMemoryApiError(e), true);
      }
    }

    row.querySelector("#memHistLoad").addEventListener("click", () => {
      loadHistoryForCard(row.querySelector("#memHistCardId").value);
    });
    row.querySelector("#memHistCardId").addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        loadHistoryForCard(row.querySelector("#memHistCardId").value);
      }
    });

    const input = row.querySelector("#memHistCardId");
    const reloadAllBtn = document.createElement("button");
    reloadAllBtn.type = "button";
    reloadAllBtn.className = "px-2 py-1 bg-gray-800 rounded border border-gray-600 text-xs text-gray-300";
    reloadAllBtn.textContent = "Recent all";
    reloadAllBtn.addEventListener("click", () => {
      input.value = "";
      loadRecentHistory();
    });
    row.appendChild(reloadAllBtn);

    if (prefilledIds.length) {
      input.value = prefilledIds[0];
      await loadHistoryForCard(prefilledIds[0]);
      if (prefilledIds.length > 1) {
        const chips = document.createElement("div");
        chips.className = "mt-2 flex flex-wrap gap-1";
        prefilledIds.forEach((cid) => {
          const chip = document.createElement("button");
          chip.type = "button";
          chip.className = "px-2 py-0.5 rounded border border-gray-700 bg-gray-900 text-[10px] text-indigo-200 hover:bg-gray-800";
          chip.textContent = cid.slice(0, 8) + "…";
          chip.title = cid;
          chip.addEventListener("click", () => {
            input.value = cid;
            loadHistoryForCard(cid);
          });
          chips.appendChild(chip);
        });
        logPanel.insertBefore(chips, out);
      }
    } else {
      await loadRecentHistory();
    }
  }

  async function loadNeighborhood(cardId, cyHost, centerLabel) {
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
      const selfLabel = String(centerLabel || "this card").slice(0, 80);
      const nodes = [{ data: { id: String(cardId), label: selfLabel } }];
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
    const consolidationDrafts = document.getElementById("memoryConsolidationDraftsPanel");
    const all = document.getElementById("memoryAllPanel");
    const log = document.getElementById("memoryLogPanel");
    const crystallizations = document.getElementById("memoryCrystallizationPanel");
    const graphAnnotator = document.getElementById("memoryGraphAnnotator");
    const detail = document.getElementById("memoryDetail");
    const cyHost = document.getElementById("memoryCytoscapeHost");
    const btnR = document.getElementById("memorySubviewReview");
    const btnCD = document.getElementById("memorySubviewConsolidationDrafts");
    const btnA = document.getElementById("memorySubviewAll");
    const btnL = document.getElementById("memorySubviewLog");
    const btnC = document.getElementById("memorySubviewCrystallizations");
    const consolidationList = document.getElementById("memoryConsolidationDraftsList");
    const consolidationStatus = document.getElementById("memoryConsolidationDraftsStatus");
    if (!review || !all || !log) return;

    let activeConsolidationDraftId = null;

    const draftTa = document.getElementById("memoryGraphDraftJson");
    const graphOut = document.getElementById("memoryGraphAnnotatorOut");
    const vBtn = document.getElementById("memoryGraphValidateBtn");
    const aBtn = document.getElementById("memoryGraphApproveBtn");
    const sBtn = document.getElementById("memoryGraphSuggestBtn");
    let memoryDraftViz = null;
    let memoryDraftForm = null;
    if (window.OrionMemoryGraphDraftUI && draftTa) {
      const cyH = document.getElementById("memoryGraphDraftCyHost");
      const det = document.getElementById("memoryGraphDraftDetail");
      const ban = document.getElementById("memoryGraphDraftParseBanner");
      const formHost = document.getElementById("memoryGraphDraftFormHost");
      if (cyH && det) {
        memoryDraftViz = window.OrionMemoryGraphDraftUI.attach({
          draftTextarea: draftTa,
          cyHost: cyH,
          detailHost: det,
          bannerEl: ban,
          onDraftJsonChange: () => {
            if (memoryDraftForm && typeof memoryDraftForm.refresh === "function") {
              memoryDraftForm.refresh();
            }
          },
        });
      }
      if (formHost && window.OrionMemoryGraphDraftForm) {
        memoryDraftForm = window.OrionMemoryGraphDraftForm.attachFormEditor({
          draftTextarea: draftTa,
          formHost,
          onDraftChange: () => {
            if (memoryDraftViz && typeof memoryDraftViz.refresh === "function") {
              memoryDraftViz.refresh();
            }
          },
        });
      }
    }
    function flushDraftEditorToJson() {
      if (memoryDraftForm && typeof memoryDraftForm.flushToTextarea === "function") {
        memoryDraftForm.flushToTextarea();
      }
    }
    function graphSetOut(obj, isErr) {
      if (!graphOut) return;
      graphOut.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
      graphOut.classList.toggle("text-red-400", !!isErr);
      graphOut.classList.toggle("text-gray-400", !isErr);
    }

    function activateSubview(key) {
      if (graphAnnotator) {
        graphAnnotator.classList.toggle("hidden", key === "crystallizations");
      }
      if (key === "crystallizations") {
        if (detail) detail.classList.add("hidden");
        if (cyHost) cyHost.classList.add("hidden");
        setStatus(statusEl, "", false);
      }
      showSubview(review, consolidationDrafts, all, log, crystallizations, key);
      styleSubviewButtons(btnR, btnCD, btnA, btnL, btnC, key);
      if (key === "review") loadReview(review, statusEl, detail, cyHost);
      else if (key === "consolidation_drafts") {
        loadConsolidationDrafts(
          consolidationList,
          consolidationStatus,
          statusEl,
          draftTa,
          memoryDraftViz,
          memoryDraftForm,
          graphSetOut,
          (draftId) => {
            activeConsolidationDraftId = draftId;
          },
          (draftId) => {
            if (activeConsolidationDraftId === draftId) {
              activeConsolidationDraftId = null;
              if (draftTa) {
                draftTa.value = "";
                if (memoryDraftViz && memoryDraftViz.refresh) memoryDraftViz.refresh();
                if (memoryDraftForm && memoryDraftForm.refresh) memoryDraftForm.refresh();
              }
              graphSetOut({ ok: true, note: "Rejected consolidation draft cleared from editor." }, false);
              setStatus(statusEl, "Rejected graph draft cleared from editor.", false);
            }
          },
        );
      }
      else if (key === "all") loadAll(all, statusEl, detail, cyHost);
      else if (key === "log") loadLog(log, statusEl, { cardIds: lastApprovedCardIds });
      else if (key === "crystallizations" && window.OrionMemoryCrystallizationUI) {
        window.OrionMemoryCrystallizationUI.activate();
      }
    }

    if (btnR) btnR.addEventListener("click", () => activateSubview("review"));
    if (btnCD) btnCD.addEventListener("click", () => activateSubview("consolidation_drafts"));
    if (btnA) btnA.addEventListener("click", () => activateSubview("all"));
    if (btnL) btnL.addEventListener("click", () => activateSubview("log"));
    if (btnC) btnC.addEventListener("click", () => activateSubview("crystallizations"));
    styleSubviewButtons(btnR, btnCD, btnA, btnL, btnC, "all");
    activateSubview("all");

    if (vBtn && draftTa) {
      vBtn.addEventListener("click", async () => {
        graphSetOut("…", false);
        try {
          flushDraftEditorToJson();
          const raw = draftTa.value.trim();
          const draftBody = JSON.parse(raw);
          const validatePayload = { draft: draftBody };
          if (activeConsolidationDraftId) {
            validatePayload.consolidation_draft_id = activeConsolidationDraftId;
          }
          const data = await apiFetch("/api/memory/graph/validate", {
            method: "POST",
            body: JSON.stringify(validatePayload),
          });
          const out = { ...data };
          if (Array.isArray(data.warnings) && data.warnings.length) {
            out.topical_warnings = data.warnings;
          }
          graphSetOut(out, !data.ok);
        } catch (e) {
          graphSetOut(e.body || e.message || String(e), true);
        }
      });
    }
    if (aBtn && draftTa) {
      aBtn.addEventListener("click", async () => {
        graphSetOut("…", false);
        try {
          flushDraftEditorToJson();
          const raw = draftTa.value.trim();
          const draftBody = JSON.parse(raw);
          const approvePayload = { draft: draftBody };
          const approvedConsolidationDraftId = activeConsolidationDraftId;
          if (approvedConsolidationDraftId) {
            approvePayload.consolidation_draft_id = approvedConsolidationDraftId;
          }
          if (memoryDraftForm && typeof memoryDraftForm.buildCardProjectionPayload === "function") {
            approvePayload.card_projection_defaults = memoryDraftForm.buildCardProjectionPayload();
          }
          const data = await apiFetch("/api/memory/graph/approve", {
            method: "POST",
            body: JSON.stringify(approvePayload),
          });
          if (!data.ok) {
            graphSetOut(data, true);
            return;
          }
          const created = [];
          if (Array.isArray(data.card_ids) && data.card_ids.length) {
            lastApprovedCardIds = data.card_ids.map((id) => String(id));
            for (const cid of lastApprovedCardIds) {
              try {
                const card = await apiFetch(`/api/memory/cards/${encodeURIComponent(cid)}`);
                created.push({
                  card_id: cid,
                  title: card.title,
                  status: card.status,
                  types: card.types,
                });
              } catch {
                created.push({ card_id: cid, title: cid });
              }
            }
          }
          graphSetOut({ ok: true, created, card_ids: data.card_ids || [] }, false);
          const titles = created.map((c) => c.title).filter(Boolean);
          if (approvedConsolidationDraftId && data.consolidation_draft_marked === false) {
            setStatus(
              statusEl,
              "Graph approved but consolidation draft inbox was not updated — check logs.",
              true,
            );
          } else {
            setStatus(
              statusEl,
              titles.length
                ? `Approved ${titles.length} active card(s): ${titles.slice(0, 2).join(" · ")}${titles.length > 2 ? " …" : ""}`
                : "Graph approved.",
              false,
            );
          }
          if (approvedConsolidationDraftId) {
            activeConsolidationDraftId = null;
            activateSubview("consolidation_drafts");
          } else {
            activateSubview("all");
          }
        } catch (e) {
          graphSetOut(e.body || e.message || String(e), true);
        }
      });
    }
    if (sBtn && draftTa) {
      sBtn.addEventListener("click", async () => {
        activeConsolidationDraftId = null;
        graphSetOut("…", false);
        try {
          const raw =
            draftTa.value.trim() ||
            "Draft a structured memory-graph JSON object for this session (ontology_version, entities, situations, edges). Output only JSON.";
          const bounded = boundedSuggestPrompt(raw);
          const payload = {
            mode: "brain",
            verbs: ["memory_graph_suggest"],
            messages: [{ role: "user", content: bounded.value }],
            use_recall: false,
            no_write: true,
            diagnostic: true,
            options: { diagnostic: true },
          };
          const controller = typeof AbortController === "function" ? new AbortController() : null;
          const timerId =
            controller != null
              ? setTimeout(() => {
                  try {
                    controller.abort();
                  } catch (_) {
                    /* ignore */
                  }
                }, memoryGraphSuggestFetchTimeoutMs())
              : null;
          let data = null;
          try {
            data = await apiFetch("/api/memory/graph/suggest", {
              method: "POST",
              body: JSON.stringify(payload),
              ...(controller ? { signal: controller.signal } : {}),
            });
          } finally {
            if (timerId != null) clearTimeout(timerId);
          }
          const ui = window.OrionMemoryGraphDraftUI || {};
          const parseDraftFn =
            typeof ui.parseMemoryGraphDraftJson === "function" ? ui.parseMemoryGraphDraftJson : null;
          const priorParsed = parseDraftFn ? parseDraftFn(draftTa.value) : { ok: false, object: null };
          const priorObj =
            priorParsed.ok && priorParsed.object && typeof priorParsed.object === "object"
              ? priorParsed.object
              : null;
          const priorIds = priorObj && Array.isArray(priorObj.utterance_ids) ? priorObj.utterance_ids : [];
          const priorText =
            priorObj &&
            priorObj.utterance_text_by_id &&
            typeof priorObj.utterance_text_by_id === "object" &&
            !Array.isArray(priorObj.utterance_text_by_id)
              ? priorObj.utterance_text_by_id
              : {};
          const coalesceFn =
            typeof ui.coalesceMemoryGraphSuggestEnvelope === "function"
              ? ui.coalesceMemoryGraphSuggestEnvelope
              : null;
          const coalesce = coalesceFn
            ? coalesceFn(data, { utteranceIds: priorIds, utteranceTextById: priorText })
            : null;
          const emptyFn =
            typeof ui.emptySuggestDraft === "function"
              ? ui.emptySuggestDraft
              : typeof ui.emptyValidSuggestDraft === "function"
                ? ui.emptyValidSuggestDraft
                : null;
          const draftText =
            coalesce && typeof coalesce.draftText === "string" && coalesce.draftText.trim()
              ? coalesce.draftText
              : JSON.stringify(
                  emptyFn
                    ? emptyFn({ utteranceIds: priorIds, utteranceTextById: priorText })
                    : {
                        ontology_version: "orionmem-2026-05",
                        utterance_ids: [],
                        entities: [],
                        situations: [],
                        edges: [],
                        dispositions: [],
                        utterance_text_by_id: {},
                      },
                  null,
                  2,
                );
          draftTa.value = draftText;
          if (memoryDraftViz && memoryDraftViz.refresh) memoryDraftViz.refresh();
          if (memoryDraftForm && memoryDraftForm.refresh) memoryDraftForm.refresh();
          const statusFn =
            typeof ui.formatSuggestCoalesceUserStatus === "function"
              ? ui.formatSuggestCoalesceUserStatus
              : null;
          const statusLine = statusFn
            ? statusFn(coalesce)
            : coalesce && coalesce.error
              ? "Extractor did not return a valid role-grounded SuggestDraftV1. Empty valid fallback draft loaded; see diagnostics."
              : "Loaded validated role-grounded SuggestDraftV1 JSON.";
          if (coalesce && coalesce.error) {
            graphSetOut(
              {
                ok: false,
                error: coalesce.error,
                diagnostics: coalesce.diagnostics || null,
                note: bounded.clipped
                  ? `${statusLine} Prompt was clipped to ${MEMORY_GRAPH_SUGGEST_INPUT_TOTAL_CHARS} chars.`
                  : statusLine,
              },
              true,
            );
            return;
          }
          graphSetOut(
            {
              ok: true,
              route_used: coalesce && coalesce.diagnostics ? coalesce.diagnostics.route_used : null,
              attempts: coalesce && coalesce.diagnostics ? coalesce.diagnostics.attempts : [],
              validation_errors:
                coalesce && coalesce.diagnostics ? coalesce.diagnostics.validation_errors : [],
              diagnostics: coalesce ? coalesce.diagnostics : null,
              note: bounded.clipped
                ? `${statusLine} Prompt input was clipped to ${MEMORY_GRAPH_SUGGEST_INPUT_TOTAL_CHARS} chars.`
                : statusLine,
            },
            false,
          );
        } catch (e) {
          if (e && e.name === "AbortError") {
            graphSetOut(
              {
                ok: false,
                error: `Suggest timed out after ${Math.round(
                  memoryGraphSuggestFetchTimeoutMs() / 1000
                )}s (browser fetch limit; server budget is typically ~180s with Quick+Brain escalation). Retry when the model gateway is responsive; reduce selected turns only if the prompt was clipped.`,
              },
              true
            );
            return;
          }
          graphSetOut(e.message || String(e), true);
        }
      });
    }

    function refreshVisibleMemorySubview() {
      if (crystallizations && !crystallizations.classList.contains("hidden")) {
        activateSubview("crystallizations");
      } else if (review && !review.classList.contains("hidden")) {
        activateSubview("review");
      } else if (all && !all.classList.contains("hidden")) {
        activateSubview("all");
      } else if (log && !log.classList.contains("hidden")) {
        activateSubview("log");
      }
    }
    window.addEventListener("orion-hub-memory-tab-activated", refreshVisibleMemorySubview);
    window.addEventListener("orion-hub-memory-graph-draft-import", () => {
      const v = sessionStorage.getItem("orion_memory_graph_draft_import");
      sessionStorage.removeItem("orion_memory_graph_draft_import");
      if (!v || !draftTa) return;
      const ui = window.OrionMemoryGraphDraftUI || {};
      const parseFn = typeof ui.parseMemoryGraphDraftJson === "function" ? ui.parseMemoryGraphDraftJson : null;
      const looksFn =
        typeof ui.looksLikeMemoryGraphDraftObject === "function" ? ui.looksLikeMemoryGraphDraftObject : null;
      const evidenceFn =
        typeof ui.looksLikeEvidenceEnvelopeOnly === "function" ? ui.looksLikeEvidenceEnvelopeOnly : null;
      const emptyFn =
        typeof ui.emptySuggestDraft === "function"
          ? ui.emptySuggestDraft
          : typeof ui.emptyValidSuggestDraft === "function"
            ? ui.emptyValidSuggestDraft
            : null;
      const parsed = parseFn ? parseFn(v) : { ok: false, object: null };
      const obj = parsed.ok && parsed.object ? parsed.object : null;
      const isEvidence = obj && evidenceFn && evidenceFn(obj);
      const isDraft = obj && looksFn && looksFn(obj);
      if (isDraft && !isEvidence) {
        draftTa.value = JSON.stringify(obj, null, 2);
        if (memoryDraftViz && memoryDraftViz.refresh) memoryDraftViz.refresh();
        if (memoryDraftForm && memoryDraftForm.refresh) memoryDraftForm.refresh();
        const cardDefaultsRaw = sessionStorage.getItem("orion_memory_graph_card_defaults_import");
        sessionStorage.removeItem("orion_memory_graph_card_defaults_import");
        if (cardDefaultsRaw && memoryDraftForm && memoryDraftForm.setCardDefaults) {
          try {
            memoryDraftForm.setCardDefaults(JSON.parse(cardDefaultsRaw));
            memoryDraftForm.refresh();
          } catch (_) {
            /* ignore */
          }
        }
        graphSetOut(
          { ok: true, note: "Loaded validated role-grounded SuggestDraftV1 JSON." },
          false,
        );
        return;
      }
      const preservedIds = obj && Array.isArray(obj.utterance_ids) ? obj.utterance_ids : [];
      const preservedText =
        obj &&
        obj.utterance_text_by_id &&
        typeof obj.utterance_text_by_id === "object" &&
        !Array.isArray(obj.utterance_text_by_id)
          ? obj.utterance_text_by_id
          : {};
      draftTa.value = JSON.stringify(
        emptyFn
          ? emptyFn({ utteranceIds: preservedIds, utteranceTextById: preservedText })
          : {
              ontology_version: "orionmem-2026-05",
              utterance_ids: preservedIds,
              entities: [],
              situations: [],
              edges: [],
              dispositions: [],
              utterance_text_by_id: preservedText,
            },
        null,
        2,
      );
      if (memoryDraftViz && memoryDraftViz.refresh) memoryDraftViz.refresh();
      if (memoryDraftForm && memoryDraftForm.refresh) memoryDraftForm.refresh();
      graphSetOut(
        {
          ok: false,
          error: "invalid_import_not_suggest_draft_v1",
          note: isEvidence
            ? "Blocked selected-turn evidence envelope from Draft JSON; evidence is request input, not graph output."
            : "Extractor did not return a valid role-grounded SuggestDraftV1. Empty valid fallback draft loaded; see diagnostics.",
        },
        true,
      );
    });
  });
})();
