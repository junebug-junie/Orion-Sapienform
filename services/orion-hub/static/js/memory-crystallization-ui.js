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
      // Include the server's own detail in the message, not just the status.
      // setStatus renders e.message, so a bare "HTTP 400" told the operator
      // nothing -- e.g. the bulk endpoint's "too_many_ids_max_500" was invisible.
      const detail = body && (body.detail || body.raw);
      const err = new Error(detail ? `HTTP ${res.status}: ${JSON.stringify(detail)}` : `HTTP ${res.status}`);
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

  // escapeHtml is safe for text nodes but leaves quotes intact, which would
  // break out of an attribute value. source_id is a free-text column, so
  // anything interpolated into an attribute goes through this instead.
  function escapeAttr(s) {
    return escapeHtml(s).replace(/"/g, "&quot;").replace(/'/g, "&#39;");
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

  function renderEvidence(evidence, editable) {
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
        // The last remaining turn has no drop control: the server rejects
        // removing it (a proposal with zero evidence is not reviewable), so
        // offering a button that always 409s would be worse than offering none.
        const canDrop = editable && chatTurns.length > 1;
        parts.push(`<div class="border border-gray-800 rounded p-2 mt-1 bg-gray-950/40">
          <div class="flex justify-between gap-2 items-start">
            <div class="text-[10px] text-gray-500">Turn ${idx + 1} · <code>${escapeHtml(ev.source_id || "")}</code></div>
            ${canDrop ? `<button type="button" data-drop-turn="${escapeAttr(ev.source_id || "")}" title="Remove this turn from the proposal" class="text-[10px] px-1 rounded border border-gray-700 text-gray-400 hover:text-red-300 hover:border-red-800">drop</button>` : ""}
          </div>
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

  function platformBadge(item) {
    const provenance = item && typeof item.provenance === "object" ? item.provenance : {};
    const platform = provenance && provenance.source_platform;
    if (!platform) return "";
    return `<span class="text-[9px] border rounded px-1 py-0.5 bg-sky-900/40 text-sky-300 border-sky-700 ml-1">${escapeHtml(platform)}</span>`;
  }

  function retirementBadge(item) {
    if (!item || !item.retirement_candidate) return "";
    return `<span class="text-[9px] border rounded px-1 py-0.5 bg-amber-900/40 text-amber-300 border-amber-700 ml-1">stale — review for archive</span>`;
  }

  function renderDetail(row, links, health) {
    const dyn = row.dynamics && typeof row.dynamics === "object" ? row.dynamics : {};
    const planning = Array.isArray(row.planning_effects) ? row.planning_effects : [];
    const retrieval = Array.isArray(row.retrieval_affordances) ? row.retrieval_affordances : [];
    const turnCount = chatTurnCount(row);
    const isActive = row.status === "active";
    const decayedActivation = row.decayed_activation != null ? Number(row.decayed_activation).toFixed(3) : "—";
    // Actions live at the TOP and stick there. They used to be the last element
    // of a max-h-72 overflow-auto pane, which meant scrolling to the bottom of
    // the detail view for every single decision and then scrolling back up to
    // reach the next item.
    const actions = `<div class="flex gap-2 sticky top-0 bg-gray-900 py-1 -mt-1 z-10 border-b border-gray-800">
        ${!isActive ? `<button type="button" data-act="approve" class="px-2 py-1 rounded border border-emerald-700 text-emerald-200">Approve</button>` : ""}
        ${!isActive ? `<button type="button" data-act="reject" class="px-2 py-1 rounded border border-red-800 text-red-200">Reject</button>` : ""}
        ${!isActive ? `<button type="button" data-act="validate" class="px-2 py-1 rounded border border-gray-600 text-gray-200">Validate</button>` : ""}
        <button type="button" data-act="sync-graphiti" class="px-2 py-1 rounded border border-sky-700 text-sky-200">Sync Graphiti</button>
        ${isActive ? `<button type="button" data-act="deprecate" class="px-2 py-1 rounded border border-amber-700 text-amber-200">Deprecate</button>` : ""}
      </div>`;
    return `<div class="space-y-2">
      ${actions}
      <div><strong>${escapeHtml(row.subject)}</strong> <span class="text-gray-500">[${escapeHtml(row.kind)}]</span>${platformBadge(row)}${retirementBadge(row)}</div>
      <div>${escapeHtml(row.summary)}</div>
      <div class="text-gray-500">Status: ${escapeHtml(row.status)} · Confidence: ${escapeHtml(row.confidence)} · Salience: ${escapeHtml(String(row.salience ?? ""))}</div>
      <div class="text-gray-500">Activation: ${escapeHtml(String(dyn.activation ?? "0"))} · Decayed activation: ${decayedActivation} · Reinforcements: ${escapeHtml(String(dyn.reinforcement_count ?? "0"))}</div>
      <div class="text-gray-500">Source turns in window: ${turnCount}</div>
      <div class="border border-gray-800 rounded p-2">${renderProvenance(row.provenance)}</div>
      ${planning.length ? `<div><span class="text-gray-500">Planning:</span><ul class="list-disc ml-4">${planning.map((p) => `<li>${escapeHtml(p)}</li>`).join("")}</ul></div>` : ""}
      ${retrieval.length ? `<div><span class="text-gray-500">Retrieval:</span> ${escapeHtml(retrieval.join(", "))}</div>` : ""}
      <div class="border border-gray-800 rounded p-2 max-h-48 overflow-y-auto">${renderEvidence(row.evidence, !isActive)}</div>
      <div class="text-gray-500">Projection refs: cards=${(row.projection_refs && row.projection_refs.memory_card_ids || []).length}, chroma=${(row.projection_refs && row.projection_refs.chroma_doc_ids || []).length}, graphiti_eps=${((row.projection_refs && row.projection_refs.graphiti_episode_ids) || []).length}, graphiti_edges=${((row.projection_refs && row.projection_refs.graphiti_edge_ids) || []).length}</div>
      <div class="text-gray-500">Links: ${(links.items || []).length}</div>
      <div class="text-gray-500">Health: chroma=${escapeHtml(health.chroma_collection || "")}, graphiti=${health.graphiti_enabled ? "on" : "off"}</div>
    </div>`;
  }

  function renderRow(item, onOpen, onToggle, checked) {
    const row = document.createElement("div");
    row.className =
      "flex items-start gap-2 border border-gray-800 rounded px-2 py-1 bg-gray-900/60 cursor-pointer hover:border-gray-600";
    const turns = chatTurnCount(item);

    if (isDecidable(item)) {
      const box = document.createElement("input");
      box.type = "checkbox";
      box.className = "mt-1 shrink-0 cursor-pointer";
      box.checked = !!checked;
      box.dataset.selectId = item.crystallization_id;
      box.setAttribute("aria-label", `select ${item.subject || item.crystallization_id}`);
      // Selection and opening are separate intents on the same row: clicking the
      // box must not also swap the detail pane out from under a bulk selection.
      box.addEventListener("click", (ev) => ev.stopPropagation());
      box.addEventListener("change", () => onToggle(item, box.checked));
      row.appendChild(box);
    } else {
      // Keeps the rows aligned without offering a control the server refuses.
      const spacer = document.createElement("span");
      spacer.className = "w-3 shrink-0";
      row.appendChild(spacer);
    }

    const body = document.createElement("div");
    body.className = "flex-1 min-w-0";
    body.innerHTML = `<div class="font-medium text-gray-100 truncate">${escapeHtml(item.subject || "")}${platformBadge(item)}${retirementBadge(item)}</div>
      <div class="text-[10px] text-gray-500">${escapeHtml(item.kind || "")} · ${escapeHtml(item.status || "")} · salience ${escapeHtml(String(item.salience ?? ""))}${turns ? ` · ${turns} turn(s)` : ""}</div>`;
    row.appendChild(body);

    const hint = document.createElement("span");
    hint.className = "text-indigo-300 text-xs shrink-0 self-center";
    hint.textContent = "Open";
    row.appendChild(hint);

    // The whole row is the open affordance now -- the old build put the only
    // handler on the "Open" label at the far right, which meant crossing the
    // full width of the panel for every single item.
    row.addEventListener("click", () => onOpen(item));
    return row;
  }

  async function loadRetirementCandidates() {
    // Best-effort: retirement surfacing (docs/superpowers/specs/2026-07-13-recall-
    // followups-loop-retirement-saturation-gate-spec.md section 2) augments the review
    // queue but must never break proposal-inbox loading if it fails.
    try {
      const data = await apiFetch("/api/memory/crystallizations?status=active");
      return ((data && data.items) || []).filter((item) => item && item.retirement_candidate);
    } catch {
      return [];
    }
  }

  // Survives across loadInbox() re-renders so a bulk selection is not silently
  // dropped by a background refresh. Pruned to what still exists on every load.
  const selected = new Set();

  // Server cap is BULK_DECIDE_MAX (500) per request. Approve is chunked far
  // smaller because each approved item also runs a card/chroma projection and
  // a second write, so 500 of them in one request is minutes of serialized I/O
  // and the browser or proxy gives up while the server keeps going.
  const REJECT_CHUNK = 200;
  const APPROVE_CHUNK = 25;

  // Only proposals can be bulk-decided. Retirement candidates are status
  // "active"; the bulk endpoint answers already_active for each, so including
  // them in select-all guarantees "N failed" on every sweep.
  function isDecidable(item) {
    return item && (item.status === "proposed" || item.status === "quarantined");
  }

  function closeDetail(detailEl) {
    // The old build re-rendered the list after a decision but left the detail
    // pane showing the item that had just been approved/rejected, with its
    // buttons still bound to that id. Clicking Approve there hit a proposal
    // whose status was no longer "proposed" and 404'd -- the "it gets pissed at
    // the next one" behavior. A decision now always closes its own pane.
    detailEl.innerHTML = "";
    detailEl.classList.add("hidden");
    detailEl.dataset.crystallizationId = "";
  }

  async function openDetail(row, listEl, statusEl, detailEl, summarize) {
    detailEl.classList.remove("hidden");
    detailEl.dataset.crystallizationId = row.crystallization_id;
    setStatus(statusEl, "Loading detail…", false);
    // crystallization_get (unlike the list endpoint) does not compute
    // decayed_activation/retirement_candidate -- carry the values already
    // known from the list row forward so the badge doesn't vanish on open.
    const full = row.retirement_candidate
      ? { ...(await apiFetch(`/api/memory/crystallizations/${encodeURIComponent(row.crystallization_id)}`)), decayed_activation: row.decayed_activation, retirement_candidate: row.retirement_candidate }
      : await apiFetch(`/api/memory/crystallizations/proposals/${encodeURIComponent(row.crystallization_id)}`);
    const links = await apiFetch(`/api/memory/crystallizations/${row.crystallization_id}/links`).catch(() => ({ items: [] }));
    const health = await apiFetch("/api/memory/crystallizations/projection/health").catch(() => ({}));
    detailEl.innerHTML = renderDetail(full, links, health);
    setStatus(statusEl, summarize(), false);
    detailEl.scrollIntoView({ behavior: "smooth", block: "nearest" });

    detailEl.querySelectorAll("button[data-act]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const act = btn.getAttribute("data-act");
        try {
          if (act === "sync-graphiti") {
            await apiFetch(`/api/memory/graphiti/sync/${row.crystallization_id}`, { method: "POST", body: "{}" });
          } else if (act === "deprecate") {
            await apiFetch(`/api/memory/crystallizations/${row.crystallization_id}/deprecate`, { method: "POST", body: "{}" });
          } else {
            await apiFetch(`/api/memory/crystallizations/proposals/${row.crystallization_id}/${act}`, { method: "POST", body: act === "validate" ? undefined : "{}" });
          }
          setStatus(statusEl, `${act} ok`, false);
          // "validate" annotates the proposal in place and leaves it in the
          // queue, so the pane must stay open for it -- only a terminal
          // decision closes.
          if (act !== "validate") {
            selected.delete(row.crystallization_id);
            closeDetail(detailEl);
          }
          await loadInbox(listEl, statusEl, detailEl);
          if (act === "validate") {
            await openDetail(row, listEl, statusEl, detailEl, summarize);
          }
        } catch (e) {
          setStatus(statusEl, e.message || String(e), true);
        }
      });
    });

    detailEl.querySelectorAll("button[data-drop-turn]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const sourceId = btn.getAttribute("data-drop-turn");
        try {
          await apiFetch(
            `/api/memory/crystallizations/${encodeURIComponent(row.crystallization_id)}/evidence/${encodeURIComponent(sourceId)}`,
            { method: "DELETE" },
          );
          setStatus(statusEl, `dropped turn ${sourceId.slice(0, 8)}`, false);
          await loadInbox(listEl, statusEl, detailEl);
          await openDetail(row, listEl, statusEl, detailEl, summarize);
        } catch (e) {
          setStatus(statusEl, e.message || String(e), true);
        }
      });
    });
  }

  function renderBulkBar(items, listEl, statusEl, detailEl, summarize) {
    const decidable = items.filter(isDecidable);
    const bar = document.createElement("div");
    bar.className =
      "flex items-center gap-2 text-xs border border-gray-800 rounded px-2 py-1 bg-gray-900/80 sticky top-0 z-10";

    const buttons = [];
    // Selection is purely local state. Reflect it by mutating the DOM in place
    // rather than calling loadInbox(): that re-fetched the whole queue on every
    // single tick, and because the call was not awaited, three quick ticks
    // launched three overlapping loads whose innerHTML="" and appends
    // interleaved -- duplicated or vanishing rows, and checkbox state rendered
    // from a stale snapshot of `selected`.
    const refreshSelectionUi = () => {
      count.textContent = selected.size ? `${selected.size} selected` : "select all";
      all.checked = decidable.length > 0 && decidable.every((i) => selected.has(i.crystallization_id));
      buttons.forEach((b) => {
        b.disabled = selected.size === 0;
      });
      listEl.querySelectorAll("input[data-select-id]").forEach((box) => {
        box.checked = selected.has(box.dataset.selectId);
      });
      setStatus(statusEl, summarize(), false);
    };

    const all = document.createElement("input");
    all.type = "checkbox";
    all.className = "cursor-pointer";
    all.setAttribute("aria-label", "select all proposals");
    all.checked = decidable.length > 0 && decidable.every((i) => selected.has(i.crystallization_id));
    all.addEventListener("change", () => {
      // `decidable`, not `items`: retirement candidates are active and the bulk
      // endpoint always refuses them.
      decidable.forEach((i) =>
        all.checked ? selected.add(i.crystallization_id) : selected.delete(i.crystallization_id),
      );
      refreshSelectionUi();
    });
    bar.appendChild(all);

    const count = document.createElement("span");
    count.className = "text-gray-400 flex-1";
    count.textContent = selected.size ? `${selected.size} selected` : "select all";
    bar.appendChild(count);

    let deciding = false;
    const decide = async (action) => {
      // In-flight guard. Without it a double-click on "Reject selected" launches
      // two concurrent chunked runs over the same id snapshot: the second gets
      // already_rejected for everything the first already decided, reports a
      // large spurious failure count, and both mutate `selected` and call
      // loadInbox() concurrently.
      if (deciding) return;
      const ids = [...selected];
      if (!ids.length) return;
      deciding = true;
      buttons.forEach((b) => {
        b.disabled = true;
      });
      // Chunked client-side. The server caps a batch at BULK_DECIDE_MAX (500)
      // and 400s the whole request past it -- which is precisely the case this
      // feature exists for, since the backlog that motivated it was 621 items.
      // Approve is far more expensive per item than reject (each one also runs
      // a card/chroma projection), so it gets a much smaller chunk to keep any
      // single request inside a normal proxy timeout.
      const chunkSize = action === "approve" ? APPROVE_CHUNK : REJECT_CHUNK;
      let succeeded = 0;
      let failed = 0;
      const firstErrors = [];
      try {
        for (let i = 0; i < ids.length; i += chunkSize) {
          const chunk = ids.slice(i, i + chunkSize);
          setStatus(
            statusEl,
            `${action}ing ${Math.min(i + chunk.length, ids.length)}/${ids.length}…`,
            false,
          );
          const res = await apiFetch("/api/memory/crystallizations/proposals/bulk", {
            method: "POST",
            body: JSON.stringify({ ids: chunk, action }),
          });
          // Only clear what actually succeeded, so a partial failure leaves the
          // still-undecided rows selected and retryable instead of silently
          // dropping them from the selection.
          (res.results || []).forEach((r) => {
            if (r.ok) selected.delete(r.crystallization_id);
            else if (firstErrors.length < 3) firstErrors.push(r.error);
          });
          succeeded += res.succeeded || 0;
          failed += res.failed || 0;
        }
        closeDetail(detailEl);
        await loadInbox(listEl, statusEl, detailEl);
        setStatus(
          statusEl,
          `${action}ed ${succeeded}/${ids.length}` +
            (failed ? ` — ${failed} failed (${firstErrors.join(", ")})` : ""),
          failed > 0,
        );
      } catch (e) {
        // A mid-run throw means earlier chunks already landed; reload so the
        // list reflects reality rather than the pre-decision state.
        await loadInbox(listEl, statusEl, detailEl).catch(() => {});
        setStatus(statusEl, `${action} stopped after ${succeeded}: ${e.message || String(e)}`, true);
      } finally {
        deciding = false;
        // loadInbox() above rebuilds the bar, so re-enabling these only matters
        // on the paths where it did not run; harmless either way.
        buttons.forEach((b) => {
          b.disabled = selected.size === 0;
        });
      }
    };

    [
      ["Approve selected", "approve", "border-emerald-700 text-emerald-200"],
      ["Reject selected", "reject", "border-red-800 text-red-200"],
    ].forEach(([label, action, cls]) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `px-2 py-0.5 rounded border ${cls} disabled:opacity-40 disabled:cursor-not-allowed`;
      btn.textContent = label;
      btn.disabled = selected.size === 0;
      btn.addEventListener("click", () => decide(action));
      buttons.push(btn);
      bar.appendChild(btn);
    });

    return { bar, refreshSelectionUi };
  }

  async function loadInbox(listEl, statusEl, detailEl) {
    setStatus(statusEl, "Loading proposals…", false);
    listEl.innerHTML = "";
    try {
      const data = await apiFetch("/api/memory/crystallizations/proposals");
      const proposalItems = (data && data.items) || [];
      const retirementItems = await loadRetirementCandidates();
      const items = [...retirementItems, ...proposalItems];
      const summarize = () =>
        `${proposalItems.length} proposal(s)` +
        (retirementItems.length ? `, ${retirementItems.length} retirement candidate(s)` : "") +
        (selected.size ? ` · ${selected.size} selected` : "");

      // Drop selections for anything that left the queue (decided here, decided
      // in another tab, or auto-activated by the formation gate) so the count
      // never claims more than the bulk call could actually act on.
      const live = new Set(items.map((i) => i.crystallization_id));
      [...selected].forEach((id) => {
        if (!live.has(id)) selected.delete(id);
      });

      // A detail pane left open on an item that is no longer in the queue is the
      // stale-button trap; close it on every refresh, not only after our own
      // decisions.
      const openId = detailEl.dataset.crystallizationId;
      if (openId && !live.has(openId)) closeDetail(detailEl);

      if (!items.length) {
        setStatus(statusEl, "No proposals in inbox.", false);
        return;
      }
      setStatus(statusEl, summarize(), false);
      const { bar, refreshSelectionUi } = renderBulkBar(items, listEl, statusEl, detailEl, summarize);
      listEl.appendChild(bar);
      items.forEach((item) => {
        listEl.appendChild(
          renderRow(
            item,
            (row) => {
              // openDetail is async and its rejections were unhandled from this
              // click path: if the proposal was decided in another tab the
              // /proposals/{id} fetch 404s and the pane is left visible-but-empty
              // with a dead id still in its dataset.
              openDetail(row, listEl, statusEl, detailEl, summarize).catch((e) => {
                closeDetail(detailEl);
                setStatus(statusEl, e.message || String(e), true);
              });
            },
            (row, isChecked) => {
              if (isChecked) selected.add(row.crystallization_id);
              else selected.delete(row.crystallization_id);
              refreshSelectionUi();
            },
            selected.has(item.crystallization_id),
          ),
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
