/* Orion Hub — read-only proposal review attention surface (Pending Decisions) */
(function () {
  const pathSegments = window.location.pathname.split("/").filter((p) => p.length > 0);
  const URL_PREFIX = pathSegments.length > 0 ? `/${pathSegments[0]}` : "";
  const API_BASE = window.location.origin + URL_PREFIX;

  function sessionHeader() {
    const sid = localStorage.getItem("orion_sid");
    return sid ? { "X-Orion-Session-Id": sid } : {};
  }

  async function apiFetch(path) {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: {
        "Content-Type": "application/json",
        ...sessionHeader(),
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
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function setStatus(el, msg, isErr) {
    if (!el) return;
    el.textContent = msg || "";
    el.classList.toggle("text-red-400", !!isErr);
    el.classList.toggle("text-gray-500", !isErr);
  }

  function renderListRow(item, onSelect) {
    const row = document.createElement("button");
    row.type = "button";
    row.className =
      "w-full text-left flex flex-col gap-0.5 border border-gray-800 rounded px-2 py-2 bg-gray-900/60 hover:border-indigo-700/50";
    row.innerHTML = `<div class="flex justify-between gap-2">
      <span class="font-medium text-gray-100">${escapeHtml(item.title || item.proposal_id || "")}</span>
      <span class="text-[10px] uppercase text-amber-300/90">${escapeHtml(item.status || "")}</span>
    </div>
    <div class="text-[10px] text-gray-500">${escapeHtml(item.proposal_type || "")} · risk ${escapeHtml(item.risk || "—")}</div>`;
    row.addEventListener("click", () => onSelect(item));
    return row;
  }

  function renderDetail(detail, eligibility) {
    const envelope = (detail && detail.envelope) || {};
    const inner = (detail && detail.inner_artifact_summary) || {};
    const execElig = eligibility || (detail && detail.execution_eligibility) || {};
    const reviewHistory = (detail && detail.review_history) || [];
    const latestReview = reviewHistory.length ? reviewHistory[reviewHistory.length - 1] : null;

    return `<div class="space-y-2">
      <div class="text-sm font-semibold text-white">${escapeHtml(envelope.title || detail.proposal_id || "")}</div>
      <div class="text-[10px] text-gray-500">${escapeHtml(detail.proposal_id || "")} · ${escapeHtml(envelope.proposal_type || "")}</div>
      <div><span class="text-gray-500">Summary:</span> ${escapeHtml(envelope.summary || "—")}</div>
      <div><span class="text-gray-500">Risk:</span> ${escapeHtml(detail.risk || envelope.risk || "—")}</div>
      <div><span class="text-gray-500">Attention:</span> ${escapeHtml(detail.attention_reason || "—")}</div>
      <div><span class="text-gray-500">Review status:</span> ${escapeHtml(detail.status || "—")}${latestReview ? ` (${escapeHtml(latestReview.decision || "")})` : ""}</div>
      <div><span class="text-gray-500">Execution eligibility:</span> ${escapeHtml(execElig.eligible === true ? "eligible" : execElig.eligible === false ? "not eligible" : "—")}${execElig.reason ? ` — ${escapeHtml(execElig.reason)}` : ""}</div>
      <div><span class="text-gray-500">Inner artifact:</span> ${escapeHtml(inner.artifact_type || envelope.artifact_type || "—")}</div>
      <div class="whitespace-pre-wrap"><span class="text-gray-500">Evidence:</span> ${escapeHtml(JSON.stringify(detail.evidence || envelope.evidence || [], null, 0))}</div>
      <div><span class="text-gray-500">Safety notes:</span> ${escapeHtml((envelope.safety_notes || []).join("; ") || "—")}</div>
      <div><span class="text-gray-500">Open questions:</span> ${escapeHtml((envelope.open_questions || []).join("; ") || "—")}</div>
    </div>`;
  }

  async function loadPendingDecisions(listEl, statusEl, detailEl, filterEl) {
    listEl.innerHTML = "";
    if (detailEl) {
      detailEl.classList.add("hidden");
      detailEl.innerHTML = "";
    }
    setStatus(statusEl, "Loading…", false);

    const filter = (filterEl && filterEl.value) || "pending_review";
    try {
      const health = await apiFetch("/api/proposal-review/health");
      if (!health.enabled) {
        setStatus(statusEl, "", false);
        listEl.innerHTML = "";
        return;
      }
      if (!health.available) {
        setStatus(statusEl, "Proposal review API unavailable.", false);
        return;
      }

      const data = await apiFetch(`/api/proposal-review/pending?status=${encodeURIComponent(filter)}`);
      if (!data.available) {
        setStatus(statusEl, "Proposal review API unavailable.", false);
        return;
      }

      const items = (data && data.proposals) || [];
      if (!items.length) {
        setStatus(statusEl, "No pending decisions.", false);
        return;
      }

      setStatus(statusEl, `${items.length} decision-worthy proposal(s)`, false);
      items.forEach((item) => {
        listEl.appendChild(
          renderListRow(item, async (row) => {
            if (!detailEl) return;
            detailEl.classList.remove("hidden");
            detailEl.innerHTML = "<div class='text-gray-500'>Loading detail…</div>";
            try {
              const detail = await apiFetch(`/api/proposal-review/proposals/${encodeURIComponent(row.proposal_id)}`);
              let eligibility = null;
              try {
                eligibility = await apiFetch(
                  `/api/proposal-review/proposals/${encodeURIComponent(row.proposal_id)}/eligibility`,
                );
              } catch {
                eligibility = detail.execution_eligibility || null;
              }
              detailEl.innerHTML = renderDetail(detail, eligibility);
            } catch (e) {
              detailEl.innerHTML = `<div class="text-red-400">${escapeHtml(e.message || String(e))}</div>`;
            }
          }),
        );
      });
    } catch (e) {
      setStatus(statusEl, "Proposal review API unavailable.", false);
    }
  }

  function wireAdvancedFilter(filterEl) {
    if (!filterEl) return;
    let clicks = 0;
    filterEl.addEventListener("mousedown", () => {
      clicks += 1;
      if (clicks >= 5) {
        filterEl.classList.remove("hidden");
      }
    });
  }

  function init() {
    const listEl = document.getElementById("proposalReviewList");
    const statusEl = document.getElementById("proposalReviewStatus");
    const detailEl = document.getElementById("proposalReviewDetail");
    const refreshBtn = document.getElementById("proposalReviewRefreshButton");
    const filterEl = document.getElementById("proposalReviewFilter");
    if (!listEl || !statusEl) return;

    wireAdvancedFilter(filterEl);
    const reload = () => loadPendingDecisions(listEl, statusEl, detailEl, filterEl);
    if (refreshBtn) refreshBtn.addEventListener("click", reload);
    if (filterEl) filterEl.addEventListener("change", reload);
    reload();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
