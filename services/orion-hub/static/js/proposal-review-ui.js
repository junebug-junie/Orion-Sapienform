/* Orion Hub — proposal review attention surface (Pending Decisions + review actions) */
(function () {
  const pathSegments = window.location.pathname.split("/").filter((p) => p.length > 0);
  const URL_PREFIX = pathSegments.length > 0 ? `/${pathSegments[0]}` : "";
  const API_BASE = window.location.origin + URL_PREFIX;

  function sessionHeader() {
    const sid = localStorage.getItem("orion_sid");
    return sid ? { "X-Orion-Session-Id": sid } : {};
  }

  async function apiFetch(path, options) {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: {
        "Content-Type": "application/json",
        ...sessionHeader(),
      },
      ...options,
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
    el.classList.toggle("text-emerald-400", !isErr && !!msg);
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

  function evidenceSummary(inner, envelope, detail) {
    const supporting = (inner && inner.supporting_evidence) || [];
    const contradicting = (inner && inner.contradicting_evidence) || [];
    const missing = (inner && inner.missing_evidence) || [];
    const parts = [];
    if (supporting.length) parts.push(`supporting (${supporting.length}): ${supporting.slice(0, 3).join("; ")}`);
    if (contradicting.length) parts.push(`contradicting (${contradicting.length}): ${contradicting.slice(0, 3).join("; ")}`);
    if (missing.length) parts.push(`missing (${missing.length}): ${missing.slice(0, 3).join("; ")}`);
    if (!parts.length) {
      const ev = (detail && detail.evidence) || (envelope && envelope.evidence) || [];
      if (ev.length) return `${ev.length} envelope evidence item(s)`;
      return "—";
    }
    return parts.join(" · ");
  }

  function renderReviewActions(proposalId, status) {
    const pending = status === "pending_review" || status === "request_changes";
    if (!pending) {
      return `<div class="text-[10px] text-gray-500 pt-2 border-t border-gray-800">Review closed (${escapeHtml(status || "—")}).</div>`;
    }
    return `<div class="pt-2 border-t border-gray-800 space-y-2" data-proposal-id="${escapeHtml(proposalId)}">
      <div class="text-[10px] uppercase tracking-wide text-gray-500">Review decision</div>
      <label class="block text-[10px] text-gray-500">Rationale (required)</label>
      <textarea id="proposalReviewRationale" rows="2" class="w-full bg-gray-800 text-gray-200 rounded border border-gray-700 px-2 py-1 text-[11px]" placeholder="Why this decision?"></textarea>
      <label class="block text-[10px] text-gray-500">Constraints (optional, approve only)</label>
      <input id="proposalReviewConstraints" type="text" class="w-full bg-gray-800 text-gray-200 rounded border border-gray-700 px-2 py-1 text-[11px]" placeholder="e.g. bounded scope" />
      <div class="flex flex-wrap gap-2">
        <button type="button" data-review-decision="approve" class="proposalReviewActionBtn text-[10px] bg-emerald-900/60 hover:bg-emerald-800/70 text-emerald-100 rounded px-2 py-1 border border-emerald-800">Approve</button>
        <button type="button" data-review-decision="reject" class="proposalReviewActionBtn text-[10px] bg-red-900/50 hover:bg-red-800/60 text-red-100 rounded px-2 py-1 border border-red-900">Reject</button>
        <button type="button" data-review-decision="request_changes" class="proposalReviewActionBtn text-[10px] bg-amber-900/50 hover:bg-amber-800/60 text-amber-100 rounded px-2 py-1 border border-amber-900">Request changes</button>
      </div>
      <div id="proposalReviewActionStatus" class="text-[10px] text-gray-500"></div>
    </div>`;
  }

  function renderDetail(detail, eligibility) {
    const envelope = (detail && detail.envelope) || {};
    const inner = (detail && detail.inner_artifact_summary) || {};
    const execElig = eligibility || (detail && detail.execution_eligibility) || {};
    const reviewHistory = (detail && detail.review_history) || [];
    const latestReview = reviewHistory.length ? reviewHistory[reviewHistory.length - 1] : null;
    const confidence = inner.confidence ?? envelope.confidence;
    const risk = detail.risk || inner.risk || envelope.risk || "—";
    const mutationAllowed = inner.mutation_allowed ?? envelope.mutation_allowed;
    const requiresHuman = inner.requires_human_approval ?? envelope.requires_human_approval;

    return `<div class="space-y-2">
      <div class="text-sm font-semibold text-white">${escapeHtml(envelope.title || detail.proposal_id || "")}</div>
      <div class="text-[10px] text-gray-500">${escapeHtml(detail.proposal_id || "")} · ${escapeHtml(envelope.proposal_type || "")}</div>
      <div><span class="text-gray-500">Current belief:</span> ${escapeHtml(inner.current_belief || "—")}</div>
      <div><span class="text-gray-500">Proposed correction:</span> ${escapeHtml(inner.correction_type || inner.proposed_belief || "—")}</div>
      <div><span class="text-gray-500">Rationale:</span> ${escapeHtml(inner.rationale || envelope.summary || "—")}</div>
      <div><span class="text-gray-500">Evidence:</span> ${escapeHtml(evidenceSummary(inner, envelope, detail))}</div>
      <div><span class="text-gray-500">Risk:</span> ${escapeHtml(risk)} · <span class="text-gray-500">Confidence:</span> ${escapeHtml(confidence ?? "—")}</div>
      <div><span class="text-gray-500">Attention:</span> ${escapeHtml(detail.attention_reason || "—")}</div>
      <div><span class="text-gray-500">Review status:</span> ${escapeHtml(detail.status || "—")}${latestReview ? ` (${escapeHtml(latestReview.decision || "")})` : ""}</div>
      <div><span class="text-gray-500">Execution eligibility:</span> ${escapeHtml(execElig.eligible === true ? "eligible" : execElig.eligible === false ? "not eligible" : "—")}${execElig.reason ? ` — ${escapeHtml(execElig.reason)}` : ""}</div>
      <div><span class="text-gray-500">Safety:</span> mutation_allowed=${escapeHtml(String(mutationAllowed))}, requires_human_approval=${escapeHtml(String(requiresHuman))}</div>
      <div><span class="text-gray-500">Open questions:</span> ${escapeHtml((envelope.open_questions || []).join("; ") || "—")}</div>
      ${renderReviewActions(detail.proposal_id, detail.status)}
    </div>`;
  }

  async function submitReviewDecision(proposalId, decision, rationale, constraintsText, actionStatusEl, reload) {
    const rationaleTrimmed = (rationale || "").trim();
    if (!rationaleTrimmed) {
      setStatus(actionStatusEl, "Rationale is required.", true);
      return;
    }

    const body = {
      decision,
      rationale: rationaleTrimmed,
      reviewer_type: "human",
      reviewer_id: "hub-operator",
    };
    if (decision === "approve" && constraintsText && constraintsText.trim()) {
      body.constraints = { note: constraintsText.trim() };
    }

    setStatus(actionStatusEl, "Submitting review…", false);
    try {
      const result = await apiFetch(
        `/api/proposal-review/proposals/${encodeURIComponent(proposalId)}/review`,
        { method: "POST", body: JSON.stringify(body) },
      );
      const status = result && result.status ? result.status : decision;
      const eligible =
        result && result.execution_eligibility && result.execution_eligibility.eligible === true;
      const msg =
        decision === "approve"
          ? `Approved (${status}). Eligible for future execution: ${eligible ? "yes" : "no"}.`
          : `Recorded ${decision.replace("_", " ")} (${status}).`;
      setStatus(actionStatusEl, msg, false);
      if (typeof reload === "function") {
        await reload(proposalId);
      }
    } catch (e) {
      const detail =
        (e.body && (e.body.detail || e.body.message)) || e.message || String(e);
      setStatus(actionStatusEl, `Review failed: ${detail}`, true);
    }
  }

  function wireReviewActions(detailEl, reload) {
    if (!detailEl) return;
    const proposalId = detailEl.querySelector("[data-proposal-id]");
    if (!proposalId) return;
    const id = proposalId.getAttribute("data-proposal-id");
    const rationaleEl = detailEl.querySelector("#proposalReviewRationale");
    const constraintsEl = detailEl.querySelector("#proposalReviewConstraints");
    const actionStatusEl = detailEl.querySelector("#proposalReviewActionStatus");
    detailEl.querySelectorAll(".proposalReviewActionBtn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const decision = btn.getAttribute("data-review-decision");
        if (!decision) return;
        submitReviewDecision(
          id,
          decision,
          rationaleEl ? rationaleEl.value : "",
          constraintsEl ? constraintsEl.value : "",
          actionStatusEl,
          reload,
        );
      });
    });
  }

  async function loadProposalDetail(proposalId, detailEl, reloadList) {
    if (!detailEl) return;
    detailEl.classList.remove("hidden");
    detailEl.innerHTML = "<div class='text-gray-500'>Loading detail…</div>";
    try {
      const detail = await apiFetch(`/api/proposal-review/proposals/${encodeURIComponent(proposalId)}`);
      let eligibility = null;
      try {
        eligibility = await apiFetch(
          `/api/proposal-review/proposals/${encodeURIComponent(proposalId)}/eligibility`,
        );
      } catch {
        eligibility = detail.execution_eligibility || null;
      }
      detailEl.innerHTML = renderDetail(detail, eligibility);
      wireReviewActions(detailEl, async (refreshedId) => {
        await loadProposalDetail(refreshedId, detailEl, reloadList);
        if (typeof reloadList === "function") {
          await reloadList();
        }
      });
    } catch (e) {
      detailEl.innerHTML = `<div class="text-red-400">${escapeHtml(e.message || String(e))}</div>`;
    }
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
      const reloadList = () => loadPendingDecisions(listEl, statusEl, detailEl, filterEl);
      items.forEach((item) => {
        listEl.appendChild(
          renderListRow(item, (row) => loadProposalDetail(row.proposal_id, detailEl, reloadList)),
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
