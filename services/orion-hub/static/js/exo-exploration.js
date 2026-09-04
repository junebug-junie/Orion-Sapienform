/**
 * Exo Exploration tab -- tech/compute finds crawled from KSL classifieds.
 * Self-contained module (same lifecycle contract as reverie-tab.js):
 * window.OrionExoExploration = { activate, deactivate, refresh }.
 *
 * Backed by services/orion-hub/scripts/exo_exploration_routes.py, which
 * proxies orion-exo-exploration's own `/finds` and `/crawl-runs`, degrading
 * to `{available: false, reason: ...}` (never a 500) if that service is
 * unreachable or not configured -- this file renders that degraded state
 * honestly instead of pretending an empty list means "no finds today".
 *
 * No iframe: this panel renders directly, unlike concept-atlas/curiosity's
 * standalone-page pattern -- there is no `/exo-exploration` page route.
 * Own small card markup here rather than reusing the cognitive-loop
 * PendingAttentionCardV1/cognitive-loop-card.js machinery, which is
 * schema-scoped to a different subsystem.
 */
(function () {
  let active = false;
  let loaded = false;
  const WORTH_A_LOOK_THRESHOLD = 1.0;

  function el(id) {
    return document.getElementById(id);
  }

  function escapeHtml(s) {
    if (s === null || s === undefined) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmtTime(iso) {
    if (!iso) return "—";
    try {
      return new Date(iso).toLocaleString();
    } catch (e) {
      return iso;
    }
  }

  function fmtPrice(price, priceRaw) {
    if (price === null || price === undefined) return escapeHtml(priceRaw || "—");
    if (price === 0) return "FREE";
    return "$" + Number(price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function categoryLabel(url) {
    if (!url) return "Unknown category";
    const m = /\/cat\/([^/?]+)/.exec(url);
    return m ? decodeURIComponent(m[1]) : url;
  }

  function setStatus(msg) {
    const s = el("exoExplorationStatus");
    if (s) s.textContent = msg;
  }

  function renderFindCard(find) {
    const reasons = Array.isArray(find.interest_reasons) ? find.interest_reasons : [];
    const reasonsHtml = reasons.length
      ? `<ul class="text-[11px] text-gray-400 list-disc list-inside">${reasons
          .map((r) => `<li>${escapeHtml(r)}</li>`)
          .join("")}</ul>`
      : `<p class="text-[11px] text-gray-600 italic">no interest rule fired</p>`;
    const statusBadge = find.is_currently_listed
      ? `<span class="px-2 py-0.5 rounded-full border border-emerald-700 text-emerald-400 text-[11px]">active</span>`
      : `<span class="px-2 py-0.5 rounded-full border border-gray-700 text-gray-500 text-[11px]">no longer listed</span>`;
    return `
      <div class="rounded-xl border border-gray-800 bg-gray-950/40 p-3 flex flex-col gap-2">
        <div class="flex justify-between items-start gap-2">
          <a href="${escapeHtml(find.url)}" target="_blank" rel="noopener noreferrer" class="text-sm font-semibold text-white hover:text-blue-400">${escapeHtml(find.title)}</a>
          ${statusBadge}
        </div>
        <div class="flex justify-between items-center text-[11px] text-gray-500">
          <span>${escapeHtml(categoryLabel(find.source_category))} · seen ${fmtTime(find.last_seen_at)}</span>
          <span class="text-gray-300 font-semibold">${fmtPrice(find.price, find.price_raw)}</span>
        </div>
        <div class="flex justify-between items-center text-[11px] text-gray-500">
          <span>interest score ${Number(find.interest_score || 0).toFixed(1)}</span>
          <span>seen ${find.times_seen || 1}x</span>
        </div>
        ${reasonsHtml}
      </div>`;
  }

  function renderFinds(finds) {
    const listEl = el("exoExplorationFindsList");
    if (!listEl) return;
    if (!finds.length) {
      listEl.innerHTML = `<div class="text-sm text-gray-500 italic p-4 col-span-full">No finds yet -- the crawl may not have run yet today.</div>`;
    } else {
      listEl.innerHTML = finds.map(renderFindCard).join("");
    }

    const worthALook = finds.filter((f) => Number(f.interest_score || 0) >= WORTH_A_LOOK_THRESHOLD);
    const worthWrap = el("exoExplorationWorthALook");
    const worthList = el("exoExplorationWorthALookList");
    if (worthWrap && worthList) {
      if (worthALook.length) {
        worthWrap.classList.remove("hidden");
        worthList.innerHTML = worthALook.map(renderFindCard).join("");
      } else {
        worthWrap.classList.add("hidden");
        worthList.innerHTML = "";
      }
    }
  }

  async function loadFinds() {
    setStatus("Loading finds…");
    const categoryFilter = el("exoExplorationCategoryFilter");
    const category = categoryFilter ? categoryFilter.value : "";
    const url = category
      ? `/api/exo-exploration/finds?category=${encodeURIComponent(category)}`
      : "/api/exo-exploration/finds";
    try {
      const resp = await fetch(url);
      const data = await resp.json();
      if (data.available === false) {
        setStatus(
          "Exo Exploration is unavailable right now (" + (data.reason || "unknown reason") + ")."
        );
        renderFinds([]);
        return;
      }
      const finds = Array.isArray(data.finds) ? data.finds : [];
      renderFinds(finds);
      setStatus(`Loaded ${finds.length} find(s).`);
      loaded = true;
    } catch (e) {
      setStatus("Failed to load finds: " + e.message);
    }
  }

  function refresh() {
    loadFinds();
  }

  let wired = false;

  function wireOnce() {
    // Idempotency guard (same reasoning as reverie-tab.js): activate() can
    // fire many times in one session, and without this guard click
    // listeners accumulate unboundedly.
    if (wired) return;
    wired = true;
    const refreshBtn = el("exoExplorationPanelRefresh");
    const categoryFilter = el("exoExplorationCategoryFilter");
    if (refreshBtn) refreshBtn.addEventListener("click", refresh);
    if (categoryFilter) categoryFilter.addEventListener("change", refresh);
  }

  function activate() {
    active = true;
    wireOnce();
    if (!loaded) loadFinds();
  }

  function deactivate() {
    active = false;
    // Rendered DOM stays in place -- same contract as Attention Organ / Reverie.
  }

  window.OrionExoExploration = { activate, deactivate, refresh };
})();
