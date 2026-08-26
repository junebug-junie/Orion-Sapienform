/**
 * Reverie tab -- historical, human-visible view of both reverie chains.
 * Self-contained module (same lifecycle contract as attention-organ.js):
 * window.OrionReverie = { activate, deactivate, refresh }.
 *
 * No continuous polling -- this is a historical browsing tool, not a live
 * telemetry dashboard (the text chain ticks every ~90s, the visual chain
 * every ~600s; a manual refresh is the honest cadence for "look back at what
 * happened", not a fast poll loop). One fetch on activate(), and whenever
 * the operator clicks Refresh or switches sub-view.
 *
 * Backed by services/orion-hub/scripts/reverie_routes.py
 * (/api/reverie/visual/recent, /api/reverie/visual/image/{sha256},
 * /api/reverie/text/recent).
 */
(function () {
  let active = false;
  let subview = "visual"; // "visual" | "text"
  let loaded = { visual: false, text: false };

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

  function setStatus(msg) {
    const s = el("reverieStatus");
    if (s) s.textContent = msg;
  }

  // --- Visual panel -----------------------------------------------------

  function renderVisualChain(chain) {
    const artifact = (chain.artifacts && chain.artifacts[0]) || null;
    const img = artifact
      ? `<img src="${artifact.image_url}" loading="lazy" class="rounded-lg w-full object-cover" style="max-height: 320px;" alt="reverie image" />`
      : `<div class="text-xs text-gray-500 italic p-4">no image stored for this chain</div>`;
    const caption = artifact && artifact.description
      ? `<p class="text-sm text-gray-200 mt-2">${escapeHtml(artifact.description)}</p>`
      : `<p class="text-xs text-gray-500 italic mt-2">not captioned (re-observation failed or was rejected -- honest null, not fabricated)</p>`;
    const promptLine = chain.prompt
      ? `<p class="text-xs text-gray-500 mt-1">prompt: <span class="text-gray-400">${escapeHtml(chain.prompt)}</span></p>`
      : "";
    const priorLine = chain.prior_description
      ? `<p class="text-xs text-gray-500 mt-1">carried forward as prior_description for the next run</p>`
      : "";
    return `
      <div class="rounded-xl border border-gray-800 bg-gray-950/40 p-3 flex flex-col gap-1">
        ${img}
        ${caption}
        ${promptLine}
        ${priorLine}
        <div class="flex justify-between items-center mt-2 text-[11px] text-gray-500">
          <span>${fmtTime(chain.created_at)}</span>
          <span class="px-2 py-0.5 rounded-full border border-gray-700">${escapeHtml(chain.terminal_reason)}</span>
        </div>
      </div>`;
  }

  async function loadVisual() {
    const container = el("reverieVisualGrid");
    if (!container) return;
    setStatus("Loading visual reverie chain…");
    try {
      const resp = await fetch("/api/reverie/visual/recent?limit=30");
      const data = await resp.json();
      if (!data.ok) throw new Error("bad response");
      if (!data.chains.length) {
        container.innerHTML = `<div class="text-sm text-gray-500 italic p-4">No visual reverie chains yet -- ORION_VISUAL_CHAIN_ENABLED may be off, or none have run.</div>`;
      } else {
        container.innerHTML = data.chains.map(renderVisualChain).join("");
      }
      loaded.visual = true;
      setStatus(`Loaded ${data.chains.length} visual chain(s).`);
    } catch (e) {
      setStatus("Failed to load visual reverie chain: " + e.message);
    }
  }

  // --- Text panel ---------------------------------------------------------

  function renderTextChain(chain) {
    const thoughts = (chain.thoughts || [])
      .map(
        (t) => `
        <div class="border-l-2 border-gray-700 pl-3 py-1">
          <p class="text-sm text-gray-200">${escapeHtml(t.interpretation)}</p>
          <p class="text-[11px] text-gray-500">${fmtTime(t.created_at)} · salience ${Number(t.salience).toFixed(2)}</p>
        </div>`
      )
      .join("");
    const badges = [];
    if (chain.downstream && chain.downstream.compaction_queued) {
      badges.push(
        `<span class="px-2 py-0.5 rounded-full border border-amber-700 text-amber-400 text-[11px]">queued for dream compaction</span>`
      );
    }
    if (chain.downstream && chain.downstream.theme_resonance_alert_count > 0) {
      badges.push(
        `<span class="px-2 py-0.5 rounded-full border border-red-700 text-red-400 text-[11px]">${chain.downstream.theme_resonance_alert_count} resonance alert(s) on this theme</span>`
      );
    }
    return `
      <div class="rounded-xl border border-gray-800 bg-gray-950/40 p-3 flex flex-col gap-2">
        <div class="flex justify-between items-center text-[11px] text-gray-500">
          <span>${fmtTime(chain.created_at)} · theme <span class="text-gray-400">${escapeHtml(chain.theme_key || "unknown")}</span></span>
          <span class="px-2 py-0.5 rounded-full border border-gray-700">${escapeHtml(chain.terminal_reason)}</span>
        </div>
        <div class="flex flex-col gap-1">${thoughts || '<span class="text-xs text-gray-500 italic">no thoughts recorded</span>'}</div>
        <div class="flex gap-2 flex-wrap">${badges.join("") || '<span class="text-[11px] text-gray-600 italic">no downstream effect yet</span>'}</div>
      </div>`;
  }

  async function loadText() {
    const container = el("reverieTextGrid");
    if (!container) return;
    setStatus("Loading text reverie chain…");
    try {
      const resp = await fetch("/api/reverie/text/recent?limit=30");
      const data = await resp.json();
      if (!data.ok) throw new Error("bad response");
      if (!data.chains.length) {
        container.innerHTML = `<div class="text-sm text-gray-500 italic p-4">No text reverie chains yet.</div>`;
      } else {
        container.innerHTML = data.chains.map(renderTextChain).join("");
      }
      loaded.text = true;
      setStatus(`Loaded ${data.chains.length} text chain(s).`);
    } catch (e) {
      setStatus("Failed to load text reverie chain: " + e.message);
    }
  }

  // --- Sub-view switching ---------------------------------------------------

  function showSubview(name) {
    subview = name;
    const visualPanel = el("reverieVisualPanel");
    const textPanel = el("reverieTextPanel");
    const visualBtn = el("reverieSubtabVisual");
    const textBtn = el("reverieSubtabText");
    if (visualPanel) visualPanel.classList.toggle("hidden", subview !== "visual");
    if (textPanel) textPanel.classList.toggle("hidden", subview !== "text");
    if (visualBtn) visualBtn.classList.toggle("bg-gray-700", subview === "visual");
    if (textBtn) textBtn.classList.toggle("bg-gray-700", subview === "text");
    if (subview === "visual" && !loaded.visual) loadVisual();
    if (subview === "text" && !loaded.text) loadText();
  }

  function refresh() {
    if (subview === "visual") {
      loaded.visual = false;
      loadVisual();
    } else {
      loaded.text = false;
      loadText();
    }
  }

  let wired = false;

  function wireOnce() {
    // Idempotency guard (review finding): activate() also calls this, and a
    // tab can be activated many times in one session -- without the guard,
    // click listeners accumulate unboundedly and a single click fires the
    // handler once per prior activation.
    if (wired) return;
    wired = true;
    const visualBtn = el("reverieSubtabVisual");
    const textBtn = el("reverieSubtabText");
    const refreshBtn = el("reverieRefreshBtn");
    if (visualBtn) visualBtn.addEventListener("click", () => showSubview("visual"));
    if (textBtn) textBtn.addEventListener("click", () => showSubview("text"));
    if (refreshBtn) refreshBtn.addEventListener("click", refresh);
  }

  function activate() {
    active = true;
    wireOnce();
    if (subview === "visual" && !loaded.visual) loadVisual();
    if (subview === "text" && !loaded.text) loadText();
  }

  function deactivate() {
    active = false;
    // Rendered DOM stays in place -- same contract as Attention Organ.
  }

  document.addEventListener("DOMContentLoaded", wireOnce);

  window.OrionReverie = { activate, deactivate, refresh };
})();
