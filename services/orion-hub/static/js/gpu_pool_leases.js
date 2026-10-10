// "GPU leases" modal on the chat screen: one switch per lendable GPU in the pool.
//
// Lending a card lets other work (agent, metacog, fast) borrow it while its owner is idle; the
// owner still takes it back. The list is whatever GET /api/gpu-pool/state reports as lendable,
// so a new lendable card in config/gpu_pool.yaml shows up here with no Hub change. A flip posts
// lend/unlend to POST /api/gpu-pool/control and the switch settles on what the pool replies,
// never on what was clicked. The full panel (holds, swaps, history) stays on the GPU pool tab.
(function () {
  "use strict";

  var STATE_URL = "/api/gpu-pool/state";
  var CONTROL_URL = "/api/gpu-pool/control";
  // The header button's "N lent" badge follows flips made elsewhere (GPU pool tab, another browser).
  var BADGE_POLL_MS = 60000;

  // Lendable cards from a pool state payload, with the roles on each card and whether any of
  // them is serving. `serving: null` means the pool listed no role for the card.
  function leaseRows(state) {
    var cards = (state && state.cards) || [];
    var roles = (state && state.roles) || [];
    return cards
      .filter(function (c) { return c && c.lendable; })
      .map(function (c) {
        var onCard = roles.filter(function (r) { return (r.cards || []).indexOf(c.card) !== -1; });
        return {
          card: String(c.card),
          host: c.host ? String(c.host) : "",
          vramGb: c.vram_gb == null ? null : Number(c.vram_gb),
          lent: !!c.lent,
          roles: onCard.map(function (r) { return String(r.role); }),
          serving: onCard.length ? onCard.some(function (r) { return r.status === "confirmed"; }) : null,
        };
      });
  }

  function lentCount(rows) {
    return rows.filter(function (r) { return r.lent; }).length;
  }

  function badgeText(rows) {
    var n = lentCount(rows);
    return n ? n + " lent" : "";
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
    });
  }

  function rowHtml(row) {
    var title = (row.host ? row.host + " · " : "") + row.card;
    var roles = row.roles.length ? row.roles.join(", ") : "no role";
    var down = row.serving === false
      ? '<span class="text-amber-400" data-lease-down="1"> · server down: lending does nothing until it answers</span>'
      : "";
    return (
      '<div class="flex items-center justify-between gap-4 rounded-xl border border-gray-800 bg-gray-900/60 px-4 py-3" data-lease-row="' + esc(row.card) + '">' +
        '<div class="min-w-0">' +
          '<div class="text-sm font-semibold text-white">' + esc(title) + "</div>" +
          '<div class="text-[11px] text-gray-400">' + esc(roles) +
            (row.vramGb != null ? " · " + esc(row.vramGb) + " GB" : "") + down + "</div>" +
          '<div class="mt-0.5 text-[11px] text-gray-500" data-lease-caption>' + esc(caption(row.lent)) + "</div>" +
        "</div>" +
        '<button type="button" role="switch" aria-checked="' + (row.lent ? "true" : "false") + '"' +
          ' aria-label="Lend ' + esc(title) + '" data-lease-card="' + esc(row.card) + '"' +
          ' class="gpu-lease-switch">' +
          '<span class="gpu-lease-knob"></span>' +
        "</button>" +
      "</div>"
    );
  }

  function caption(lent) {
    return lent ? "Lent: other work may borrow it; its owner still takes it back." : "Owner only.";
  }

  function setSwitch(btn, lent) {
    btn.setAttribute("aria-checked", lent ? "true" : "false");
    var row = btn.closest("[data-lease-row]");
    var cap = row && row.querySelector("[data-lease-caption]");
    if (cap) cap.textContent = caption(lent);
  }

  var els = {};
  var rows = [];

  function setBadge() {
    if (!els.badge) return;
    var text = badgeText(rows);
    els.badge.textContent = text;
    els.badge.hidden = !text;
  }

  function setStatus(text, isError) {
    if (!els.status) return;
    els.status.textContent = text || "";
    els.status.className = "text-xs " + (isError ? "text-red-400" : "text-gray-400");
  }

  async function refresh() {
    var res = await fetch(STATE_URL, { credentials: "same-origin", cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    rows = leaseRows(await res.json());
    setBadge();
    return rows;
  }

  function renderList() {
    els.list.innerHTML = rows.length
      ? rows.map(rowHtml).join("")
      : '<div class="text-xs text-gray-400">No lendable GPUs in config/gpu_pool.yaml.</div>';
  }

  async function openModal() {
    els.modal.hidden = false;
    els.list.innerHTML = '<div class="text-xs text-gray-400">Loading…</div>';
    setStatus("");
    try {
      await refresh();
      renderList();
    } catch (err) {
      els.list.innerHTML = "";
      setStatus("Could not read the GPU pool: " + (err && err.message ? err.message : err), true);
    }
    els.close.focus();
  }

  function closeModal() {
    els.modal.hidden = true;
    els.open.focus();
  }

  async function flip(btn) {
    var card = btn.getAttribute("data-lease-card");
    var was = btn.getAttribute("aria-checked") === "true";
    var want = !was;
    setSwitch(btn, want);   // move the knob now; the pool's reply has the final say
    btn.disabled = true;
    setStatus("");
    try {
      var res = await fetch(CONTROL_URL, {
        method: "POST",
        credentials: "same-origin",
        // X-Requested-With: the pool control route refuses requests without it (CSRF guard).
        headers: { "Content-Type": "application/json", "X-Requested-With": "orion-hub" },
        body: JSON.stringify({ verb: want ? "lend" : "unlend", card: card }),
      });
      var body = await res.json().catch(function () { return null; });
      if (!res.ok || !(body && body.ok)) {
        var detail = body && (body.detail || body.reason) ? (body.detail || body.reason) : "HTTP " + res.status;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      var lent = !!(body.detail && body.detail.lent);
      setSwitch(btn, lent);
      rows.forEach(function (r) { if (r.card === card) r.lent = lent; });
      setBadge();
    } catch (err) {
      setSwitch(btn, was);
      setStatus("Could not " + (want ? "lend " : "take back ") + card + ": " + (err && err.message ? err.message : err), true);
    } finally {
      btn.disabled = false;
    }
  }

  function init() {
    els.open = document.getElementById("gpuLeasesOpen");
    els.badge = document.getElementById("gpuLeasesBadge");
    els.modal = document.getElementById("gpuLeasesModal");
    els.list = document.getElementById("gpuLeasesList");
    els.status = document.getElementById("gpuLeasesStatus");
    els.close = document.getElementById("gpuLeasesClose");
    if (!els.open || !els.modal || !els.list || !els.close) return;

    els.open.addEventListener("click", openModal);
    els.close.addEventListener("click", closeModal);
    els.modal.addEventListener("click", function (ev) {
      if (ev.target === els.modal) closeModal();   // backdrop click
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && !els.modal.hidden) closeModal();
    });
    els.list.addEventListener("click", function (ev) {
      var btn = ev.target.closest("button[data-lease-card]");
      if (btn && !btn.disabled) flip(btn);
    });

    function poll() {
      if (document.hidden || !els.modal.hidden) return;
      refresh().catch(function () { /* badge keeps its last known value */ });
    }
    poll();
    setInterval(poll, BADGE_POLL_MS);
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", init);
    } else {
      init();
    }
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { leaseRows: leaseRows, lentCount: lentCount, badgeText: badgeText, rowHtml: rowHtml };
  }
})();
