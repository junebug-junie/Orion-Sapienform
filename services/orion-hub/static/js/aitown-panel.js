(function () {
  const API_BASE = (window.HUB_CFG && window.HUB_CFG.apiBaseOverride) || "";
  const pollMs = 10000;
  let pollTimer = null;
  let iframeLoaded = false;

  function el(id) {
    return document.getElementById(id);
  }

  async function refreshStatus() {
    const strip = el("aitownStatusStrip");
    if (!strip) return;
    try {
      const resp = await fetch(`${API_BASE}/api/aitown/status`);
      const data = await resp.json();
      const running = data.engine_running || data.convex_reachable;
      const dot = running ? "●" : "○";
      const label = running ? "Running" : "Offline";
      strip.textContent = `${dot} ${label} · players ${data.player_count || 0} · agents ${data.agent_count || 0} · gen ${data.generation ?? "—"}`;
    } catch (err) {
      strip.textContent = `○ Offline · ${err.message || err}`;
    }
  }

  function ensureIframe() {
    if (iframeLoaded) return;
    const frame = el("aitownFrame");
    if (!frame) return;
    frame.src = "/aitown/";
    iframeLoaded = true;
  }

  function startPolling() {
    if (pollTimer) return;
    refreshStatus();
    pollTimer = window.setInterval(refreshStatus, pollMs);
  }

  function stopPolling() {
    if (pollTimer) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  window.OrionAitownPanel = {
    activate() {
      ensureIframe();
      startPolling();
    },
    deactivate() {
      stopPolling();
    },
    refresh: refreshStatus,
  };

  const refreshBtn = el("aitownRefreshBtn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => refreshStatus());
  }
})();
