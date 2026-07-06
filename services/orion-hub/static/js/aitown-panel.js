(function () {
  const API_BASE = (window.HUB_CFG && window.HUB_CFG.apiBaseOverride) || "";
  const pollMs = 10000;
  const IFRAME_BASE = "/ai-town/?hub_embed=1";
  const READY_TIMEOUT_MS = 20000;
  let pollTimer = null;
  let readyTimer = null;
  let loadToken = 0;

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

  function setLoading(loading, message) {
    const overlay = el("aitownFrameLoading");
    if (!overlay) return;
    overlay.classList.toggle("hidden", !loading);
    overlay.textContent = message || (loading ? "Loading AI Town…" : "");
  }

  function clearReadyTimer() {
    if (readyTimer) {
      window.clearTimeout(readyTimer);
      readyTimer = null;
    }
  }

  function readFrameState(frame) {
    try {
      const doc = frame.contentDocument;
      if (!doc) return { ready: false };
      const canvas = doc.querySelector("canvas");
      const h1 = doc.querySelector("h1");
      const bodyLen = (doc.body?.innerText || "").length;
      return {
        ready: true,
        canvasW: canvas ? canvas.width : 0,
        canvasH: canvas ? canvas.height : 0,
        h1: h1 ? h1.textContent : "",
        bodyLen,
      };
    } catch (err) {
      return { ready: false, error: String(err) };
    }
  }

  function finishLoading(stripMessage) {
    clearReadyTimer();
    setLoading(false);
    if (stripMessage) {
      const strip = el("aitownStatusStrip");
      if (strip) strip.textContent = stripMessage;
    }
  }

  function waitForGameReady(frame, token) {
    clearReadyTimer();
    const started = Date.now();

    const tick = () => {
      if (token !== loadToken) return;
      const state = readFrameState(frame);
      if (state.canvasW > 0 && state.canvasH > 0) {
        finishLoading();
        return;
      }
      if (state.h1 && state.bodyLen > 60 && Date.now() - started > 2500) {
        finishLoading();
        return;
      }
      if (Date.now() - started >= READY_TIMEOUT_MS) {
        finishLoading("○ AI Town timed out · try Refresh or Open standalone");
        return;
      }
      readyTimer = window.setTimeout(tick, 250);
    };

    tick();
  }

  function loadIframe() {
    const frame = el("aitownFrame");
    const panel = el("ai-town");
    if (!frame || !panel) return;

    loadToken += 1;
    const token = loadToken;
    clearReadyTimer();
    setLoading(true, "Loading AI Town…");

    const bust = `${IFRAME_BASE}&hub=${Date.now()}`;
    const start = () => {
      if (token !== loadToken) return;
      frame.src = "about:blank";
      window.setTimeout(() => {
        if (token !== loadToken) return;
        frame.src = bust;
        waitForGameReady(frame, token);
      }, 50);
    };

    requestAnimationFrame(() => requestAnimationFrame(start));
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
      loadIframe();
      startPolling();
    },
    deactivate() {
      stopPolling();
    },
    refresh: refreshStatus,
  };

  const refreshBtn = el("aitownRefreshBtn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      refreshStatus();
      loadIframe();
    });
  }
})();
