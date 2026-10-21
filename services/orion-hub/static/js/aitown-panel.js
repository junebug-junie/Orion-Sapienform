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
    refreshSimulation();
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

  let simState = null;
  let eventId = null;
  async function refreshSimulation() {
    const label = el("townSimState");
    if (!label) return;
    try {
      const resp = await fetch(`${API_BASE}/api/aitown/simulation`);
      if (!resp.ok) throw new Error("Simulation unavailable");
      const data = await resp.json();
      simState = data.simulation?.state;
      if (!simState) throw new Error("Simulation not initialized");
      label.textContent = `Game minute ${simState.minute} · ${data.simulation.enabled ? "Running" : "Paused"} · workshop parts ${simState.stock["workshop.parts"]}`;
      const select = el("townEventForm").elements.participants;
      const selected = new Set(Array.from(select.selectedOptions).map(o => o.value));
      select.replaceChildren(...Object.entries(simState.agents).map(([id, a]) => {
        const option = document.createElement("option"); option.value = id; option.textContent = a.name;
        option.selected = selected.has(id); return option;
      }));
      el("townQuestList").textContent = Object.values(simState.quests).map(q => `${q.title}: ${q.status} (deadline ${q.deadline})${q.effect_error ? " — reward/penalty failed" : ""}`).join("\n") || "No quests yet.";
    } catch (err) { label.textContent = err.message; simState = null; }
  }
  const eventForm = el("townEventForm");
  if (eventForm) eventForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const feedback = el("townEventResult");
    if (!simState) { feedback.textContent = "Wait for the simulation to connect."; return; }
    const kind = eventForm.elements.kind.value;
    const title = eventForm.elements.title.value.trim();
    const participants = Array.from(eventForm.elements.participants.selectedOptions).map(o => o.value);
    if (kind === "quest" && participants.length < 2) { feedback.textContent = "Select at least two quest participants."; return; }
    eventId = eventId || crypto.randomUUID();
    const body = {kind, title, correlation_id:eventId, effects:kind === "delivery" ? [{op:"stock_delta",target:"workshop.parts",value:3}] : []};
    if (kind === "quest") body.quest = {id:eventId,title,participants,deadline:simState.minute + Number(eventForm.elements.duration.value),goal:[{target:"fact:generator.fixed",cmp:"eq",value:true}],steps:[],reward:[],penalty:[]};
    const button = eventForm.querySelector("button"); button.disabled = true;
    try {
      const response = await fetch(`${API_BASE}/api/aitown/events`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Invalid event");
      feedback.textContent = result.accepted ? "Added to the town." : `Refused: ${result.reason}`;
      if (result.accepted) { eventId = null; await refreshSimulation(); }
    } catch (err) { feedback.textContent = `${err.message}. Retry uses the same event ID.`; }
    finally { button.disabled = false; }
  });

  window.OrionAitownPanel = {
    activate() {
      loadIframe();
      startPolling();
      refreshSimulation();
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
