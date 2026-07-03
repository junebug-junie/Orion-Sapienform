(function () {
  const pathSegments = window.location.pathname.split('/').filter((p) => p.length > 0);
  const URL_PREFIX = pathSegments.length > 0 ? `/${pathSegments[0]}` : "";
  const API_BASE_URL = window.location.origin + URL_PREFIX;

  const SUMMARY_ENDPOINT = "/api/substrate/observability/summary";
  const PANEL_HASH = "#self-observability";
  const AUTO_REFRESH_MS = 30000;
  const MAX_GAP_SIGNALS = 5;
  const NO_DATA_TEXT = "no data yet";

  document.addEventListener("DOMContentLoaded", () => {
    const panel = document.getElementById("self-observability");
    const tabButton = document.getElementById("selfObservabilityTabButton");
    const statusEl = document.getElementById("selfObsStatus");
    const refreshBtn = document.getElementById("selfObsRefresh");
    const attentionTypeEl = document.getElementById("selfObsAttentionType");
    const dwellTicksEl = document.getElementById("selfObsDwellTicks");
    const nodeCountEl = document.getElementById("selfObsNodeCount");
    const coalitionDescEl = document.getElementById("selfObsCoalitionDesc");
    const stabilityEl = document.getElementById("selfObsStability");
    const attendedCountEl = document.getElementById("selfObsAttendedCount");
    const gapCountEl = document.getElementById("selfObsGapCount");
    const gapListEl = document.getElementById("selfObsGapList");
    const presenceHealthEl = document.getElementById("selfObsPresenceHealth");
    const lastTurnAgeEl = document.getElementById("selfObsLastTurnAge");
    const turnsPerMinEl = document.getElementById("selfObsTurnsPerMin");

    if (!panel || !tabButton || !statusEl) {
      return;
    }

    let autoRefreshTimer = null;
    let fetchInFlight = false;

    function setStatus(text, isError) {
      statusEl.textContent = text;
      statusEl.classList.toggle("text-red-400", Boolean(isError));
      statusEl.classList.toggle("text-gray-400", !isError);
    }

    function setText(el, value) {
      if (el) el.textContent = value;
    }

    function truncate(text, maxLen) {
      const raw = typeof text === "string" ? text : "";
      if (raw.length <= maxLen) return raw;
      return `${raw.slice(0, maxLen - 1)}…`;
    }

    function humanizeAgeSeconds(seconds) {
      const value = Number(seconds);
      if (!Number.isFinite(value) || value < 0) return "—";
      if (value < 90) return `${Math.round(value)}s`;
      return `${(value / 60).toFixed(1)}m`;
    }

    function renderGapListPlaceholder(text) {
      if (!gapListEl) return;
      gapListEl.innerHTML = "";
      const li = document.createElement("li");
      li.className = "text-gray-500 list-none";
      li.textContent = text;
      gapListEl.appendChild(li);
    }

    function renderAttentionSchema(selfState) {
      if (!selfState || typeof selfState !== "object") {
        setText(attentionTypeEl, NO_DATA_TEXT);
        setText(dwellTicksEl, "—");
        setText(nodeCountEl, "—");
        return;
      }
      setText(attentionTypeEl, selfState.attention_schema_type || "none");
      setText(dwellTicksEl, String(selfState.attention_dwell_ticks ?? "—"));
      setText(nodeCountEl, String(selfState.attention_node_count ?? "—"));
    }

    function renderCoalitionFocus(broadcast) {
      if (!broadcast || typeof broadcast !== "object") {
        setText(coalitionDescEl, NO_DATA_TEXT);
        setText(stabilityEl, "—");
        setText(attendedCountEl, "—");
        return;
      }
      setText(coalitionDescEl, broadcast.selected_description || "(no selected coalition)");
      const stability = Number(broadcast.coalition_stability_score);
      setText(stabilityEl, Number.isFinite(stability) ? stability.toFixed(2) : "—");
      const attended = Array.isArray(broadcast.attended_node_ids) ? broadcast.attended_node_ids.length : 0;
      setText(attendedCountEl, String(attended));
    }

    function renderCuriosityGaps(curiosity) {
      if (!curiosity || typeof curiosity !== "object") {
        setText(gapCountEl, "—");
        renderGapListPlaceholder(NO_DATA_TEXT);
        return;
      }
      setText(gapCountEl, String(curiosity.gap_count ?? 0));
      const signals = Array.isArray(curiosity.signals) ? curiosity.signals : [];
      if (signals.length === 0) {
        renderGapListPlaceholder("no open curiosity signals");
        return;
      }
      gapListEl.innerHTML = "";
      signals.slice(0, MAX_GAP_SIGNALS).forEach((signal) => {
        const li = document.createElement("li");
        const strength = Number(signal && signal.signal_strength);
        const strengthLabel = Number.isFinite(strength) ? strength.toFixed(2) : "?";
        const signalType = (signal && signal.signal_type) || "unknown";
        const summary = truncate((signal && signal.evidence_summary) || "", 96);
        li.textContent = summary ? `${signalType} · ${strengthLabel} — ${summary}` : `${signalType} · ${strengthLabel}`;
        gapListEl.appendChild(li);
      });
    }

    function renderHubPresence(presence) {
      if (!presence || typeof presence !== "object") {
        setText(presenceHealthEl, NO_DATA_TEXT);
        setText(lastTurnAgeEl, "—");
        setText(turnsPerMinEl, "—");
        return;
      }
      setText(presenceHealthEl, presence.connection_health || "unknown");
      setText(lastTurnAgeEl, humanizeAgeSeconds(presence.last_turn_age_sec));
      const tpm = Number(presence.turns_per_minute);
      setText(turnsPerMinEl, Number.isFinite(tpm) ? tpm.toFixed(1) : "—");
    }

    async function refreshSummary() {
      if (fetchInFlight) return;
      fetchInFlight = true;
      setStatus("Loading self-observability summary...", false);
      try {
        const response = await fetch(`${API_BASE_URL}${SUMMARY_ENDPOINT}`);
        if (!response.ok) {
          throw new Error(`status=${response.status}`);
        }
        const payload = await response.json();
        renderAttentionSchema(payload && payload.self_state);
        renderCoalitionFocus(payload && payload.attention_broadcast);
        renderCuriosityGaps(payload && payload.curiosity);
        renderHubPresence(payload && payload.hub_presence);
        const generatedAt = payload && payload.generated_at ? ` (generated ${payload.generated_at})` : "";
        setStatus(`Updated ${new Date().toLocaleTimeString()}${generatedAt}`, false);
      } catch (err) {
        console.warn("[SelfObservability] summary fetch failed", err);
        setStatus(`Self-observability fetch failed: ${err.message || err}`, true);
      } finally {
        fetchInFlight = false;
      }
    }

    function styleTabButton(button, isActive) {
      if (!button) return;
      button.classList.toggle("bg-indigo-600", isActive);
      button.classList.toggle("text-white", isActive);
      button.classList.toggle("border-indigo-500", isActive);
      button.classList.toggle("bg-gray-800", !isActive);
      button.classList.toggle("text-gray-200", !isActive);
      button.classList.toggle("border-gray-700", !isActive);
    }

    function startAutoRefresh() {
      if (autoRefreshTimer) return;
      autoRefreshTimer = setInterval(() => {
        if (panel.classList.contains("hidden")) return;
        refreshSummary();
      }, AUTO_REFRESH_MS);
    }

    function stopAutoRefresh() {
      if (!autoRefreshTimer) return;
      clearInterval(autoRefreshTimer);
      autoRefreshTimer = null;
    }

    function activatePanel() {
      document.querySelectorAll("#appPanels section[data-panel]").forEach((section) => {
        const key = section.getAttribute("data-panel");
        section.classList.toggle("hidden", key !== "self-observability");
      });
      document.querySelectorAll("a[data-hash-target]").forEach((anchor) => {
        styleTabButton(anchor, anchor === tabButton);
      });
      history.replaceState(null, "", PANEL_HASH);
      startAutoRefresh();
      refreshSummary();
    }

    function deactivatePanel() {
      stopAutoRefresh();
      panel.classList.add("hidden");
      styleTabButton(tabButton, false);
    }

    tabButton.addEventListener("click", (event) => {
      event.preventDefault();
      activatePanel();
    });

    // When any other tab button is clicked, app.js switches panels via
    // setActiveTab without knowing about this panel — hide it ourselves.
    document.querySelectorAll("a[data-hash-target]").forEach((anchor) => {
      if (anchor === tabButton) return;
      anchor.addEventListener("click", () => {
        deactivatePanel();
      });
    });

    // Direct hash navigation. Defer so app.js's hashchange handler
    // (which falls back to the hub tab for unknown hashes) runs first.
    window.addEventListener("hashchange", () => {
      setTimeout(() => {
        if (window.location.hash === PANEL_HASH) {
          activatePanel();
        } else if (!panel.classList.contains("hidden")) {
          deactivatePanel();
        }
      }, 0);
    });

    if (window.location.hash === PANEL_HASH) {
      setTimeout(() => activatePanel(), 0);
    }
  });
})();
