// Collapse Mirror — standalone init
// Extracted from app.js so the Collapse Mirror tab can live in its own file.
// Works whether loaded before or after DOM ready.
(function () {
  function initCollapseMirror() {
    const guidedBtn = document.getElementById("collapseModeGuided");
    const rawBtn = document.getElementById("collapseModeRaw");
    const guidedSection = document.getElementById("collapseGuidedSection");
    const rawSection = document.getElementById("collapseRawSection");

    const tooltipToggle = document.getElementById("collapseTooltipToggle");
    const tooltip = document.getElementById("collapseTooltip");

    const statusEl = document.getElementById("collapseStatus");

    const guidedSubmit =
      document.getElementById("collapseGuidedSubmit") ||
      document.getElementById("collapseSubmit"); // legacy id fallback

    const rawSubmit =
      document.getElementById("collapseRawSubmit") ||
      document.getElementById("collapseSubmitRaw"); // legacy id fallback

    const rawJsonEl = document.getElementById("collapseRawJson");

    // Guided inputs
    const el = (id) => document.getElementById(id);
    const observerEl = el("collapseObserver");
    const typeEl = el("collapseType");
    const triggerEl = el("collapseTrigger");
    const observerStateEl = el("collapseObserverState");
    const fieldResonanceEl = el("collapseFieldResonance");
    const emergentEntityEl = el("collapseEmergentEntity");
    const summaryEl = el("collapseSummary");
    const mantraEl = el("collapseMantra");
    const causalEchoEl = el("collapseCausalEcho");

    function setCollapseStatus(msg, isError=false) {
      if (statusEl) {
        statusEl.textContent = msg;
        statusEl.classList.toggle("text-red-400", !!isError);
        statusEl.classList.toggle("text-gray-400", !isError);
        return;
      }
      // If template is missing status element, do something visible.
      console[isError ? "error" : "log"]("[collapse]", msg);
      if (isError) alert(msg);
    }

    function showGuided() {
      if (guidedSection) guidedSection.classList.remove("hidden");
      if (rawSection) rawSection.classList.add("hidden");
      if (guidedBtn) {
        guidedBtn.classList.add("bg-gray-700", "text-gray-100");
        guidedBtn.classList.remove("text-gray-300");
      }
      if (rawBtn) {
        rawBtn.classList.remove("bg-gray-700", "text-gray-100");
        rawBtn.classList.add("text-gray-300");
      }
    }

    function showRaw() {
      if (rawSection) rawSection.classList.remove("hidden");
      if (guidedSection) guidedSection.classList.add("hidden");
      if (rawBtn) {
        rawBtn.classList.add("bg-gray-700", "text-gray-100");
        rawBtn.classList.remove("text-gray-300");
      }
      if (guidedBtn) {
        guidedBtn.classList.remove("bg-gray-700", "text-gray-100");
        guidedBtn.classList.add("text-gray-300");
      }
    }

    if (guidedBtn) guidedBtn.addEventListener("click", showGuided);
    if (rawBtn) rawBtn.addEventListener("click", showRaw);

    if (tooltipToggle && tooltip) {
      tooltipToggle.addEventListener("click", () => tooltip.classList.toggle("hidden"));
    }

    async function postCollapse(payload) {
      setCollapseStatus("Submitting…");
      try {
        const resp = await fetch("/submit-collapse", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const txt = await resp.text();
        let data = null;
        try { data = txt ? JSON.parse(txt) : null; } catch { data = { raw: txt }; }

        if (!resp.ok) {
          const detail =
            (data && (data.error || data.detail || data.message)) ||
            `HTTP ${resp.status}`;
          setCollapseStatus(`❌ ${detail}`, true);
          return;
        }

        if (data && data.success === false) {
          setCollapseStatus(`❌ ${data.error || "Unknown error"}`, true);
          return;
        }

        setCollapseStatus("✅ Collapse submitted.");
      } catch (e) {
        setCollapseStatus(`❌ Network error: ${e}`, true);
      }
    }

    function buildGuidedPayload() {
      // If the guided inputs don't exist, fail loudly.
      const missing = [];
      const must = [
        ["observer", observerEl],
        ["type", typeEl],
        ["trigger", triggerEl],
        ["observer_state", observerStateEl],
        ["field_resonance", fieldResonanceEl],
        ["emergent_entity", emergentEntityEl],
        ["summary", summaryEl],
        ["mantra", mantraEl],
      ];
      for (const [name, node] of must) if (!node) missing.push(name);
      if (missing.length) {
        setCollapseStatus(`❌ UI missing required inputs: ${missing.join(", ")}`, true);
        return null;
      }

      const observer_state = (observerStateEl.value || "")
        .split(/\r?\n/)
        .map((s) => s.trim())
        .filter(Boolean);

      const payload = {
        observer: observerEl.value.trim(),
        trigger: triggerEl.value.trim(),
        observer_state,
        field_resonance: fieldResonanceEl.value.trim(),
        type: typeEl.value.trim(),
        emergent_entity: emergentEntityEl.value.trim(),
        summary: summaryEl.value.trim(),
        mantra: mantraEl.value.trim(),
        causal_echo: (causalEchoEl && causalEchoEl.value.trim()) || null,
      };

      // Basic validation (FastAPI will also validate)
      const req = [
        ["observer", payload.observer],
        ["trigger", payload.trigger],
        ["observer_state", payload.observer_state.length ? "ok" : ""],
        ["field_resonance", payload.field_resonance],
        ["type", payload.type],
        ["emergent_entity", payload.emergent_entity],
        ["summary", payload.summary],
        ["mantra", payload.mantra],
      ];
      const missing2 = req.filter(([_, v]) => !v).map(([k]) => k);
      if (missing2.length) {
        setCollapseStatus(`❌ Missing: ${missing2.join(", ")}`, true);
        return null;
      }

      return payload;
    }

    if (guidedSubmit) {
      guidedSubmit.addEventListener("click", async (e) => {
        e.preventDefault();
        const payload = buildGuidedPayload();
        if (!payload) return;
        await postCollapse(payload);
      });
    }

    if (rawSubmit) {
      rawSubmit.addEventListener("click", async (e) => {
        e.preventDefault();
        const raw = (rawJsonEl && rawJsonEl.value) ? rawJsonEl.value.trim() : "";
        if (!raw) {
          setCollapseStatus("❌ Paste JSON first.", true);
          return;
        }
        try {
          const payload = JSON.parse(raw);
          await postCollapse(payload);
        } catch (err) {
          setCollapseStatus(`❌ Invalid JSON: ${err}`, true);
        }
      });
    }

    // default to guided
    showGuided();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCollapseMirror);
  } else {
    initCollapseMirror();
  }
})();
