(function () {
  // The Hub Surface tab body is a standalone iframe (/hub-surface, served by
  // hub_surface_routes.py) that fetches and renders its own data. app.js's
  // tab router does not know about #hub-surface, so this script's only job
  // is panel show/hide + hash lifecycle + the iframe refresh button,
  // matching the pattern self_observability.js established for #self-observability.
  const PANEL_HASH = "#hub-surface";

  document.addEventListener("DOMContentLoaded", () => {
    const panel = document.getElementById("hub-surface");
    const tabButton = document.getElementById("hubSurfaceTabButton");
    const panelFrame = document.getElementById("hubSurfacePanelFrame");
    const panelRefresh = document.getElementById("hubSurfacePanelRefresh");

    if (!panel || !tabButton) {
      return;
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

    function activatePanel() {
      document.querySelectorAll("#appPanels section[data-panel]").forEach((section) => {
        const key = section.getAttribute("data-panel");
        section.classList.toggle("hidden", key !== "hub-surface");
      });
      document.querySelectorAll("a[data-hash-target]").forEach((anchor) => {
        styleTabButton(anchor, anchor === tabButton);
      });
      history.replaceState(null, "", PANEL_HASH);
    }

    function deactivatePanel() {
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

    if (panelRefresh && panelFrame) {
      panelRefresh.addEventListener("click", () => {
        panelFrame.contentWindow?.location.reload();
      });
    }
  });
})();
