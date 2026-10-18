(function () {
  // Use the existing self-routing Hub tab pattern. Unload on exit to revoke play.
  const PANEL_HASH = "#retro";

  document.addEventListener("DOMContentLoaded", () => {
    const panel = document.getElementById("retro");
    const tabButton = document.getElementById("retroTabButton");
    const panelFrame = document.getElementById("retroPanelFrame");

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
        section.classList.toggle("hidden", key !== "retro");
      });
      document.querySelectorAll("a[data-hash-target]").forEach((anchor) => {
        styleTabButton(anchor, anchor === tabButton);
      });
      if (panelFrame.getAttribute("src") !== "/retro") panelFrame.src = "/retro";
      history.replaceState(null, "", PANEL_HASH);
    }

    function deactivatePanel() {
      panelFrame.src = "about:blank";
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
        } else {
          deactivatePanel();
        }
      }, 0);
    });

    if (window.location.hash === PANEL_HASH) {
      setTimeout(() => activatePanel(), 0);
    }

  });
})();
