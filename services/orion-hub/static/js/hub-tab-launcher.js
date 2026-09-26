// Hub tab launcher: the section tabs live in a grid modal (Gmail-apps style)
// opened from the header button, instead of a scrolling pill strip.
//
// The tab anchors themselves are unchanged -- same ids, hrefs and
// data-hash-target -- so app.js / *_tab.js keep wiring clicks and toggling the
// active classes on them. This module only (1) orders them (Hub pinned first,
// the rest alphabetical), (2) opens/closes/filters the modal, and (3) mirrors
// the active tab's label onto the header button.
(function (root) {
  "use strict";

  const PINNED_ID = "hubTabButton";
  const ACTIVE_CLASS = "bg-indigo-600";

  function tabLabel(anchor) {
    return String((anchor && anchor.textContent) || "").replace(/\s+/g, " ").trim();
  }

  // Pure ordering: pinned id first, everything else case-insensitive A->Z.
  function orderHubTabs(items, pinnedId) {
    const pin = pinnedId || PINNED_ID;
    const pinned = items.filter((item) => item.id === pin);
    const rest = items
      .filter((item) => item.id !== pin)
      .sort((a, b) => a.label.localeCompare(b.label, undefined, { sensitivity: "base" }));
    return pinned.concat(rest);
  }

  function matchesFilter(label, query) {
    const q = String(query || "").trim().toLowerCase();
    if (!q) return true;
    return label.toLowerCase().includes(q);
  }

  function initHubTabLauncher(doc) {
    const nav = doc.getElementById("hubPrimaryNav");
    const button = doc.getElementById("hubTabLauncherButton");
    const modal = doc.getElementById("hubTabLauncherModal");
    const current = doc.getElementById("hubTabLauncherCurrent");
    const filter = doc.getElementById("hubTabLauncherFilter");
    const empty = doc.getElementById("hubTabLauncherEmpty");
    if (!nav || !button || !modal) return null;

    const anchors = Array.from(nav.querySelectorAll("a[data-hash-target]"));
    const ordered = orderHubTabs(
      anchors.map((anchor) => ({ id: anchor.id, label: tabLabel(anchor), anchor })),
      PINNED_ID,
    );
    ordered.forEach(({ anchor, label, id }) => {
      anchor.dataset.initial = label.charAt(0).toUpperCase();
      anchor.classList.toggle("hub-tab-pinned", id === PINNED_ID);
      nav.appendChild(anchor);
    });

    function visibleAnchors() {
      return ordered.map((o) => o.anchor).filter((a) => !a.hidden);
    }

    function applyFilter() {
      const query = filter ? filter.value : "";
      let shown = 0;
      ordered.forEach(({ anchor, label }) => {
        const ok = matchesFilter(label, query);
        anchor.hidden = !ok;
        if (ok) shown += 1;
      });
      if (empty) empty.hidden = shown > 0;
    }

    function syncCurrent() {
      if (!current) return;
      const active = ordered.find((o) => o.anchor.classList.contains(ACTIVE_CLASS));
      current.textContent = active ? active.label : "Hub";
    }

    function isOpen() {
      return !modal.hidden;
    }

    function open() {
      if (isOpen()) return;
      modal.hidden = false;
      button.setAttribute("aria-expanded", "true");
      if (filter) {
        filter.value = "";
        applyFilter();
        filter.focus();
      }
    }

    function close(restoreFocus) {
      if (!isOpen()) return;
      modal.hidden = true;
      button.setAttribute("aria-expanded", "false");
      if (restoreFocus) button.focus();
    }

    button.addEventListener("click", (event) => {
      event.preventDefault();
      if (isOpen()) close(false);
      else open();
    });

    // Backdrop click (the overlay itself, not the panel) closes. The overlay
    // starts below the header, so a click on another header control (e.g. the
    // runtime marquee) must close it too.
    const panel = modal.firstElementChild;
    modal.addEventListener("click", (event) => {
      if (event.target === modal) close(false);
    });
    doc.addEventListener("click", (event) => {
      if (!isOpen()) return;
      const t = event.target;
      if (button.contains(t) || modal.contains(t)) return;
      close(false);
    });

    // Picking a tab closes the launcher; app.js's own click handler on the
    // anchor still runs and switches the panel. Focus returns to the button,
    // since the chosen anchor is now hidden and would drop focus to <body>.
    ordered.forEach(({ anchor }) => {
      anchor.addEventListener("click", () => close(true));
    });

    doc.addEventListener("keydown", (event) => {
      if (!isOpen()) return;
      if (event.key === "Escape") {
        event.preventDefault();
        close(true);
        return;
      }
      // aria-modal: keep Tab cycling inside the launcher.
      if (event.key === "Tab") {
        const stops = (filter ? [filter] : []).concat(visibleAnchors());
        if (!stops.length) return;
        const first = stops[0];
        const last = stops[stops.length - 1];
        const at = doc.activeElement;
        const inside = panel ? panel.contains(at) : stops.includes(at);
        if (event.shiftKey && (at === first || !inside)) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && (at === last || !inside)) {
          event.preventDefault();
          first.focus();
        }
      }
    });

    if (filter) {
      filter.addEventListener("input", applyFilter);
      filter.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" || event.isComposing) return;
        const first = visibleAnchors()[0];
        if (first) {
          event.preventDefault();
          first.click();
        }
      });
    }

    // Active state is toggled by several scripts via class changes (and
    // clicks use history.replaceState, which fires no hashchange), so watch
    // the classes directly.
    const Observer = doc.defaultView && doc.defaultView.MutationObserver;
    if (Observer) {
      new Observer(syncCurrent).observe(nav, {
        attributes: true,
        attributeFilter: ["class"],
        subtree: true,
      });
    }
    syncCurrent();

    return { open, close, isOpen, applyFilter, syncCurrent, ordered };
  }

  const api = { orderHubTabs, matchesFilter, initHubTabLauncher, tabLabel };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  if (root && root.document) {
    root.HubTabLauncher = api;
    const start = () => initHubTabLauncher(root.document);
    if (root.document.readyState === "loading") {
      root.document.addEventListener("DOMContentLoaded", start);
    } else {
      start();
    }
  }
})(typeof window !== "undefined" ? window : null);
