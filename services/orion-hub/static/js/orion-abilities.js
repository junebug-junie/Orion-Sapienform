// "Ask Orion" modal: a catalog of things Orion can do from a chat message.
//
// Juniper clicks the button above Send, picks a card, and the card's message is
// put into the chat box at the cursor. Nothing is sent: she edits and sends.
//
// The catalog lives in one checked-in file, static/catalog/orion-abilities.json.
// tests/test_orion_abilities_catalog.py fails CI when a dig command or an
// Orion MCP tool has no entry there, so this list cannot quietly drift.
//
// Pure helpers are exported for node:test (orion-abilities.test.js); the DOM
// wiring runs only in a browser.
(function () {
  "use strict";

  var CATALOG_URL = "/static/catalog/orion-abilities.json";

  // Full message text for one entry: the plain-English ask, then an optional
  // hint line with the exact command (Orion routes by topic, not by phrase).
  function composeMessage(entry) {
    return composeInsert(entry).text;
  }

  // The text to insert and where the caret should land inside it. A message
  // that ends with ":" asks Juniper to fill something in (a date, a link), so
  // the caret lands right after it, not after the hint line.
  function composeInsert(entry) {
    var msg = String((entry && entry.message) || "").trim();
    var hint = String((entry && entry.hint) || "").trim();
    if (/:$/.test(msg)) {
      return { text: msg + " " + (hint ? "\n" + hint : ""), caretIn: msg.length + 1 };
    }
    var text = hint ? msg + "\n" + hint : msg;
    return { text: text, caretIn: text.length };
  }

  // Insert `text` into `value` at `caret` without deleting anything. Blank
  // lines separate it from text already typed before or after. Returns the new
  // value and the caret position right after the inserted text.
  function insertAtCaret(value, caret, text) {
    var v = String(value || "");
    var t = String(text || "");
    var at = typeof caret === "number" && caret >= 0 && caret <= v.length ? caret : v.length;
    var before = v.slice(0, at);
    var after = v.slice(at);
    var lead = "";
    if (before.trim()) {
      if (!/\n\n$/.test(before)) lead = /\n$/.test(before) ? "\n" : "\n\n";
    } else {
      before = before.replace(/\s+$/, "");
    }
    var tail = "";
    if (after.trim() && !/^\n\n/.test(after)) tail = /^\n/.test(after) ? "\n" : "\n\n";
    var next = before + lead + t + tail + after;
    var start = before.length + lead.length;
    return { value: next, start: start, caret: start + t.length };
  }

  function entryHaystack(section, entry) {
    return [
      section.title, section.blurb, entry.title, entry.description, entry.message, entry.hint,
    ].concat(entry.keywords || []).join(" ").toLowerCase();
  }

  // Sections with only the entries that match every word of `query`.
  function filterCatalog(catalog, query) {
    var words = String(query || "").toLowerCase().split(/\s+/).filter(Boolean);
    var sections = (catalog && catalog.sections) || [];
    var out = [];
    sections.forEach(function (section) {
      var entries = (section.entries || []).filter(function (entry) {
        if (!words.length) return true;
        var hay = entryHaystack(section, entry);
        return words.every(function (w) { return hay.indexOf(w) !== -1; });
      });
      if (entries.length) out.push({ section: section, entries: entries });
    });
    return out;
  }

  function countEntries(groups) {
    return groups.reduce(function (n, g) { return n + g.entries.length; }, 0);
  }

  var api = {
    CATALOG_URL: CATALOG_URL,
    composeMessage: composeMessage,
    composeInsert: composeInsert,
    insertAtCaret: insertAtCaret,
    filterCatalog: filterCatalog,
    countEntries: countEntries,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window === "undefined" || typeof document === "undefined") return;
  window.OrionAbilities = api;

  // ---- DOM wiring -------------------------------------------------------

  var FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function init() {
    var openBtn = document.getElementById("orionAbilitiesButton");
    var root = document.getElementById("orionAbilitiesModalRoot");
    var dialog = document.getElementById("orionAbilitiesDialog");
    var list = document.getElementById("orionAbilitiesList");
    var search = document.getElementById("orionAbilitiesSearch");
    var count = document.getElementById("orionAbilitiesCount");
    var nav = document.getElementById("orionAbilitiesNav");
    var closeBtn = document.getElementById("orionAbilitiesClose");
    var backdrop = document.getElementById("orionAbilitiesBackdrop");
    var input = document.getElementById("chatInput");
    if (!openBtn || !root || !dialog || !list || !search || !input) return;

    var catalog = null;
    var loading = null;
    var returnFocus = null;
    // Where the caret sat in the chat box when the modal opened. A textarea
    // keeps its selection after blur, but read it now so nothing moves it.
    var savedCaret = null;
    var countTimer = null;
    // A textarea nobody has clicked or typed in reports caret 0 even when code filled it;
    // only trust its caret once Juniper has put it somewhere herself.
    var caretTouched = false;
    ["focus", "keyup", "mouseup", "select"].forEach(function (ev) {
      input.addEventListener(ev, function () { caretTouched = true; });
    });
    // While open, everything else on the page is inert (aria-modal alone is weak in some readers).
    var inerted = [];
    function setBackgroundInert(on) {
      if (on) {
        // The modal sits inside the page's own containers, so inert every sibling along the
        // path from the modal up to <body>, never one of its ancestors.
        inerted = [];
        for (var node = root; node && node !== document.body && node.parentElement; node = node.parentElement) {
          Array.prototype.forEach.call(node.parentElement.children, function (n) {
            if (n !== node && !n.hasAttribute("inert") && n.tagName !== "SCRIPT" && n.tagName !== "STYLE") inerted.push(n);
          });
        }
        inerted.forEach(function (n) { n.setAttribute("inert", ""); });
      } else {
        inerted.forEach(function (n) { n.removeAttribute("inert"); });
        inerted = [];
      }
    }

    function load() {
      if (catalog) return Promise.resolve(catalog);
      if (loading) return loading;
      var version = (document.body && document.body.getAttribute("data-ui-version")) || "";
      loading = fetch(CATALOG_URL + (version ? "?v=" + encodeURIComponent(version) : ""), { cache: "no-cache" })
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then(function (data) { catalog = data; return data; })
        .catch(function (err) { loading = null; throw err; });
      return loading;
    }

    function insert(entry) {
      var piece = composeInsert(entry);
      if (!piece.text) return;
      var caret = savedCaret == null ? input.value.length : savedCaret;
      var res = insertAtCaret(input.value, caret, piece.text);
      var land = res.start + piece.caretIn;
      input.value = res.value;
      // Same event typing makes: keeps the expanded editor and any listeners in step.
      input.dispatchEvent(new Event("input", { bubbles: true }));
      close(false);
      input.focus();
      try { input.setSelectionRange(land, land); } catch (_) { /* not a text control */ }
    }

    function render() {
      list.textContent = "";
      if (nav) nav.textContent = "";
      if (!catalog) return;
      var groups = filterCatalog(catalog, search.value);
      var total = countEntries(filterCatalog(catalog, ""));
      var shown = countEntries(groups);
      // aria-live: say the count once typing pauses, not on every keystroke
      var countText = search.value.trim() ? shown + " of " + total + " shown" : total + " things to ask";
      if (count) {
        clearTimeout(countTimer);
        countTimer = setTimeout(function () { count.textContent = countText; }, search.value.trim() ? 300 : 0);
      }
      if (!groups.length) {
        list.appendChild(el("p", "oa-empty", "Nothing matches “" + search.value.trim() + "”. Try a plainer word, like body, cost or past."));
        return;
      }
      groups.forEach(function (g) {
        var sec = el("section", "oa-section");
        sec.id = "oa-sec-" + g.section.id;
        sec.setAttribute("aria-labelledby", sec.id + "-h");
        var head = el("div", "oa-section-head");
        var h = el("h3", "oa-section-title", g.section.title);
        h.id = sec.id + "-h";
        h.tabIndex = -1;
        head.appendChild(h);
        if (g.section.blurb) head.appendChild(el("p", "oa-section-blurb", g.section.blurb));
        sec.appendChild(head);
        var grid = el("div", "oa-grid");
        g.entries.forEach(function (entry) {
          var card = el("article", "oa-card");
          card.setAttribute("data-entry-id", entry.id);
          card.appendChild(el("h4", "oa-card-title", entry.title));
          card.appendChild(el("p", "oa-card-desc", entry.description));
          card.appendChild(el("p", "oa-card-preview", entry.message));
          if (entry.hint) card.appendChild(el("code", "oa-card-hint", entry.hint));
          var btn = el("button", "oa-insert", "Insert");
          btn.type = "button";
          btn.setAttribute("aria-label", "Insert into chat: " + entry.title);
          btn.addEventListener("click", function () { insert(entry); });
          card.appendChild(btn);
          grid.appendChild(card);
        });
        sec.appendChild(grid);
        list.appendChild(sec);
        if (nav) {
          var chip = el("button", "oa-chip", g.section.title + " · " + g.entries.length);
          chip.type = "button";
          chip.addEventListener("click", function () {
            var calm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
            sec.scrollIntoView({ behavior: calm ? "auto" : "smooth", block: "start" });
            h.focus({ preventScroll: true });
          });
          nav.appendChild(chip);
        }
      });
    }

    function open() {
      returnFocus = document.activeElement && document.activeElement !== document.body ? document.activeElement : openBtn;
      savedCaret = caretTouched && typeof input.selectionEnd === "number" ? input.selectionEnd : null;
      root.classList.remove("hidden");
      root.setAttribute("aria-hidden", "false");
      setBackgroundInert(true);
      openBtn.setAttribute("aria-expanded", "true");
      document.documentElement.classList.add("oa-scroll-lock");
      search.value = "";
      list.textContent = "";
      list.appendChild(el("p", "oa-empty", "Loading…"));
      search.focus();
      load().then(render).catch(function (err) {
        list.textContent = "";
        list.appendChild(el("p", "oa-empty oa-error", "Could not load the list (" + err.message + "). Close and try again."));
      });
    }

    function close(restoreFocus) {
      if (root.classList.contains("hidden")) return;
      setBackgroundInert(false);
      root.classList.add("hidden");
      root.setAttribute("aria-hidden", "true");
      openBtn.setAttribute("aria-expanded", "false");
      document.documentElement.classList.remove("oa-scroll-lock");
      if (restoreFocus !== false && returnFocus && typeof returnFocus.focus === "function") returnFocus.focus();
    }

    openBtn.addEventListener("click", open);
    if (closeBtn) closeBtn.addEventListener("click", function () { close(); });
    if (backdrop) backdrop.addEventListener("click", function () { close(); });
    search.addEventListener("input", render);
    search.addEventListener("keydown", function (e) {
      // Enter in the search box inserts the first card shown.
      if (e.key === "Enter" && !e.isComposing) {
        e.preventDefault();
        var first = list.querySelector(".oa-insert");
        if (first) first.click();
      }
    });
    root.addEventListener("keydown", function (e) {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        close();
        return;
      }
      if (e.key !== "Tab") return;
      var items = Array.prototype.filter.call(dialog.querySelectorAll(FOCUSABLE), function (n) {
        return n.offsetParent !== null || n === document.activeElement;
      });
      if (!items.length) return;
      var first = items[0];
      var last = items[items.length - 1];
      if (e.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (document.activeElement === last || !dialog.contains(document.activeElement))) {
        e.preventDefault();
        first.focus();
      }
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
