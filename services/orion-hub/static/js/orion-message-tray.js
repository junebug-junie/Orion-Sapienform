// Orion's own messages in the Hub notification tray, plus the self-clearing
// alert rule. Pure, DOM-free logic only -- app.js does the rendering.
//
// Exposed as window.OrionMessageTray in the browser and as a CommonJS module
// under node:test. Must load before app.js.
(function () {
  "use strict";

  var OUTREACH_TAG = "endogenous_outreach";
  var CLEARED_SUFFIX = "_cleared";

  function timeOf(item) {
    var raw = (item && (item.received_at || item.created_at)) || 0;
    var t = Date.parse(raw);
    return isNaN(t) ? 0 : t;
  }

  // True when a tray item is one of Orion's own outreach messages: the live
  // in-app event (notification_type) or any item tagged as outreach.
  function isOrionMessage(item) {
    if (!item) return false;
    if (item.notification_type === OUTREACH_TAG) return true;
    var tags = Array.isArray(item.tags) ? item.tags : [];
    return tags.indexOf(OUTREACH_TAG) !== -1;
  }

  // One shape for both sources: a durable row from /api/chat/messages
  // (ChatMessageState) and a live in-app event (HubNotificationEvent).
  function normalizeMessage(item) {
    if (!item || !item.message_id) return null;
    return {
      message_id: String(item.message_id),
      session_id: item.session_id || "",
      correlation_id: item.correlation_id || "",
      title: item.title || "Orion reached out",
      text: String(item.full_text || item.body_text || item.preview_text || ""),
      created_at: item.created_at || item.received_at || null,
    };
  }

  // Open Orion messages, newest first, one per message_id. `durable` comes
  // from orion-notify (survives restarts); `live` are tray items that arrived
  // on this page (covers orion-notify being down). `dismissed` holds ids
  // Juniper dismissed on this page, so a slow receipt write cannot bring a
  // message back on the next refresh.
  function mergeOpenMessages(durable, live, dismissed) {
    var gone = dismissed || new Set();
    var byId = new Map();
    function add(item) {
      var m = normalizeMessage(item);
      if (!m || gone.has(m.message_id)) return;
      var prev = byId.get(m.message_id);
      // Durable rows carry the full text; prefer whichever has more.
      if (!prev || (m.text || "").length > (prev.text || "").length) byId.set(m.message_id, m);
    }
    (durable || []).forEach(add);
    (live || []).filter(isOrionMessage).forEach(add);
    return Array.from(byId.values()).sort(function (a, b) {
      return timeOf(b) - timeOf(a);
    });
  }

  // Self-clearing alert rule.
  //
  // A producer that sends `<kind>_cleared` has told us `<kind>` is a condition,
  // not an event: it starts, it stops. An alert for a condition that has
  // already stopped asks nothing of Juniper. So, in the tray:
  //   1. `*_cleared` items are hidden (an all-clear needs no action);
  //   2. an item of the same family (kind == base, or kind starts with
  //      base + "_", e.g. `..._tripwire_still_blocked`) is hidden when a
  //      clear of that family arrived at or after it;
  //   3. of a family that is still active, only the newest item is shown.
  // Kinds with no `_cleared` sibling in the list are untouched. If the clear
  // was evicted from the window, the alert stays visible -- we cannot prove it
  // cleared. Returns { visible, hidden, groups } so the tray can say what it
  // hid and dismiss a whole family from its one visible card.
  function collapseSelfClearing(items) {
    var list = Array.isArray(items) ? items : [];
    var bases = new Set();
    list.forEach(function (n) {
      var kind = String((n && n.event_kind) || "");
      if (kind.length > CLEARED_SUFFIX.length && kind.slice(-CLEARED_SUFFIX.length) === CLEARED_SUFFIX) {
        bases.add(kind.slice(0, -CLEARED_SUFFIX.length));
      }
    });
    if (bases.size === 0) return { visible: list.slice(), hidden: 0, groups: new Map() };

    function familyOf(kind) {
      var best = null;
      bases.forEach(function (base) {
        if (kind === base || kind.indexOf(base + "_") === 0) {
          if (best === null || base.length > best.length) best = base;
        }
      });
      return best;
    }

    var lastClear = new Map();
    var newestActive = new Map();
    list.forEach(function (n) {
      var kind = String((n && n.event_kind) || "");
      var fam = familyOf(kind);
      if (fam === null) return;
      if (kind === fam + CLEARED_SUFFIX) {
        lastClear.set(fam, Math.max(lastClear.get(fam) || 0, timeOf(n)));
      }
    });
    list.forEach(function (n) {
      var kind = String((n && n.event_kind) || "");
      var fam = familyOf(kind);
      if (fam === null || kind === fam + CLEARED_SUFFIX) return;
      var cleared = lastClear.has(fam) && lastClear.get(fam) >= timeOf(n);
      if (cleared) return;
      var cur = newestActive.get(fam);
      if (!cur || timeOf(n) > timeOf(cur)) newestActive.set(fam, n);
    });

    var visible = [];
    // groups: visible item -> every item of its family (itself included), so
    // dismissing the one shown card also drops the hidden ones it stands for.
    var groups = new Map();
    list.forEach(function (n) {
      var fam = familyOf(String((n && n.event_kind) || ""));
      if (fam === null || newestActive.get(fam) === n) visible.push(n);
    });
    newestActive.forEach(function (n, fam) {
      groups.set(
        n,
        list.filter(function (m) {
          return familyOf(String((m && m.event_kind) || "")) === fam;
        })
      );
    });
    return { visible: visible, hidden: list.length - visible.length, groups: groups };
  }

  var api = {
    OUTREACH_TAG: OUTREACH_TAG,
    isOrionMessage: isOrionMessage,
    normalizeMessage: normalizeMessage,
    mergeOpenMessages: mergeOpenMessages,
    collapseSelfClearing: collapseSelfClearing,
  };
  if (typeof window !== "undefined") window.OrionMessageTray = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
