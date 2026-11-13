// services/orion-hub/static/js/hub-attention-tracker.js
//
// Hub attention minutes: tell the Hub, once per minute, that Juniper is
// actually looking at it. A minute counts only when ALL three hold:
//   1. the page is visible   (document.visibilityState === 'visible')
//   2. the window has focus  (document.hasFocus(); true also when focus is
//                             inside a child iframe of this page)
//   3. real input happened in the last 120 s (pointermove, pointerdown,
//      keydown, wheel, scroll, touchstart)
// No heartbeat is sent when any of them fails. Rows land server-side in
// hub_attention_minute via POST /api/hub-attention/heartbeat.
//
// Loaded only by templates/index.html (the top-level Hub shell). Several Hub
// tabs are same-origin iframes (/reading, /gpu-pool, /temporal-self, ...);
// input events inside them do not reach this window, so the tracker also
// attaches its input listeners to every same-origin iframe window it can see
// (on load and on a 15 s rescan, nested frames included). Missed on purpose:
// cross-origin iframes (e.g. an external Grafana embed) cannot be listened to,
// and a sub-page opened directly (not inside the Hub shell) does not load this
// script.
//
// Best-effort: every path is wrapped; nothing here may throw into the page.
// No dependencies.
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  } else if (root && typeof root === 'object') {
    root.HubAttentionTracker = api;
    try {
      api.start(root);
    } catch (_) { /* never break the Hub page */ }
  }
})(typeof window !== 'undefined' ? window : globalThis, function () {
  const INPUT_WINDOW_MS = 120 * 1000;
  const RESCAN_MS = 15 * 1000;
  // Beat at local second :30 of every minute, so one beat lands in each server
  // minute bucket even with a modest client/server clock offset.
  const BEAT_SECOND_MS = 30 * 1000;
  const INPUT_EVENTS = ['pointermove', 'pointerdown', 'keydown', 'wheel', 'scroll', 'touchstart'];
  const HEARTBEAT_PATH = '/api/hub-attention/heartbeat';

  function msUntilNextBeat(nowMs) {
    const intoMinute = nowMs % 60000;
    let wait = BEAT_SECOND_MS - intoMinute;
    if (wait <= 0) wait += 60000;
    return wait;
  }

  function classifyClient(win) {
    try {
      if (win.matchMedia && win.matchMedia('(pointer: coarse)').matches) return 'mobile';
    } catch (_) { /* fall through */ }
    return 'desktop';
  }

  function resolveApiBase(win) {
    try {
      const cfg = win.__HUB_CFG__ || {};
      const override = String(cfg.apiBaseOverride || '').trim();
      if (override) return override.replace(/\/+$/, '');
      return String((win.location && win.location.origin) || '').replace(/\/+$/, '');
    } catch (_) {
      return '';
    }
  }

  function createTracker(win, opts) {
    const o = opts || {};
    const now = o.now || (() => Date.now());
    const fetchFn = o.fetch || ((...args) => win.fetch(...args));
    const doc = win.document;
    const state = {
      lastInputAt: 0,
      stopped: false,
      attached: typeof WeakSet === 'function' ? new WeakSet() : null,
      attachedCount: 0,
      sent: 0,
    };

    function onInput() {
      state.lastInputAt = now();
    }

    function attachTo(target) {
      try {
        if (!target || !target.addEventListener) return;
        if (state.attached) {
          if (state.attached.has(target)) return;
          state.attached.add(target);
        }
        INPUT_EVENTS.forEach((name) => {
          target.addEventListener(name, onInput, { capture: true, passive: true });
        });
        state.attachedCount += 1;
      } catch (_) { /* cross-origin or torn down */ }
    }

    // Walk same-origin iframes (recursively). Reading contentWindow.document
    // throws for cross-origin frames; those are skipped.
    function scanFrames(d, depth) {
      if (!d || depth > 3) return;
      let frames = [];
      try {
        frames = Array.from(d.querySelectorAll('iframe'));
      } catch (_) {
        return;
      }
      frames.forEach((frame) => {
        try {
          const fw = frame.contentWindow;
          const fd = fw && fw.document;
          if (!fd) return;
          // Key on the Document, not the window: a navigated iframe keeps the
          // same WindowProxy but gets a fresh Document without our listeners.
          attachTo(fd);
          scanFrames(fd, depth + 1);
        } catch (_) { /* cross-origin: input there is not counted */ }
      });
    }

    function isAttended() {
      try {
        if (doc.visibilityState !== 'visible') return false;
        if (typeof doc.hasFocus === 'function' && !doc.hasFocus()) return false;
        if (!state.lastInputAt) return false;
        return now() - state.lastInputAt <= INPUT_WINDOW_MS;
      } catch (_) {
        return false;
      }
    }

    function currentTab() {
      try {
        const h = String((win.location && win.location.hash) || '').replace(/^#/, '');
        return h ? h.slice(0, 64) : null;
      } catch (_) {
        return null;
      }
    }

    async function beat() {
      if (state.stopped || !isAttended()) return false;
      const body = JSON.stringify({ client: classifyClient(win), tab: currentTab() });
      try {
        const resp = await fetchFn(`${resolveApiBase(win)}${HEARTBEAT_PATH}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body,
          keepalive: true,
          credentials: 'same-origin',
        });
        state.sent += 1;
        if (resp && resp.ok && typeof resp.json === 'function') {
          const data = await resp.json().catch(() => null);
          // Server flag off: stop for this page load.
          if (data && data.enabled === false) state.stopped = true;
        }
        return true;
      } catch (_) {
        return false;
      }
    }

    return { attachTo, scanFrames, isAttended, beat, currentTab, state, onInput };
  }

  function start(win) {
    if (!win || !win.document || win.__hubAttentionTrackerStarted) return null;
    if (win.top && win.top !== win) return null; // top-level Hub shell only
    win.__hubAttentionTrackerStarted = true;
    const tracker = createTracker(win);
    const doc = win.document;
    // Capture-phase listeners on the Document see every input event in it,
    // including element scrolls (scroll does not bubble, but it does capture).
    tracker.attachTo(doc);
    const rescan = () => tracker.scanFrames(doc, 0);
    try {
      // 'load' does not bubble, but capture-phase listeners still see each iframe's load.
      doc.addEventListener('load', (ev) => {
        if (ev && ev.target && ev.target.tagName === 'IFRAME') rescan();
      }, true);
    } catch (_) { /* ignore */ }
    if (doc.readyState === 'loading') {
      doc.addEventListener('DOMContentLoaded', rescan, { once: true });
    } else {
      rescan();
    }
    win.setInterval(rescan, RESCAN_MS);
    const loop = () => {
      tracker.beat().catch(() => {});
      if (!tracker.state.stopped) win.setTimeout(loop, msUntilNextBeat(Date.now()));
    };
    win.setTimeout(loop, msUntilNextBeat(Date.now()));
    win.__hubAttentionTracker = tracker;
    return tracker;
  }

  return { createTracker, start, msUntilNextBeat, classifyClient, INPUT_WINDOW_MS, INPUT_EVENTS };
});
