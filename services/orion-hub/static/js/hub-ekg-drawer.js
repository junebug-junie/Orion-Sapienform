(function (global) {
  // Inner Sky drawer: three widths, remembered. Layout lives in style.css (.hub-ekg-drawer).
  //   open      -- half the row, chat beside it (default)
  //   collapsed -- a thin rail; chat takes the row        (◀ from open)
  //   wide      -- the whole row; chat is hidden           (▶ from open, or the sky's own "Widen" link)
  // Stored under the old key so a remembered collapse survives: '1' collapsed, '0' open, '2' wide.
  var KEY = 'orion.hub.ekgDrawerCollapsed';
  var WIDEN_MESSAGE = 'orion-inner-sky-widen';
  var STATE_MESSAGE = 'orion-inner-sky-drawer'; // drawer -> sky: {type, state, phone}
  var CODE = { collapsed: '1', open: '0', wide: '2' };

  function readStored(storage) {
    var v = null;
    try { v = storage.getItem(KEY); } catch (_e) { v = null; }
    return v === '1' ? 'collapsed' : v === '2' ? 'wide' : 'open';
  }
  function writeStored(storage, state) {
    try { storage.setItem(KEY, CODE[state]); } catch (_e) { /* private mode: not remembered */ }
  }

  function skyFrame(drawer) {
    return drawer && drawer.querySelector ? drawer.querySelector('iframe') : null;
  }
  // The sky picks its layout from this, not from its own width: compact unless the drawer is wide.
  function tellSky(drawer, state, win) {
    var frame = skyFrame(drawer);
    if (!frame || !frame.contentWindow || !win || !win.location) return;
    var phone = !!(win.matchMedia && win.matchMedia('(max-width: 767px)').matches);
    try { frame.contentWindow.postMessage({ type: STATE_MESSAGE, state: state, phone: phone }, win.location.origin); } catch (_e) { /* frame not ready */ }
  }

  function apply(doc, state, win) {
    var drawer = doc.getElementById('ekgDrawer');
    if (!drawer) return;
    drawer.classList.toggle('is-collapsed', state === 'collapsed');
    drawer.classList.toggle('is-wide', state === 'wide');
    var toggle = doc.getElementById('ekgDrawerToggle');
    if (toggle) {
      toggle.setAttribute('aria-expanded', state === 'collapsed' ? 'false' : 'true');
      toggle.setAttribute('title', state === 'wide' ? 'Bring the chat back beside the Inner Sky' : 'Slide the Inner Sky away and give the chat the full width');
      toggle.setAttribute('aria-label', state === 'wide' ? 'Show chat beside the Inner Sky' : 'Collapse Inner Sky drawer');
    }
    var widen = doc.getElementById('ekgDrawerWiden');
    if (widen) {
      widen.setAttribute('aria-pressed', state === 'wide' ? 'true' : 'false');
      widen.hidden = state === 'wide';
    }
    tellSky(drawer, state, win);
  }

  function mount(doc, storage, win) {
    var drawer = doc.getElementById('ekgDrawer');
    if (!drawer) return null;
    var state = readStored(storage);
    apply(doc, state, win);
    var frame = skyFrame(drawer);
    if (frame && frame.addEventListener) frame.addEventListener('load', function () { tellSky(drawer, state, win); });
    function set(next) {
      state = next;
      apply(doc, state, win);
      writeStored(storage, state);
    }
    var toggle = doc.getElementById('ekgDrawerToggle');
    var rail = doc.getElementById('ekgDrawerRail');
    var widen = doc.getElementById('ekgDrawerWiden');
    if (toggle) toggle.addEventListener('click', function () { set(state === 'wide' ? 'open' : 'collapsed'); });
    if (rail) rail.addEventListener('click', function () { set('open'); });
    if (widen) widen.addEventListener('click', function () { set('wide'); });
    // The sky (same-origin iframe) asks to be widened from its own "Widen" link.
    if (win && win.addEventListener) {
      win.addEventListener('message', function (ev) {
        var sameOrigin = !win.location || ev.origin === win.location.origin;
        var f = skyFrame(drawer);
        var fromSky = !f || !f.contentWindow || ev.source === f.contentWindow; // not another same-origin frame
        if (!sameOrigin || !fromSky || !ev.data) return;
        if (ev.data.type === WIDEN_MESSAGE) set('wide');
        else if (ev.data.type === STATE_MESSAGE + '?') tellSky(drawer, state, win); // the sky asks after it loads
      });
    }
    return { set: set, state: function () { return state; }, isCollapsed: function () { return state === 'collapsed'; } };
  }

  var api = { KEY: KEY, WIDEN_MESSAGE: WIDEN_MESSAGE, STATE_MESSAGE: STATE_MESSAGE, mount: mount };
  global.OrionEkgDrawer = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  var storage = null;
  try { storage = typeof localStorage !== 'undefined' ? localStorage : null; } catch (_e) { storage = null; }
  if (typeof document !== 'undefined' && storage) {
    var go = function () { mount(document, storage, typeof window !== 'undefined' ? window : null); };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
  }
})(typeof window !== 'undefined' ? window : globalThis);
