(function (global) {
  // Cognitive EKG drawer: collapse/expand toggle with remembered state.
  // Layout lives in style.css (.hub-ekg-drawer); this only flips `is-collapsed`.
  var KEY = 'orion.hub.ekgDrawerCollapsed';

  function readStored(storage) {
    try { return storage.getItem(KEY) === '1'; } catch (_e) { return false; }
  }
  function writeStored(storage, collapsed) {
    try { storage.setItem(KEY, collapsed ? '1' : '0'); } catch (_e) { /* private mode: not remembered */ }
  }

  function apply(doc, collapsed) {
    var drawer = doc.getElementById('ekgDrawer');
    if (!drawer) return;
    drawer.classList.toggle('is-collapsed', collapsed);
    var toggle = doc.getElementById('ekgDrawerToggle');
    if (toggle) toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
  }

  function mount(doc, storage) {
    var drawer = doc.getElementById('ekgDrawer');
    if (!drawer) return null;
    var collapsed = readStored(storage);
    apply(doc, collapsed);
    function set(next) {
      collapsed = next;
      apply(doc, collapsed);
      writeStored(storage, collapsed);
    }
    var toggle = doc.getElementById('ekgDrawerToggle');
    var rail = doc.getElementById('ekgDrawerRail');
    if (toggle) toggle.addEventListener('click', function () { set(true); });
    if (rail) rail.addEventListener('click', function () { set(false); });
    return { set: set, isCollapsed: function () { return collapsed; } };
  }

  var api = { KEY: KEY, mount: mount };
  global.OrionEkgDrawer = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  var storage = null;
  try { storage = typeof localStorage !== 'undefined' ? localStorage : null; } catch (_e) { storage = null; }
  if (typeof document !== 'undefined' && storage) {
    var go = function () { mount(document, storage); };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
  }
})(typeof window !== 'undefined' ? window : globalThis);
