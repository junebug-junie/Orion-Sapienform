/* Hub section launcher: tabs moved from a scrolling pill strip into a grid
 * modal, Hub pinned first, rest alphabetical. No jsdom in CI, so the DOM
 * behaviour runs against a minimal fake document, and the real template/CSS
 * are checked as text.
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { orderHubTabs, matchesFilter, initHubTabLauncher } = require('./hub-tab-launcher.js');

const TEMPLATE = fs.readFileSync(path.join(__dirname, '..', '..', 'templates', 'index.html'), 'utf8');
const CSS = fs.readFileSync(path.join(__dirname, '..', 'css', 'style.css'), 'utf8');

/* ── minimal fake DOM ─────────────────────────────────────────────────── */
class FakeClassList {
  constructor(initial) { this.set = new Set(initial); }
  contains(c) { return this.set.has(c); }
  toggle(c, force) {
    const on = force === undefined ? !this.set.has(c) : !!force;
    if (on) this.set.add(c); else this.set.delete(c);
    return on;
  }
  add(c) { this.set.add(c); }
  remove(c) { this.set.delete(c); }
}

class FakeEl {
  constructor(id, { text = '', classes = [], hashTarget = null } = {}) {
    this.id = id;
    this.textContent = text;
    this.classList = new FakeClassList(classes);
    this.dataset = {};
    this.hidden = false;
    this.value = '';
    this.children = [];
    this.parent = null;
    this.attrs = {};
    this.hashTarget = hashTarget;
    this.listeners = {};
  }
  get firstElementChild() { return this.children[0] || null; }
  contains(other) {
    for (let n = other; n; n = n.parent) if (n === this) return true;
    return false;
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k]; }
  appendChild(child) {
    if (child.parent) child.parent.children = child.parent.children.filter((c) => c !== child);
    child.parent = this;
    this.children.push(child);
  }
  querySelectorAll(sel) {
    assert.equal(sel, 'a[data-hash-target]');
    return this.children.filter((c) => c.hashTarget);
  }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  dispatch(type, extra = {}) {
    const event = { type, target: this, preventDefault() { this.defaultPrevented = true; }, ...extra };
    (this.listeners[type] || []).forEach((fn) => fn(event));
    return event;
  }
  click() { return this.dispatch('click'); }
  focus() { this.ownerDoc.activeElement = this; }
}

function buildDoc(tabs) {
  const doc = { els: {}, listeners: {}, activeElement: null, defaultView: null };
  const make = (id, opts) => {
    const el = new FakeEl(id, opts);
    el.ownerDoc = doc;
    doc.els[id] = el;
    return el;
  };
  const header = make('header');
  const button = make('hubTabLauncherButton');
  header.appendChild(button);
  button.appendChild(make('hubTabLauncherCurrent'));
  header.appendChild(make('runtimeMarquee'));
  const modal = make('hubTabLauncherModal');
  modal.hidden = true;
  header.appendChild(modal);
  const panel = make('panel');
  modal.appendChild(panel);
  const nav = make('hubPrimaryNav');
  panel.appendChild(make('hubTabLauncherFilter'));
  panel.appendChild(nav);
  panel.appendChild(make('hubTabLauncherEmpty'));
  tabs.forEach(([id, text, active]) => {
    nav.appendChild(make(id, { text, hashTarget: '#' + id, classes: [active ? 'bg-indigo-600' : 'bg-gray-800'] }));
  });
  doc.getElementById = (id) => doc.els[id] || null;
  doc.addEventListener = (type, fn) => { (doc.listeners[type] ||= []).push(fn); };
  doc.key = (key, extra = {}) => {
    const ev = { key, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...extra };
    (doc.listeners.keydown || []).forEach((fn) => fn(ev));
    return ev;
  };
  // Bubble a click from an element up to the document, as a browser would.
  doc.clickFrom = (el) => {
    el.click();
    (doc.listeners.click || []).forEach((fn) => fn({ type: 'click', target: el }));
  };
  return doc;
}

/* Real tab list, pulled from the template so the test tracks the page. */
function templateTabs() {
  const nav = TEMPLATE.split('id="hubPrimaryNav"')[1].split('</nav>')[0];
  const out = [];
  const re = /<a\b[^>]*\bid="([^"]+)"[^>]*>([^<]*)<\/a>/gs;
  let m;
  while ((m = re.exec(nav))) out.push([m[1], m[2].trim()]);
  return out;
}

/* ── ordering ─────────────────────────────────────────────────────────── */
test('Hub is pinned first and the rest are case-insensitive alphabetical', () => {
  const items = [
    { id: 'z', label: 'Substrate' },
    { id: 'hubTabButton', label: 'Hub' },
    { id: 'a', label: 'attention Organ' },
    { id: 'g', label: 'GPU pool' },
    { id: 'c', label: 'Causal Geometry' },
  ];
  assert.deepEqual(orderHubTabs(items).map((i) => i.label),
    ['Hub', 'attention Organ', 'Causal Geometry', 'GPU pool', 'Substrate']);
});

test('ordering does not drop or duplicate tabs, and tolerates a missing pin', () => {
  const items = [{ id: 'b', label: 'B' }, { id: 'a', label: 'A' }];
  assert.deepEqual(orderHubTabs(items).map((i) => i.id), ['a', 'b']);
});

test('every tab in the real template ends up ordered: Hub first, then A->Z', () => {
  const tabs = templateTabs();
  assert.ok(tabs.length >= 20, `expected the full tab set, parsed ${tabs.length}`);
  const ordered = orderHubTabs(tabs.map(([id, label]) => ({ id, label })));
  assert.equal(ordered[0].id, 'hubTabButton');
  assert.equal(ordered.length, tabs.length);
  const rest = ordered.slice(1).map((o) => o.label.toLowerCase());
  assert.deepEqual(rest, [...rest].sort((a, b) => a.localeCompare(b)));
});

test('filter is a case-insensitive substring match; blank shows everything', () => {
  assert.equal(matchesFilter('Substrate Atlas', 'atl'), true);
  assert.equal(matchesFilter('Substrate Atlas', '  ATLAS '), true);
  assert.equal(matchesFilter('Memory', 'atlas'), false);
  assert.equal(matchesFilter('Memory', ''), true);
});

/* ── DOM behaviour ────────────────────────────────────────────────────── */
function launcher(tabs) {
  const doc = buildDoc(tabs || [
    ['memoryTabButton', 'Memory'],
    ['hubTabButton', 'Hub', true],
    ['conceptAtlasTabButton', 'Concept Atlas'],
    ['substrateAtlasTabButton', 'Substrate Atlas'],
  ]);
  const api = initHubTabLauncher(doc);
  return { doc, api, el: (id) => doc.getElementById(id) };
}

test('init physically reorders the anchors in the nav and marks Hub pinned', () => {
  const { el } = launcher();
  assert.deepEqual(el('hubPrimaryNav').children.map((c) => c.id),
    ['hubTabButton', 'conceptAtlasTabButton', 'memoryTabButton', 'substrateAtlasTabButton']);
  assert.equal(el('hubTabButton').classList.contains('hub-tab-pinned'), true);
  assert.equal(el('memoryTabButton').classList.contains('hub-tab-pinned'), false);
  assert.equal(el('memoryTabButton').dataset.initial, 'M');
});

test('button toggles the modal and aria-expanded; Escape closes and returns focus', () => {
  const { doc, el } = launcher();
  el('hubTabLauncherButton').click();
  assert.equal(el('hubTabLauncherModal').hidden, false);
  assert.equal(el('hubTabLauncherButton').getAttribute('aria-expanded'), 'true');
  assert.equal(doc.activeElement, el('hubTabLauncherFilter'));
  doc.key('Escape');
  assert.equal(el('hubTabLauncherModal').hidden, true);
  assert.equal(el('hubTabLauncherButton').getAttribute('aria-expanded'), 'false');
  assert.equal(doc.activeElement, el('hubTabLauncherButton'));
  el('hubTabLauncherButton').click();
  el('hubTabLauncherButton').click();
  assert.equal(el('hubTabLauncherModal').hidden, true);
});

test('backdrop click closes, a click inside the panel does not', () => {
  const { el } = launcher();
  const modal = el('hubTabLauncherModal');
  el('hubTabLauncherButton').click();
  modal.dispatch('click', { target: el('hubTabLauncherFilter') });
  assert.equal(modal.hidden, false);
  modal.dispatch('click', { target: modal });
  assert.equal(modal.hidden, true);
});

test('choosing a tab closes the launcher and hands focus back to the button', () => {
  const { doc, el } = launcher();
  el('hubTabLauncherButton').click();
  el('memoryTabButton').click();
  assert.equal(el('hubTabLauncherModal').hidden, true);
  assert.equal(doc.activeElement, el('hubTabLauncherButton'), 'hidden anchor must not keep focus');
});

test('a click elsewhere in the header closes it; clicks on the button or panel do not', () => {
  const { doc, el } = launcher();
  doc.clickFrom(el('hubTabLauncherButton'));
  assert.equal(el('hubTabLauncherModal').hidden, false, 'opening click must not self-close');
  doc.clickFrom(el('hubTabLauncherFilter'));
  assert.equal(el('hubTabLauncherModal').hidden, false);
  doc.clickFrom(el('runtimeMarquee'));
  assert.equal(el('hubTabLauncherModal').hidden, true);
});

test('Tab cycles inside the open launcher (aria-modal focus trap)', () => {
  const { doc, el } = launcher();
  el('hubTabLauncherButton').click();
  assert.equal(doc.activeElement, el('hubTabLauncherFilter'));
  el('substrateAtlasTabButton').focus(); // last visible tile
  let ev = doc.key('Tab');
  assert.equal(ev.defaultPrevented, true);
  assert.equal(doc.activeElement, el('hubTabLauncherFilter'));
  ev = doc.key('Tab', { shiftKey: true });
  assert.equal(doc.activeElement, el('substrateAtlasTabButton'));
  el('memoryTabButton').focus(); // middle: browser handles it
  ev = doc.key('Tab');
  assert.equal(ev.defaultPrevented, false);
  el('runtimeMarquee').focus(); // escaped somehow: pulled back in
  doc.key('Tab');
  assert.equal(doc.activeElement, el('hubTabLauncherFilter'));
});

test('Tab is left alone while the launcher is closed', () => {
  const { doc } = launcher();
  assert.equal(doc.key('Tab').defaultPrevented, false);
});

test('typing filters tiles, empty state shows on no match, Enter opens first match', () => {
  const { el } = launcher();
  const filter = el('hubTabLauncherFilter');
  el('hubTabLauncherButton').click();
  filter.value = 'atlas';
  filter.dispatch('input');
  assert.equal(el('hubTabButton').hidden, true);
  assert.equal(el('memoryTabButton').hidden, true);
  assert.equal(el('conceptAtlasTabButton').hidden, false);
  assert.equal(el('substrateAtlasTabButton').hidden, false);
  assert.equal(el('hubTabLauncherEmpty').hidden, true);

  let opened = null;
  el('conceptAtlasTabButton').addEventListener('click', () => { opened = 'concept'; });
  el('substrateAtlasTabButton').addEventListener('click', () => { opened = 'substrate'; });
  filter.dispatch('keydown', { key: 'Enter' });
  assert.equal(opened, 'concept', 'first alphabetical match wins');
  assert.equal(el('hubTabLauncherModal').hidden, true);

  el('hubTabLauncherButton').click();
  filter.value = 'atlas';
  filter.dispatch('input');
  filter.dispatch('keydown', { key: 'Enter', isComposing: true });
  assert.equal(el('hubTabLauncherModal').hidden, false, 'IME confirm Enter must not navigate');
  el('hubTabLauncherButton').click();
  el('hubTabLauncherButton').click();
  assert.equal(filter.value, '', 'reopening clears the old query');
  assert.equal(el('memoryTabButton').hidden, false);
  filter.value = 'zzz';
  filter.dispatch('input');
  assert.equal(el('hubTabLauncherEmpty').hidden, false);
});

test('header label mirrors whichever tab carries the active class', () => {
  const { api, el } = launcher();
  assert.equal(el('hubTabLauncherCurrent').textContent, 'Hub');
  el('hubTabButton').classList.remove('bg-indigo-600');
  el('memoryTabButton').classList.add('bg-indigo-600');
  api.syncCurrent();
  assert.equal(el('hubTabLauncherCurrent').textContent, 'Memory');
});

test('missing launcher markup is a no-op, not a crash', () => {
  const doc = buildDoc([['hubTabButton', 'Hub', true]]);
  delete doc.els.hubTabLauncherModal;
  doc.getElementById = (id) => doc.els[id] || null;
  assert.equal(initHubTabLauncher(doc), null);
});

/* ── real template / stylesheet wiring ────────────────────────────────── */
test('template: nav lives inside the hidden launcher modal and the script is loaded', () => {
  const modal = TEMPLATE.split('id="hubTabLauncherModal"')[1];
  assert.ok(modal, 'launcher modal missing');
  assert.match(TEMPLATE, /id="hubTabLauncherModal"[^>]*\bhidden\b/);
  assert.ok(modal.indexOf('id="hubPrimaryNav"') < modal.indexOf('</nav>'));
  assert.ok(TEMPLATE.indexOf('id="hubTabLauncherButton"') < TEMPLATE.indexOf('id="hubPrimaryNav"'));
  assert.ok(TEMPLATE.includes('{{HUB_AITOWN_TAB_NAV}}'), 'AI Town tab slot must stay inside the nav');
  assert.ok(TEMPLATE.split('id="hubPrimaryNav"')[1].split('</nav>')[0].includes('{{HUB_AITOWN_TAB_NAV}}'));
  assert.match(TEMPLATE, /<script src="\/static\/js\/hub-tab-launcher\.js\?v=\{\{HUB_UI_ASSET_VERSION\}\}" defer><\/script>/);
});

test('css: [hidden] overrides beat the display rules they undo (see runtime modal incident)', () => {
  // Same failure class as runtime-activity-modal-hidden.test.js: an author
  // `display` rule silently beats the UA `[hidden]{display:none}`.
  assert.match(CSS, /\.hub-tab-launcher-overlay\[hidden\]\s*\{\s*display:\s*none;/);
  assert.match(CSS, /\.hub-tab-nav a\[hidden\]\s*\{\s*display:\s*none;/);
  assert.match(CSS, /\.hub-tab-launcher-empty\[hidden\]\s*\{\s*display:\s*none;/);
});
