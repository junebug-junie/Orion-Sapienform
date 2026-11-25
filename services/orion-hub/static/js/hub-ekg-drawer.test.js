const test = require('node:test');
const assert = require('node:assert/strict');
const drawer = require('./hub-ekg-drawer.js');

function fakeDoc() {
  const classes = new Set();
  const attrs = {};
  const widenAttrs = {};
  const handlers = {};
  const widen = { hidden: false, setAttribute: (k, v) => { widenAttrs[k] = v; }, addEventListener: (_e, f) => { handlers.widen = f; } };
  const els = {
    ekgDrawer: { classList: { toggle: (c, on) => (on ? classes.add(c) : classes.delete(c)) } },
    ekgDrawerToggle: { setAttribute: (k, v) => { attrs[k] = v; }, addEventListener: (_e, f) => { handlers.toggle = f; } },
    ekgDrawerRail: { addEventListener: (_e, f) => { handlers.rail = f; } },
    ekgDrawerWiden: widen,
  };
  return { doc: { getElementById: (id) => els[id] || null }, classes, attrs, widen, widenAttrs, handlers };
}
function fakeStorage(init) {
  const m = new Map(Object.entries(init || {}));
  return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, v), m };
}
function fakeWin(origin) {
  const h = {};
  return { location: { origin }, addEventListener: (e, f) => { h[e] = f; }, matchMedia: () => ({ matches: false }), h };
}
function withSkyFrame(f) {
  const sent = [];
  const frame = { contentWindow: { postMessage: (m, o) => sent.push([m, o]) }, addEventListener: () => {} };
  f.doc.getElementById('ekgDrawer').querySelector = () => frame;
  return { frame, sent };
}

test('drawer starts open, collapses on toggle, reopens from the rail, and remembers', () => {
  const f = fakeDoc();
  const st = fakeStorage();
  drawer.mount(f.doc, st);
  assert.equal(f.classes.has('is-collapsed'), false);
  f.handlers.toggle();
  assert.equal(f.classes.has('is-collapsed'), true);
  assert.equal(f.attrs['aria-expanded'], 'false');
  assert.equal(st.m.get(drawer.KEY), '1');
  f.handlers.rail();
  assert.equal(f.classes.has('is-collapsed'), false);
  assert.equal(st.m.get(drawer.KEY), '0');
});

test('a remembered collapsed state is restored on load', () => {
  const f = fakeDoc();
  drawer.mount(f.doc, fakeStorage({ [drawer.KEY]: '1' }));
  assert.equal(f.classes.has('is-collapsed'), true);
});

test('widen hides chat; the back arrow from wide returns to open, not collapsed', () => {
  const f = fakeDoc();
  const st = fakeStorage();
  drawer.mount(f.doc, st);
  f.handlers.widen();
  assert.equal(f.classes.has('is-wide'), true);
  assert.equal(f.classes.has('is-collapsed'), false);
  assert.equal(f.widen.hidden, true);
  assert.equal(st.m.get(drawer.KEY), '2');
  f.handlers.toggle();
  assert.equal(f.classes.has('is-wide'), false);
  assert.equal(f.classes.has('is-collapsed'), false);
  assert.equal(f.widen.hidden, false);
  assert.equal(st.m.get(drawer.KEY), '0');
});

test('a remembered wide state is restored on load', () => {
  const f = fakeDoc();
  drawer.mount(f.doc, fakeStorage({ [drawer.KEY]: '2' }));
  assert.equal(f.classes.has('is-wide'), true);
});

test('the sky can ask to be widened, but only from its own frame on the same origin', () => {
  const f = fakeDoc();
  const sky = withSkyFrame(f);
  const w = fakeWin('https://hub.local');
  drawer.mount(f.doc, fakeStorage(), w);
  const src = sky.frame.contentWindow;
  w.h.message({ origin: 'https://evil.example', source: src, data: { type: drawer.WIDEN_MESSAGE } });
  assert.equal(f.classes.has('is-wide'), false);
  w.h.message({ origin: 'https://hub.local', source: {}, data: { type: drawer.WIDEN_MESSAGE } }); // another same-origin frame
  assert.equal(f.classes.has('is-wide'), false);
  w.h.message({ origin: 'https://hub.local', source: src, data: { type: 'something-else' } });
  assert.equal(f.classes.has('is-wide'), false);
  w.h.message({ origin: 'https://hub.local', source: src, data: { type: drawer.WIDEN_MESSAGE } });
  assert.equal(f.classes.has('is-wide'), true);
});

test('the drawer tells the sky its state on every change and when the sky asks', () => {
  const f = fakeDoc();
  const sky = withSkyFrame(f);
  const w = fakeWin('https://hub.local');
  drawer.mount(f.doc, fakeStorage(), w);
  assert.deepEqual(sky.sent.at(-1), [{ type: drawer.STATE_MESSAGE, state: 'open', phone: false }, 'https://hub.local']);
  f.handlers.widen();
  assert.equal(sky.sent.at(-1)[0].state, 'wide');
  const before = sky.sent.length;
  w.h.message({ origin: 'https://hub.local', source: sky.frame.contentWindow, data: { type: drawer.STATE_MESSAGE + '?' } });
  assert.equal(sky.sent.length, before + 1);
});
