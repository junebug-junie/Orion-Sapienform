const test = require('node:test');
const assert = require('node:assert/strict');
const drawer = require('./hub-ekg-drawer.js');

function fakeDoc() {
  const classes = new Set();
  const attrs = {};
  const handlers = {};
  const els = {
    ekgDrawer: { classList: { toggle: (c, on) => (on ? classes.add(c) : classes.delete(c)) } },
    ekgDrawerToggle: { setAttribute: (k, v) => { attrs[k] = v; }, addEventListener: (_e, f) => { handlers.toggle = f; } },
    ekgDrawerRail: { addEventListener: (_e, f) => { handlers.rail = f; } },
  };
  return { doc: { getElementById: (id) => els[id] || null }, classes, attrs, handlers };
}
function fakeStorage(init) {
  const m = new Map(Object.entries(init || {}));
  return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, v), m };
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
