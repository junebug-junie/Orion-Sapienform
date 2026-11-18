const test = require('node:test');
const assert = require('node:assert/strict');
const tracker = require('./hub-attention-tracker.js');

function fakeDoc({ visible = true, focused = true, frames = [] } = {}) {
  const listeners = {};
  return {
    visibilityState: visible ? 'visible' : 'hidden',
    hasFocus: () => focused,
    listeners,
    addEventListener(name, fn) {
      (listeners[name] = listeners[name] || []).push(fn);
    },
    dispatch(name) {
      (listeners[name] || []).forEach((fn) => fn({ type: name }));
    },
    querySelectorAll(sel) {
      return sel === 'iframe' ? frames : [];
    },
  };
}

function fakeWin(doc, { coarse = false, hash = '#reading' } = {}) {
  return {
    document: doc,
    location: { origin: 'http://hub.test', hash },
    matchMedia: (q) => ({ matches: coarse && q === '(pointer: coarse)' }),
    __HUB_CFG__: {},
  };
}

function setup(docOpts, winOpts) {
  let clock = 1_000_000;
  const calls = [];
  const doc = fakeDoc(docOpts);
  const win = fakeWin(doc, winOpts);
  const t = tracker.createTracker(win, {
    now: () => clock,
    fetch: async (url, init) => {
      calls.push({ url, init, body: JSON.parse(init.body) });
      return { ok: true, json: async () => ({ ok: true, enabled: true }) };
    },
  });
  t.attachTo(doc);
  return { t, doc, win, calls, advance: (ms) => { clock += ms; } };
}

test('no heartbeat before any input, even when visible and focused', async () => {
  const { t, calls } = setup();
  assert.equal(t.isAttended(), false);
  assert.equal(await t.beat(), false);
  assert.equal(calls.length, 0);
});

test('visible + focused + recent input sends one heartbeat with client and tab', async () => {
  const { t, doc, calls } = setup();
  doc.dispatch('pointermove');
  assert.equal(await t.beat(), true);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, 'http://hub.test/api/hub-attention/heartbeat');
  assert.equal(calls[0].init.method, 'POST');
  assert.deepEqual(calls[0].body, { client: 'desktop', tab: 'reading' });
});

test('every listed input event counts', () => {
  for (const name of tracker.INPUT_EVENTS) {
    const { t, doc } = setup();
    doc.dispatch(name);
    assert.equal(t.isAttended(), true, name);
  }
});

test('input older than 120 s does not count', async () => {
  const { t, doc, calls, advance } = setup();
  doc.dispatch('keydown');
  advance(tracker.INPUT_WINDOW_MS + 1);
  assert.equal(await t.beat(), false);
  assert.equal(calls.length, 0);
});

test('hidden page or unfocused window sends nothing', async () => {
  for (const opts of [{ visible: false }, { focused: false }]) {
    const { t, doc, calls } = setup(opts);
    doc.dispatch('pointerdown');
    assert.equal(await t.beat(), false);
    assert.equal(calls.length, 0);
  }
});

test('coarse pointer reports mobile', async () => {
  const { t, doc, calls } = setup({}, { coarse: true, hash: '' });
  doc.dispatch('touchstart');
  await t.beat();
  assert.deepEqual(calls[0].body, { client: 'mobile', tab: null });
});

test('input inside a same-origin iframe counts; cross-origin frames are skipped without throwing', () => {
  const inner = fakeDoc();
  const sameOrigin = { contentWindow: { document: inner } };
  const crossOrigin = {
    get contentWindow() {
      return { get document() { throw new Error('SecurityError'); } };
    },
  };
  const { t, doc } = setup({ frames: [crossOrigin, sameOrigin] });
  t.scanFrames(doc, 0);
  t.scanFrames(doc, 0); // rescan must not double-attach
  assert.equal(inner.listeners.wheel.length, 1);
  inner.dispatch('wheel');
  assert.equal(t.isAttended(), true);
});

test('server flag off stops further heartbeats; fetch failure never throws', async () => {
  const { t, doc, calls } = setup();
  doc.dispatch('keydown');
  let failing = true;
  const t2 = tracker.createTracker({ ...fakeWin(doc), document: doc }, {
    fetch: async () => { if (failing) throw new Error('network'); return { ok: true, json: async () => ({ enabled: false }) }; },
  });
  t2.attachTo(doc);
  doc.dispatch('keydown');
  assert.equal(await t2.beat(), false);
  failing = false;
  assert.equal(await t2.beat(), true);
  assert.equal(t2.state.stopped, true);
  assert.equal(await t2.beat(), false);
  assert.equal(calls.length, 0);
  assert.equal(t.isAttended(), true);
});

test('beats are scheduled at second :30 of each minute', () => {
  assert.equal(tracker.msUntilNextBeat(0), 30000);
  assert.equal(tracker.msUntilNextBeat(30000), 60000);
  assert.equal(tracker.msUntilNextBeat(45000), 45000);
});
