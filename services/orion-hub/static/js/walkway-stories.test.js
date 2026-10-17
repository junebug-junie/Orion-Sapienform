const test = require('node:test');
const assert = require('node:assert/strict');
const ws = require('./walkway-stories.js');

const H = (c) => c.repeat(64);

test('tally line shows progress against the 50 / 80% / zero-wrong-name gate', () => {
  const t = { checked: 12, right: 10, wrong_name: 0, pct_right: 0.8333, named_checked: 2,
              gate: { min_checked: 50, min_right: 0.8, max_wrong_names: 0, passed: false } };
  assert.equal(ws.tallyLine(t),
    'Checked 12 of 50 · right 10 (83%, need 80%) · wrong names 0 (must be 0) · named stories checked 2');
  assert.equal(ws.gateLine(t), 'Hub only for now: 38 more to check.');
});

test('gate line names every reason it is not passed, and says when it is', () => {
  const gate = { min_checked: 50, min_right: 0.8, max_wrong_names: 0, passed: false };
  assert.equal(ws.gateLine({ checked: 60, right: 40, pct_right: 0.66, wrong_name: 1, gate }),
    'Hub only for now: too many wrong, a wrong name was found.');
  assert.match(ws.gateLine({ checked: 50, gate: { ...gate, passed: true } }), /^Gate passed/);
  assert.equal(ws.tallyLine({}).startsWith('Checked 0 of 50 · right 0 (–'), true);
});

test('thumbs: only real sha256 refs, first / middle / last when there are many', () => {
  const ev = [{ thumb: H('a') }, { thumb: null }, { thumb: '../etc' }, { thumb: H('b') }, { thumb: H('c') },
              { thumb: H('d') }];
  assert.deepEqual(ws.pickThumbs(ev), [H('a'), H('b'), H('d')]);
  assert.deepEqual(ws.pickThumbs([{ thumb: H('e') }]), [H('e')]);
  assert.deepEqual(ws.pickThumbs([]), []);
});

test('box map numbers boxes in time order and draws the path through unclipped feet only', () => {
  const m = ws.boxMapModel([
    { box: [100, 200, 180, 400], clipped: false },
    { box: [300, 210, 380, 410], clipped: false },
    { box: [2480, 200, 2560, 400], clipped: true },
    { box: ['x'] },
  ]);
  assert.equal(m.boxes.length, 3);
  assert.deepEqual(m.boxes.map((b) => b.i), [1, 2, 3]);
  assert.equal(m.boxes[2].clipped, true);
  assert.equal(m.path, '140,400 340,410');
});

test('verdict buttons are the four Juniper asked for', () => {
  assert.deepEqual(ws.VERDICTS.map((v) => v.value), ['right', 'wrong_name', 'wrong_motion', 'wrong_other']);
  assert.equal(ws.VIEW_VALUE, 'walkway-live');
});
