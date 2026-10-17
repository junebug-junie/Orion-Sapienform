const test = require('node:test');
const assert = require('node:assert/strict');
const ws = require('./walkway-stories.js');

const H = (c) => c.repeat(64);

test('tally line shows progress against the 50 / 80% / zero-wrong-name gate', () => {
  const t = { checked: 12, right: 10, wrong_name: 0, pct_right: 0.8333, named_checked: 2,
              gate: { min_checked: 50, min_right: 0.8, max_wrong_names: 0, passed: false } };
  assert.equal(ws.tallyLine(t),
    'Checked 12 of 50 · right 10 (83%, need 80%) · wrong names 0 (must be 0) · named stories checked 2 · ' +
    "can't tell 0 (not counted)");
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

test('strip model: the moment\'s frame and crops, only whitelisted names, missing ones say why', () => {
  const id = 'a'.repeat(32);
  const m = ws.stripModel({ story_id: id, dark: false, evidence_strip: {
    frame: { file: 'frame.jpg', t: '2026-10-17T00:47:09Z' },
    crops: [{ file: 'crop_0.jpg', t: 't0' }, { missing: 'frame_gone', t: 't1' }, { missing: 'weird', t: 't2' },
            { file: '../../etc/passwd', t: 't3' }],
  } });
  assert.equal(m.frame.url, `/api/vision/motion-stories/${id}/strip/frame.jpg`);
  assert.deepEqual(m.crops.map((c) => c.url || c.missing),
    [`/api/vision/motion-stories/${id}/strip/crop_0.jpg`, 'frame_gone', 'not_captured', 'not_captured']);
  assert.equal(ws.hasPicture(m), true);
});

test('strip model: no strip, a pruned strip, or a bad story id is "no picture"', () => {
  assert.equal(ws.hasPicture(ws.stripModel({ story_id: 'b'.repeat(32), evidence_strip: null })), false);
  const pruned = ws.stripModel({ story_id: 'b'.repeat(32),
    evidence_strip: { pruned_at: 'x', frame: { file: 'frame.jpg' }, crops: [] } });
  assert.equal(pruned.pruned, true);
  assert.equal(ws.hasPicture(pruned), false);
  assert.equal(ws.stripModel({ story_id: '../x', evidence_strip: { frame: { file: 'frame.jpg' } } }).frame, null);
});

test('verdict buttons are the four Juniper asked for, plus Can\'t tell', () => {
  assert.deepEqual(ws.VERDICTS.map((v) => v.value), ['right', 'wrong_name', 'wrong_motion', 'wrong_other']);
  assert.deepEqual(ws.CANT_TELL, { value: 'cant_tell', label: "Can't tell" });
  assert.equal(ws.VIEW_VALUE, 'walkway-live');
});
