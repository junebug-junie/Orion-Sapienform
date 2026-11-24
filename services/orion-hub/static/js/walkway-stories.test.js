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
            { file: '../../etc/passwd', t: 't3' }, { missing: 'prop_overlap', t: 't4' }],
  } });
  assert.equal(m.frame.url, `/api/vision/motion-stories/${id}/strip/frame.jpg`);
  assert.deepEqual(m.crops.map((c) => c.url || c.missing),
    [`/api/vision/motion-stories/${id}/strip/crop_0.jpg`, 'frame_gone', 'not_captured', 'not_captured',
     'prop_overlap']);
  assert.equal(ws.hasPicture(m), true);
});

test('strip model: no strip, a pruned strip, or a bad story id is "no picture"', () => {
  assert.equal(ws.hasPicture(ws.stripModel({ story_id: 'b'.repeat(32), evidence_strip: null })), false);
  const pruned = ws.stripModel({ story_id: 'b'.repeat(32),
    evidence_strip: { pruned_at: 'x', frame: { file: 'frame.jpg' }, crops: [] } });
  assert.equal(pruned.pruned, true);
  assert.equal(pruned.retentionDays, null);
  assert.equal(ws.stripModel({ story_id: 'b'.repeat(32), evidence_strip: { pruned_at: 'x', retention_days: 30 } })
    .retentionDays, 30);
  assert.equal(ws.hasPicture(pruned), false);
  assert.equal(ws.stripModel({ story_id: '../x', evidence_strip: { frame: { file: 'frame.jpg' } } }).frame, null);
});

test('verdict buttons are the four Juniper asked for, plus Can\'t tell', () => {
  assert.deepEqual(ws.VERDICTS.map((v) => v.value), ['right', 'wrong_name', 'wrong_motion', 'wrong_other']);
  assert.deepEqual(ws.CANT_TELL, { value: 'cant_tell', label: "Can't tell" });
  assert.equal(ws.VIEW_VALUE, 'walkway-live');
});

test('gate line says how many dark or picture-less stories the gate does not cover', () => {
  const gate = { min_checked: 50, min_right: 0.8, max_wrong_names: 0, passed: false };
  assert.equal(ws.gateLine({ checked: 10, not_checkable: 3, gate }),
    'Hub only for now: 40 more to check. The gate covers daylight stories with a picture; ' +
    '3 dark or picture-less stories are not in it.');
  assert.equal(ws.gateLine({ checked: 10, not_checkable: 0, gate }), 'Hub only for now: 40 more to check.');
});

// Each person's own pictures (2026-11-24, Juniper: "how the f would i know which is which").
test('story people: one row per individual, lettered in order', () => {
  const got = ws.storyPeople({ people: [{ individual_id: 'a' }, null, { individual_id: 'b' }, { individual_id: 'a' },
    { individual_id: '' }, { individual_id: 'c' }] });
  assert.deepEqual(got.map((e) => [e.person.individual_id, e.letter]), [['a', 'A'], ['b', 'B'], ['c', 'C']]);
  assert.deepEqual(ws.storyPeople({}), []);
  const many = ws.storyPeople({ people: Array.from({ length: 28 }, (_, i) => ({ individual_id: `p${i}` })) });
  assert.deepEqual(many.slice(24).map((e) => e.letter), ['Y', 'Z', 'A1', 'B1']);
});

test('person pictures: only whitelisted strip crops, else their own thumbnail, else none', () => {
  const sid = 'a'.repeat(32);
  const p = { pictures: [{ story_id: sid, file: 'crop_1.jpg', t: 't1', crop_id: 'c1' },
    { story_id: '../x', file: 'crop_0.jpg' }, { story_id: sid, file: '../../etc/passwd' }], thumb: H('f') };
  assert.deepEqual(ws.personPictures(p), [{ url: `/api/vision/motion-stories/${sid}/strip/crop_1.jpg`, t: 't1', crop_id: 'c1' }]);
  assert.deepEqual(ws.personPictures({ pictures: [], thumb: H('f'), thumb_crop_id: 'c9' }),
    [{ url: `/api/vision/crop-thumbs/${H('f')}`, t: null, crop_id: 'c9' }]);
  assert.equal(ws.isPictured({ pictures: [], thumb: 'nope', pictured: true }), false);
  assert.equal(ws.isPictured({ pictures: [{ story_id: sid, file: 'crop_0.jpg' }], pictured: true }), true);
  // The server decides: a thumbnail it could not tie to a crop is shown nowhere as askable.
  assert.equal(ws.isPictured({ pictures: [], thumb: H('f'), pictured: false }), false);
  assert.equal(ws.isPictured({ pictures: [{ story_id: sid, file: 'crop_0.jpg' }] }), false);
});

test('boxes text says which numbered boxes are theirs, or that they are not boxed', () => {
  assert.equal(ws.boxesText({ boxes: [{ n: 2 }, { n: 3 }] }, true), 'Boxes 2, 3 on the big picture.');
  assert.equal(ws.boxesText({ boxes: [{ n: 5 }] }, true), 'Box 5 on the big picture.');
  assert.equal(ws.boxesText({ boxes: [], role: 'companion' }, true),
    'Not boxed on the big picture (they walked alongside).');
  assert.equal(ws.boxesText({ boxes: [{ n: 1 }] }, false), '');
});

test('crop owners: each time crop gets the letter of the person it belongs to', () => {
  const entries = ws.storyPeople({ people: [{ individual_id: 'a', crop_ids: ['c0', 'c2'] },
    { individual_id: 'b', crop_ids: ['c1'] }] });
  const owners = ws.cropOwners(entries);
  assert.deepEqual(['c0', 'c1', 'c2'].map((c) => owners[c].letter), ['A', 'B', 'A']);
  assert.equal(owners.c3, undefined);
});

test('a yes on a named row reads back as a yes', () => {
  assert.equal(ws.actionLine({ action: 'confirm', name: 'Sasha' }), 'You said yes, this is Sasha.');
});
