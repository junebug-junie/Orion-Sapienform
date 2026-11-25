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

test('linked persons: one row per person, "seen as N camera groups", one not-same choice per direct link', () => {
  const sasha = { person_key: 'g1', individual_ids: ['g1', 'g2'], label: 'Sasha', pictured: true,
    pictures: [{ thumb: H('a'), crop_id: 'c1' }],
    groups: [{ individual_id: 'g1', short_id: 'g1aaaaaa' }, { individual_id: 'g2', short_id: 'g2bbbbbb' }],
    links: [{ a: 'g1', b: 'g2' }, { a: 'g1', b: 'outside' }] };
  const entries = ws.storyPeople({ persons: [sasha, { person_key: 'z', individual_ids: ['z'] }],
    people: [{ individual_id: 'g1' }, { individual_id: 'g2' }, { individual_id: 'z' }] });
  assert.deepEqual(entries.map((e) => [e.person.individual_id, e.person.grouped, e.letter]),
    [['g1', true, 'A'], ['z', true, 'B']]);
  assert.match(ws.groupsText(sasha), /^Seen as 2 camera groups in this story/);
  assert.equal(ws.groupsText({ individual_ids: ['z'] }), '');
  assert.deepEqual(ws.notSameChoices(sasha),
    [{ a: 'g1', b: 'g2', label: 'group g1aaaaaa and group g2bbbbbb are different people' }]);
  // A thumbnail-only picture of a linked person is shown from vision-host's thumbs.
  assert.deepEqual(ws.personPictures(sasha).map((c) => c.url), [`/api/vision/crop-thumbs/${H('a')}`]);
  // An older server (people only): one row per camera group, never `grouped`.
  assert.deepEqual(ws.storyPeople({ people: [{ individual_id: 'g1' }] }).map((e) => e.person.grouped), [undefined]);
});

test('one at a time: the question, the names offered, and what each answer sends', () => {
  assert.equal(ws.questionText({ label: 'Alex' }), 'Is this Alex?');
  assert.equal(ws.questionText({ label: null }), 'Who is this?');
  assert.equal(ws.questionText({ label: null, role: 'companion' }), 'Walking with them: Who is this?');
  const names = [{ name: 'Theo' }, { name: 'alex' }, { name: ' Sasha ' }, { name: 'theo' }, { name: '' }];
  // Never the person's own name (that is "yes"); one per name; only for people.
  assert.deepEqual(ws.nameChoices({ label: 'Alex', kind: 'person' }, names), ['Theo', 'Sasha']);
  assert.deepEqual(ws.nameChoices({ label: null, kind: 'dog' }, names), []);
  assert.deepEqual(ws.nameChoices({ label: null }, Array.from({ length: 20 }, (_, i) => ({ name: `n${i}` }))).length, 8);
  const alex = { label: 'Alex' };
  const anon = { label: null };
  assert.deepEqual(ws.answerSpec(alex, 'yes'), { action: 'confirm', name: 'Alex' });
  assert.deepEqual(ws.answerSpec(alex, 'pick', 'Cora'), { action: 'someone_else', name: 'Cora' });
  assert.deepEqual(ws.answerSpec(anon, 'pick', ' Theo  Ray '), { action: 'name', name: 'Theo Ray' });
  assert.deepEqual(ws.answerSpec(alex, 'dont_know'), { action: 'someone_else', name: '' });
  assert.deepEqual(ws.answerSpec(anon, 'dont_know'), { action: 'unsure', name: '', dontKnow: true });
  assert.deepEqual(ws.answerSpec(anon, 'cant_see'), { action: 'unsure', name: '' });
  assert.equal(ws.pendingText(alex, { name: 'Cora' }), 'Changed to Cora (not Alex)');
  assert.equal(ws.pendingText(anon, { name: 'Theo' }), 'Named Theo');
});

test("tally line counts can't-see answers apart when the server sends them", () => {
  assert.match(ws.tallyLine({ checked: 1, person_answers: 10, cant_see: 4 }), /can't see them 4 of 10 person answers$/);
  assert.doesNotMatch(ws.tallyLine({ checked: 1 }), /can't see them/);
});

test('the cursor is a person: it follows the person, else the first one not yet answered', () => {
  const e = (k, said) => ({ person: { individual_id: k, story_action: said ? { action: 'unsure' } : null } });
  const askable = [e('a', true), e('b', false), e('c', false)];
  assert.equal(ws.cursorIndex(askable, 'c'), 2);
  assert.equal(ws.cursorIndex(askable, 'gone'), 1);      // left the list: the first unanswered, not a shifted index
  assert.equal(ws.cursorIndex(askable, undefined), 1);
  assert.equal(ws.cursorIndex([e('a', true)], 'x'), 0);
});

test('sentence parts: only the recorded ones, and the tally says where each verdict came from', () => {
  assert.deepEqual(ws.storyParts({ parts: { stated: ['people', 'place', 'length'] } }), ['people', 'place']);
  assert.equal(ws.storyParts({ parts: { stated: null } }), null);
  assert.equal(ws.storyParts({}), null);
  const t = { by_source: { button: { right: 260, wrong_name: 5, cant_tell: 1 }, parts: { right: 3, wrong_motion: 1 } },
              wrong_name_from_corrections: 1, half_graded: 2 };
  assert.equal(ws.sourceText(t), ' · from parts 4, from the old buttons 265 · wrong names held by your corrections 1 · ' +
    'half-graded 2 (not counted yet)');
  assert.equal(ws.sourceText({}), '');
});

test('a link this card made says where it stands', () => {
  assert.equal(ws.linkState({ applied: true }), 'Linked as the same person on this card.');
  assert.equal(ws.linkState({ applied: false }), 'Same person: Orion links them on its next pass.');
  assert.equal(ws.linkState({ undo_requested_at: 't' }), 'Undo asked for; Orion applies it on its next pass.');
  assert.equal(ws.linkState({ undo_refused: 'still joined' }), 'Undo refused: still joined');
  assert.equal(ws.linkState({ undone_at: 't', undo_refused: 'x' }), 'Link undone.');
});

// 2026-11-25 card fixes: what "Right people?" asks, the automatic no after a name correction,
// and the result line under graded parts.
test('people part says what Orion claimed', () => {
  assert.equal(ws.partQuestion({ name_label: 'Addy' }, 'people'), 'Right person? (Orion said Addy)');
  assert.equal(ws.partQuestion({ name_label: null }, 'people'), 'Right number of people?');
  assert.equal(ws.partQuestion({}, 'direction'), 'Right direction?');
});

test('a name correction on the walker pre-marks people as no; sent or held', () => {
  const story = { story_id: 's1', name_label: 'Addy',
    persons: [{ person_key: 'k1', role: 'walker', label: 'Addy', story_action: null }] };
  assert.equal(ws.correctedName(story, null), null);
  const held = { story_id: 's1', person_key: 'k1', spec: { action: 'someone_else', name: 'Juniper' } };
  assert.equal(ws.correctedName(story, held), 'Addy');
  const sent = Object.assign({}, story, { persons: [Object.assign({}, story.persons[0],
    { story_action: { action: 'name', name: 'Juniper' } })] });
  assert.equal(ws.correctedName(sent, null), 'Addy');
  // A yes, a companion, or a sentence that named nobody is not a correction.
  const yes = Object.assign({}, story, { persons: [Object.assign({}, story.persons[0],
    { story_action: { action: 'confirm', name: 'Addy' } })] });
  assert.equal(ws.correctedName(yes, null), null);
  const comp = Object.assign({}, story, { persons: [Object.assign({}, story.persons[0],
    { role: 'companion', story_action: { action: 'someone_else', name: 'X' } })] });
  assert.equal(ws.correctedName(comp, null), null);
  assert.equal(ws.correctedName(Object.assign({}, sent, { name_label: null }), null), null);
});

test('result line says why a story is can\'t tell', () => {
  const story = { verdict: 'cant_tell', name_label: null, parts: { answers: {
    people: { verdict: 'cant_tell' }, direction: { verdict: 'yes' }, place: { verdict: 'yes' } } } };
  assert.equal(ws.resultLine(story, ['people', 'direction', 'place']),
    "This story counts as can't tell (you said can't tell for: Right number of people?).");
  assert.equal(ws.resultLine({ verdict: null, parts: { answers: {} } }, ['people']), null);
});

test('a verdict from the old buttons shows while parts are still open', () => {
  const story = { verdict: 'right', parts: { answers: { people: { verdict: 'yes' } } } };
  assert.equal(ws.resultLine(story, ['people', 'direction']),
    'Counted as right so far; answering every part replaces it.');
});
