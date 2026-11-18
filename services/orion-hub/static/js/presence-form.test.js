const test = require('node:test');
const assert = require('node:assert/strict');
const form = require('./presence-form.js');

test('each fixed-value field is a labeled dropdown, not a free-text box', () => {
  const html = form.rowHtml({ display_name: 'Pip', relationship: 'child' }, 0);
  for (const field of ['relationship', 'age_band', 'role']) {
    assert.match(html, new RegExp(`<select data-presence-field="${field}"`));
    assert.doesNotMatch(html, new RegExp(`<input data-presence-field="${field}"`));
  }
  assert.match(html, />Name</);
  assert.match(html, />Who they are</);
  assert.match(html, />Age</);
});

test('dropdown values match PresenceCompanionV1 literals exactly', () => {
  assert.deepEqual(form.CHOICES.relationship.map((p) => p[0]), ['child', 'spouse', 'friend', 'coworker', 'guest', 'other']);
  assert.deepEqual(form.CHOICES.age_band.map((p) => p[0]), ['child', 'teen', 'adult', 'unknown']);
  assert.deepEqual(form.CHOICES.role.map((p) => p[0]), ['listener', 'asker', 'participant', 'nearby']);
});

test('a child defaults to age child and the saved values are selected', () => {
  const html = form.rowHtml({ display_name: 'Pip', relationship: 'child' }, 0);
  assert.match(html, /<option value="child" selected>Child<\/option><option value="spouse">/);
  assert.match(html, /data-presence-field="age_band"[^>]*>(<option[^>]*>[^<]*<\/option>)*?<option value="child" selected>/);
});

test('names are escaped, not injected as markup', () => {
  const html = form.rowHtml({ display_name: '"><img src=x onerror=alert(1)>' }, 0);
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /&quot;&gt;&lt;img/);
});

test('collectRows reads selects and inputs back per row and drops nameless rows', () => {
  const el = (i, f, v) => ({ dataset: { presenceIndex: String(i), presenceField: f }, value: v });
  const rows = form.collectRows([
    el(0, 'display_name', ' Pip '), el(0, 'relationship', 'child'), el(0, 'age_band', 'child'), el(0, 'role', 'listener'),
    el(1, 'display_name', ''), el(1, 'relationship', 'other'),
  ]);
  assert.deepEqual(rows, [{ display_name: 'Pip', relationship: 'child', age_band: 'child', role: 'listener' }]);
});

test('a saved note is shown and survives the read-back', () => {
  const html = form.rowHtml({ display_name: 'Pip', relationship: 'child', context_note: 'loves <rockets>' }, 0);
  assert.match(html, /data-presence-field="context_note"[^>]*value="loves &lt;rockets&gt;"/);
  assert.match(html, />Note \(optional\)</);
  const el = (f, v) => ({ dataset: { presenceIndex: '0', presenceField: f }, value: v });
  const rows = form.collectRows([el('display_name', 'Pip'), el('context_note', 'loves rockets')]);
  assert.equal(rows[0].context_note, 'loves rockets');
});
