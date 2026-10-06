const test = require('node:test');
const assert = require('node:assert/strict');

const api = require('./draft-revision.js');

// Minimal DOM: just the surface draft-revision.js uses. jsdom is declared in
// package.json but not installed on the hosts that run this suite.
class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.dataset = {};
    this.className = '';
    this._text = '';
  }
  get parentElement() { return this.parentNode; }
  appendChild(child) {
    if (child.parentNode) child.parentNode.removeChild(child);
    this.children.push(child);
    child.parentNode = this;
    return child;
  }
  insertBefore(child, ref) {
    if (child.parentNode) child.parentNode.removeChild(child);
    if (ref == null) return this.appendChild(child);
    const i = this.children.indexOf(ref);
    if (i < 0) throw new Error('ref not a child');
    this.children.splice(i, 0, child);
    child.parentNode = this;
    return child;
  }
  removeChild(child) {
    const i = this.children.indexOf(child);
    if (i < 0) throw new Error('not a child');
    this.children.splice(i, 1);
    child.parentNode = null;
    return child;
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(''); }
}
const doc = { createElement: (t) => new El(t) };

function finalMessage(text) {
  // Stand-in for app.js::appendMessage's node: header row then body.
  const div = doc.createElement('div');
  div.dataset.turnId = 'turn-final';
  const header = doc.createElement('div');
  header.textContent = 'Orion';
  const body = doc.createElement('p');
  body.textContent = text;
  div.appendChild(header);
  div.appendChild(body);
  return div;
}

function conversation() {
  const c = doc.createElement('div');
  const you = doc.createElement('div');
  you.textContent = 'You: hi';
  c.appendChild(you);
  return c;
}

test('draft is shown as a provisional Orion message with a checking badge', () => {
  const c = conversation();
  const node = api.showDraft(c, { correlation_id: 'corr-1', draft_text: 'hello Juniper' }, { document: doc });
  assert.equal(c.children.length, 2);
  assert.equal(node.dataset.draftPreviewFor, 'corr-1');
  assert.equal(node.dataset.draftState, 'checking');
  assert.match(node.textContent, /hello Juniper/);
  assert.match(node.textContent, /draft, still being checked/);
  // Not a memory-graph turn: the draft must never be picked up as a turn.
  assert.equal(node.dataset.turnId, undefined);
  // Repeated frame for the same turn does not add a second bubble.
  api.showDraft(c, { correlation_id: 'corr-1', draft_text: 'hello Juniper' }, { document: doc });
  assert.equal(c.children.length, 2);
});

test('revision replaces the draft in place and marks it revised with the reason', () => {
  const c = conversation();
  const draft = api.showDraft(c, { correlation_id: 'corr-2', draft_text: 'You -- the containers are all up' }, { document: doc });
  // Something else lands after the draft before the final arrives.
  const sys = doc.createElement('div');
  sys.textContent = 'System: note';
  c.appendChild(sys);
  // appendMessage appends the final at the end; settle moves it into place.
  const fin = c.appendChild(finalMessage('Juniper -- I checked two containers'));
  const found = api.findDraft(c, 'corr-2');
  assert.equal(found, draft);
  const result = api.settleWithFinal(c, found, fin, { revised: true, revised_reason: 'strain_unresolved' }, { document: doc });
  assert.equal(result, 'revised');
  assert.equal(c.children.length, 3);
  assert.equal(c.children[1], fin, 'final takes the draft position');
  assert.equal(c.children[2], sys);
  assert.equal(api.findDraft(c, 'corr-2'), null, 'draft is gone');
  assert.equal(fin.children[1].dataset.revisedMarker, '1', 'marker sits under the header');
  assert.equal(fin.children[1].textContent, 'revised: the check found an unresolved strain');
  assert.equal(fin.dataset.revised, '1');
});

test('no-revision case swaps the draft for the final with no marker', () => {
  const c = conversation();
  const draft = api.showDraft(c, { correlation_id: 'corr-3', draft_text: 'same text' }, { document: doc });
  const fin = c.appendChild(finalMessage('same text'));
  const result = api.settleWithFinal(c, draft, fin, { revised: false, revised_reason: null }, { document: doc });
  assert.equal(result, 'replaced');
  assert.equal(c.children.length, 2);
  assert.equal(c.children[1], fin);
  assert.equal(fin.children.length, 2, 'no marker added');
  assert.equal(fin.dataset.revised, undefined);
});

test('turns with no draft are untouched (judge-first and flag-off paths)', () => {
  const c = conversation();
  const fin = c.appendChild(finalMessage('final only'));
  assert.equal(api.findDraft(c, 'corr-4'), null);
  assert.equal(api.settleWithFinal(c, null, fin, { revised: false }, { document: doc }), 'no_draft');
  assert.equal(c.children.length, 2);
  assert.equal(c.children[1], fin);
});

test('a turn that fails after the draft keeps it, labelled not checked', () => {
  const c = conversation();
  const draft = api.showDraft(c, { correlation_id: 'corr-5', draft_text: 'partial' }, { document: doc });
  assert.equal(api.settleWithoutFinal(draft), true);
  assert.equal(draft.dataset.draftState, 'unchecked');
  assert.match(draft.textContent, /not checked/);
});

test('empty or unkeyed draft frames render nothing', () => {
  const c = conversation();
  assert.equal(api.showDraft(c, { correlation_id: 'corr-6', draft_text: '   ' }, { document: doc }), null);
  assert.equal(api.showDraft(c, { draft_text: 'x' }, { document: doc }), null);
  assert.equal(c.children.length, 1);
});

test('revisedLabel names known reasons and falls back to the raw one', () => {
  assert.equal(api.revisedLabel('misaligned'), 'revised: the check found it out of line with the stance');
  assert.equal(api.revisedLabel('uncertain'), 'revised: the check was unsure about it');
  assert.equal(api.revisedLabel('finalize_changed'), 'revised: the final pass changed it');
  assert.equal(api.revisedLabel('something_new'), 'revised: something_new');
  assert.equal(api.revisedLabel(null), 'revised: the check changed it');
});
