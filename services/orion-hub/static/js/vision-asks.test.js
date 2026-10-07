const test = require('node:test');
const assert = require('node:assert/strict');
const asks = require('./vision-asks.js');

test('image ref: web URL renders as a picture, a disk path as text, empty as nothing', () => {
  assert.deepEqual(asks.askImageView('https://x/y.jpg'), { kind: 'img', ref: 'https://x/y.jpg' });
  assert.deepEqual(asks.askImageView('/mnt/telemetry/vision/frames/a.jpg'), { kind: 'text', ref: '/mnt/telemetry/vision/frames/a.jpg' });
  assert.deepEqual(asks.askImageView('javascript:alert(1)'), { kind: 'text', ref: 'javascript:alert(1)' });
  assert.deepEqual(asks.askImageView(null), { kind: 'none', ref: '' });
});

test('thumb ref maps to the Hub thumbnail route; a malformed one is text', () => {
  const h = 'ab'.repeat(32);
  assert.deepEqual(asks.askImageView('thumb:' + h), { kind: 'img', ref: '/api/vision/crop-thumbs/' + h });
  assert.equal(asks.askImageView('thumb:../../etc/passwd').kind, 'text');
  assert.equal(asks.askImageView('thumb:' + h.toUpperCase()).kind, 'text');
  assert.equal(asks.askImageView('crop:cropobs:art:0').kind, 'text');
});

test('status line is plain English and counts', () => {
  assert.equal(asks.statusLine([]), 'Orion has no open questions for you.');
  assert.equal(asks.statusLine([{}]), 'Orion has 1 question for you.');
  assert.equal(asks.statusLine([{}, {}]), 'Orion has 2 questions for you.');
});

test('409 explains that it was already answered or expired', () => {
  assert.match(asks.errorLine(409), /already answered|expired/);
});

test('submitAction posts the trimmed answer to the answer route', async () => {
  const calls = [];
  const fetchFn = async (url, init) => {
    calls.push({ url, init });
    return { ok: true, status: 200, json: async () => ({ ok: true }) };
  };
  const res = await asks.submitAction(fetchFn, 'ask 1', 'answer', '  the mail carrier ');
  assert.equal(res.ok, true);
  assert.equal(calls[0].url, '/api/asks/ask%201/answer');
  assert.equal(calls[0].init.method, 'POST');
  assert.deepEqual(JSON.parse(calls[0].init.body), { answer: 'the mail carrier' });
});

test('submitAction dismiss sends no body', async () => {
  const calls = [];
  const fetchFn = async (url, init) => {
    calls.push({ url, init });
    return { ok: false, status: 409, json: async () => ({ detail: 'ask_not_open:answered' }) };
  };
  const res = await asks.submitAction(fetchFn, 'a2', 'dismiss');
  assert.equal(res.status, 409);
  assert.equal(calls[0].url, '/api/asks/a2/dismiss');
  assert.equal(calls[0].init.body, undefined);
});

// Minimal DOM stand-in: enough to exercise mount() -> render -> click -> POST -> refresh.
function fakeDoc() {
  const byId = {};
  function mk(tag) {
    const node = {
      tagName: tag, className: '', textContent: '', children: [], attrs: {}, listeners: {}, disabled: false,
      classList: { remove() {}, add() {} },
      value: '',
      setAttribute(k, v) { this.attrs[k] = v; },
      getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
      appendChild(c) { this.children.push(c); return c; },
      removeChild(c) { this.children = this.children.filter((x) => x !== c); },
      get firstChild() { return this.children[0] || null; },
      addEventListener(ev, fn) { this.listeners[ev] = fn; },
    };
    return node;
  }
  byId.visionAsksList = mk('div');
  byId.visionAsksStatus = mk('div');
  return { hidden: false, activeElement: null, createElement: mk, getElementById: (id) => byId[id] || null, byId };
}

function find(node, pred) {
  if (pred(node)) return node;
  for (const c of node.children || []) {
    const hit = find(c, pred);
    if (hit) return hit;
  }
  return null;
}

test('mount renders open asks and Answer posts then refreshes', async () => {
  const doc = fakeDoc();
  let open = [{ ask_id: 'a1', question: 'Do you know who this is?', image_ref: '/mnt/x.jpg', source_kind: 'vision_individual', source_ref: 'ind-1' }];
  const posts = [];
  const fetchFn = async (url, init) => {
    if (init && init.method === 'POST') {
      posts.push({ url, body: init.body });
      open = [];
      return { ok: true, status: 200, json: async () => ({ ok: true, published: true }) };
    }
    return { ok: true, status: 200, json: async () => ({ asks: open }) };
  };
  const handle = asks.mount(doc, fetchFn);
  await handle.refresh();
  assert.equal(doc.byId.visionAsksStatus.textContent, 'Orion has 1 question for you.');
  const card = doc.byId.visionAsksList.children[0];
  assert.equal(card.attrs['data-ask-id'], 'a1');
  assert.ok(find(card, (n) => n.textContent === 'Picture: /mnt/x.jpg'));
  const input = find(card, (n) => n.attrs && n.attrs['data-ask-input'] === 'a1');
  input.value = 'the mail carrier';
  const answerBtn = find(card, (n) => n.attrs && n.attrs['data-ask-action'] === 'answer');
  await answerBtn.listeners.click();
  // click handler kicks off an async chain; let it settle.
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(posts.length, 1);
  assert.equal(posts[0].url, '/api/asks/a1/answer');
  assert.deepEqual(JSON.parse(posts[0].body), { answer: 'the mail carrier' });
  assert.equal(doc.byId.visionAsksStatus.textContent, 'Orion has no open questions for you.');
  handle.stop();
});

test('Answer with an empty box does not POST', async () => {
  const doc = fakeDoc();
  let posted = false;
  const fetchFn = async (url, init) => {
    if (init && init.method === 'POST') posted = true;
    return { ok: true, status: 200, json: async () => ({ asks: [{ ask_id: 'a1', question: 'q' }] }) };
  };
  const handle = asks.mount(doc, fetchFn);
  await handle.refresh();
  const card = doc.byId.visionAsksList.children[0];
  const note = card.children[card.children.length - 1];
  await handle.onAction('a1', 'answer', '   ', note);
  assert.equal(posted, false);
  assert.match(note.textContent, /Type an answer/);
  handle.stop();
});

test('poll is suppressed while an answer is being typed', async () => {
  const doc = fakeDoc();
  const fetchFn = async () => ({ ok: true, status: 200, json: async () => ({ asks: [{ ask_id: 'a1', question: 'q' }] }) });
  const handle = asks.mount(doc, fetchFn);
  await handle.refresh();
  assert.equal(handle.isTyping(), false);
  const card = doc.byId.visionAsksList.children[0];
  const input = find(card, (n) => n.attrs && n.attrs['data-ask-input'] === 'a1');
  input.value = 'half-typ';
  assert.equal(handle.isTyping(), true);
  input.value = '';
  doc.activeElement = input;
  assert.equal(handle.isTyping(), true);
  handle.stop();
});

test('buttons are disabled while the request is in flight', async () => {
  const doc = fakeDoc();
  let release;
  const gate = new Promise((r) => { release = r; });
  const fetchFn = async (url, init) => {
    if (init && init.method === 'POST') {
      await gate;
      return { ok: true, status: 200, json: async () => ({}) };
    }
    return { ok: true, status: 200, json: async () => ({ asks: [{ ask_id: 'a1', question: 'q' }] }) };
  };
  const handle = asks.mount(doc, fetchFn);
  await handle.refresh();
  const card = doc.byId.visionAsksList.children[0];
  const dismissBtn = find(card, (n) => n.attrs && n.attrs['data-ask-action'] === 'dismiss');
  const answerBtn = find(card, (n) => n.attrs && n.attrs['data-ask-action'] === 'answer');
  const pending = dismissBtn.listeners.click();
  assert.equal(dismissBtn.disabled, true);
  assert.equal(answerBtn.disabled, true);
  release();
  await pending;
  assert.equal(dismissBtn.disabled, false);
  handle.stop();
});

test('a non-string error detail (FastAPI 422 list) is not shown as [object Object]', () => {
  assert.equal(asks.errorLine(422, [{ msg: 'x' }]), 'Something went wrong.');
});

test('a thumb ask renders an <img> pointing at the Hub thumbnail route', () => {
  const doc = fakeDoc();
  const h = '0f'.repeat(32);
  const card = asks.renderAsk(doc, { ask_id: 'a9', question: 'Who?', image_ref: 'thumb:' + h }, () => {});
  const img = find(card, (n) => n.tagName === 'img');
  assert.ok(img, 'expected an img node');
  assert.equal(img.attrs.src, '/api/vision/crop-thumbs/' + h);
});

test('a thumbnail that fails to load is hidden, not shown broken', () => {
  const doc = fakeDoc();
  const card = asks.renderAsk(doc, { ask_id: 'a8', question: 'Who?', image_ref: 'thumb:' + '1a'.repeat(32) }, () => {});
  const img = find(card, (n) => n.tagName === 'img');
  assert.ok(img.listeners.error, 'expected an error handler');
  img.listeners.error();
  assert.equal(img.hidden, true);
});

// --- memory confirmation cards (2026-10-06) ---------------------------------

const MEM_ASK = {
  ask_id: 'm1', question: 'You told me something on Oct 3, and I wrote it down like this: “X.” Want me to remember that?',
  source_kind: 'memory_confirmation', source_ref: 'memory-confirm-00000000-0000-0000-0000-000000000007',
  memory_statement: 'Juniper told me that X happened today.',
};

function memMount(postReply) {
  const doc = fakeDoc();
  let open = [MEM_ASK];
  const posts = [];
  const fetchFn = async (url, init) => {
    if (init && init.method === 'POST') {
      posts.push({ url, body: JSON.parse(init.body || '{}') });
      const reply = postReply ? postReply(posts) : { ok: true, status: 200, body: { ok: true } };
      if (reply.ok) open = [];
      return { ok: reply.ok, status: reply.status, json: async () => reply.body };
    }
    return { ok: true, status: 200, json: async () => ({ asks: open }) };
  };
  return { doc, posts, handle: asks.mount(doc, fetchFn) };
}

const settle = async () => { for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0)); };
const action = (card, a) => find(card, (n) => n.attrs && n.attrs['data-ask-action'] === a);

test('a memory card offers Confirm / Revise / Reject, not Answer / Dismiss', async () => {
  const { doc, handle } = memMount();
  await handle.refresh();
  const card = doc.byId.visionAsksList.children[0];
  assert.equal(card.attrs['data-ask-kind'], 'resolvable');
  for (const a of ['confirm', 'revise', 'reject']) assert.ok(action(card, a), a);
  assert.equal(action(card, 'answer'), null);
  assert.equal(action(card, 'dismiss'), null);
  assert.ok(asks.isResolvable(MEM_ASK) && !asks.isResolvable({ source_kind: 'vision_individual' }));
  handle.stop();
});

for (const [btn, resolution] of [['confirm', 'confirmed'], ['reject', 'rejected']]) {
  test(`${btn} posts {resolution: ${resolution}} to the resolve route and refreshes`, async () => {
    const { doc, posts, handle } = memMount();
    await handle.refresh();
    await action(doc.byId.visionAsksList.children[0], btn).listeners.click();
    await settle();
    assert.deepEqual(posts, [{ url: '/api/asks/m1/resolve', body: { resolution, note: '' } }]);
    assert.equal(doc.byId.visionAsksStatus.textContent, 'Orion has no open questions for you.');
    handle.stop();
  });
}

test('Revise opens a box prefilled with the current wording and posts the edit', async () => {
  const { doc, posts, handle } = memMount();
  await handle.refresh();
  const card = doc.byId.visionAsksList.children[0];
  const box = find(card, (n) => n.tagName === 'textarea');
  assert.equal(box.value, '');
  const panel = find(card, (n) => n.attrs && n.attrs['data-ask-revise'] === 'm1');
  assert.equal(panel.hidden, true);
  action(card, 'revise').listeners.click();
  assert.equal(panel.hidden, false);
  assert.equal(box.value, 'Juniper told me that X happened today.');
  box.value = '  Juniper told me Y happened, not X. ';
  await action(card, 'save-revision').listeners.click();
  await settle();
  assert.deepEqual(posts, [{ url: '/api/asks/m1/resolve', body: { resolution: 'revised', note: 'Juniper told me Y happened, not X.' } }]);
  handle.stop();
});

test('Revise with an empty box does not POST and says why', async () => {
  const { doc, posts, handle } = memMount();
  await handle.refresh();
  const card = doc.byId.visionAsksList.children[0];
  find(card, (n) => n.tagName === 'textarea').value = '   ';
  await action(card, 'save-revision').listeners.click();
  await settle();
  assert.equal(posts.length, 0);
  assert.ok(find(card, (n) => n.textContent === 'Write how I should remember it first.'));
  handle.stop();
});

test('a stale memory card (409) explains and refreshes', async () => {
  const { doc, handle } = memMount(() => ({ ok: false, status: 409, body: { detail: 'ask_not_open:answered' } }));
  await handle.refresh();
  const card = doc.byId.visionAsksList.children[0];
  await action(card, 'confirm').listeners.click();
  await settle();
  assert.ok(find(card, (n) => /already answered/.test(n.textContent)));
  handle.stop();
});

test('revisionProblem mirrors the server: empty, too short, unchanged (structural, no word list)', () => {
  assert.equal(asks.revisionProblem('   ', 'a b c d e f'), 'revised_needs_note');
  assert.equal(asks.revisionProblem("no that's wrong", 'a b c d e f'), 'revised_too_short');
  assert.equal(asks.revisionProblem(' Juniper told me X   happened today ', 'juniper told me x happened today'), 'revised_unchanged');
  assert.equal(asks.revisionProblem('Juniper told me Y happened, not X.', 'Juniper told me X happened.'), null);
});

test('a meta-note revision is refused in the card, points at Reject, and does not POST', async () => {
  const { doc, posts, handle } = memMount();
  await handle.refresh();
  const card = doc.byId.visionAsksList.children[0];
  action(card, 'revise').listeners.click();
  find(card, (n) => n.tagName === 'textarea').value = 'no, wrong';
  await action(card, 'save-revision').listeners.click();
  await settle();
  assert.equal(posts.length, 0);
  assert.ok(find(card, (n) => /press Reject/.test(n.textContent)));
  find(card, (n) => n.tagName === 'textarea').value = ' juniper told me that X happened today. ';  // the prefill, unchanged
  await action(card, 'save-revision').listeners.click();
  await settle();
  assert.equal(posts.length, 0);
  assert.ok(find(card, (n) => /same as what I have/.test(n.textContent)));
  handle.stop();
});

test('asks list: only the first two cards are visible, the rest scroll', () => {
  assert.equal(asks.visibleAsksMaxHeight([100, 120, 90, 300], 8), 228);
  assert.equal(asks.visibleAsksMaxHeight([100, 120], 8), null);
  assert.equal(asks.visibleAsksMaxHeight([], 8), null);
  assert.equal(asks.visibleAsksMaxHeight([50, 60, 70], 0), 110);
});
