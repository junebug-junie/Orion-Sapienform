const test = require('node:test');
const assert = require('node:assert/strict');

let JSDOM;
try {
  ({ JSDOM } = require('jsdom'));
} catch (err) {
  JSDOM = null;
}

function fresh() {
  const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
  global.window = dom.window;
  global.document = dom.window.document;
  delete require.cache[require.resolve('./chat-attachments.js')];
  return { dom, api: require('./chat-attachments.js') };
}

const describe = JSDOM ? test : test.skip;

// ── formatBytes ──────────────────────────────────────────────────────────
// Expected strings are hand-computed, not read back from the implementation.

test('formatBytes uses the right unit at each boundary', () => {
  const { formatBytes } = require('./chat-attachments.js');
  assert.equal(formatBytes(0), '0 B');
  assert.equal(formatBytes(1023), '1023 B');
  assert.equal(formatBytes(1024), '1 KB');
  assert.equal(formatBytes(1536), '2 KB');          // 1.5 rounds to 2 at 0 dp
  assert.equal(formatBytes(1024 * 1024 - 1), '1024 KB');
  assert.equal(formatBytes(1024 * 1024), '1.0 MB');
  assert.equal(formatBytes(2621440), '2.5 MB');     // 2.5 * 1024 * 1024
  assert.equal(formatBytes(undefined), '0 B');
  assert.equal(formatBytes(null), '0 B');
});

// ── imageFilesFrom ───────────────────────────────────────────────────────

// A real File, not a stub: FormData.append rejects plain objects, so a stub
// would make every upload path throw for a reason the code under test never
// causes. (Found the hard way -- five tests failed on it.)
function fakeFile(type, name, bytes = 100) {
  return new File([new Uint8Array(bytes)], name, { type });
}

test('imageFilesFrom reads clipboard items and ignores non-images', () => {
  const { imageFilesFrom } = require('./chat-attachments.js');
  const png = fakeFile('image/png', 'a.png');
  const clipboard = {
    items: [
      { kind: 'string', getAsFile: () => null },
      { kind: 'file', getAsFile: () => png },
      { kind: 'file', getAsFile: () => fakeFile('application/pdf', 'b.pdf') },
    ],
  };
  assert.deepEqual(imageFilesFrom(clipboard), [png]);
});

test('imageFilesFrom falls back to .files when items yields nothing', () => {
  const { imageFilesFrom } = require('./chat-attachments.js');
  const jpg = fakeFile('image/jpeg', 'a.jpg');
  const drop = {
    items: [{ kind: 'string', getAsFile: () => null }],
    files: [jpg, fakeFile('text/plain', 'notes.txt')],
  };
  assert.deepEqual(imageFilesFrom(drop), [jpg]);
});

test('imageFilesFrom tolerates null and empty sources', () => {
  const { imageFilesFrom } = require('./chat-attachments.js');
  assert.deepEqual(imageFilesFrom(null), []);
  assert.deepEqual(imageFilesFrom({}), []);
  assert.deepEqual(imageFilesFrom({ items: [], files: [] }), []);
});

// ── controller ───────────────────────────────────────────────────────────

function buildDom(dom) {
  const d = dom.window.document;
  const chipRow = d.createElement('div');
  const statusEl = d.createElement('div');
  const attachButton = d.createElement('button');
  const fileInput = d.createElement('input');
  fileInput.type = 'file';
  const dropZone = d.createElement('div');
  const pasteTarget = d.createElement('textarea');
  [chipRow, statusEl, attachButton, fileInput, dropZone, pasteTarget].forEach((n) => d.body.appendChild(n));
  return { chipRow, statusEl, attachButton, fileInput, dropZone, pasteTarget };
}

function ref(sha, extra) {
  return Object.assign({
    kind: 'image',
    sha256: sha,
    mime: 'image/png',
    bytes: 2048,
    width: 64,
    height: 64,
    source_url: `http://100.92.216.81:8080/api/chat/attachments/${sha}`,
    filename: `${sha.slice(0, 4)}.png`,
  }, extra || {});
}

describe('attach button is disabled when the route reports no vision', () => {
  const { dom, api } = fresh();
  const nodes = buildDom(dom);
  api.createController(Object.assign({ visionAvailable: () => false }, nodes));
  assert.equal(nodes.attachButton.disabled, true);
  assert.match(nodes.attachButton.title, /no vision capability/);
});

describe('attach button is enabled when the route can see', () => {
  const { dom, api } = fresh();
  const nodes = buildDom(dom);
  api.createController(Object.assign({ visionAvailable: () => true }, nodes));
  assert.equal(nodes.attachButton.disabled, false);
});

describe('takePending returns refs once and clears the composer', async () => {
  const { dom, api } = fresh();
  const nodes = buildDom(dom);
  let uploaded = 0;
  global.fetch = async () => ({
    ok: true,
    json: async () => { uploaded += 1; return ref('a'.repeat(64)); },
  });
  const ctl = api.createController(Object.assign({ visionAvailable: () => true }, nodes));
  await ctl.addFiles([fakeFile('image/gif', 'x.gif')]);   // gif skips canvas work

  assert.equal(uploaded, 1);
  assert.equal(ctl.peekPending().length, 1);
  assert.equal(nodes.chipRow.querySelectorAll('.om-chip').length, 1);

  const taken = ctl.takePending();
  assert.equal(taken.length, 1);
  assert.equal(ctl.peekPending().length, 0, 'pending survived takePending');
  assert.deepEqual(ctl.takePending(), [], 'second take re-sent the same images');
  assert.equal(nodes.chipRow.querySelectorAll('.om-chip').length, 0);
});

describe('the same image attached twice is stored once', async () => {
  const { dom, api } = fresh();
  const nodes = buildDom(dom);
  global.fetch = async () => ({ ok: true, json: async () => ref('b'.repeat(64)) });
  const ctl = api.createController(Object.assign({ visionAvailable: () => true }, nodes));
  await ctl.addFiles([fakeFile('image/gif', 'x.gif')]);
  await ctl.addFiles([fakeFile('image/gif', 'copy.gif')]);
  assert.equal(ctl.peekPending().length, 1, 'content-addressed dedup did not hold');
});

describe('per-turn limit drops extras and says so', async () => {
  const { dom, api } = fresh();
  const nodes = buildDom(dom);
  let n = 0;
  global.fetch = async () => ({
    ok: true,
    json: async () => { n += 1; return ref(String(n).repeat(64).slice(0, 64)); },
  });
  const ctl = api.createController(Object.assign(
    { visionAvailable: () => true, options: { maxPerTurn: 2 } },
    nodes,
  ));
  await ctl.addFiles([
    fakeFile('image/gif', '1.gif'),
    fakeFile('image/gif', '2.gif'),
    fakeFile('image/gif', '3.gif'),
  ]);
  assert.equal(ctl.peekPending().length, 2);
  assert.match(nodes.statusEl.textContent, /dropped 1/);
  assert.equal(nodes.attachButton.disabled, true, 'button stayed enabled at the limit');
});

describe('a failed upload surfaces the server detail and attaches nothing', async () => {
  const { dom, api } = fresh();
  const nodes = buildDom(dom);
  global.fetch = async () => ({
    ok: false,
    status: 415,
    json: async () => ({ detail: 'unsupported attachment type' }),
  });
  const ctl = api.createController(Object.assign({ visionAvailable: () => true }, nodes));
  await ctl.addFiles([fakeFile('image/gif', 'x.gif')]);
  assert.equal(ctl.peekPending().length, 0);
  assert.match(nodes.statusEl.textContent, /unsupported attachment type/);
});

describe('adding files while blind refuses instead of uploading', async () => {
  const { dom, api } = fresh();
  const nodes = buildDom(dom);
  let called = false;
  global.fetch = async () => { called = true; return { ok: true, json: async () => ref('c'.repeat(64)) }; };
  const ctl = api.createController(Object.assign({ visionAvailable: () => false }, nodes));
  await ctl.addFiles([fakeFile('image/gif', 'x.gif')]);
  assert.equal(called, false, 'uploaded to a blind route');
  assert.match(nodes.statusEl.textContent, /cannot accept images/);
});

describe('removing a chip drops that ref', async () => {
  const { dom, api } = fresh();
  const nodes = buildDom(dom);
  global.fetch = async () => ({ ok: true, json: async () => ref('d'.repeat(64)) });
  const ctl = api.createController(Object.assign({ visionAvailable: () => true }, nodes));
  await ctl.addFiles([fakeFile('image/gif', 'x.gif')]);
  nodes.chipRow.querySelector('.om-chip-remove').click();
  assert.equal(ctl.peekPending().length, 0);
});

describe('downscale passes GIFs through untouched', async () => {
  const { api } = fresh();
  const gif = fakeFile('image/gif', 'anim.gif');
  // Would flatten an animation to one frame if it went through canvas.
  assert.equal(await api.downscale(gif, api.DEFAULTS), gif);
});

describe('downscale returns the original when the image cannot be decoded', async () => {
  const { api } = fresh();
  global.window.createImageBitmap = async () => { throw new Error('bad image'); };
  const png = fakeFile('image/png', 'x.png');
  assert.equal(await api.downscale(png, api.DEFAULTS), png);
});

// ── dragover protected mode ──────────────────────────────────────────────

test('dragCarriesImage reads kind/type, which is all that is readable mid-drag', () => {
  const { dragCarriesImage } = require('./chat-attachments.js');
  // During dragover the DataTransfer is protected: getAsFile() returns null and
  // .files is empty. Only kind and type are visible.
  const protectedDrag = {
    items: [{ kind: 'file', type: 'image/png', getAsFile: () => null }],
    files: [],
  };
  assert.equal(dragCarriesImage(protectedDrag), true);
  assert.equal(
    require('./chat-attachments.js').imageFilesFrom(protectedDrag).length,
    0,
    'imageFilesFrom must NOT be usable for the dragover decision',
  );
});

test('dragCarriesImage accepts a typeless file drag and rejects plain text', () => {
  const { dragCarriesImage } = require('./chat-attachments.js');
  assert.equal(dragCarriesImage({ items: [{ kind: 'file', type: '' }] }), true);
  assert.equal(dragCarriesImage({ items: [{ kind: 'string', type: 'text/plain' }] }), false);
  assert.equal(dragCarriesImage({ types: ['Files'] }), true);
  assert.equal(dragCarriesImage({ types: ['text/plain'] }), false);
  assert.equal(dragCarriesImage(null), false);
});
