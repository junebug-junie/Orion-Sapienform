/**
 * Markdown rendering + sanitization.
 *
 * Uses the SAME vendored marked/DOMPurify the browser loads, driven through
 * jsdom, rather than stubs. A sanitizer test against a fake sanitizer proves
 * nothing.
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

let JSDOM;
try {
  ({ JSDOM } = require('jsdom'));
} catch (err) {
  JSDOM = null;
}

function loadInDom() {
  const dom = new JSDOM('<!doctype html><html><body></body></html>', {
    runScripts: 'outside-only',
    url: 'http://localhost/',
  });
  const { window } = dom;
  global.window = window;
  global.document = window.document;
  global.navigator = window.navigator;

  // Vendored bundles, evaluated against the jsdom window exactly as a browser
  // would evaluate them.
  const vm = require('node:vm');
  const fs = require('node:fs');
  const ctx = dom.getInternalVMContext();
  for (const file of ['vendor/marked-15.0.7.min.js', 'vendor/dompurify-3.2.4.min.js']) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, file), 'utf8'), ctx);
  }
  delete require.cache[require.resolve('./chat-markdown.js')];
  const api = require('./chat-markdown.js');
  return { window, api };
}

const describe = JSDOM ? test : test.skip;

describe('renders headings, lists, tables and inline code', () => {
  const { window, api } = loadInDom();
  window.marked = window.marked || global.marked;
  const frag = api.renderMarkdown('# Title\n\n- one\n- two\n\n| a | b |\n| - | - |\n| 1 | 2 |\n\nuse `x = 1` here');
  assert.ok(frag, 'expected a rendered fragment');
  const host = window.document.createElement('div');
  host.appendChild(frag);
  assert.equal(host.querySelectorAll('h1').length, 1);
  assert.equal(host.querySelectorAll('li').length, 2);
  assert.equal(host.querySelectorAll('table').length, 1);
  assert.equal(host.querySelectorAll('td').length, 2);
  assert.equal(host.querySelectorAll('code').length, 1);
});

describe('a script tag in model output does not survive rendering', () => {
  const { window, api } = loadInDom();
  const hostile = 'hello\n\n<script>window.__pwned = true;</script>\n\n<img src=x onerror="window.__pwned=true">';
  const frag = api.renderMarkdown(hostile);
  const host = window.document.createElement('div');
  host.appendChild(frag);
  assert.equal(host.querySelectorAll('script').length, 0, 'script tag survived');
  const img = host.querySelector('img');
  if (img) assert.equal(img.getAttribute('onerror'), null, 'onerror survived');
  assert.equal(window.__pwned, undefined, 'inline script executed');
});

describe('javascript: hrefs are stripped', () => {
  const { window, api } = loadInDom();
  const frag = api.renderMarkdown('[click](javascript:alert(1))');
  const host = window.document.createElement('div');
  host.appendChild(frag);
  const a = host.querySelector('a');
  if (a) {
    assert.ok(!String(a.getAttribute('href') || '').toLowerCase().startsWith('javascript:'));
  }
});

describe('external links get noopener', () => {
  const { window, api } = loadInDom();
  const frag = api.renderMarkdown('[orion](https://example.com)');
  const host = window.document.createElement('div');
  host.appendChild(frag);
  const a = host.querySelector('a');
  assert.ok(a, 'expected an anchor');
  assert.equal(a.getAttribute('target'), '_blank');
  assert.match(a.getAttribute('rel'), /noopener/);
});

describe('fenced code gets a language label and a copy button', () => {
  const { window, api } = loadInDom();
  const frag = api.renderMarkdown('```python\nprint("hi")\n```');
  const host = window.document.createElement('div');
  host.appendChild(frag);
  assert.equal(host.querySelectorAll('.om-code').length, 1);
  assert.equal(host.querySelector('.om-code-lang').textContent, 'python');
  assert.equal(host.querySelectorAll('.om-code .om-copy').length, 1);
  assert.match(host.querySelector('pre').textContent, /print\("hi"\)/);
});

describe('emoji survive rendering unchanged', () => {
  const { window, api } = loadInDom();
  const frag = api.renderMarkdown('shipped it 🚀✨ — 🎉');
  const host = window.document.createElement('div');
  host.appendChild(frag);
  assert.match(host.textContent, /🚀✨/);
  assert.match(host.textContent, /🎉/);
});

describe('empty or blank source renders nothing', () => {
  const { api } = loadInDom();
  assert.equal(api.renderMarkdown(''), null);
  assert.equal(api.renderMarkdown('   \n  '), null);
  assert.equal(api.renderMarkdown(null), null);
});

// Runs without jsdom: the fail-closed path is the one that must hold even in a
// bare environment, and it must not depend on the libraries being loadable.
test('renderMarkdown returns null when the sanitizer is absent', () => {
  const saved = { window: global.window, document: global.document };
  global.window = undefined;
  global.document = undefined;
  delete require.cache[require.resolve('./chat-markdown.js')];
  const api = require('./chat-markdown.js');
  assert.equal(api.librariesReady(), false);
  assert.equal(api.renderMarkdown('# heading'), null, 'rendered without a sanitizer');
  global.window = saved.window;
  global.document = saved.document;
});

describe('transcriptToMarkdown reads stashed source, not rendered text', () => {
  const { window, api } = loadInDom();
  const conversation = window.document.createElement('div');
  const turn = window.document.createElement('div');
  turn.setAttribute('data-role', 'assistant');
  // Rendered DOM says one thing; the stashed markdown says another. Only the
  // stashed markdown is correct to copy.
  turn.setAttribute('data-md-source', '| a | b |\n| - | - |\n| 1 | 2 |');
  turn.innerHTML = '<p data-message-body="1">a b 1 2</p>';
  conversation.appendChild(turn);

  const md = api.transcriptToMarkdown(conversation);
  assert.match(md, /### Orion/);
  assert.match(md, /\| a \| b \|/, 'copied rendered text instead of markdown source');
});

describe('transcriptToMarkdown labels speakers and skips empty turns', () => {
  const { window, api } = loadInDom();
  const conversation = window.document.createElement('div');
  for (const [role, src] of [['user', 'hi'], ['assistant', 'hello'], ['system', '   ']]) {
    const node = window.document.createElement('div');
    node.setAttribute('data-role', role);
    node.setAttribute('data-md-source', src);
    conversation.appendChild(node);
  }
  const md = api.transcriptToMarkdown(conversation);
  assert.match(md, /### Juniper\n\nhi/);
  assert.match(md, /### Orion\n\nhello/);
  assert.ok(!md.includes('### System'), 'blank turn was included');
});
