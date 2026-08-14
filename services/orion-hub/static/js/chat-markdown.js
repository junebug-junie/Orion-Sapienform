/**
 * Markdown rendering for Orion's chat replies.
 *
 * Three rules, all of them load-bearing:
 *
 * 1. **Assistant turns only.** What Juniper typed is rendered with
 *    `textContent`, exactly as before. Rendering the human's own input as
 *    markdown would mangle pasted code and gains nothing.
 *
 * 2. **Never render without a sanitizer.** This module is the first place the
 *    Hub moves from `textContent` to `innerHTML`, and the content is model
 *    output -- partially influenceable through recall hits and any web content
 *    that reaches the context. If DOMPurify is missing for any reason, we fall
 *    back to plain text rather than rendering unsanitized HTML. Failing closed
 *    costs formatting; failing open costs the session.
 *
 * 3. **Copy yields source, not rendered text.** Copying a rendered table via
 *    `innerText` produces space-mangled garbage. Every copy path here returns
 *    the original markdown.
 */
(function (global) {
  'use strict';

  const COPY_RESET_MS = 1400;

  function librariesReady() {
    return Boolean(
      global.marked
      && typeof global.marked.parse === 'function'
      && global.DOMPurify
      && typeof global.DOMPurify.sanitize === 'function'
    );
  }

  let configured = false;
  function configureOnce() {
    if (configured || !librariesReady()) return;
    global.marked.setOptions({
      gfm: true,
      breaks: true,   // chat convention: a single newline is a line break
      pedantic: false,
    });
    configured = true;
  }

  // Allowlist, not a denylist. Anything not named here is stripped.
  const ALLOWED_TAGS = [
    'p', 'br', 'hr', 'span', 'div',
    'strong', 'em', 'b', 'i', 'u', 's', 'del', 'ins', 'mark', 'sup', 'sub',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li',
    'blockquote', 'pre', 'code',
    'table', 'thead', 'tbody', 'tr', 'th', 'td',
    'a', 'img',
  ];
  const ALLOWED_ATTR = ['href', 'title', 'alt', 'src', 'class', 'colspan', 'rowspan', 'start'];

  function sanitize(html) {
    return global.DOMPurify.sanitize(html, {
      ALLOWED_TAGS,
      ALLOWED_ATTR,
      // Blocks javascript:, data: (except images, handled below), vbscript:.
      ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto):|[^a-z]|[a-z+.-]+(?:[^a-z+.\-:]|$))/i,
      FORBID_TAGS: ['style', 'script', 'iframe', 'object', 'embed', 'form', 'input', 'button'],
      FORBID_ATTR: ['style', 'onerror', 'onload', 'onclick'],
    });
  }

  /**
   * Render markdown to a sanitized DocumentFragment.
   * Returns null when rendering is unavailable or the source is empty, so the
   * caller can fall back to its existing textContent path.
   */
  function renderMarkdown(source) {
    const text = String(source == null ? '' : source);
    if (!text.trim()) return null;
    configureOnce();
    if (!librariesReady()) return null;

    let html;
    try {
      html = global.marked.parse(text);
    } catch (err) {
      // A malformed reply must degrade to plain text, never to a broken panel.
      if (global.console) global.console.warn('[chat-markdown] parse failed', err);
      return null;
    }

    const clean = sanitize(html);
    const host = document.createElement('div');
    host.innerHTML = clean;
    decorateLinks(host);
    decorateCodeBlocks(host);

    const fragment = document.createDocumentFragment();
    while (host.firstChild) fragment.appendChild(host.firstChild);
    return fragment;
  }

  function decorateLinks(root) {
    root.querySelectorAll('a[href]').forEach((anchor) => {
      // A reply can contain a link Orion read somewhere. Opening it must not
      // hand the opener a handle back to the Hub window.
      anchor.setAttribute('target', '_blank');
      anchor.setAttribute('rel', 'noopener noreferrer nofollow');
      anchor.classList.add('om-link');
    });
  }

  function languageOf(codeEl) {
    const cls = String(codeEl.getAttribute('class') || '');
    const match = cls.match(/language-([A-Za-z0-9_+-]+)/);
    return match ? match[1] : '';
  }

  function decorateCodeBlocks(root) {
    root.querySelectorAll('pre > code').forEach((codeEl) => {
      const pre = codeEl.parentElement;
      const wrapper = document.createElement('div');
      wrapper.className = 'om-code';

      const header = document.createElement('div');
      header.className = 'om-code-head';

      const label = document.createElement('span');
      label.className = 'om-code-lang';
      label.textContent = languageOf(codeEl) || 'text';
      header.appendChild(label);

      // Read the source once, now: the button must copy the code even if the
      // DOM is later re-rendered underneath it.
      const source = codeEl.textContent || '';
      header.appendChild(buildCopyButton(source, 'Copy'));

      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(header);
      wrapper.appendChild(pre);
      pre.classList.add('om-code-body');
    });
  }

  async function writeClipboard(text) {
    if (global.navigator && global.navigator.clipboard && global.navigator.clipboard.writeText) {
      await global.navigator.clipboard.writeText(text);
      return true;
    }
    // Fallback for non-secure contexts, where navigator.clipboard is undefined.
    const scratch = document.createElement('textarea');
    scratch.value = text;
    scratch.setAttribute('readonly', '');
    scratch.style.position = 'fixed';
    scratch.style.opacity = '0';
    document.body.appendChild(scratch);
    scratch.select();
    let ok = false;
    try {
      ok = document.execCommand('copy');
    } catch (err) {
      ok = false;
    }
    document.body.removeChild(scratch);
    return ok;
  }

  function buildCopyButton(text, label) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'om-copy';
    button.textContent = label;
    button.addEventListener('click', async (event) => {
      event.preventDefault();
      event.stopPropagation();
      const ok = await writeClipboard(typeof text === 'function' ? text() : text);
      button.textContent = ok ? 'Copied' : 'Failed';
      button.classList.toggle('om-copy-ok', ok);
      global.setTimeout(() => {
        button.textContent = label;
        button.classList.remove('om-copy-ok');
      }, COPY_RESET_MS);
    });
    return button;
  }

  /**
   * Serialize the whole conversation to markdown for the clipboard.
   *
   * Reads `data-md-source` (the original markdown, stashed at render time)
   * rather than the rendered DOM, so a copied transcript round-trips.
   */
  function transcriptToMarkdown(conversationEl) {
    if (!conversationEl) return '';
    const blocks = [];
    conversationEl.querySelectorAll('[data-role]').forEach((node) => {
      const role = node.getAttribute('data-role') || 'system';
      const speaker = role === 'assistant' ? 'Orion' : (role === 'user' ? 'Juniper' : 'System');
      const body = node.getAttribute('data-md-source');
      const text = body != null ? body : textOfMessageBody(node);
      if (!String(text || '').trim()) return;
      blocks.push(`### ${speaker}\n\n${String(text).trim()}`);
    });
    return blocks.join('\n\n---\n\n');
  }

  function textOfMessageBody(node) {
    const body = node.querySelector('[data-message-body]');
    return body ? (body.innerText || body.textContent || '') : '';
  }

  global.ChatMarkdown = {
    renderMarkdown,
    transcriptToMarkdown,
    buildCopyButton,
    writeClipboard,
    librariesReady,
    // Exported for tests.
    _sanitize: (html) => (librariesReady() ? sanitize(html) : html),
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = global.ChatMarkdown;
  }
}(typeof window !== 'undefined' ? window : globalThis));
