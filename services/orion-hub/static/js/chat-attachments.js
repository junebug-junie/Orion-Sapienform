/**
 * Composer attachments: pick, paste, drop, preview, send, view.
 *
 * Notes that matter:
 *
 * * **Downscale on the client.** The server does no resizing (no Pillow in the
 *   Hub's dependency set, deliberately). Downscaling here bounds three things at
 *   once: upload time, bytes on disk, and vision tokens eaten out of the chat
 *   lane's context window.
 *
 * * **Paste is the primary path.** Screenshot to clipboard, ctrl-V, send is how
 *   people actually share an image. The file picker and drag-drop are there for
 *   completeness.
 *
 * * **Upload happens at attach time, not send time.** By the time the message is
 *   sent, the bytes are already stored and the payload carries only refs, so
 *   sending stays fast and a failed upload surfaces while there is still a
 *   composer to show the error in.
 */
(function (global) {
  'use strict';

  const DEFAULTS = {
    endpoint: '/api/chat/attachments',
    maxPerTurn: 4,
    maxEdge: 1568,          // matches common vision-tower tiling; larger buys nothing
    maxBytes: 8 * 1024 * 1024,
    jpegQuality: 0.85,
    accept: 'image/png,image/jpeg,image/webp,image/gif',
  };

  function formatBytes(n) {
    const bytes = Number(n) || 0;
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  /**
   * Downscale an image file so its longest edge is at most `maxEdge`.
   *
   * Returns the ORIGINAL file untouched when it is already small enough, or when
   * anything goes wrong -- a failed resize must not block sending. GIFs are
   * always passed through: canvas re-encoding would silently flatten an
   * animation to its first frame, which is a worse outcome than a larger upload.
   */
  async function downscale(file, options) {
    const opts = options || DEFAULTS;
    if (!file || !file.type) return file;
    if (file.type === 'image/gif') return file;

    let bitmap;
    try {
      bitmap = await global.createImageBitmap(file);
    } catch (err) {
      return file;
    }
    const { width, height } = bitmap;
    const longest = Math.max(width, height);
    if (longest <= opts.maxEdge) {
      if (bitmap.close) bitmap.close();
      return file;
    }

    const scale = opts.maxEdge / longest;
    const targetW = Math.max(1, Math.round(width * scale));
    const targetH = Math.max(1, Math.round(height * scale));
    const canvas = document.createElement('canvas');
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      if (bitmap.close) bitmap.close();
      return file;
    }
    ctx.drawImage(bitmap, 0, 0, targetW, targetH);
    if (bitmap.close) bitmap.close();

    // PNG stays PNG so screenshots of text keep their crisp edges; everything
    // else becomes JPEG, where the quality/size trade is worth taking.
    const outType = file.type === 'image/png' ? 'image/png' : 'image/jpeg';
    const blob = await new Promise((resolve) => {
      canvas.toBlob(resolve, outType, opts.jpegQuality);
    });
    if (!blob) return file;
    if (blob.size >= file.size) return file;  // resizing made it worse; keep the original
    return new File([blob], file.name || 'image', { type: outType });
  }

  async function uploadOne(file, options) {
    const opts = options || DEFAULTS;
    const prepared = await downscale(file, opts);
    if (prepared.size > opts.maxBytes) {
      throw new Error(`${prepared.name || 'image'} is ${formatBytes(prepared.size)}, limit is ${formatBytes(opts.maxBytes)}`);
    }
    const form = new FormData();
    form.append('file', prepared, prepared.name || 'image');
    const response = await fetch(opts.endpoint, { method: 'POST', body: form });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (body && body.detail) detail = String(body.detail);
      } catch (err) { /* keep the status-code message */ }
      throw new Error(detail);
    }
    return response.json();
  }

  /**
   * Does this drag carry an image? Answerable during dragenter/dragover.
   *
   * The DataTransfer is in *protected mode* during a drag: `getAsFile()` returns
   * null and `.files` is empty until the drop actually fires. Only `item.kind`
   * and `item.type` are readable. `imageFilesFrom` therefore always returns []
   * during dragover, so gating `preventDefault()` on it meant preventDefault was
   * never called -- and without it the browser rejects the drop and performs its
   * default action instead: navigating away to the dropped image, discarding
   * whatever was in the composer.
   */
  function dragCarriesImage(dataTransfer) {
    if (!dataTransfer) return false;
    const items = dataTransfer.items;
    if (items && items.length) {
      for (let i = 0; i < items.length; i += 1) {
        const item = items[i];
        if (item.kind === 'file' && String(item.type || '').startsWith('image/')) return true;
      }
      // Some browsers report kind='file' with an empty type mid-drag; accept a
      // plain file drag and let the drop handler do the real filtering.
      for (let i = 0; i < items.length; i += 1) {
        if (items[i].kind === 'file') return true;
      }
      return false;
    }
    const types = dataTransfer.types;
    if (types && Array.prototype.indexOf.call(types, 'Files') !== -1) return true;
    return false;
  }

  function imageFilesFrom(source) {
    const out = [];
    if (!source) return out;
    const items = source.items;
    if (items && items.length) {
      for (let i = 0; i < items.length; i += 1) {
        const item = items[i];
        if (item.kind === 'file') {
          const file = item.getAsFile();
          if (file && String(file.type || '').startsWith('image/')) out.push(file);
        }
      }
      if (out.length) return out;
    }
    const files = source.files;
    if (files && files.length) {
      for (let i = 0; i < files.length; i += 1) {
        if (String(files[i].type || '').startsWith('image/')) out.push(files[i]);
      }
    }
    return out;
  }

  /**
   * Bind the composer.
   *
   * `deps` supplies the DOM nodes and a `visionAvailable()` predicate. The
   * predicate reads the LLM gateway's per-route /props report (published on
   * GET /routes) -- runtime truth, not a config flag -- so the button reflects
   * whether the model that will answer can actually see.
   */
  function createController(deps) {
    const opts = Object.assign({}, DEFAULTS, deps.options || {});
    const chipRow = deps.chipRow;
    const statusEl = deps.statusEl;
    const attachButton = deps.attachButton;
    const fileInput = deps.fileInput;
    const dropZone = deps.dropZone;
    const visionAvailable = deps.visionAvailable || (() => true);
    const onOpenViewer = deps.onOpenViewer || (() => {});

    let pending = [];   // AttachmentRefV1 objects returned by the Hub
    let busy = 0;

    function setStatus(message, tone) {
      if (!statusEl) return;
      statusEl.textContent = message || '';
      statusEl.className = 'om-attach-status'
        + (tone === 'error' ? ' om-attach-status-error' : '')
        + (tone === 'busy' ? ' om-attach-status-busy' : '');
    }

    function refreshButton() {
      if (!attachButton) return;
      const capable = Boolean(visionAvailable());
      const full = pending.length >= opts.maxPerTurn;
      attachButton.disabled = !capable || full || busy > 0;
      if (!capable) {
        attachButton.title = 'The model serving this route reports no vision capability';
      } else if (full) {
        attachButton.title = `Limit is ${opts.maxPerTurn} images per message`;
      } else {
        attachButton.title = 'Attach an image (or just paste one)';
      }
      attachButton.classList.toggle('om-attach-disabled', attachButton.disabled);
    }

    function renderChips() {
      if (!chipRow) return;
      chipRow.innerHTML = '';
      chipRow.classList.toggle('hidden', pending.length === 0);
      pending.forEach((ref) => {
        const chip = document.createElement('div');
        chip.className = 'om-chip';
        chip.dataset.sha256 = ref.sha256;

        const thumb = document.createElement('img');
        thumb.className = 'om-chip-thumb';
        thumb.src = `${opts.endpoint}/${ref.sha256}`;
        thumb.alt = ref.filename || 'attachment';
        thumb.addEventListener('click', () => onOpenViewer(ref));
        chip.appendChild(thumb);

        const meta = document.createElement('div');
        meta.className = 'om-chip-meta';
        const name = document.createElement('div');
        name.className = 'om-chip-name';
        name.textContent = ref.filename || `${ref.mime}`;
        const size = document.createElement('div');
        size.className = 'om-chip-size';
        size.textContent = ref.width && ref.height
          ? `${ref.width}x${ref.height} · ${formatBytes(ref.bytes)}`
          : formatBytes(ref.bytes);
        meta.appendChild(name);
        meta.appendChild(size);
        chip.appendChild(meta);

        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'om-chip-remove';
        remove.setAttribute('aria-label', `Remove ${ref.filename || 'attachment'}`);
        remove.textContent = '×';
        remove.addEventListener('click', () => {
          pending = pending.filter((p) => p.sha256 !== ref.sha256);
          renderChips();
          refreshButton();
          setStatus('');
        });
        chip.appendChild(remove);
        chipRow.appendChild(chip);
      });
    }

    async function addFiles(files) {
      const list = Array.from(files || []);
      if (!list.length) return;
      if (!visionAvailable()) {
        setStatus('This route cannot accept images right now.', 'error');
        return;
      }
      const room = opts.maxPerTurn - pending.length;
      if (room <= 0) {
        setStatus(`Limit is ${opts.maxPerTurn} images per message.`, 'error');
        return;
      }
      const accepted = list.slice(0, room);
      const dropped = list.length - accepted.length;

      busy += 1;
      refreshButton();
      setStatus(accepted.length > 1 ? `Uploading ${accepted.length} images…` : 'Uploading…', 'busy');
      const errors = [];
      for (const file of accepted) {
        try {
          const ref = await uploadOne(file, opts);
          // Content-addressed: attaching the same image twice is one attachment.
          if (!pending.some((p) => p.sha256 === ref.sha256)) pending.push(ref);
        } catch (err) {
          errors.push(err && err.message ? err.message : String(err));
        }
      }
      busy -= 1;
      renderChips();
      refreshButton();

      if (errors.length) setStatus(errors.join(' · '), 'error');
      else if (dropped > 0) setStatus(`Attached ${accepted.length}; dropped ${dropped} over the ${opts.maxPerTurn}-image limit.`, 'error');
      else setStatus('');
    }

    function discardPending(reason) {
      if (!pending.length) return 0;
      const dropped = pending.length;
      pending = [];
      renderChips();
      refreshButton();
      setStatus(reason || 'Attachments cleared.', 'error');
      return dropped;
    }

    function takePending() {
      const refs = pending.slice();
      pending = [];
      renderChips();
      refreshButton();
      setStatus('');
      return refs;
    }

    function peekPending() {
      return pending.slice();
    }

    // --- wiring ---
    if (attachButton && fileInput) {
      attachButton.addEventListener('click', () => fileInput.click());
      fileInput.setAttribute('accept', opts.accept);
      fileInput.addEventListener('change', () => {
        addFiles(fileInput.files);
        fileInput.value = '';   // so re-picking the same file fires change again
      });
    }

    if (dropZone) {
      ['dragenter', 'dragover'].forEach((evt) => {
        dropZone.addEventListener(evt, (e) => {
          // dragCarriesImage, not imageFilesFrom: the DataTransfer is protected
          // during a drag, so file contents are unreadable until drop.
          if (!dragCarriesImage(e.dataTransfer)) return;
          e.preventDefault();
          dropZone.classList.add('om-drop-active');
        });
      });
      ['dragleave', 'drop'].forEach((evt) => {
        dropZone.addEventListener(evt, () => dropZone.classList.remove('om-drop-active'));
      });
      dropZone.addEventListener('drop', (e) => {
        const files = imageFilesFrom(e.dataTransfer);
        if (!files.length) return;
        e.preventDefault();
        addFiles(files);
      });
    }

    if (deps.pasteTarget) {
      deps.pasteTarget.addEventListener('paste', (e) => {
        const files = imageFilesFrom(e.clipboardData);
        if (!files.length) return;   // let ordinary text paste through untouched
        e.preventDefault();
        addFiles(files);
      });
    }

    refreshButton();
    renderChips();

    return { addFiles, takePending, peekPending, discardPending, refreshButton, setStatus };
  }

  global.ChatAttachments = {
    createController,
    downscale,
    uploadOne,
    imageFilesFrom,
    dragCarriesImage,
    formatBytes,
    DEFAULTS,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = global.ChatAttachments;
  }
}(typeof window !== 'undefined' ? window : globalThis));
