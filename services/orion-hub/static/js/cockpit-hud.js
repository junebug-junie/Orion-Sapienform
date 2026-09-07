(function (global) {
  // Soft HUD cockpit: visor + hop rail + scrubber + inspector.
  // No framework. State is {hopsBySeq, order, selectedSeq, followLive, playing}.
  // Task 7 wires the Cockpit button / WS ingest onto this API.

  const PLAY_INTERVAL_MS = 750;

  function emptyState() {
    return {
      correlationId: '',
      apiBaseUrl: '',
      hopsBySeq: {},
      order: [],
      selectedSeq: null,
      followLive: true,
      playing: false,
      complete: false,
      loading: false,
      error: null,
    };
  }

  let state = emptyState();
  let playTimer = null;
  let escapeBound = false;

  function escapeText(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function escapeHtml(value) {
    return escapeText(value).replace(/"/g, '&quot;');
  }

  function hopSeq(hop) {
    const n = Number(hop && hop.seq);
    return Number.isFinite(n) ? n : null;
  }

  function hopsInOrder(snapshot) {
    return (snapshot.order || [])
      .map(function (seq) { return snapshot.hopsBySeq[seq]; })
      .filter(Boolean);
  }

  function selectedHop(snapshot) {
    if (snapshot.selectedSeq == null) {
      const hops = hopsInOrder(snapshot);
      return hops.length ? hops[hops.length - 1] : null;
    }
    return snapshot.hopsBySeq[snapshot.selectedSeq] || null;
  }

  function rebuildOrder(snapshot) {
    snapshot.order = Object.keys(snapshot.hopsBySeq)
      .map(function (key) { return Number(key); })
      .filter(function (n) { return Number.isFinite(n); })
      .sort(function (a, b) { return a - b; });
  }

  function ingestInto(snapshot, hop) {
    if (!hop || typeof hop !== 'object') return snapshot;
    const seq = hopSeq(hop);
    if (seq == null) return snapshot;
    snapshot.hopsBySeq[seq] = hop;
    rebuildOrder(snapshot);
    if (snapshot.followLive && snapshot.order.length) {
      snapshot.selectedSeq = snapshot.order[snapshot.order.length - 1];
    }
    return snapshot;
  }

  function stateFromFixture(fixture) {
    const snapshot = emptyState();
    snapshot.followLive = false;
    snapshot.correlationId = (fixture && fixture.correlationId) || '';
    const hops = (fixture && Array.isArray(fixture.hops)) ? fixture.hops : [];
    hops.forEach(function (hop) { ingestInto(snapshot, hop); });
    if (fixture && fixture.selectedSeq != null) {
      snapshot.selectedSeq = Number(fixture.selectedSeq);
    }
    return snapshot;
  }

  function buildVisorHtml(snapshot) {
    const hops = hopsInOrder(snapshot);
    const current = selectedHop(snapshot);
    let line = 'No hops yet';
    if (snapshot.loading) line = 'Loading sighting…';
    else if (snapshot.error) line = 'Sighting unavailable · ' + snapshot.error;
    else if (current) line = current.visor_line || current.stage || line;
    const stream = hops.slice(-8).map(function (hop) {
      const gap = hop.status === 'gap';
      const isCurrent = current && hop.seq === current.seq;
      const cls = 'cockpit-visor-line' + (gap ? ' cockpit-visor-line-gap' : '') + (isCurrent ? ' is-current' : '');
      return '<div class="' + cls + '">' + escapeText(hop.visor_line || hop.stage || '') + '</div>';
    }).join('');
    return [
      '<div class="cockpit-visor" id="cockpitVisor">',
      '<div class="cockpit-visor-kicker">Cockpit</div>',
      '<div class="cockpit-visor-current">' + escapeText(line) + '</div>',
      '<div class="cockpit-visor-stream">' + stream + '</div>',
      '</div>',
    ].join('');
  }

  function buildRailHtml(snapshot) {
    const hops = hopsInOrder(snapshot);
    const current = selectedHop(snapshot);
    const beads = hops.map(function (hop) {
      const gap = hop.status === 'gap';
      const selected = current && hop.seq === current.seq;
      const cls = [
        'cockpit-hop',
        gap ? 'cockpit-hop-gap' : '',
        selected ? 'is-selected' : '',
      ].filter(Boolean).join(' ');
      return (
        '<button type="button" class="' + cls + '"'
        + ' data-seq="' + escapeHtml(String(hop.seq)) + '"'
        + ' data-status="' + escapeHtml(hop.status || '') + '"'
        + ' data-stage="' + escapeHtml(hop.stage || '') + '"'
        + ' title="' + escapeHtml(hop.stage || '') + '"'
        + '>' + escapeText(hop.stage || String(hop.seq)) + '</button>'
      );
    }).join('');
    return '<div class="cockpit-rail" id="cockpitRail" role="list">' + beads + '</div>';
  }

  function buildInspectorHtml(snapshot) {
    const hop = selectedHop(snapshot);
    if (!hop) {
      return '<div class="cockpit-inspector" id="cockpitInspector"><div class="cockpit-inspector-empty">Select a hop.</div></div>';
    }
    const raw = hop.raw && typeof hop.raw === 'object' ? hop.raw : {};
    const summary = hop.summary && typeof hop.summary === 'object' ? hop.summary : {};
    const keys = Object.keys(summary);
    const summaryRows = keys.length
      ? keys.map(function (key) {
        return '<div class="cockpit-inspector-row"><span>' + escapeText(key) + '</span><span>' + escapeText(String(summary[key])) + '</span></div>';
      }).join('')
      : '<div class="cockpit-inspector-muted">No summary fields.</div>';
    const promptText = (raw && typeof raw.prompt === 'string') ? raw.prompt : '';
    const promptSection = promptText
      ? (
          '<div class="cockpit-inspector-section" data-cockpit-section="prompt">' +
            '<div class="cockpit-inspector-section-title">Prompt/Prefix</div>' +
            '<pre class="cockpit-inspector-prompt">' + escapeText(promptText) + '</pre>' +
          '</div>'
        )
      : '';
    return [
      '<div class="cockpit-inspector" id="cockpitInspector">',
      '<div class="cockpit-inspector-meta">',
      '<div>seq ' + escapeText(String(hop.seq)) + '</div>',
      '<div>' + escapeText(hop.stage || '') + '</div>',
      '<div data-status="' + escapeHtml(hop.status || '') + '">' + escapeText(hop.status || '') + '</div>',
      '</div>',
      '<div class="cockpit-inspector-summary">' + summaryRows + '</div>',
      promptSection,
      '<pre class="cockpit-inspector-raw">' + escapeText(JSON.stringify(raw, null, 2)) + '</pre>',
      '</div>',
    ].join('');
  }

  function buildScrubberHtml(snapshot) {
    const hops = hopsInOrder(snapshot);
    const max = Math.max(0, hops.length - 1);
    let idx = hops.findIndex(function (hop) { return hop.seq === snapshot.selectedSeq; });
    if (idx < 0) idx = hops.length ? hops.length - 1 : 0;
    const label = hops.length ? (idx + 1) + ' / ' + hops.length : '0 / 0';
    return [
      '<div class="cockpit-scrubber" id="cockpitScrubber">',
      '<button type="button" data-action="step-back"' + (idx <= 0 ? ' disabled' : '') + '>Back</button>',
      '<button type="button" data-action="play">' + (snapshot.playing ? 'Pause' : 'Play') + '</button>',
      '<button type="button" data-action="step-forward"' + (idx >= max || !hops.length ? ' disabled' : '') + '>Next</button>',
      '<input type="range" min="0" max="' + max + '" value="' + idx + '" data-action="scrub"' + (hops.length ? '' : ' disabled') + ' />',
      '<button type="button" data-action="jump-live"' + (snapshot.followLive ? ' disabled' : '') + '>Live</button>',
      '<span class="cockpit-scrubber-meta">' + escapeText(label) + (snapshot.complete ? ' · complete' : '') + '</span>',
      '</div>',
    ].join('');
  }

  function renderFrame(snapshot) {
    return [
      '<div class="cockpit-hud" data-correlation-id="' + escapeHtml(snapshot.correlationId || '') + '" role="dialog" aria-modal="true" aria-label="Cockpit Soft HUD">',
      '<div class="cockpit-hud-backdrop" data-action="close"></div>',
      '<div class="cockpit-hud-vignette" aria-hidden="true"></div>',
      '<div class="cockpit-hud-glass">',
      '<header class="cockpit-hud-header">',
      '<div class="cockpit-hud-title">Turn sighting <span class="cockpit-hud-corr">' + escapeText(snapshot.correlationId || '') + '</span></div>',
      '<button type="button" class="cockpit-hud-close" data-action="close">Close</button>',
      '</header>',
      '<div class="cockpit-hud-body">',
      buildVisorHtml(snapshot),
      buildRailHtml(snapshot),
      buildInspectorHtml(snapshot),
      '</div>',
      buildScrubberHtml(snapshot),
      '</div>',
      '</div>',
    ].join('');
  }

  function buildShell(opts) {
    const snapshot = emptyState();
    snapshot.correlationId = (opts && opts.correlationId) || '';
    return renderFrame(snapshot);
  }

  function render(snapshot) {
    return renderFrame(snapshot || state);
  }

  function renderFixture(fixture) {
    return render(stateFromFixture(fixture || {}));
  }

  function getRoot() {
    if (typeof document === 'undefined') return null;
    return document.getElementById('cockpitHudRoot');
  }

  function isOpen() {
    const root = getRoot();
    return !!(root && !root.classList.contains('hidden'));
  }

  function bindRoot(root) {
    root.querySelectorAll('[data-action="close"]').forEach(function (el) {
      el.addEventListener('click', close);
    });
    root.querySelectorAll('.cockpit-hop').forEach(function (el) {
      el.addEventListener('click', function () {
        const seq = Number(el.getAttribute('data-seq'));
        if (!Number.isFinite(seq)) return;
        state.followLive = false;
        state.selectedSeq = seq;
        paint();
      });
    });
    const play = root.querySelector('[data-action="play"]');
    if (play) play.addEventListener('click', togglePlay);
    const back = root.querySelector('[data-action="step-back"]');
    if (back) back.addEventListener('click', function () { step(-1); });
    const fwd = root.querySelector('[data-action="step-forward"]');
    if (fwd) fwd.addEventListener('click', function () { step(1); });
    const live = root.querySelector('[data-action="jump-live"]');
    if (live) live.addEventListener('click', jumpLive);
    const scrub = root.querySelector('[data-action="scrub"]');
    if (scrub) {
      scrub.addEventListener('input', function () {
        const hops = hopsInOrder(state);
        const hop = hops[Number(scrub.value)];
        if (!hop) return;
        state.followLive = hop.seq === hops[hops.length - 1].seq;
        state.selectedSeq = hop.seq;
        paint();
      });
    }
  }

  function paint() {
    const root = getRoot();
    if (!root) return;
    root.innerHTML = render(state);
    bindRoot(root);
  }

  function step(delta) {
    const hops = hopsInOrder(state);
    if (!hops.length) return;
    let idx = hops.findIndex(function (hop) { return hop.seq === state.selectedSeq; });
    if (idx < 0) idx = hops.length - 1;
    const next = hops[Math.max(0, Math.min(hops.length - 1, idx + delta))];
    state.selectedSeq = next.seq;
    state.followLive = next.seq === hops[hops.length - 1].seq;
    if (state.playing && state.followLive) stopPlay();
    paint();
  }

  function togglePlay() {
    if (state.playing) stopPlay();
    else startPlay();
  }

  function startPlay() {
    state.playing = true;
    paint();
    if (playTimer) clearInterval(playTimer);
    playTimer = setInterval(function () {
      const hops = hopsInOrder(state);
      if (!hops.length) return;
      let idx = hops.findIndex(function (hop) { return hop.seq === state.selectedSeq; });
      if (idx < 0) idx = -1;
      if (idx >= hops.length - 1) {
        stopPlay();
        return;
      }
      state.selectedSeq = hops[idx + 1].seq;
      state.followLive = state.selectedSeq === hops[hops.length - 1].seq;
      paint();
    }, PLAY_INTERVAL_MS);
  }

  function stopPlay() {
    state.playing = false;
    if (playTimer) {
      clearInterval(playTimer);
      playTimer = null;
    }
    if (isOpen()) paint();
  }

  function jumpLive() {
    const hops = hopsInOrder(state);
    state.followLive = true;
    state.selectedSeq = hops.length ? hops[hops.length - 1].seq : null;
    paint();
  }

  function joinApi(base, path) {
    const root = String(base || '').replace(/\/+$/, '');
    return root + path;
  }

  function ensureEscape() {
    if (escapeBound || typeof document === 'undefined') return;
    escapeBound = true;
    document.addEventListener('keydown', function (ev) {
      if (ev.key !== 'Escape') return;
      if (!isOpen()) return;
      close();
    });
  }

  async function open(opts) {
    const correlationId = String((opts && opts.correlationId) || '').trim();
    const apiBaseUrl = String((opts && opts.apiBaseUrl) || '').trim();
    state = emptyState();
    state.correlationId = correlationId;
    state.apiBaseUrl = apiBaseUrl;
    state.loading = true;
    ensureEscape();
    const root = getRoot();
    if (root && typeof document !== 'undefined') {
      if (root.parentElement !== document.body) {
        document.body.appendChild(root);
      }
      root.classList.remove('hidden');
      root.setAttribute('aria-hidden', 'false');
      root.style.position = 'fixed';
      root.style.inset = '0';
      root.style.zIndex = '2147483646';
    }
    paint();
    if (typeof window.syncDebugModalScrollLock === 'function') {
      window.syncDebugModalScrollLock();
    }
    if (!correlationId) {
      state.loading = false;
      state.error = 'missing_correlation_id';
      paint();
      return;
    }
    try {
      const url = joinApi(apiBaseUrl, '/api/chat/turn/' + encodeURIComponent(correlationId) + '/cockpit');
      const response = await fetch(url);
      if (state.correlationId !== correlationId) return;
      if (!response.ok) throw new Error('http_' + response.status);
      const body = await response.json();
      if (state.correlationId !== correlationId) return;
      const hops = (body && Array.isArray(body.hops)) ? body.hops : [];
      hops.forEach(function (hop) { ingestInto(state, hop); });
      if (body && body.complete) state.complete = true;
      state.loading = false;
      state.error = null;
    } catch (err) {
      if (state.correlationId !== correlationId) return;
      state.loading = false;
      state.error = (err && err.message) ? err.message : 'fetch_failed';
    }
    paint();
  }

  function close() {
    stopPlay();
    const root = getRoot();
    if (root) {
      root.classList.add('hidden');
      root.setAttribute('aria-hidden', 'true');
      root.innerHTML = '';
    }
    if (typeof window.syncDebugModalScrollLock === 'function') {
      window.syncDebugModalScrollLock();
    }
  }

  function ingestHop(hop) {
    ingestInto(state, hop);
    if (isOpen()) paint();
  }

  function markComplete() {
    state.complete = true;
    if (isOpen()) paint();
  }

  const api = {
    open: open,
    close: close,
    ingestHop: ingestHop,
    markComplete: markComplete,
    buildShell: buildShell,
    render: render,
    renderFixture: renderFixture,
  };
  global.OrionCockpitHud = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
