/* Runtime activity: the header marquee + "what's running right now" modal.
 *
 * Data: GET /api/runtime-activity (snapshot) and its SSE sibling
 * /api/runtime-activity/stream (one frame per change, keepalives between).
 * Shape is produced by orion/hub/runtime_activity.py's snapshot(). This file
 * only renders; every number shown is one the server already computed, except
 * the live "elapsed" counters, which tick locally from the server's ISO stamps
 * so the display keeps moving between frames.
 *
 * Pure functions (marqueeItems, formatDuration, laneSummary, ...) are exported
 * for node tests; DOM wiring lives in init() and is skipped under node.
 */
(function (global) {
  'use strict';

  const LINE_LABELS = { investigate: 'curiosity', self_inquiry: 'self-inquiry' };
  const NODE_LABELS = {
    harness_turn: 'thinking (harness turn)',
    read_turn_result: 'reading what it found',
    publish_attention_row: 'noting what it attended to',
    journal: 'writing the journal entry',
    finish: 'finishing',
  };

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function formatDuration(sec) {
    const s = Math.max(0, Math.floor(Number(sec) || 0));
    if (s < 60) return s + 's';
    const m = Math.floor(s / 60);
    const rem = s % 60;
    if (m < 60) return m + 'm ' + String(rem).padStart(2, '0') + 's';
    const h = Math.floor(m / 60);
    return h + 'h ' + String(m % 60).padStart(2, '0') + 'm';
  }

  /* Seconds elapsed since an ISO stamp, or null when the stamp is absent.
   * Absent stays absent -- a missing first_step_at means "not started", not 0. */
  function elapsedSince(iso, nowMs) {
    if (!iso) return null;
    const t = Date.parse(iso);
    if (!Number.isFinite(t)) return null;
    return Math.max(0, (nowMs - t) / 1000);
  }

  function runLabel(run) {
    return LINE_LABELS[run.line] || (run.line ? String(run.line) : 'curiosity');
  }

  function nodeLabel(node) {
    return NODE_LABELS[node] || (node ? String(node) : 'dispatched, waiting for the runner');
  }

  /* Live duration for a run: server's duration_sec at frame time, advanced
   * locally for active runs. Finished runs keep the server's number. */
  function liveRunDuration(run, snapshotAtMs, nowMs) {
    const base = Number(run.duration_sec) || 0;
    if (!run.active) return base;
    return base + Math.max(0, (nowMs - snapshotAtMs) / 1000);
  }

  function liveTurnElapsed(turn, snapshotAtMs, nowMs) {
    const base = Number(turn.elapsed_sec) || 0;
    if (turn.phase === 'finished') return base;
    return base + Math.max(0, (nowMs - snapshotAtMs) / 1000);
  }

  /* One marquee chip per thing worth watching, most urgent first:
   * active curiosity runs, then busy governor lanes, then gateway lanes with
   * anyone waiting. Empty list => "idle". */
  function marqueeItems(snapshot, nowMs, snapshotAtMs) {
    const items = [];
    if (!snapshot) return items;
    const at = snapshotAtMs == null ? nowMs : snapshotAtMs;
    (snapshot.curiosity_runs || []).forEach(function (run) {
      if (!run.active) return;
      const turn = run.turn;
      const lane = turn && turn.lane && turn.lane !== 'unknown' ? turn.lane + ' lane' : null;
      items.push({
        kind: 'run',
        key: 'run:' + run.run_id,
        tone: run.status === 'resumed' ? 'warn' : 'live',
        text: runLabel(run) + ' · ' + nodeLabel(run.node) + ' · ' + formatDuration(liveRunDuration(run, at, nowMs)) + (lane ? ' · ' + lane : ''),
      });
    });
    const lanes = snapshot.lanes || {};
    Object.keys(lanes).forEach(function (lane) {
      const l = lanes[lane] || {};
      const running = (l.running || []).filter(function (t) {
        // A turn already shown as a curiosity run chip is not repeated here.
        return !(snapshot.curiosity_runs || []).some(function (r) { return r.active && r.correlation_id === t.correlation_id; });
      });
      const queued = l.queued || [];
      if (!running.length && !queued.length) return;
      const parts = [];
      running.forEach(function (t) {
        parts.push((t.source || 'chat') + ' running ' + formatDuration(liveTurnElapsed(t, at, nowMs)));
      });
      if (queued.length) parts.push(queued.length + ' queued');
      items.push({ kind: 'lane', key: 'lane:' + lane, tone: queued.length ? 'warn' : 'live', text: lane + ' lane · ' + parts.join(' · ') });
    });
    const gw = snapshot.gateway && snapshot.gateway.snapshot;
    (gw && gw.lanes ? gw.lanes : []).forEach(function (u) {
      const waiting = Number(u.waiting) || 0;
      const inflight = Number(u.inflight) || 0;
      if (!waiting && !inflight) return;
      const names = (u.routes || []).map(function (r) { return r.id; }).filter(Boolean);
      const label = names.length ? names.join('/') : String(u.upstream || 'upstream');
      items.push({
        kind: 'gateway',
        key: 'gw:' + u.upstream,
        tone: waiting ? 'warn' : 'quiet',
        text: label + ' · ' + inflight + '/' + (u.max_inflight == null ? '?' : u.max_inflight) + ' in flight' + (waiting ? ' · ' + waiting + ' waiting' : ''),
      });
    });
    return items;
  }

  function laneSummary(snapshot) {
    const lanes = (snapshot && snapshot.lanes) || {};
    return Object.keys(lanes).map(function (name) {
      const l = lanes[name];
      return { lane: name, running: (l.running || []).length, queued: (l.queued || []).length, recent: (l.recent || []).length };
    });
  }

  /* ---------------------------------------------------------------- modal */

  function turnRows(turns, snapshotAtMs, nowMs, opts) {
    const showLane = opts && opts.showLane;
    if (!turns || !turns.length) return '<div class="ra-empty">none</div>';
    return turns.map(function (t) {
      const elapsed = formatDuration(liveTurnElapsed(t, snapshotAtMs, nowMs));
      const steps = t.recent_steps && t.recent_steps.length ? t.recent_steps.slice(-3).join(' → ') : '';
      // Plain text here; escaped exactly once where it is printed below.
      const outcome = t.phase === 'finished'
        ? (t.ok ? (t.compliance_verdict || 'ok') + (t.fcc_elapsed_sec != null ? ' · fcc ' + formatDuration(t.fcc_elapsed_sec) : '') : 'failed: ' + (t.error || 'unknown'))
        : (t.phase === 'queued' ? 'waiting for the governor · ' + formatDuration(t.queued_sec) : 'step ' + (t.step_count || 0));
      return '<div class="ra-row ra-row--' + escapeHtml(t.phase) + '">' +
        '<div class="ra-row-main"><span class="ra-tag">' + escapeHtml(t.source || 'chat') + '</span>' +
        (showLane ? '<span class="ra-tag ra-tag--lane">' + escapeHtml(t.lane) + '</span>' : '') +
        '<span class="ra-tag">' + escapeHtml(t.mode || 'orion') + '</span>' +
        (t.model_label ? '<span class="ra-dim">' + escapeHtml(t.model_label) + '</span>' : '') +
        (t.served_model ? '<span class="ra-dim">served by ' + escapeHtml(t.served_model) + '</span>' : '') +
        '</div>' +
        '<div class="ra-row-sub">' + escapeHtml(outcome) + ' · ' + elapsed + (steps ? ' · ' + escapeHtml(steps) : '') + '</div>' +
        '<div class="ra-row-id" title="correlation id">corr ' + escapeHtml(t.correlation_id) + '</div>' +
        '</div>';
    }).join('');
  }

  function runCard(run, snapshotAtMs, nowMs) {
    const dur = formatDuration(liveRunDuration(run, snapshotAtMs, nowMs));
    const turn = run.turn;
    const transitions = (run.transitions || []).slice(-8).map(function (t) {
      return '<li><span class="ra-dim">' + escapeHtml((t.at || '').replace('T', ' ').replace(/\.\d+Z$/, 'Z')) + '</span> ' + escapeHtml(t.node || '?') + ' <span class="ra-status ra-status--' + escapeHtml(t.status) + '">' + escapeHtml(t.status) + '</span>' + (t.backfilled ? ' <span class="ra-dim">(from db after restart)</span>' : '') + '</li>';
    }).join('');
    const finish = run.finish || {};
    return '<div class="ra-card ra-card--' + (run.active ? 'active' : 'done') + '">' +
      '<div class="ra-card-head">' +
      '<span class="ra-status ra-status--' + escapeHtml(run.status) + '">' + escapeHtml(run.status) + '</span> ' +
      '<strong>' + escapeHtml(runLabel(run)) + '</strong> · ' + escapeHtml(nodeLabel(run.node)) + ' · ' + dur +
      (run.resumed_from_node ? ' · <span class="ra-warn">resumed at ' + escapeHtml(run.resumed_from_node) + '</span>' : '') +
      '</div>' +
      '<div class="ra-card-ids">run ' + escapeHtml(run.run_id) + ' · corr ' + escapeHtml(run.correlation_id) + ' · started ' + escapeHtml(run.started_at || '?') + '</div>' +
      (turn ? '<div class="ra-card-turn"><div class="ra-label">its harness turn</div>' + turnRows([turn], snapshotAtMs, nowMs, { showLane: true }) + '</div>'
            : '<div class="ra-card-turn ra-dim">no harness turn seen for this run yet' + (run.active ? ' (still before the governor, or a Hub restart lost the handoff)' : '') + '</div>') +
      (run.error ? '<div class="ra-error">' + escapeHtml(run.error) + '</div>' : '') +
      (finish.finding_text ? '<div class="ra-finding">' + escapeHtml(finish.finding_text) + (finish.reach_out ? ' <span class="ra-tag">wants to reach out</span>' : '') + '</div>' : '') +
      (transitions ? '<ul class="ra-transitions">' + transitions + '</ul>' : '') +
      '</div>';
  }

  function gatewayRows(gateway) {
    if (!gateway) return '<div class="ra-empty">no gateway data yet</div>';
    const snap = gateway.snapshot;
    const err = gateway.error ? '<div class="ra-error">gateway poll failed: ' + escapeHtml(gateway.error) + (snap ? ' (showing last good read)' : '') + '</div>' : '';
    if (!snap || !snap.lanes) return err || '<div class="ra-empty">no gateway data yet</div>';
    const rows = snap.lanes.map(function (u) {
      const names = (u.routes || []).map(function (r) {
        return '<span class="ra-tag' + (r.priority === 'background' ? ' ra-tag--bg' : '') + (r.status && r.status !== 'up' ? ' ra-tag--down' : '') + '">' + escapeHtml(r.id) + '</span>';
      }).join('');
      const served = (u.routes || []).map(function (r) { return r.served_by; }).filter(Boolean)[0];
      const busy = (Number(u.inflight) || 0) + (Number(u.waiting) || 0) > 0;
      return '<div class="ra-row' + (busy ? ' ra-row--running' : '') + '">' +
        '<div class="ra-row-main">' + (names || '<span class="ra-dim">unnamed upstream</span>') + (served ? '<span class="ra-dim">' + escapeHtml(served) + '</span>' : '') + '</div>' +
        '<div class="ra-row-sub">' + (Number(u.inflight) || 0) + '/' + (u.max_inflight == null ? '?' : escapeHtml(u.max_inflight)) + ' in flight · ' + (Number(u.waiting) || 0) + ' waiting · shed ' + (Number(u.shed) || 0) + ' · longest wait ' + formatDuration(u.longest_wait_s) + '</div>' +
        '<div class="ra-row-id">' + escapeHtml(u.upstream) + '</div>' +
        '</div>';
    }).join('');
    const ledger = snap.ledger || {};
    const ledgerLine = ledger.checked != null
      ? '<div class="ra-dim ra-ledger">last 5 min: ' + escapeHtml(ledger.checked) + ' checked · ' + escapeHtml(ledger.queued || 0) + ' queued · ' + escapeHtml(ledger.deferrals || 0) + ' deferred · longest ' + formatDuration(ledger.longest_wait_s) + (gateway.polled_at ? ' · polled ' + escapeHtml(gateway.polled_at) : '') + '</div>'
      : '';
    return err + rows + ledgerLine;
  }

  function renderModalBody(snapshot, nowMs, snapshotAtMs) {
    if (!snapshot) return '<div class="ra-empty">no data</div>';
    const at = snapshotAtMs == null ? nowMs : snapshotAtMs;
    const runs = snapshot.curiosity_runs || [];
    const active = runs.filter(function (r) { return r.active; });
    const recent = runs.filter(function (r) { return !r.active; });
    const lanes = snapshot.lanes || {};
    const laneHtml = Object.keys(lanes).map(function (lane) {
      const l = lanes[lane];
      return '<div class="ra-lane"><div class="ra-lane-head">' + escapeHtml(lane) + ' lane <span class="ra-dim">(governor runs one turn at a time here)</span></div>' +
        '<div class="ra-label">running</div>' + turnRows(l.running, at, nowMs) +
        '<div class="ra-label">queued</div>' + turnRows(l.queued, at, nowMs) +
        (l.recent && l.recent.length ? '<div class="ra-label">recent</div>' + turnRows(l.recent, at, nowMs) : '') +
        '</div>';
    }).join('');
    return '<section class="ra-section"><h3>Curiosity runs</h3>' +
      (active.length ? active.map(function (r) { return runCard(r, at, nowMs); }).join('') : '<div class="ra-empty">nothing running</div>') +
      (recent.length ? '<details class="ra-details"><summary>' + recent.length + ' recent</summary>' + recent.map(function (r) { return runCard(r, at, nowMs); }).join('') + '</details>' : '') +
      '</section>' +
      '<section class="ra-section"><h3>Harness lanes</h3><div class="ra-lanes">' + laneHtml + '</div></section>' +
      '<section class="ra-section"><h3>LLM gateway lanes</h3><div class="ra-dim">Per worker: how many requests are inside it and how many are waiting for a slot. Route names come from the gateway\'s catalog; metacog/quick_background are the background-class lanes.</div>' + gatewayRows(snapshot.gateway) + '</section>' +
      '<div class="ra-foot ra-dim">snapshot v' + escapeHtml(snapshot.version) + ' at ' + escapeHtml(snapshot.generated_at || '?') + '</div>';
  }

  /* ------------------------------------------------------------------ DOM */

  function init(doc) {
    const strip = doc.getElementById('runtimeMarquee');
    const modal = doc.getElementById('runtimeActivityModal');
    if (!strip || !modal) return null;
    const track = strip.querySelector('.ra-marquee-track');
    const body = modal.querySelector('.ra-modal-body');
    const state = { snapshot: null, snapshotAtMs: 0, source: null, connected: false, retryMs: 2000, open: false };

    function render() {
      const now = Date.now();
      const items = marqueeItems(state.snapshot, now, state.snapshotAtMs);
      strip.classList.toggle('ra-marquee--busy', items.length > 0);
      strip.classList.toggle('ra-marquee--offline', !state.connected);
      if (!items.length) {
        track.innerHTML = '<span class="ra-chip ra-chip--idle">' + (state.connected ? 'idle' : 'activity feed offline') + '</span>';
      } else {
        track.innerHTML = items.map(function (it) {
          return '<span class="ra-chip ra-chip--' + escapeHtml(it.tone) + '" data-key="' + escapeHtml(it.key) + '">' + escapeHtml(it.text) + '</span>';
        }).join('');
      }
      if (state.open) body.innerHTML = renderModalBody(state.snapshot, now, state.snapshotAtMs);
    }

    function apply(snapshot) {
      state.snapshot = snapshot;
      state.snapshotAtMs = Date.now();
      render();
    }

    function connect() {
      if (typeof EventSource === 'undefined') return;
      try {
        const es = new EventSource('/api/runtime-activity/stream');
        state.source = es;
        es.addEventListener('snapshot', function (e) {
          state.connected = true;
          state.retryMs = 2000;
          try { apply(JSON.parse(e.data)); } catch (_) { /* ignore a bad frame */ }
        });
        es.onerror = function () {
          state.connected = false;
          render();
          es.close();
          setTimeout(connect, state.retryMs);
          state.retryMs = Math.min(30000, state.retryMs * 2);
        };
      } catch (_) {
        state.connected = false;
        render();
      }
    }

    function open() {
      state.open = true;
      modal.hidden = false;
      render();
    }
    function close() {
      state.open = false;
      modal.hidden = true;
    }

    strip.addEventListener('click', open);
    strip.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
    modal.querySelector('.ra-modal-close').addEventListener('click', close);
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });
    doc.addEventListener('keydown', function (e) { if (e.key === 'Escape' && state.open) close(); });

    fetch('/api/runtime-activity').then(function (r) { return r.ok ? r.json() : null; }).then(function (s) {
      if (s) { state.connected = true; apply(s); }
    }).catch(function () { /* stream will report */ });
    connect();
    setInterval(render, 1000);
    render();
    return state;
  }

  const api = { marqueeItems, formatDuration, elapsedSince, laneSummary, renderModalBody, runCard, turnRows, gatewayRows, escapeHtml, init };
  global.OrionRuntimeActivity = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
  if (typeof document !== 'undefined' && typeof module === 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', function () { init(document); });
    } else {
      init(document);
    }
  }
})(typeof window !== 'undefined' ? window : globalThis);
