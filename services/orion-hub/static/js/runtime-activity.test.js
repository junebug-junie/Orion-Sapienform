const test = require('node:test');
const assert = require('node:assert');

const { marqueeItems, formatDuration, renderModalBody, laneSummary, escapeHtml } = require('./runtime-activity.js');

const NOW = Date.parse('2026-09-09T05:10:00Z');

function snapshot(overrides) {
  return Object.assign({
    version: 7,
    generated_at: '2026-09-09T05:10:00Z',
    busy: true,
    curiosity_runs: [],
    active_run_count: 0,
    lanes: { chat: { running: [], queued: [], recent: [] }, agent: { running: [], queued: [], recent: [] } },
    gateway: { snapshot: null, error: null, polled_at: null },
  }, overrides || {});
}

test('formatDuration reads like a clock, not a float', () => {
  assert.strictEqual(formatDuration(0), '0s');
  assert.strictEqual(formatDuration(59.9), '59s');
  assert.strictEqual(formatDuration(61), '1m 01s');
  assert.strictEqual(formatDuration(3725), '1h 02m');
  assert.strictEqual(formatDuration(null), '0s');
});

test('idle snapshot renders no chips', () => {
  assert.deepStrictEqual(marqueeItems(snapshot(), NOW, NOW), []);
  assert.deepStrictEqual(marqueeItems(null, NOW, NOW), []);
});

test('an active curiosity run is one chip, with its lane and a duration that keeps ticking', () => {
  const s = snapshot({
    curiosity_runs: [{
      run_id: 'abc', correlation_id: 'c1', line: 'self_inquiry', status: 'running', active: true,
      node: 'harness_turn', duration_sec: 120,
      turn: { correlation_id: 'c1', lane: 'agent', phase: 'running', elapsed_sec: 100, source: 'curiosity_investigation' },
    }],
    lanes: {
      chat: { running: [], queued: [], recent: [] },
      agent: { running: [{ correlation_id: 'c1', lane: 'agent', phase: 'running', elapsed_sec: 100, source: 'curiosity_investigation' }], queued: [], recent: [] },
    },
  });
  const items = marqueeItems(s, NOW + 30_000, NOW);
  assert.strictEqual(items.length, 1, 'the run chip absorbs its own harness turn; the lane is not repeated');
  assert.strictEqual(items[0].kind, 'run');
  assert.strictEqual(items[0].text, 'self-inquiry · thinking (harness turn) · 2m 30s · agent lane');
});

test('a finished run leaves the marquee; a resumed one is flagged', () => {
  const s = snapshot({
    curiosity_runs: [
      { run_id: 'done', correlation_id: 'x', line: 'investigate', status: 'completed', active: false, node: 'finish', duration_sec: 900 },
      { run_id: 'res', correlation_id: 'y', line: 'investigate', status: 'resumed', active: true, node: 'journal', duration_sec: 10 },
    ],
  });
  const items = marqueeItems(s, NOW, NOW);
  assert.strictEqual(items.length, 1);
  assert.strictEqual(items[0].tone, 'warn');
  assert.match(items[0].text, /^curiosity · writing the journal entry/);
});

test('governor lanes show running and queued counts; queued turns turn the chip amber', () => {
  const s = snapshot({
    lanes: {
      chat: {
        running: [{ correlation_id: 'a', lane: 'chat', phase: 'running', elapsed_sec: 5, source: 'chat' }],
        queued: [{ correlation_id: 'b', lane: 'chat', phase: 'queued', elapsed_sec: 1, queued_sec: 1, source: 'chat' }],
        recent: [],
      },
      agent: { running: [], queued: [], recent: [] },
    },
  });
  const items = marqueeItems(s, NOW, NOW);
  assert.strictEqual(items.length, 1);
  assert.strictEqual(items[0].text, 'chat lane · chat running 5s · 1 queued');
  assert.strictEqual(items[0].tone, 'warn');
});

test('gateway lanes appear only when busy, named by their routes', () => {
  const s = snapshot({
    gateway: {
      error: null, polled_at: '2026-09-09T05:09:58Z',
      snapshot: {
        lanes: [
          { upstream: 'http://w:8012', inflight: 0, waiting: 0, max_inflight: 8, routes: [{ id: 'metacog' }, { id: 'metacog_background' }] },
          { upstream: 'http://w:8013', inflight: 2, waiting: 3, max_inflight: 8, routes: [{ id: 'quick' }, { id: 'quick_background' }] },
          { upstream: 'http://w:9999', inflight: 1, waiting: 0, max_inflight: 8, routes: [] },
        ],
        ledger: {},
      },
    },
  });
  const items = marqueeItems(s, NOW, NOW);
  assert.deepStrictEqual(items.map((i) => i.text), [
    'quick/quick_background · 2/8 in flight · 3 waiting',
    'http://w:9999 · 1/8 in flight',
  ]);
  assert.strictEqual(items[0].tone, 'warn');
  assert.strictEqual(items[1].tone, 'quiet');
});

test('modal body escapes everything it prints and reports absence as absence', () => {
  const s = snapshot({
    curiosity_runs: [{
      run_id: 'r<script>', correlation_id: 'c', line: 'investigate', status: 'failed', active: false, node: 'harness_turn',
      duration_sec: 3, error: '<b>boom</b>', transitions: [{ node: 'harness_turn', status: 'failed', at: '2026-09-09T05:00:00Z' }],
    }],
    gateway: { snapshot: null, error: 'ConnectionError: refused', polled_at: null },
  });
  const html = renderModalBody(s, NOW, NOW);
  assert.ok(!html.includes('<script>'), 'run id is escaped');
  assert.ok(html.includes('&lt;b&gt;boom&lt;/b&gt;'), 'error text is escaped');
  assert.ok(html.includes('nothing running'));
  assert.ok(html.includes('no harness turn seen for this run yet'));
  assert.ok(html.includes('gateway poll failed: ConnectionError: refused'));
  assert.ok(html.includes('snapshot v7'));
});

test('a queued turn in the modal says what it is waiting for', () => {
  const s = snapshot({
    lanes: {
      chat: { running: [], queued: [{ correlation_id: 'q1', lane: 'chat', phase: 'queued', elapsed_sec: 12, queued_sec: 12, source: 'world_pulse_read', mode: 'orion', model_label: 'MODEL_SONNET' }], recent: [] },
      agent: { running: [], queued: [], recent: [] },
    },
  });
  const html = renderModalBody(s, NOW, NOW);
  assert.ok(html.includes('waiting for the governor · 12s'));
  assert.ok(html.includes('corr q1'));
  assert.ok(html.includes('MODEL_SONNET'));
});

test('a failed turn error is escaped exactly once', () => {
  const s = snapshot({
    lanes: {
      chat: { running: [], queued: [], recent: [{ correlation_id: 'f', lane: 'chat', phase: 'finished', elapsed_sec: 3, ok: false, error: '<x>&', source: 'chat' }] },
      agent: { running: [], queued: [], recent: [] },
    },
  });
  const html = renderModalBody(s, NOW, NOW);
  assert.ok(html.includes('failed: &lt;x&gt;&amp;'));
  assert.ok(!html.includes('&amp;lt;'), 'no double escaping');
});

test('laneSummary counts per lane', () => {
  const s = snapshot({ lanes: { chat: { running: [{}], queued: [{}, {}], recent: [] }, agent: { running: [], queued: [], recent: [{}] } } });
  assert.deepStrictEqual(laneSummary(s), [
    { lane: 'chat', running: 1, queued: 2, recent: 0 },
    { lane: 'agent', running: 0, queued: 0, recent: 1 },
  ]);
});

test('escapeHtml covers the five characters that matter', () => {
  assert.strictEqual(escapeHtml('<a href="x">&\'</a>'), '&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;');
});
