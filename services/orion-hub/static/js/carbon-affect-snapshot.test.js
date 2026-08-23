const test = require('node:test');
const assert = require('node:assert/strict');
const { renderAffectSnapshotHtml, createLatestWinsGate } = require('./carbon-affect-snapshot.js');

test('createLatestWinsGate drops a stale response that resolves after a newer one already won', () => {
  // Regression, round 3: two concurrent fetches (dropdown-selection fetch
  // + the pre-existing 15s poll) can resolve out of order. The older one
  // resolving LATER must not be allowed to overwrite the newer one's
  // already-painted result.
  const gate = createLatestWinsGate();
  const tokenA = gate.issue(); // issued first (older)
  const tokenB = gate.issue(); // issued second (newer)
  // B resolves first and paints -- correct, it's current:
  assert.equal(gate.isCurrent(tokenB), true);
  // A resolves afterward, even though it was issued first -- must lose:
  assert.equal(gate.isCurrent(tokenA), false);
});

test('createLatestWinsGate lets a lone in-order fetch win', () => {
  const gate = createLatestWinsGate();
  const token = gate.issue();
  assert.equal(gate.isCurrent(token), true);
});

test('renderAffectSnapshotHtml shows an in-progress capture distinctly, not stale content', () => {
  const html = renderAffectSnapshotHtml({ last_attempt_at: 100, tick_in_progress: true }, 105);
  assert.match(html, /Capturing now/);
});

test('renderAffectSnapshotHtml shows the raw response text and a relative timestamp on a real success', () => {
  const html = renderAffectSnapshotHtml(
    {
      last_attempt_at: 100,
      tick_in_progress: false,
      last_result_ok: true,
      last_raw_response: 'sad, contemplative',
      last_trigger: 'ambient',
    },
    110
  );
  assert.match(html, /sad, contemplative/);
  assert.match(html, /10s ago/);
  assert.match(html, /ambient tick/);
});

test('renderAffectSnapshotHtml recomputes the elapsed-time text fresh on every call (no staleness)', () => {
  // Regression, round 2: an earlier draft skipped repainting when a
  // dedup key was unchanged, so the displayed "X ago" text froze at
  // whatever it was on the last actual repaint even though the caller
  // polls every ~15s. This module has no dedup of its own -- the SAME
  // status object rendered at two different "now" values must produce
  // two different elapsed-time strings.
  const status = { last_attempt_at: 100, tick_in_progress: false, last_result_ok: true, last_raw_response: 'calm' };
  const soon = renderAffectSnapshotHtml(status, 110);
  const later = renderAffectSnapshotHtml(status, 2600);
  assert.match(soon, /10s ago/);
  assert.match(later, /42m ago/);
});

test('renderAffectSnapshotHtml distinguishes a textless success from a real failure (review finding)', () => {
  const success = renderAffectSnapshotHtml(
    { last_attempt_at: 100, tick_in_progress: false, last_result_ok: true, last_raw_response: '' },
    110
  );
  assert.match(success, /succeeded but returned no text/);
  assert.doesNotMatch(success, /had no successful result/);

  const failure = renderAffectSnapshotHtml(
    { last_attempt_at: 100, tick_in_progress: false, last_result_ok: false, last_error: '404 Client Error' },
    110
  );
  assert.match(failure, /had no successful result: 404 Client Error/);
});

test('renderAffectSnapshotHtml fully escapes HTML in the model response (&, <, >)', () => {
  // Review finding, round 2: an earlier draft only escaped "<", matching
  // the pre-existing pre-fix code but weaker than app.js's own
  // escapeHtml() helper, which also escapes "&" and ">".
  const html = renderAffectSnapshotHtml(
    { last_attempt_at: 100, tick_in_progress: false, last_result_ok: true, last_raw_response: '<script>a && b</script>' },
    110
  );
  assert.ok(!html.includes('<script>'));
  assert.ok(html.includes('&lt;script&gt;a &amp;&amp; b&lt;/script&gt;'));
});

test('renderAffectSnapshotHtml also escapes the error text (pre-existing code never did)', () => {
  const html = renderAffectSnapshotHtml(
    { last_attempt_at: 100, tick_in_progress: false, last_result_ok: false, last_error: '<b>boom</b> & retry' },
    110
  );
  assert.ok(!html.includes('<b>boom</b>'));
  assert.ok(html.includes('&lt;b&gt;boom&lt;/b&gt; &amp; retry'));
});

test('renderAffectSnapshotHtml reports the never-ran and unavailable states distinctly', () => {
  assert.match(renderAffectSnapshotHtml(null, 0), /unavailable/);
  assert.match(renderAffectSnapshotHtml({}, 0), /No affect check has run yet/);
});
