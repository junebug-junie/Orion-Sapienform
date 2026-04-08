const test = require('node:test');
const assert = require('node:assert/strict');
const scheduleUi = require('./workflow-schedule-ui.js');

test('normalizeSchedule provides bounded inventory model', () => {
  const model = scheduleUi.normalizeSchedule({
    schedule_id: 'sch-1',
    workflow_id: 'journal_pass',
    workflow_display_name: 'Journal Pass',
    state: 'scheduled',
    notify_on: 'completion',
    next_run_at: '2026-03-25T22:00:00Z',
    last_run_at: '2026-03-24T22:00:00Z',
    last_result_status: 'completed',
    execution_policy: {
      schedule: { kind: 'recurring', cadence: 'daily', hour_local: 22, minute_local: 0 },
    },
    analytics: {
      health: 'healthy',
      needs_attention: false,
      recent_run_count: 5,
      recent_success_count: 4,
      recent_failure_count: 1,
      recent_outcomes: ['completed', 'completed', 'failed'],
    },
  });

  assert.equal(model.schedule_id, 'sch-1');
  assert.equal(model.cadence_summary, 'daily @ 22:00');
  assert.equal(model.state, 'scheduled');
  assert.equal(model.analytics.health, 'healthy');
  assert.equal(model.analytics.trend_text, '4/5 recent succeeded');
});

test('stateChipClass gives deterministic style buckets', () => {
  assert.match(scheduleUi.stateChipClass('paused'), /amber/);
  assert.match(scheduleUi.stateChipClass('cancelled'), /gray/);
});


test('normalizeSchedule exposes short id for disambiguation cues', () => {
  const model = scheduleUi.normalizeSchedule({ schedule_id: '1234567890abcdef', workflow_id: 'self_review' });
  assert.equal(model.schedule_id_short, '90abcdef');
});

test('historyStatusClass maps failed and dispatched statuses', () => {
  assert.match(scheduleUi.historyStatusClass('failed'), /red/);
  assert.match(scheduleUi.historyStatusClass('dispatched'), /sky/);
});

test('normalizeAnalytics handles overdue and missing fields conservatively', () => {
  const analytics = scheduleUi.normalizeAnalytics({
    health: 'degraded',
    is_overdue: true,
    overdue_seconds: 7200,
    recent_run_count: 0,
  });
  assert.equal(analytics.health, 'degraded');
  assert.equal(analytics.is_overdue, true);
  assert.equal(analytics.needs_attention, true);
  assert.equal(analytics.trend_text, null);
});

test('healthChipClass maps backend health enum without deriving health client-side', () => {
  assert.match(scheduleUi.healthChipClass('healthy'), /emerald/);
  assert.match(scheduleUi.healthChipClass('degraded'), /amber/);
  assert.match(scheduleUi.healthChipClass('failing'), /red/);
});

test('normalizeSchedule tolerates schedules without analytics', () => {
  const model = scheduleUi.normalizeSchedule({
    schedule_id: 'sch-missing-analytics',
    workflow_id: 'concept_induction',
    execution_policy: { schedule: { kind: 'one_shot', run_at_utc: '2026-03-25T12:00:00Z' } },
  });
  assert.equal(model.analytics, null);
});

test('normalizeSchedule clamps malformed enum-like values to safe defaults', () => {
  const model = scheduleUi.normalizeSchedule({
    schedule_id: 'sch-bad-enums',
    workflow_id: 'journal_pass',
    state: 'totally_broken_state',
    last_result_status: 'wat',
    execution_policy: { schedule: { kind: 'recurring', cadence: 'daily' } },
    analytics: { health: 'catastrophic', most_recent_result_status: 'bad' },
  });
  assert.equal(model.state, 'scheduled');
  assert.equal(model.last_result_status, 'unknown');
  assert.equal(model.analytics.health, 'idle');
  assert.equal(model.analytics.most_recent_result_status, 'unknown');
});

test('normalizeHistoryItem and normalizeEventItem fail-soft on malformed entries', () => {
  assert.equal(scheduleUi.normalizeHistoryItem(null), null);
  const run = scheduleUi.normalizeHistoryItem({ status: 'failed', error: 42 });
  assert.equal(run.status, 'failed');
  assert.equal(run.error, '42');
  const event = scheduleUi.normalizeEventItem({ kind: '', occurred_at: null });
  assert.equal(event.kind, 'event');
});
