(function (global) {
  function toObject(value) {
    return value && typeof value === 'object' ? value : null;
  }

  function toLower(value, fallback) {
    const normalized = String(value || fallback || '').trim().toLowerCase();
    return normalized || String(fallback || '').toLowerCase();
  }

  function oneOf(value, allowed, fallback) {
    const normalized = toLower(value, fallback);
    return allowed.includes(normalized) ? normalized : fallback;
  }

  function cadenceSummary(spec) {
    if (spec.kind === 'recurring') {
      const hasTime = typeof spec.hour_local === 'number';
      const hour = String(spec.hour_local || 0).padStart(2, '0');
      const minute = String(spec.minute_local || 0).padStart(2, '0');
      return `${spec.cadence || 'recurring'}${hasTime ? ` @ ${hour}:${minute}` : ''}`;
    }
    return spec.run_at_utc ? 'one-shot' : 'immediate';
  }

  function normalizeRecentOutcomes(outcomes) {
    if (!Array.isArray(outcomes)) return [];
    return outcomes
      .map((value) => String(value || '').trim().toLowerCase())
      .filter((value) => value.length > 0)
      .slice(0, 5);
  }

  function normalizeAnalytics(analyticsLike) {
    const analytics = toObject(analyticsLike);
    if (!analytics) return null;
    const health = oneOf(analytics.health, ['healthy', 'degraded', 'failing', 'paused', 'idle', 'cancelled'], 'idle');
    const recentRunCount = Number(analytics.recent_run_count || 0);
    const recentSuccessCount = Number(analytics.recent_success_count || 0);
    const recentFailureCount = Number(analytics.recent_failure_count || 0);
    const recentOutcomes = normalizeRecentOutcomes(analytics.recent_outcomes);
    const isOverdue = Boolean(analytics.is_overdue);
    const overdueSeconds = Number.isFinite(Number(analytics.overdue_seconds)) ? Number(analytics.overdue_seconds) : null;
    const missedRunCount = Number(analytics.missed_run_count || 0);
    const needsAttention = Boolean(analytics.needs_attention || isOverdue);
    let trendText = null;
    if (recentRunCount > 0) {
      trendText = `${recentSuccessCount}/${recentRunCount} recent succeeded`;
    }

    return {
      health,
      needs_attention: needsAttention,
      last_success_at: analytics.last_success_at || null,
      last_failure_at: analytics.last_failure_at || null,
      recent_run_count: recentRunCount,
      recent_success_count: recentSuccessCount,
      recent_failure_count: recentFailureCount,
      recent_outcomes: recentOutcomes,
      most_recent_result_status: analytics.most_recent_result_status
        ? oneOf(analytics.most_recent_result_status, ['unknown', 'completed', 'failed', 'cancelled', 'dispatched'], 'unknown')
        : null,
      is_overdue: isOverdue,
      overdue_seconds: overdueSeconds,
      missed_run_count: missedRunCount,
      history_window_runs: Number(analytics.history_window_runs || 5),
      trend_text: trendText,
    };
  }

  function normalizeSchedule(entryLike) {
    const entry = toObject(entryLike);
    if (!entry) return null;
    const scheduleId = String(entry.schedule_id || '').trim();
    if (!scheduleId) return null;
    const policy = toObject(entry.execution_policy) || {};
    const spec = toObject(policy.schedule) || {};
    const analytics = normalizeAnalytics(entry.analytics);

    return {
      schedule_id: scheduleId,
      schedule_id_short: scheduleId.length > 8 ? scheduleId.slice(-8) : scheduleId,
      workflow_id: String(entry.workflow_id || 'workflow').trim(),
      workflow_display_name: String(entry.workflow_display_name || entry.workflow_id || 'Workflow').trim(),
      state: oneOf(entry.state, ['scheduled', 'due', 'dispatched', 'completed', 'failed', 'cancelled', 'paused'], 'scheduled'),
      notify_on: String(entry.notify_on || policy.notify_on || 'none').toLowerCase(),
      next_run_at: entry.next_run_at || null,
      last_run_at: entry.last_run_at || null,
      last_result_status: oneOf(entry.last_result_status, ['unknown', 'completed', 'failed', 'cancelled', 'dispatched'], 'unknown'),
      cadence_summary: cadenceSummary(spec),
      execution_policy: policy,
      source_service: entry.source_service || null,
      source_kind: entry.source_kind || null,
      revision: Number(entry.revision || 0),
      analytics,
      raw: entry,
    };
  }

  function normalizeHistoryItem(itemLike) {
    const item = toObject(itemLike);
    if (!item) return null;
    return {
      run_id: String(item.run_id || '').trim() || null,
      status: oneOf(item.status, ['unknown', 'completed', 'failed', 'cancelled', 'dispatched'], 'unknown'),
      dispatch_at: item.dispatch_at || null,
      completed_at: item.completed_at || null,
      error: item.error ? String(item.error) : null,
    };
  }

  function normalizeEventItem(itemLike) {
    const item = toObject(itemLike);
    if (!item) return null;
    return {
      event_id: String(item.event_id || '').trim() || null,
      kind: String(item.kind || 'event').trim() || 'event',
      occurred_at: item.occurred_at || null,
    };
  }

  function stateChipClass(state) {
    const normalized = String(state || '').toLowerCase();
    if (normalized === 'scheduled') return 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200';
    if (normalized === 'paused') return 'border-amber-500/40 bg-amber-500/10 text-amber-200';
    if (normalized === 'cancelled') return 'border-gray-600/50 bg-gray-800/70 text-gray-300';
    if (normalized === 'failed') return 'border-red-500/40 bg-red-500/10 text-red-200';
    if (normalized === 'completed') return 'border-sky-500/40 bg-sky-500/10 text-sky-200';
    return 'border-cyan-500/40 bg-cyan-500/10 text-cyan-200';
  }

  function healthChipClass(health) {
    const normalized = toLower(health, 'idle');
    if (normalized === 'healthy') return 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200';
    if (normalized === 'degraded') return 'border-amber-500/40 bg-amber-500/10 text-amber-200';
    if (normalized === 'failing') return 'border-red-500/40 bg-red-500/10 text-red-200';
    if (normalized === 'paused') return 'border-amber-500/40 bg-amber-500/10 text-amber-200';
    if (normalized === 'cancelled') return 'border-gray-600/50 bg-gray-800/70 text-gray-300';
    if (normalized === 'idle') return 'border-slate-500/40 bg-slate-500/10 text-slate-200';
    return 'border-cyan-500/40 bg-cyan-500/10 text-cyan-200';
  }

  function historyStatusClass(status) {
    const normalized = String(status || '').toLowerCase();
    if (normalized === 'completed') return 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200';
    if (normalized === 'failed') return 'border-red-500/40 bg-red-500/10 text-red-200';
    if (normalized === 'cancelled') return 'border-gray-600/50 bg-gray-800/70 text-gray-300';
    if (normalized === 'dispatched') return 'border-sky-500/40 bg-sky-500/10 text-sky-200';
    if (normalized === 'queued' || normalized === 'due') return 'border-amber-500/40 bg-amber-500/10 text-amber-200';
    return 'border-cyan-500/40 bg-cyan-500/10 text-cyan-200';
  }

  function asLocal(value) {
    if (!value) return '--';
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return '--';
    return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
  }

  const api = {
    normalizeAnalytics,
    normalizeSchedule,
    normalizeHistoryItem,
    normalizeEventItem,
    stateChipClass,
    healthChipClass,
    historyStatusClass,
    asLocal,
  };

  global.OrionWorkflowScheduleUI = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
