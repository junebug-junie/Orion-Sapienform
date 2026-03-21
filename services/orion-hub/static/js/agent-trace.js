(function (global) {
  const FAMILY_ORDER = [
    'planning',
    'recall',
    'reasoning',
    'communication',
    'runtime',
    'external',
    'memory',
    'orchestration',
    'device',
    'unknown',
  ];

  function normalizeSummary(summary) {
    return summary && typeof summary === 'object' ? summary : null;
  }

  function shouldShowAgentTrace(summary) {
    const normalized = normalizeSummary(summary);
    if (!normalized) return false;
    if (normalized.mode !== 'agent') return false;
    const hasSteps = Array.isArray(normalized.steps) && normalized.steps.length > 0;
    const hasTools = Array.isArray(normalized.tools) && normalized.tools.length > 0;
    return hasSteps || hasTools || Boolean(normalized.summary_text);
  }

  function formatDuration(durationMs) {
    const ms = Number(durationMs || 0);
    if (!Number.isFinite(ms) || ms <= 0) return '--';
    if (ms < 1000) return `${ms} ms`;
    const seconds = ms / 1000;
    if (seconds < 60) return `${seconds.toFixed(seconds >= 10 ? 1 : 2)} s`;
    const minutes = Math.floor(seconds / 60);
    const remSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remSeconds}s`;
  }

  function groupToolsByFamily(tools) {
    const list = Array.isArray(tools) ? tools : [];
    const buckets = new Map();
    list.forEach((tool) => {
      if (!tool || typeof tool !== 'object') return;
      const family = tool.tool_family || 'unknown';
      if (!buckets.has(family)) {
        buckets.set(family, { family, count: 0, duration_ms: 0, tools: [] });
      }
      const bucket = buckets.get(family);
      const count = Number(tool.count || 0);
      const duration = Number(tool.duration_ms || 0);
      bucket.count += Number.isFinite(count) ? count : 0;
      bucket.duration_ms += Number.isFinite(duration) ? duration : 0;
      bucket.tools.push(tool);
    });
    return Array.from(buckets.values()).sort((a, b) => {
      const aIdx = FAMILY_ORDER.includes(a.family) ? FAMILY_ORDER.indexOf(a.family) : 999;
      const bIdx = FAMILY_ORDER.includes(b.family) ? FAMILY_ORDER.indexOf(b.family) : 999;
      if (aIdx !== bIdx) return aIdx - bIdx;
      return a.family.localeCompare(b.family);
    });
  }

  function buildTimelineRows(summary) {
    const normalized = normalizeSummary(summary);
    if (!normalized || !Array.isArray(normalized.steps)) return [];
    return normalized.steps.map((step, index) => ({
      index: Number(step.index ?? index),
      event_type: step.event_type || 'unknown',
      tool_id: step.tool_id || '--',
      tool_family: step.tool_family || 'unknown',
      action_kind: step.action_kind || 'unknown',
      effect_kind: step.effect_kind || 'unknown',
      status: step.status || 'unknown',
      duration_label: formatDuration(step.duration_ms),
      summary: step.summary || '--',
    }));
  }

  const api = {
    normalizeSummary,
    shouldShowAgentTrace,
    formatDuration,
    groupToolsByFamily,
    buildTimelineRows,
  };

  global.OrionAgentTrace = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
