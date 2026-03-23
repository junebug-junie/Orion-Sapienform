(function (global) {
  const BLOCKED_RE = /\b(sealed|private|password|secret|ssn|mirror|journal)\b/i;
  const SAFE_OMISSION_RE = /blocked\/private (summaries|material) omitted/i;
  const SECTION_LABELS = {
    context_window: 'Context window',
    claims: 'Claims / consensus / divergence',
    commitments: 'Commitments',
    routing: 'Routing',
    repair: 'Repair',
    deliberation: 'Deliberation',
    floor: 'Handoff / closure',
    calibration: 'Calibration',
    freshness: 'Freshness',
    resumptive: 'Re-entry / snapshot',
    epistemic: 'Epistemic stance',
    artifact_dialogue: 'Artifact dialogue',
    gif: 'GIF policy',
    safety: 'Safety omissions',
  };

  function safeText(value, limit = 220) {
    const text = String(value ?? '').replace(/\s+/g, ' ').trim();
    if (!text) return '';
    if (BLOCKED_RE.test(text) && !SAFE_OMISSION_RE.test(text)) return '';
    return text.slice(0, limit);
  }

  function cleanList(values, limit = 8) {
    const items = Array.isArray(values) ? values : [];
    const out = [];
    items.forEach((value) => {
      const text = safeText(value, 240);
      if (text && !out.includes(text) && out.length < limit) out.push(text);
    });
    return out;
  }

  function normalizeSnapshot(snapshot) {
    if (!snapshot || typeof snapshot !== 'object') return null;
    const sections = Array.isArray(snapshot.sections)
      ? snapshot.sections
          .map((section) => normalizeSection(section))
          .filter(Boolean)
      : [];
    return {
      snapshot_id: safeText(snapshot.snapshot_id, 180),
      platform: safeText(snapshot.platform, 80),
      room_id: safeText(snapshot.room_id, 120),
      thread_key: safeText(snapshot.thread_key, 180),
      participant_id: safeText(snapshot.participant_id, 120),
      summary: safeText(snapshot.summary, 220),
      sections,
      decision_traces: Array.isArray(snapshot.decision_traces) ? snapshot.decision_traces : [],
      metadata: snapshot.metadata && typeof snapshot.metadata === 'object' ? snapshot.metadata : {},
      built_at: safeText(snapshot.built_at, 80),
    };
  }

  function normalizeSection(section) {
    if (!section || typeof section !== 'object') return null;
    const kind = safeText(section.section_kind, 80);
    if (!kind) return null;
    return {
      section_id: safeText(section.section_id, 180),
      section_kind: kind,
      label: SECTION_LABELS[kind] || kind.replace(/_/g, ' '),
      why_this_mattered: safeText(section.why_this_mattered, 260),
      included_artifact_summaries: cleanList(section.included_artifact_summaries, 10),
      selected_state: cleanList(section.selected_state, 8),
      softened_state: cleanList(section.softened_state, 8),
      excluded_state: cleanList(section.excluded_state, 8),
      freshness_hints: cleanList(section.freshness_hints, 6),
      confidence_hints: cleanList(section.confidence_hints, 6),
      decision_traces: Array.isArray(section.decision_traces)
        ? section.decision_traces.map((trace) => normalizeTrace(trace)).filter(Boolean)
        : [],
      metadata: section.metadata && typeof section.metadata === 'object' ? section.metadata : {},
    };
  }

  function normalizeTrace(trace) {
    if (!trace || typeof trace !== 'object') return null;
    const rawSummary = String(trace.summary ?? '').replace(/\s+/g, ' ').trim();
    const summary = safeText(rawSummary, 220);
    if (rawSummary && !summary) return null;
    const why = safeText(trace.why_it_mattered, 240);
    if (!summary && !why) return null;
    return {
      trace_kind: safeText(trace.trace_kind, 80),
      decision_state: safeText(trace.decision_state, 40),
      summary,
      why_it_mattered: why,
      freshness_hint: safeText(trace.freshness_hint, 80),
      source_ref: safeText(trace.source_ref, 120),
    };
  }

  function getSection(snapshot, kind) {
    const normalized = normalizeSnapshot(snapshot);
    if (!normalized) return null;
    return normalized.sections.find((section) => section.section_kind === kind) || null;
  }

  function countStateItems(section, state) {
    if (!section || !Array.isArray(section[state])) return 0;
    return section[state].length;
  }

  function formatCountLabel(count, singular, plural) {
    const value = Number(count || 0);
    return `${value} ${value === 1 ? singular : plural}`;
  }

  function resolveInspectionQuery(routeDebug) {
    const debug = routeDebug && typeof routeDebug === 'object' ? routeDebug : {};
    const live = debug.social_inspection && typeof debug.social_inspection === 'object' ? debug.social_inspection : {};
    const platform = safeText(live.platform, 80);
    const roomId = safeText(live.room_id, 120);
    const participantId = safeText(live.participant_id, 120);
    const threadKey = safeText(live.thread_key, 180);
    return {
      available: Boolean(platform && roomId),
      platform,
      room_id: roomId,
      participant_id: participantId,
      thread_key: threadKey,
      cache_key: [platform || 'none', roomId || 'none', participantId || 'room', threadKey || 'thread'].join('::'),
    };
  }

  function buildOperatorSummary(routeDebug, liveSnapshot, memorySnapshot) {
    const debug = routeDebug && typeof routeDebug === 'object' ? routeDebug : {};
    const live = normalizeSnapshot(liveSnapshot);
    const memory = normalizeSnapshot(memorySnapshot);
    const routing = debug.social_thread_routing && typeof debug.social_thread_routing === 'object' ? debug.social_thread_routing : {};
    const epistemicDecision = debug.social_epistemic_decision && typeof debug.social_epistemic_decision === 'object' ? debug.social_epistemic_decision : {};
    const epistemicSignal = debug.social_epistemic_signal && typeof debug.social_epistemic_signal === 'object' ? debug.social_epistemic_signal : {};
    const repairDecision = debug.social_repair_decision && typeof debug.social_repair_decision === 'object' ? debug.social_repair_decision : {};
    const gifPolicy = debug.social_gif_policy && typeof debug.social_gif_policy === 'object' ? debug.social_gif_policy : {};
    const gifInterpretation = debug.social_gif_interpretation && typeof debug.social_gif_interpretation === 'object' ? debug.social_gif_interpretation : {};
    const liveContext = getSection(live, 'context_window');
    const liveClaims = getSection(live, 'claims');
    const liveCommitments = getSection(live, 'commitments');
    const liveFreshness = getSection(live, 'freshness');
    const liveSafety = getSection(live, 'safety');
    const liveResumptive = getSection(live, 'resumptive');
    const route = safeText(routing.routing_decision, 80) || '--';
    const audience = safeText(routing.audience_scope, 80) || '--';
    const primaryThread = safeText(routing.primary_thread_summary || routing.thread_summary || (live && live.thread_key), 220) || '--';
    const epistemic = safeText(epistemicDecision.decision_kind || epistemicSignal.signal_kind, 120) || '--';
    const repair = safeText(repairDecision.decision_kind, 120) || '--';
    const gif = safeText(gifPolicy.decision || gifPolicy.decision_kind || gifInterpretation.cue_disposition, 120) || '--';
    const safetyOmissions = countStateItems(liveSafety, 'excluded_state') || Number(live && live.metadata && live.metadata.safety_omissions || 0) || 0;

    const badges = [
      { label: 'Route', value: route, tone: 'indigo' },
      { label: 'Audience', value: audience, tone: 'sky' },
      { label: 'Epistemic', value: epistemic, tone: 'amber' },
      { label: 'Context', value: formatCountLabel(countStateItems(liveContext, 'selected_state'), 'selected', 'selected'), tone: 'emerald' },
      { label: 'Claims', value: formatCountLabel(countStateItems(liveClaims, 'selected_state'), 'active', 'active'), tone: 'violet' },
      { label: 'Commitments', value: formatCountLabel(countStateItems(liveCommitments, 'selected_state'), 'open', 'open'), tone: 'fuchsia' },
      { label: 'Freshness', value: formatCountLabel(countStateItems(liveFreshness, 'excluded_state'), 're-ground', 're-ground'), tone: 'orange' },
      { label: 'Safety', value: safetyOmissions ? `${safetyOmissions} omitted` : 'bounded', tone: safetyOmissions ? 'rose' : 'gray' },
    ];

    const summaryRows = [
      { label: 'Route', value: route },
      { label: 'Audience', value: audience },
      { label: 'Primary thread', value: primaryThread },
      { label: 'Epistemic stance', value: epistemic },
      { label: 'Repair / closure', value: [repair, safeText((debug.social_handoff_signal || {}).handoff_kind, 80), safeText((((debug.social_room_continuity || {}).closure_signal || {}).decision_kind), 80)].filter(Boolean).join(' · ') || '--' },
      { label: 'GIF policy', value: gif },
      { label: 'Re-entry anchor', value: safeText(liveResumptive && (liveResumptive.selected_state[0] || liveResumptive.included_artifact_summaries[0]), 220) || '--' },
      { label: 'Memory source', value: memory ? `${memory.platform || '--'} / ${memory.room_id || '--'}` : 'Not loaded' },
    ];

    return {
      badges,
      summaryRows,
      liveSnapshot: live,
      memorySnapshot: memory,
      query: resolveInspectionQuery(debug),
    };
  }

  function buildSurfaceModel(snapshot, fallbackTitle) {
    const normalized = normalizeSnapshot(snapshot);
    if (!normalized) {
      return {
        title: fallbackTitle,
        summary: 'No inspection payload available.',
        sections: [],
      };
    }
    return {
      title: fallbackTitle,
      summary: normalized.summary || 'No summary available.',
      built_at: normalized.built_at,
      sections: normalized.sections.map((section) => ({
        title: section.label,
        kind: section.section_kind,
        why: section.why_this_mattered,
        selected: section.selected_state,
        softened: section.softened_state,
        excluded: section.excluded_state,
        included: section.included_artifact_summaries,
        freshness: section.freshness_hints,
        confidence: section.confidence_hints,
        traces: section.decision_traces,
      })),
    };
  }

  function shouldShowSocialInspection(routeDebug) {
    const debug = routeDebug && typeof routeDebug === 'object' ? routeDebug : {};
    return Boolean(debug.social_inspection && normalizeSnapshot(debug.social_inspection));
  }

  const api = {
    safeText,
    cleanList,
    normalizeSnapshot,
    normalizeSection,
    normalizeTrace,
    resolveInspectionQuery,
    buildOperatorSummary,
    buildSurfaceModel,
    shouldShowSocialInspection,
    sectionLabels: SECTION_LABELS,
  };

  global.OrionSocialInspection = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
