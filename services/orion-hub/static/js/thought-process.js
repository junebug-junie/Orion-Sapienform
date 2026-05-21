(function (global) {
  function cleanText(value) {
    const text = typeof value === 'string' ? value : value === null || value === undefined ? '' : String(value);
    const trimmed = text.trim();
    return trimmed ? trimmed : null;
  }

  function asObject(value) {
    return value && typeof value === 'object' && !Array.isArray(value) ? value : null;
  }

  function asList(value) {
    return Array.isArray(value) ? value : [];
  }

  function firstText(values) {
    for (let i = 0; i < values.length; i += 1) {
      const candidate = cleanText(values[i]);
      if (candidate) return candidate;
    }
    return null;
  }

  function collectBase(messageOrMeta) {
    const root = asObject(messageOrMeta) || {};
    const raw = asObject(root.raw) || {};
    const rawNested = asObject(raw.raw) || {};
    const rawMeta = asObject(raw.metadata) || {};
    const routing = asObject(root.routingDebug) || asObject(root.routing_debug) || {};
    const metacog = asList(root.metacogTraces || root.metacog_traces || raw.metacog_traces || rawMeta.metacog_traces);
    return { root, raw, rawNested, rawMeta, routing, metacog };
  }

  function pickMetacogReasoningTrace(traces) {
    for (let i = 0; i < traces.length; i += 1) {
      const trace = asObject(traces[i]);
      if (!trace) continue;
      const role = String(trace.trace_role || trace.role || trace.trace_type || trace.type || '').toLowerCase();
      if (!role.includes('reason') && !role.includes('think')) continue;
      const content = firstText([trace.content, trace.text]);
      if (!content) continue;
      return { content, trace, index: i };
    }
    return null;
  }

  function selectThoughtProcess(messageOrMeta) {
    const { root, raw, rawNested, rawMeta, routing, metacog } = collectBase(messageOrMeta);
    const reasoningTrace = asObject(root.reasoning_trace || root.reasoningTrace || raw.reasoning_trace || raw.reasoningTrace);
    const reasoningContent = firstText([
      root.reasoning_content,
      root.reasoningContent,
      root.reasoning,
    ]);
    const providerReasoning = firstText([
      rawMeta.reasoning_content,
      raw.reasoning_content,
      rawNested.reasoning_content,
    ]);
    const parsedThink = firstText([
      root.inline_think_content,
      root.inlineThinkContent,
      root.parsed_think,
      root.parsedThink,
      rawMeta.inline_think_content,
      raw.inline_think_content,
    ]);
    const metacogChoice = pickMetacogReasoningTrace(metacog);

    let text = null;
    let source = null;
    let traceMeta = null;
    if (reasoningTrace) {
      text = firstText([reasoningTrace.content]);
      if (text) {
        source = 'reasoning_trace';
        traceMeta = reasoningTrace;
      }
    }
    if (!text && reasoningTrace) {
      text = firstText([reasoningTrace.text]);
      if (text) {
        source = 'reasoning_trace';
        traceMeta = reasoningTrace;
      }
    }
    if (!text && reasoningContent) {
      text = reasoningContent;
      source = 'reasoning_content';
    }
    if (!text && metacogChoice) {
      text = metacogChoice.content;
      source = 'metacog';
      traceMeta = metacogChoice.trace;
    }
    if (!text && providerReasoning) {
      text = providerReasoning;
      source = 'provider';
    }
    if (!text && parsedThink) {
      text = parsedThink;
      source = 'parsed_think';
    }

    const model = firstText([root.model, rawMeta.model, raw.model]);
    const provider = firstText([root.provider, rawMeta.provider, raw.provider]);
    const mode = firstText([root.mode, routing.mode, raw.mode]);
    const tokenCountRaw = (traceMeta && (traceMeta.token_count || traceMeta.tokens)) || rawMeta.reasoning_tokens || root.reasoning_tokens;
    const numericTokenCount = Number(tokenCountRaw);
    const tokenCount = Number.isFinite(numericTokenCount) && numericTokenCount > 0 ? numericTokenCount : null;
    const charCount = text ? text.length : 0;

    return {
      text: text || null,
      source: source || null,
      metadata: {
        mode,
        model,
        provider,
        token_count: tokenCount,
        char_count: charCount || null,
      },
    };
  }

  function escapeHtml(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function resolveCorrelationId(meta) {
    const root = asObject(meta) || {};
    const raw = asObject(root.raw) || {};
    const routing = asObject(root.routingDebug) || asObject(root.routing_debug) || {};
    const linkage = asObject(root.trace_linkage) || asObject(root.traceLinkage) || {};
    return firstText([
      root.root_correlation_id,
      root.rootCorrelationId,
      linkage.root_correlation_id,
      linkage.correlation_id,
      raw.root_correlation_id,
      routing.root_correlation_id,
      root.correlationId,
      root.correlation_id,
      root.turnId,
      root.turn_id,
      raw.correlation_id,
    ]);
  }

  async function fetchCognitionTrace(apiBaseUrl, correlationId) {
    const corr = cleanText(correlationId);
    if (!corr) return { error: 'missing_correlation_id' };
    const base = String(apiBaseUrl || '').replace(/\/$/, '');
    const res = await fetch(`${base}/api/cognition/trace/${encodeURIComponent(corr)}`);
    if (res.status === 404) return { error: 'trace_not_cached' };
    if (!res.ok) return { error: `http_${res.status}` };
    return { body: await res.json() };
  }

  function stepServices(step) {
    const row = asObject(step) || {};
    const listed = asList(row.services).map((name) => cleanText(name)).filter(Boolean);
    if (listed.length) return listed;
    const result = asObject(row.result) || {};
    return Object.keys(result);
  }

  function statusBadgeClass(status) {
    const normalized = String(status || '').toLowerCase();
    if (normalized === 'success') return 'border-emerald-500/40 bg-emerald-500/15 text-emerald-100';
    if (normalized === 'error' || normalized === 'failed' || normalized === 'failure') {
      return 'border-rose-500/40 bg-rose-500/15 text-rose-100';
    }
    return 'border-gray-600 bg-gray-800/80 text-gray-200';
  }

  function wireExecutionStepsPanelActions(host) {
    if (!host || typeof host.querySelector !== 'function') return;
    host.querySelectorAll('[data-orion-action="open-organ-signals"]').forEach((btn) => {
      if (btn.dataset.orionWired === '1') return;
      btn.dataset.orionWired = '1';
      btn.addEventListener('click', (event) => {
        event.preventDefault();
        const corr = cleanText(btn.getAttribute('data-correlation-id'));
        if (!corr) return;
        const opener = global.OrionHubOpenOrganSignals;
        if (typeof opener === 'function') {
          opener(corr);
          return;
        }
        const url = new URL(global.location.href);
        url.searchParams.set('correlation_id', corr);
        url.hash = '#signals';
        global.location.assign(`${url.pathname}${url.search}${url.hash}`);
      });
    });
  }

  function buildExecutionStepsPanel({ correlationId, apiBaseUrl, trace, debug, error, loading }) {
    const corr = cleanText(correlationId) || '--';
    const traceObj = asObject(trace) || {};
    const steps = asList(traceObj.steps).slice().sort((a, b) => {
      const left = Number((asObject(a) || {}).order);
      const right = Number((asObject(b) || {}).order);
      if (Number.isFinite(left) && Number.isFinite(right)) return left - right;
      return 0;
    });
    const verb = cleanText(traceObj.verb) || '--';
    const complete = traceObj.complete !== false;
    const gaps = asList(traceObj.gaps);

    let body = '';
    if (loading) {
      body = '<div class="text-[11px] text-gray-400">Loading execution trace…</div>';
    } else if (error) {
      body = `<div class="text-[11px] text-amber-200">Trace unavailable (${escapeHtml(error)}).</div>`;
    } else if (!steps.length) {
      body = '<div class="text-[11px] text-gray-400">No execution steps cached for this turn.</div>';
    } else {
      const rows = steps.map((step) => {
        const row = asObject(step) || {};
        const stepName = cleanText(row.step_name) || 'step';
        const status = cleanText(row.status) || '--';
        const latency = Number(row.latency_ms);
        const latencyLabel = Number.isFinite(latency) ? `${latency} ms` : '--';
        const services = stepServices(row);
        const servicesLabel = services.length ? services.join(', ') : '--';
        const showExpand = Boolean(debug) && (row.error || asList(row.log_tail).length);
        let expandHtml = '';
        if (showExpand) {
          const errorText = cleanText(row.error) || '';
          const logTail = asList(row.log_tail).slice(0, 5).map((line) => escapeHtml(line)).join('\n');
          expandHtml = [
            '<details class="mt-1 rounded border border-gray-800 bg-gray-950/50 p-2">',
            '<summary class="cursor-pointer text-[10px] text-gray-400">Debug details</summary>',
            errorText ? `<pre class="mt-1 whitespace-pre-wrap break-words text-[10px] text-rose-200">${escapeHtml(errorText)}</pre>` : '',
            logTail ? `<pre class="mt-1 whitespace-pre-wrap break-words text-[10px] text-gray-300">${logTail}</pre>` : '',
            '</details>',
          ].join('');
        }
        return [
          '<li class="rounded-lg border border-gray-800 bg-gray-950/50 px-2 py-2">',
          '<div class="flex flex-wrap items-center justify-between gap-2">',
          `<div class="font-mono text-[11px] text-gray-100">${escapeHtml(stepName)}</div>`,
          `<span class="rounded-full border px-2 py-0.5 text-[10px] ${statusBadgeClass(status)}">${escapeHtml(status)}</span>`,
          '</div>',
          `<div class="mt-1 text-[10px] text-gray-400">${escapeHtml(latencyLabel)} · ${escapeHtml(servicesLabel)}</div>`,
          expandHtml,
          '</li>',
        ].join('');
      }).join('');
      const metaBits = [
        `verb ${escapeHtml(verb)}`,
        complete ? 'complete' : 'partial',
        gaps.length ? `${gaps.length} gap(s)` : null,
      ].filter(Boolean).join(' · ');
      body = [
        `<div class="text-[10px] text-gray-500">${metaBits}</div>`,
        `<ol class="mt-2 space-y-2">${rows}</ol>`,
      ].join('');
    }

    const footer = [
      '<div class="mt-3 border-t border-gray-800 pt-2 space-y-1.5">',
      `<div class="text-[10px] text-gray-500 font-mono break-all">correlation_id: ${escapeHtml(corr)}</div>`,
      `<button type="button" class="text-[11px] text-indigo-300 hover:text-indigo-100 underline-offset-2 hover:underline" data-orion-action="open-organ-signals" data-correlation-id="${escapeHtml(corr)}">View in Organ Signals</button>`,
      '</div>',
    ].join('');

    return [
      '<details class="execution-steps-panel mt-2 rounded-xl border border-sky-500/30 bg-sky-500/5 p-3">',
      '<summary class="cursor-pointer text-[10px] uppercase tracking-wide text-sky-200">Execution Steps</summary>',
      `<div class="mt-2 space-y-2">${body}${footer}</div>`,
      '</details>',
    ].join('');
  }

  async function mountExecutionStepsPanel(parent, { meta, apiBaseUrl, debug } = {}) {
    if (!parent || typeof parent.appendChild !== 'function') return null;
    const correlationId = resolveCorrelationId(meta);
    if (!correlationId) return null;

    const host = document.createElement('div');
    host.className = 'execution-steps-host';
    host.innerHTML = buildExecutionStepsPanel({
      correlationId,
      apiBaseUrl,
      trace: null,
      debug,
      loading: true,
    });
    wireExecutionStepsPanelActions(host);
    parent.appendChild(host);

    try {
      const result = await fetchCognitionTrace(apiBaseUrl, correlationId);
      host.innerHTML = buildExecutionStepsPanel({
        correlationId,
        apiBaseUrl,
        trace: result.body || null,
        debug,
        error: result.error || null,
      });
      wireExecutionStepsPanelActions(host);
    } catch (_err) {
      host.innerHTML = buildExecutionStepsPanel({
        correlationId,
        apiBaseUrl,
        trace: null,
        debug,
        error: 'fetch_failed',
      });
      wireExecutionStepsPanelActions(host);
    }
    return host;
  }

  const api = {
    selectThoughtProcess,
    resolveCorrelationId,
    fetchCognitionTrace,
    buildExecutionStepsPanel,
    mountExecutionStepsPanel,
  };
  global.OrionThoughtProcess = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);

(function (global) {
  const INLINE_CARD_ID = 'chatStanceCognitiveProjectionInspectCard';
  const MODAL_CARD_ID = 'chatStanceCognitiveProjectionInspectModalCard';
  const INLINE_COMPARE_ID = 'chatStanceCognitiveComparisonInspectCard';
  const MODAL_COMPARE_ID = 'chatStanceCognitiveComparisonInspectModalCard';

  function asObject(value) {
    return value && typeof value === 'object' && !Array.isArray(value) ? value : null;
  }

  function asList(value) {
    return Array.isArray(value) ? value : [];
  }

  function short(value, fallback = '--') {
    const text = String(value === null || value === undefined ? '' : value).trim();
    return text || fallback;
  }

  function parseJson(text) {
    const raw = String(text || '').trim();
    if (!raw || raw === '--' || raw.startsWith('No chat stance debug')) return null;
    try {
      const parsed = JSON.parse(raw);
      return asObject(parsed);
    } catch (_err) {
      return null;
    }
  }

  function firstPresent(values) {
    for (let i = 0; i < values.length; i += 1) {
      const value = values[i];
      if (value === null || value === undefined) continue;
      if (typeof value === 'string' && !value.trim()) continue;
      if (Array.isArray(value) && !value.length) continue;
      if (typeof value === 'object' && !Array.isArray(value) && !Object.keys(value).length) continue;
      return value;
    }
    return null;
  }

  function projectionBundle(payload) {
    const root = asObject(payload) || {};
    return asObject(root.cognitive_projection) || asObject(asObject(root.raw)?.cognitive_projection) || null;
  }

  function normalizeProjectionBundle(bundle) {
    const b = asObject(bundle) || {};
    const sharedSpine = asObject(b.shared_spine) || {};
    const projectionDebug = asObject(b.projection_debug) || {};
    const projection = asObject(b.projection) || null;
    const anchors = asObject(projection && projection.anchors) || {};
    const anchorNames = Object.keys(anchors);
    const notes = asList(projectionDebug.notes).length ? asList(projectionDebug.notes) : asList(projection && projection.notes);
    const coldAnchors = asList(projectionDebug.cold_anchors).length ? asList(projectionDebug.cold_anchors) : asList(sharedSpine.cold_anchors);
    const degraded = asList(projectionDebug.degraded_producers).length ? asList(projectionDebug.degraded_producers) : asList(sharedSpine.degraded_producers);
    return {
      sharedSpine,
      projectionDebug,
      projection,
      anchors,
      anchorNames,
      notes,
      coldAnchors,
      degraded,
      used: Boolean(sharedSpine.enabled && sharedSpine.beliefs_present),
      present: Boolean(projectionDebug.present),
      projectionId: projectionDebug.projection_id || (projection && projection.projection_id),
      itemCount: projectionDebug.item_count ?? (projection && projection.item_count) ?? 0,
      anchorCount: projectionDebug.anchor_count ?? anchorNames.length,
      lineage: asList(sharedSpine.lineage),
    };
  }

  function makeEl(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined && text !== null) el.textContent = String(text);
    return el;
  }

  function metric(label, value) {
    const card = makeEl('div', 'rounded-lg border border-gray-800 bg-gray-950/60 px-3 py-2');
    card.appendChild(makeEl('div', 'text-[10px] uppercase tracking-wide text-gray-500', label));
    card.appendChild(makeEl('div', 'mt-1 text-[11px] text-gray-100 break-words', short(value)));
    return card;
  }

  function compactProjectionItems(model, limit = 8) {
    const out = [];
    const anchors = asObject(model.anchors) || {};
    Object.keys(anchors).forEach((anchor) => {
      const items = asList(asObject(anchors[anchor])?.items);
      items.forEach((item) => {
        if (!asObject(item)) return;
        out.push({ anchor, ...item });
      });
    });
    return out.slice(0, limit);
  }

  function normalizeLegacyChatStance(payload) {
    const root = asObject(payload) || {};
    const raw = asObject(root.raw) || {};
    const overview = asObject(root.overview) || asObject(raw.overview) || {};
    const sourceInputs = asObject(root.source_inputs) || asObject(raw.source_inputs) || {};
    const synthesizedBrief = firstPresent([
      root.synthesized_brief,
      raw.synthesized_brief,
      root.chat_stance_brief,
      raw.chat_stance_brief,
      root.brief,
      raw.brief,
    ]);
    const finalPromptContract = firstPresent([
      root.final_prompt_contract,
      raw.final_prompt_contract,
      root.prompt_contract,
      raw.prompt_contract,
    ]);
    const categories = asList(overview.categories_present);
    return {
      present: Boolean(synthesizedBrief || finalPromptContract || categories.length || Object.keys(sourceInputs).length),
      categories,
      fallbackInvoked: Boolean(overview.fallback_invoked),
      normalizedApplied: Boolean(overview.normalized_applied),
      qualityModified: Boolean(overview.quality_enforcement_modified),
      semanticFallback: Boolean(overview.semantic_fallback),
      sourceInputCount: Object.keys(sourceInputs).length,
      synthesizedBrief,
      finalPromptContract,
    };
  }

  function normalizeMindHandoff(payload) {
    const root = asObject(payload) || {};
    const raw = asObject(root.raw) || {};
    const sourceInputs = asObject(root.source_inputs) || asObject(raw.source_inputs) || {};
    const finalPromptContract = asObject(root.final_prompt_contract) || asObject(raw.final_prompt_contract) || {};
    const mindHandoff = firstPresent([
      root.mind_handoff,
      raw.mind_handoff,
      sourceInputs.mind_handoff,
      sourceInputs.mind,
      finalPromptContract.mind_handoff,
    ]);
    const quality = firstPresent([
      root.mind_quality,
      raw.mind_quality,
      sourceInputs.mind_quality,
      finalPromptContract.mind_quality,
      asObject(mindHandoff)?.mind_quality,
    ]);
    const runOk = firstPresent([
      root.mind_run_ok,
      raw.mind_run_ok,
      sourceInputs.mind_run_ok,
      finalPromptContract.mind_run_ok,
    ]);
    const contractOnly = firstPresent([
      root.mind_contract_only,
      raw.mind_contract_only,
      sourceInputs.mind_contract_only,
      finalPromptContract.mind_contract_only,
    ]);
    const skipStance = firstPresent([
      root.mind_skip_stance_synthesis,
      raw.mind_skip_stance_synthesis,
      sourceInputs.mind_skip_stance_synthesis,
      finalPromptContract.mind_skip_stance_synthesis,
    ]);
    const handoffObject = asObject(mindHandoff) || {};
    const summary = firstPresent([
      handoffObject.summary_one_paragraph,
      handoffObject.summary,
      handoffObject.stance_summary,
      sourceInputs.mind_summary,
    ]);
    const projectionItemCount = firstPresent([
      root['mind.cognitive_projection_item_count'],
      raw['mind.cognitive_projection_item_count'],
      sourceInputs['mind.cognitive_projection_item_count'],
      finalPromptContract['mind.cognitive_projection_item_count'],
      asObject(mindHandoff)?.machine_contract?.['mind.cognitive_projection_item_count'],
    ]);
    const projectionStarved = firstPresent([
      root['mind.projection_starved'],
      raw['mind.projection_starved'],
      sourceInputs['mind.projection_starved'],
      finalPromptContract['mind.projection_starved'],
      asObject(mindHandoff)?.machine_contract?.['mind.projection_starved'],
    ]);
    const degradedContractOnly = quality === 'fallback_contract_only' || contractOnly === true;
    const visiblyDegraded = Boolean(
      degradedContractOnly
      && (projectionStarved === true || Number(projectionItemCount) === 0),
    );
    const orchBeforeExec = firstPresent([
      root['mind.projection_resolution.orch_before_exec'],
      raw['mind.projection_resolution.orch_before_exec'],
      sourceInputs['mind.projection_resolution.orch_before_exec'],
      finalPromptContract['mind.projection_resolution.orch_before_exec'],
      asObject(mindHandoff)?.machine_contract?.['mind.projection_resolution.orch_before_exec'],
      root.mind_orch_before_exec,
      raw.mind_orch_before_exec,
      sourceInputs.mind_orch_before_exec,
    ]);
    return {
      present: Boolean(mindHandoff || quality || runOk !== null || contractOnly !== null || skipStance !== null),
      quality,
      runOk,
      contractOnly,
      skipStance,
      summary,
      handoff: mindHandoff,
      projectionItemCount,
      projectionStarved,
      visiblyDegraded,
      orchBeforeExec,
    };
  }

  function normalizeMindShadowSynthesis(payload) {
    const root = asObject(payload) || {};
    const raw = asObject(root.raw) || {};
    const sourceInputs = asObject(root.source_inputs) || asObject(raw.source_inputs) || {};
    const finalPromptContract = asObject(root.final_prompt_contract) || asObject(raw.final_prompt_contract) || {};
    const rootHandoff = asObject(root.mind_handoff) || {};
    const rawHandoff = asObject(raw.mind_handoff) || {};
    const sourceHandoff = asObject(sourceInputs.mind_handoff) || asObject(sourceInputs.mind) || {};
    const contractHandoff = asObject(finalPromptContract.mind_handoff) || {};
    const shadow = firstPresent([
      root.mind_shadow_synthesis,
      raw.mind_shadow_synthesis,
      sourceInputs.mind_shadow_synthesis,
      finalPromptContract.mind_shadow_synthesis,
      rootHandoff.shadow_synthesis,
      rawHandoff.shadow_synthesis,
      sourceHandoff.shadow_synthesis,
      contractHandoff.shadow_synthesis,
    ]);
    const shadowObject = asObject(shadow) || {};
    const presentFlag = firstPresent([
      root.mind_shadow_synthesis_present,
      raw.mind_shadow_synthesis_present,
      sourceInputs.mind_shadow_synthesis_present,
      finalPromptContract.mind_shadow_synthesis_present,
      shadowObject.present,
    ]);
    const authorizedForStanceSkip = firstPresent([
      root.mind_authorized_for_stance_skip,
      raw.mind_authorized_for_stance_skip,
      sourceInputs.mind_authorized_for_stance_skip,
      finalPromptContract.mind_authorized_for_stance_skip,
      shadowObject.authorized_for_stance_skip,
    ]);
    return {
      present: Boolean(shadow || presentFlag),
      schemaVersion: shadowObject.schema_version || '--',
      authorizedForStanceSkip,
      confidence: shadowObject.confidence,
      attentionFocus: asList(shadowObject.attention_focus),
      curiosityCandidate: asList(shadowObject.curiosity_candidate),
      relationshipFrame: shadowObject.relationship_frame,
      projectionRefsUsed: asList(shadowObject.projection_refs_used),
      hazards: asList(shadowObject.hazards),
      rationale: shadowObject.rationale,
      stanceCandidate: asObject(shadowObject.stance_candidate),
      shadow: shadowObject,
    };
  }

  function renderProjectionCard(model, opts = {}) {
    const card = makeEl('section', 'rounded-xl border border-cyan-500/30 bg-cyan-500/5 p-3 space-y-3');
    card.id = opts.modal ? MODAL_CARD_ID : INLINE_CARD_ID;

    const header = makeEl('div', 'flex items-center justify-between gap-2');
    header.appendChild(makeEl('div', 'text-[10px] uppercase tracking-wide text-cyan-200', 'Cognitive Projection'));
    const status = model.used ? 'shared spine used' : 'not active';
    header.appendChild(makeEl('div', model.used ? 'text-[10px] text-emerald-200' : 'text-[10px] text-amber-200', status));
    card.appendChild(header);

    const grid = makeEl('div', 'grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4');
    grid.appendChild(metric('Shared spine', model.used ? 'yes' : 'no'));
    grid.appendChild(metric('Projection id', model.projectionId || '--'));
    grid.appendChild(metric('Items', model.itemCount));
    grid.appendChild(metric('Anchors', model.anchorCount));
    grid.appendChild(metric('Cold anchors', model.coldAnchors.length ? model.coldAnchors.join(', ') : '--'));
    grid.appendChild(metric('Degraded producers', model.degraded.length ? model.degraded.join(', ') : '--'));
    grid.appendChild(metric('Lineage', model.lineage.length ? model.lineage.slice(0, 3).join(' · ') : '--'));
    grid.appendChild(metric('Notes', model.notes.length ? model.notes.join(', ') : '--'));
    card.appendChild(grid);

    if (opts.modal && model.projection) {
      const items = compactProjectionItems(model.projection, 10);
      const detail = makeEl('details', 'rounded-lg border border-cyan-500/20 bg-gray-950/50 p-2');
      detail.open = true;
      const summary = makeEl('summary', 'cursor-pointer text-[11px] text-cyan-100', `Top projection items (${items.length})`);
      detail.appendChild(summary);
      if (!items.length) {
        detail.appendChild(makeEl('div', 'mt-2 text-[11px] text-gray-400', 'No compact projection items were present.'));
      } else {
        const list = makeEl('div', 'mt-2 space-y-2');
        items.forEach((item) => {
          const row = makeEl('div', 'rounded border border-gray-800 bg-gray-900/50 px-2 py-1 text-[11px] text-gray-200');
          const label = short(item.label || item.summary || item.node_id, 'projection item');
          const meta = [item.anchor, item.bucket, item.node_kind, item.salience !== undefined ? `sal=${item.salience}` : null, item.confidence !== undefined ? `conf=${item.confidence}` : null]
            .filter(Boolean)
            .join(' · ');
          row.appendChild(makeEl('div', 'text-gray-100', label));
          row.appendChild(makeEl('div', 'text-[10px] text-gray-500', meta));
          list.appendChild(row);
        });
        detail.appendChild(list);
      }
      card.appendChild(detail);
    }

    if (opts.modal) {
      const raw = makeEl('details', 'rounded-lg border border-gray-800 bg-gray-950/50 p-2');
      raw.appendChild(makeEl('summary', 'cursor-pointer text-[10px] uppercase tracking-wide text-gray-500', 'Raw cognitive projection'));
      const pre = makeEl('pre', 'mt-2 max-h-80 overflow-y-auto whitespace-pre-wrap break-words text-[10px] text-gray-300');
      pre.textContent = JSON.stringify(model.projection || model.projectionDebug || {}, null, 2);
      raw.appendChild(pre);
      card.appendChild(raw);
    }
    return card;
  }

  function comparisonColumn(title, status, rows, toneClass) {
    const section = makeEl('section', `rounded-xl border ${toneClass.border} ${toneClass.bg} p-3 space-y-2`);
    const header = makeEl('div', 'flex items-center justify-between gap-2');
    header.appendChild(makeEl('div', `text-[10px] uppercase tracking-wide ${toneClass.title}`, title));
    header.appendChild(makeEl('div', 'text-[10px] text-gray-300', status));
    section.appendChild(header);
    rows.forEach(([label, value]) => {
      const row = makeEl('div', 'rounded-lg border border-gray-800 bg-gray-950/50 px-2 py-1');
      row.appendChild(makeEl('div', 'text-[10px] uppercase tracking-wide text-gray-500', label));
      row.appendChild(makeEl('div', 'mt-0.5 text-[11px] text-gray-100 break-words', short(value)));
      section.appendChild(row);
    });
    return section;
  }

  function renderComparisonCard(payload, opts = {}) {
    const bundle = projectionBundle(payload);
    const projection = bundle ? normalizeProjectionBundle(bundle) : null;
    const legacy = normalizeLegacyChatStance(payload);
    const mind = normalizeMindHandoff(payload);
    const shadow = normalizeMindShadowSynthesis(payload);
    const execRichMindStarved = Boolean(
      projection
      && Number(projection.itemCount) > 0
      && mind.visiblyDegraded,
    );
    const card = makeEl('section', 'rounded-xl border border-fuchsia-500/30 bg-fuchsia-500/5 p-3 space-y-3');
    card.id = opts.modal ? MODAL_COMPARE_ID : INLINE_COMPARE_ID;

    const header = makeEl('div', 'flex items-center justify-between gap-2');
    header.appendChild(makeEl('div', 'text-[10px] uppercase tracking-wide text-fuchsia-200', 'Cognitive Comparison'));
    header.appendChild(makeEl('div', 'text-[10px] text-gray-400', 'read-only · no promotion'));
    card.appendChild(header);
    if (execRichMindStarved) {
      card.appendChild(
        makeEl(
          'div',
          'rounded-lg border border-rose-500/35 bg-rose-500/10 px-3 py-2 text-[11px] text-rose-100',
          'Mind starved before Exec rebuild — comparison projection is from post-Exec materialization; Mind snapshot used an empty Orch-time shell.',
        ),
      );
    }

    const grid = makeEl('div', 'grid grid-cols-1 gap-3 xl:grid-cols-4');
    grid.appendChild(comparisonColumn(
      'Shared projection',
      projection ? (projection.used ? 'active' : 'present / inactive') : 'absent',
      [
        ['projection id', projection && projection.projectionId],
        ['items / anchors', projection ? `${projection.itemCount} / ${projection.anchorCount}` : '--'],
        ['cold anchors', projection && projection.coldAnchors.length ? projection.coldAnchors.join(', ') : '--'],
        ['degraded producers', projection && projection.degraded.length ? projection.degraded.join(', ') : '--'],
      ],
      { border: 'border-cyan-500/25', bg: 'bg-cyan-500/5', title: 'text-cyan-200' },
    ));
    grid.appendChild(comparisonColumn(
      'Legacy ChatStanceBrief',
      legacy.present ? 'present' : 'absent',
      [
        ['categories', legacy.categories.length ? legacy.categories.join(', ') : '--'],
        ['source input count', legacy.sourceInputCount],
        ['fallback / semantic fallback', `${legacy.fallbackInvoked ? 'yes' : 'no'} / ${legacy.semanticFallback ? 'yes' : 'no'}`],
        ['quality modified', legacy.qualityModified ? 'yes' : 'no'],
      ],
      { border: 'border-indigo-500/25', bg: 'bg-indigo-500/5', title: 'text-indigo-200' },
    ));
    grid.appendChild(comparisonColumn(
      'Mind handoff / quality',
      mind.visiblyDegraded ? 'degraded / contract-only' : (mind.present ? 'present' : 'absent'),
      [
        ['mind quality', mind.quality || '--'],
        ['run ok', mind.runOk === null || mind.runOk === undefined ? '--' : String(mind.runOk)],
        ['contract only', mind.contractOnly === null || mind.contractOnly === undefined ? '--' : String(mind.contractOnly)],
        ['projection items', mind.projectionItemCount === null || mind.projectionItemCount === undefined ? '--' : String(mind.projectionItemCount)],
        ['projection starved', mind.projectionStarved === null || mind.projectionStarved === undefined ? '--' : String(mind.projectionStarved)],
        ['skip stance synthesis', mind.skipStance === null || mind.skipStance === undefined ? '--' : String(mind.skipStance)],
      ],
      {
        border: mind.visiblyDegraded ? 'border-rose-500/35' : 'border-emerald-500/25',
        bg: mind.visiblyDegraded ? 'bg-rose-500/10' : 'bg-emerald-500/5',
        title: mind.visiblyDegraded ? 'text-rose-200' : 'text-emerald-200',
      },
    ));
    grid.appendChild(comparisonColumn(
      'Mind shadow synthesis',
      shadow.present ? 'present' : 'absent',
      [
        ['schema', shadow.schemaVersion || '--'],
        ['authorized for skip', shadow.authorizedForStanceSkip === null || shadow.authorizedForStanceSkip === undefined ? '--' : String(shadow.authorizedForStanceSkip)],
        ['confidence', shadow.confidence === null || shadow.confidence === undefined ? '--' : shadow.confidence],
        ['attention focus', shadow.attentionFocus.length ? shadow.attentionFocus.slice(0, 3).join(' · ') : '--'],
        ['curiosity candidate', shadow.curiosityCandidate.length ? shadow.curiosityCandidate.slice(0, 2).join(' · ') : '--'],
        ['projection refs', shadow.projectionRefsUsed.length ? shadow.projectionRefsUsed.slice(0, 4).join(' · ') : '--'],
        ['hazards', shadow.hazards.length ? shadow.hazards.slice(0, 3).join(' · ') : '--'],
      ],
      { border: 'border-amber-500/25', bg: 'bg-amber-500/5', title: 'text-amber-200' },
    ));
    card.appendChild(grid);

    if (opts.modal) {
      const raw = makeEl('details', 'rounded-lg border border-gray-800 bg-gray-950/50 p-2');
      raw.appendChild(makeEl('summary', 'cursor-pointer text-[10px] uppercase tracking-wide text-gray-500', 'Raw comparison sources'));
      const pre = makeEl('pre', 'mt-2 max-h-80 overflow-y-auto whitespace-pre-wrap break-words text-[10px] text-gray-300');
      pre.textContent = JSON.stringify({
        cognitive_projection: bundle || null,
        legacy_chat_stance: {
          synthesized_brief: legacy.synthesizedBrief || null,
          final_prompt_contract: legacy.finalPromptContract || null,
        },
        mind: {
          quality: mind.quality || null,
          run_ok: mind.runOk,
          contract_only: mind.contractOnly,
          skip_stance_synthesis: mind.skipStance,
          summary: mind.summary || null,
          handoff: mind.handoff || null,
        },
        mind_shadow_synthesis: {
          present: shadow.present,
          authorized_for_stance_skip: shadow.authorizedForStanceSkip,
          confidence: shadow.confidence,
          attention_focus: shadow.attentionFocus,
          curiosity_candidate: shadow.curiosityCandidate,
          relationship_frame: shadow.relationshipFrame,
          projection_refs_used: shadow.projectionRefsUsed,
          hazards: shadow.hazards,
          rationale: shadow.rationale,
          stance_candidate: shadow.stanceCandidate || null,
          raw: shadow.shadow || null,
        },
      }, null, 2);
      raw.appendChild(pre);
      card.appendChild(raw);
    }
    return card;
  }

  function payloadFromRawPre() {
    const rawPre = document.getElementById('chatStanceDebugRaw');
    return parseJson(rawPre ? rawPre.textContent : '');
  }

  function removeCards() {
    [INLINE_CARD_ID, MODAL_CARD_ID, INLINE_COMPARE_ID, MODAL_COMPARE_ID].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.remove();
    });
  }

  function renderFromPayload(payload) {
    const modelPayload = asObject(payload);
    removeCards();
    if (!modelPayload || !Object.keys(modelPayload).length) return false;

    const bundle = projectionBundle(modelPayload);
    const projection = bundle ? normalizeProjectionBundle(bundle) : null;
    const overview = document.getElementById('chatStanceDebugOverview');
    if (overview) {
      if (projection) overview.appendChild(renderProjectionCard(projection, { modal: false }));
      overview.appendChild(renderComparisonCard(modelPayload, { modal: false }));
    }
    const modalBody = document.getElementById('chatStanceDebugModalBody');
    if (modalBody) {
      const first = modalBody.firstChild || null;
      modalBody.insertBefore(renderComparisonCard(modelPayload, { modal: true }), first);
      if (projection) modalBody.insertBefore(renderProjectionCard(projection, { modal: true }), modalBody.firstChild || null);
    }
    return true;
  }

  function refresh() {
    return renderFromPayload(payloadFromRawPre());
  }

  function attach() {
    const rawPre = document.getElementById('chatStanceDebugRaw');
    if (rawPre && typeof MutationObserver !== 'undefined') {
      const observer = new MutationObserver(() => refresh());
      observer.observe(rawPre, { childList: true, characterData: true, subtree: true });
    }
    const modalButton = document.getElementById('chatStanceDebugOpenModal');
    if (modalButton) {
      modalButton.addEventListener('click', () => {
        setTimeout(() => refresh(), 0);
        setTimeout(() => refresh(), 50);
      });
    }
    refresh();
  }

  const api = { attach, refresh, normalizeProjectionBundle, normalizeLegacyChatStance, normalizeMindHandoff, normalizeMindShadowSynthesis, renderFromPayload };
  global.OrionChatStanceProjectionInspect = api;
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', attach);
    else attach();
  }
})(typeof window !== 'undefined' ? window : globalThis);

(function (global) {
  const INLINE_EVAL_ID = 'chatStanceMindShadowEvaluationCard';
  const MODAL_EVAL_ID = 'chatStanceMindShadowEvaluationModalCard';

  function asObject(value) {
    return value && typeof value === 'object' && !Array.isArray(value) ? value : null;
  }

  function asList(value) {
    return Array.isArray(value) ? value : [];
  }

  function short(value, fallback = '--') {
    const text = String(value === null || value === undefined ? '' : value).trim();
    return text || fallback;
  }

  function makeEl(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined && text !== null) el.textContent = String(text);
    return el;
  }

  function firstPresent(values) {
    for (let i = 0; i < values.length; i += 1) {
      const value = values[i];
      if (value === null || value === undefined) continue;
      if (typeof value === 'string' && !value.trim()) continue;
      if (Array.isArray(value) && !value.length) continue;
      if (typeof value === 'object' && !Array.isArray(value) && !Object.keys(value).length) continue;
      return value;
    }
    return null;
  }

  function parseJson(text) {
    const raw = String(text || '').trim();
    if (!raw || raw === '--' || raw.startsWith('No chat stance debug')) return null;
    try {
      const parsed = JSON.parse(raw);
      return asObject(parsed);
    } catch (_err) {
      return null;
    }
  }

  function tokensFrom(value) {
    const seen = new Set();
    const visit = (item) => {
      if (item === null || item === undefined) return;
      if (Array.isArray(item)) {
        item.forEach(visit);
        return;
      }
      if (typeof item === 'object') {
        Object.keys(item).forEach((key) => visit(item[key]));
        return;
      }
      String(item).toLowerCase().split(/[^a-z0-9_]+/).forEach((token) => {
        if (token.length >= 3) seen.add(token);
      });
    };
    visit(value);
    return seen;
  }

  function normalizeLegacy(payload) {
    const root = asObject(payload) || {};
    const raw = asObject(root.raw) || {};
    const overview = asObject(root.overview) || asObject(raw.overview) || {};
    const sourceInputs = asObject(root.source_inputs) || asObject(raw.source_inputs) || {};
    const synthesizedBrief = firstPresent([
      root.synthesized_brief,
      raw.synthesized_brief,
      root.chat_stance_brief,
      raw.chat_stance_brief,
      root.brief,
      raw.brief,
    ]);
    const finalPromptContract = firstPresent([
      root.final_prompt_contract,
      raw.final_prompt_contract,
      root.prompt_contract,
      raw.prompt_contract,
    ]);
    return {
      present: Boolean(synthesizedBrief || finalPromptContract || asList(overview.categories_present).length || Object.keys(sourceInputs).length),
      categories: asList(overview.categories_present),
      synthesizedBrief,
      finalPromptContract,
      sourceInputs,
    };
  }

  function normalizeShadow(payload) {
    const root = asObject(payload) || {};
    const raw = asObject(root.raw) || {};
    const sourceInputs = asObject(root.source_inputs) || asObject(raw.source_inputs) || {};
    const finalPromptContract = asObject(root.final_prompt_contract) || asObject(raw.final_prompt_contract) || {};
    const rootHandoff = asObject(root.mind_handoff) || {};
    const rawHandoff = asObject(raw.mind_handoff) || {};
    const sourceHandoff = asObject(sourceInputs.mind_handoff) || asObject(sourceInputs.mind) || {};
    const contractHandoff = asObject(finalPromptContract.mind_handoff) || {};
    const shadow = firstPresent([
      root.mind_shadow_synthesis,
      raw.mind_shadow_synthesis,
      sourceInputs.mind_shadow_synthesis,
      finalPromptContract.mind_shadow_synthesis,
      rootHandoff.shadow_synthesis,
      rawHandoff.shadow_synthesis,
      sourceHandoff.shadow_synthesis,
      contractHandoff.shadow_synthesis,
    ]);
    const shadowObject = asObject(shadow) || {};
    const presentFlag = firstPresent([
      root.mind_shadow_synthesis_present,
      raw.mind_shadow_synthesis_present,
      sourceInputs.mind_shadow_synthesis_present,
      finalPromptContract.mind_shadow_synthesis_present,
      shadowObject.present,
    ]);
    const authorizedForStanceSkip = firstPresent([
      root.mind_authorized_for_stance_skip,
      raw.mind_authorized_for_stance_skip,
      sourceInputs.mind_authorized_for_stance_skip,
      finalPromptContract.mind_authorized_for_stance_skip,
      shadowObject.authorized_for_stance_skip,
    ]);
    return {
      present: Boolean(shadow || presentFlag),
      authorizedForStanceSkip,
      confidence: shadowObject.confidence,
      attentionFocus: asList(shadowObject.attention_focus),
      curiosityCandidate: asList(shadowObject.curiosity_candidate),
      relationshipFrame: shadowObject.relationship_frame,
      projectionRefsUsed: asList(shadowObject.projection_refs_used),
      hazards: asList(shadowObject.hazards),
      rationale: shadowObject.rationale,
      stanceCandidate: asObject(shadowObject.stance_candidate),
      raw: shadowObject,
    };
  }

  function evaluateMindShadow(payload) {
    const legacy = normalizeLegacy(payload);
    const shadow = normalizeShadow(payload);
    const shadowTokens = tokensFrom([
      shadow.attentionFocus,
      shadow.curiosityCandidate,
      shadow.relationshipFrame,
      shadow.projectionRefsUsed,
      shadow.hazards,
      shadow.rationale,
      shadow.stanceCandidate,
    ]);
    const categoryHits = legacy.categories.filter((category) => shadowTokens.has(String(category).toLowerCase()));
    const categoryGaps = legacy.categories.filter((category) => !shadowTokens.has(String(category).toLowerCase()));
    const authorityFlag = shadow.authorizedForStanceSkip === true || String(shadow.authorizedForStanceSkip).toLowerCase() === 'true';
    const notices = [];
    if (!shadow.present) notices.push('No Mind shadow candidate emitted for this turn.');
    if (!legacy.present) notices.push('Legacy ChatStanceBrief comparison source is absent.');
    if (authorityFlag) notices.push('Unexpected authority flag observed; shadow remains display-only in Hub.');
    if (shadow.hazards.length) notices.push(`${shadow.hazards.length} shadow hazard(s) surfaced.`);
    if (shadow.projectionRefsUsed.length) notices.push(`${shadow.projectionRefsUsed.length} projection ref(s) used by shadow.`);
    if (!notices.length) notices.push('Read-only comparison available; legacy stance remains authoritative.');

    return {
      status: shadow.present ? (legacy.present ? 'comparison available' : 'shadow only') : 'no shadow candidate',
      authorityBoundary: authorityFlag ? 'inspect upstream flag' : 'read-only / no promotion',
      confidence: shadow.confidence,
      legacyCategoryHits: categoryHits,
      legacyCategoryGaps: categoryGaps,
      stanceCandidateKeys: Object.keys(shadow.stanceCandidate || {}),
      projectionRefsUsed: shadow.projectionRefsUsed,
      hazards: shadow.hazards,
      notices,
      rationale: shadow.rationale,
      legacy,
      shadow,
    };
  }

  function renderEvaluationCard(evaluation, opts = {}) {
    const card = makeEl('section', 'rounded-xl border border-rose-500/25 bg-rose-500/5 p-3 space-y-3');
    card.id = opts.modal ? MODAL_EVAL_ID : INLINE_EVAL_ID;
    const header = makeEl('div', 'flex items-center justify-between gap-2');
    header.appendChild(makeEl('div', 'text-[10px] uppercase tracking-wide text-rose-200', 'Mind shadow evaluation'));
    header.appendChild(makeEl('div', 'text-[10px] text-gray-400', 'operator comparison · read-only'));
    card.appendChild(header);

    const grid = makeEl('div', 'grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-4');
    [
      ['Status', evaluation.status],
      ['Authority boundary', evaluation.authorityBoundary],
      ['Confidence', evaluation.confidence === null || evaluation.confidence === undefined ? '--' : evaluation.confidence],
      ['Legacy category hits', evaluation.legacyCategoryHits.length ? evaluation.legacyCategoryHits.join(', ') : '--'],
      ['Legacy category gaps', evaluation.legacyCategoryGaps.length ? evaluation.legacyCategoryGaps.join(', ') : '--'],
      ['Stance candidate keys', evaluation.stanceCandidateKeys.length ? evaluation.stanceCandidateKeys.slice(0, 6).join(', ') : '--'],
      ['Projection refs', evaluation.projectionRefsUsed.length ? evaluation.projectionRefsUsed.slice(0, 6).join(', ') : '--'],
      ['Hazards', evaluation.hazards.length ? evaluation.hazards.slice(0, 4).join(' · ') : '--'],
    ].forEach(([label, value]) => {
      const row = makeEl('div', 'rounded-lg border border-gray-800 bg-gray-950/50 px-2 py-1');
      row.appendChild(makeEl('div', 'text-[10px] uppercase tracking-wide text-gray-500', label));
      row.appendChild(makeEl('div', 'mt-0.5 text-[11px] text-gray-100 break-words', short(value)));
      grid.appendChild(row);
    });
    card.appendChild(grid);

    const noticeList = makeEl('div', 'space-y-1');
    evaluation.notices.slice(0, 5).forEach((notice) => {
      noticeList.appendChild(makeEl('div', 'rounded border border-rose-500/20 bg-gray-950/40 px-2 py-1 text-[11px] text-gray-200', notice));
    });
    card.appendChild(noticeList);

    if (opts.modal) {
      const raw = makeEl('details', 'rounded-lg border border-gray-800 bg-gray-950/50 p-2');
      raw.appendChild(makeEl('summary', 'cursor-pointer text-[10px] uppercase tracking-wide text-gray-500', 'Raw Mind shadow evaluation'));
      const pre = makeEl('pre', 'mt-2 max-h-80 overflow-y-auto whitespace-pre-wrap break-words text-[10px] text-gray-300');
      pre.textContent = JSON.stringify(evaluation, null, 2);
      raw.appendChild(pre);
      card.appendChild(raw);
    }
    return card;
  }

  function payloadFromRawPre() {
    const rawPre = document.getElementById('chatStanceDebugRaw');
    return parseJson(rawPre ? rawPre.textContent : '');
  }

  function removeCards() {
    [INLINE_EVAL_ID, MODAL_EVAL_ID].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.remove();
    });
  }

  function insertAfter(parent, anchorId, card) {
    const anchor = document.getElementById(anchorId);
    if (anchor && anchor.parentNode === parent) {
      parent.insertBefore(card, anchor.nextSibling);
      return;
    }
    parent.appendChild(card);
  }

  function renderFromPayload(payload) {
    const modelPayload = asObject(payload);
    removeCards();
    if (!modelPayload || !Object.keys(modelPayload).length) return false;
    const evaluation = evaluateMindShadow(modelPayload);
    const overview = document.getElementById('chatStanceDebugOverview');
    if (overview) insertAfter(overview, 'chatStanceCognitiveComparisonInspectCard', renderEvaluationCard(evaluation, { modal: false }));
    const modalBody = document.getElementById('chatStanceDebugModalBody');
    if (modalBody) insertAfter(modalBody, 'chatStanceCognitiveComparisonInspectModalCard', renderEvaluationCard(evaluation, { modal: true }));
    return true;
  }

  function refresh() {
    return renderFromPayload(payloadFromRawPre());
  }

  function attach() {
    const rawPre = document.getElementById('chatStanceDebugRaw');
    if (rawPre && typeof MutationObserver !== 'undefined') {
      const observer = new MutationObserver(() => refresh());
      observer.observe(rawPre, { childList: true, characterData: true, subtree: true });
    }
    const modalButton = document.getElementById('chatStanceDebugOpenModal');
    if (modalButton) {
      modalButton.addEventListener('click', () => {
        setTimeout(() => refresh(), 0);
        setTimeout(() => refresh(), 50);
      });
    }
    refresh();
  }

  const api = { attach, refresh, evaluateMindShadow, renderEvaluationCard, renderFromPayload };
  global.OrionMindShadowEvaluation = api;
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', attach);
    else attach();
  }
})(typeof window !== 'undefined' ? window : globalThis);

(function (global) {
  const LANE_GROUNDED_SMALL = 'grounded_small';
  const LANE_QUICK = 'quick';
  const LANE_BRAIN = 'brain';

  function asObject(value) {
    return value && typeof value === 'object' && !Array.isArray(value) ? value : null;
  }

  function isChatPayload(payload) {
    return payload && typeof payload === 'object' && (
      Object.prototype.hasOwnProperty.call(payload, 'text_input')
      || Object.prototype.hasOwnProperty.call(payload, 'messages')
      || Object.prototype.hasOwnProperty.call(payload, 'audio')
    );
  }

  function getLane() {
    if (typeof document === 'undefined' || !document.body) return LANE_GROUNDED_SMALL;
    return document.body.dataset.orionChatLane || LANE_GROUNDED_SMALL;
  }

  function setLane(lane) {
    if (typeof document === 'undefined' || !document.body) return;
    document.body.dataset.orionChatLane = lane || LANE_GROUNDED_SMALL;
  }

  function normalizeOutboundPayload(payload) {
    if (!isChatPayload(payload)) return payload;
    const lane = getLane();
    const options = asObject(payload.options) ? { ...payload.options } : {};
    const verbs = Array.isArray(payload.verbs) ? payload.verbs.map((v) => String(v || '').trim()).filter(Boolean) : [];
    const isLegacyQuickStance = verbs.length === 1 && verbs[0] === 'chat_quick' && options.chat_quick_full_stance === true;
    const isDefaultBrainSend = String(payload.mode || '').toLowerCase() === 'brain' && verbs.length === 0;

    if (lane === LANE_GROUNDED_SMALL || isLegacyQuickStance) {
      payload.mode = 'brain';
      payload.verbs = [];
      options.llm_route = 'quick';
      delete options.chat_quick_full_stance;
      payload.options = options;
      payload.surface_context = {
        ...(asObject(payload.surface_context) || {}),
        hub_chat_lane: LANE_GROUNDED_SMALL,
      };
      payload.context = asObject(payload.context) || {};
      payload.context.metadata = asObject(payload.context.metadata) || {};
      payload.context.metadata.mind_enabled = true;
      return payload;
    }

    if (lane === LANE_QUICK && verbs.length === 1 && verbs[0] === 'chat_quick') {
      delete options.chat_quick_full_stance;
      payload.options = Object.keys(options).length ? options : undefined;
      return payload;
    }

    if (lane === LANE_BRAIN && isDefaultBrainSend) {
      delete options.llm_route;
      delete options.chat_quick_full_stance;
      payload.options = Object.keys(options).length ? options : undefined;
    }
    return payload;
  }

  function patchWebSocketSend() {
    if (!global.WebSocket || global.WebSocket.prototype._orionGroundedSmallPatched) return;
    const originalSend = global.WebSocket.prototype.send;
    global.WebSocket.prototype.send = function patchedSend(data) {
      if (typeof data === 'string') {
        try {
          const parsed = JSON.parse(data);
          normalizeOutboundPayload(parsed);
          return originalSend.call(this, JSON.stringify(parsed));
        } catch (_err) {
          return originalSend.call(this, data);
        }
      }
      return originalSend.call(this, data);
    };
    global.WebSocket.prototype._orionGroundedSmallPatched = true;
  }

  function patchFetch() {
    if (!global.fetch || global.fetch._orionGroundedSmallPatched) return;
    const originalFetch = global.fetch;
    global.fetch = function patchedFetch(input, init) {
      const url = typeof input === 'string' ? input : (input && input.url ? String(input.url) : '');
      if (url.includes('/api/chat') && init && typeof init.body === 'string') {
        try {
          const parsed = JSON.parse(init.body);
          normalizeOutboundPayload(parsed);
          init = { ...init, body: JSON.stringify(parsed) };
        } catch (_err) {
          // Leave non-JSON bodies untouched.
        }
      }
      return originalFetch.call(this, input, init);
    };
    global.fetch._orionGroundedSmallPatched = true;
  }

  function ensureBrainButton(group, afterNode) {
    let brain = document.getElementById('brainDeepModeBtn');
    if (brain) return brain;
    brain = document.createElement('button');
    brain.type = 'button';
    brain.id = 'brainDeepModeBtn';
    brain.className = 'mode-btn px-2 py-1 rounded bg-gray-700 text-gray-200 hover:bg-gray-600 transition-colors';
    brain.dataset.mode = 'brain';
    brain.dataset.llmRoute = 'chat';
    brain.title = 'Brain (stance + deep chat lane)';
    brain.textContent = 'Brain';
    if (afterNode && afterNode.parentNode === group) group.insertBefore(brain, afterNode.nextSibling);
    else group.appendChild(brain);
    return brain;
  }

  function configureModeRow() {
    setLane(LANE_GROUNDED_SMALL);
    const buttons = Array.from(document.querySelectorAll('.mode-btn'));
    const originalBrain = buttons.find((btn) => {
      const label = String(btn.textContent || '').trim().toLowerCase();
      return label === 'brain' && (btn.dataset.mode || '') === 'brain' && !btn.dataset.verbOverride;
    });
    if (originalBrain) {
      originalBrain.id = originalBrain.id || 'groundedSmallModeBtn';
      originalBrain.dataset.mode = 'brain';
      originalBrain.dataset.llmRoute = 'quick';
      originalBrain.textContent = 'Grounded Small';
      originalBrain.title = 'Grounded Small (stance + small/quick final lane)';
      ensureBrainButton(originalBrain.parentNode, originalBrain);
    }

    const quick = document.getElementById('quickModeBtn');
    if (quick) {
      quick.textContent = 'Quick';
      quick.title = 'Quick (small single-pass lane; no stance synthesis)';
    }
    document.querySelectorAll('.quick-variant-item').forEach((item) => {
      const variant = String(item.getAttribute('data-quick-variant') || '').trim();
      if (variant === 'fast') item.textContent = 'Quick';
      if (variant === 'stance') item.textContent = 'Grounded Small';
    });

    document.addEventListener('click', (event) => {
      const target = event.target && event.target.closest ? event.target.closest('button') : null;
      if (!target) return;
      if (target.id === 'brainDeepModeBtn') {
        setLane(LANE_BRAIN);
        return;
      }
      if (target.id === 'quickModeBtn') {
        setLane(LANE_QUICK);
        return;
      }
      if (target.classList && target.classList.contains('quick-variant-item')) {
        const variant = String(target.getAttribute('data-quick-variant') || '').trim();
        setLane(variant === 'stance' ? LANE_GROUNDED_SMALL : LANE_QUICK);
        return;
      }
      if (target.classList && target.classList.contains('mode-btn')) {
        const mode = String(target.dataset.mode || '').trim().toLowerCase();
        const verb = String(target.dataset.verbOverride || '').trim().toLowerCase();
        const route = String(target.dataset.llmRoute || '').trim().toLowerCase();
        if (mode === 'brain' && route === 'quick') setLane(LANE_GROUNDED_SMALL);
        else if (mode === 'brain' && !verb) setLane(LANE_BRAIN);
        else if (verb === 'chat_quick') setLane(LANE_QUICK);
        else setLane(mode || LANE_GROUNDED_SMALL);
      }
    }, true);
  }

  patchWebSocketSend();
  patchFetch();
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', configureModeRow);
    else configureModeRow();
  }
  global.OrionHubGroundedSmallLane = { normalizeOutboundPayload, configureModeRow };
})(typeof window !== 'undefined' ? window : globalThis);
