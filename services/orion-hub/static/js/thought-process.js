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

  const api = { selectThoughtProcess };
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
    return {
      present: Boolean(mindHandoff || quality || runOk !== null || contractOnly !== null || skipStance !== null),
      quality,
      runOk,
      contractOnly,
      skipStance,
      summary,
      handoff: mindHandoff,
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
    const card = makeEl('section', 'rounded-xl border border-fuchsia-500/30 bg-fuchsia-500/5 p-3 space-y-3');
    card.id = opts.modal ? MODAL_COMPARE_ID : INLINE_COMPARE_ID;

    const header = makeEl('div', 'flex items-center justify-between gap-2');
    header.appendChild(makeEl('div', 'text-[10px] uppercase tracking-wide text-fuchsia-200', 'Cognitive Comparison'));
    header.appendChild(makeEl('div', 'text-[10px] text-gray-400', 'read-only · no promotion'));
    card.appendChild(header);

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
      mind.present ? 'present' : 'absent',
      [
        ['mind quality', mind.quality || '--'],
        ['run ok', mind.runOk === null || mind.runOk === undefined ? '--' : String(mind.runOk)],
        ['contract only', mind.contractOnly === null || mind.contractOnly === undefined ? '--' : String(mind.contractOnly)],
        ['skip stance synthesis', mind.skipStance === null || mind.skipStance === undefined ? '--' : String(mind.skipStance)],
      ],
      { border: 'border-emerald-500/25', bg: 'bg-emerald-500/5', title: 'text-emerald-200' },
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
