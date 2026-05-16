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

  function payloadFromRawPre() {
    const rawPre = document.getElementById('chatStanceDebugRaw');
    return parseJson(rawPre ? rawPre.textContent : '');
  }

  function renderFromPayload(payload) {
    const bundle = projectionBundle(payload);
    const existingInline = document.getElementById(INLINE_CARD_ID);
    const existingModal = document.getElementById(MODAL_CARD_ID);
    if (existingInline) existingInline.remove();
    if (existingModal) existingModal.remove();
    if (!bundle) return false;
    const model = normalizeProjectionBundle(bundle);
    const overview = document.getElementById('chatStanceDebugOverview');
    if (overview) {
      overview.appendChild(renderProjectionCard(model, { modal: false }));
    }
    const modalBody = document.getElementById('chatStanceDebugModalBody');
    if (modalBody) {
      modalBody.insertBefore(renderProjectionCard(model, { modal: true }), modalBody.firstChild || null);
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

  const api = { attach, refresh, normalizeProjectionBundle, renderFromPayload };
  global.OrionChatStanceProjectionInspect = api;
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', attach);
    else attach();
  }
})(typeof window !== 'undefined' ? window : globalThis);
