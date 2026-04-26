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
