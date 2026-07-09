(function (global) {
  const _liveClaudeSteps = new Map();
  const LIVE_ANCHOR_ID = 'conversation';
  const PREVIEW_MAX = 240;

  function resolveAnchor(doc) {
    const root = doc || (typeof document !== 'undefined' ? document : null);
    return root && root.getElementById ? root.getElementById(LIVE_ANCHOR_ID) : null;
  }

  function clip(text, maxLen) {
    const limit = Number.isFinite(maxLen) ? maxLen : PREVIEW_MAX;
    const value = String(text || '').replace(/\s+/g, ' ').trim();
    if (!value) return '';
    return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`;
  }

  function basename(path) {
    const raw = String(path || '').trim();
    if (!raw) return '';
    const parts = raw.split(/[/\\]/);
    return parts[parts.length - 1] || raw;
  }

  function formatToolInput(name, input) {
    const tool = String(name || 'tool').trim() || 'tool';
    const args = input && typeof input === 'object' ? input : {};
    if (tool === 'Read' || tool === 'Write' || tool === 'Edit') {
      const path = args.file_path || args.path || args.notebook_path;
      return path ? `${tool} ${basename(path)}` : tool;
    }
    if (tool === 'Glob') {
      return args.pattern ? `Glob ${clip(args.pattern, 80)}` : 'Glob';
    }
    if (tool === 'Grep') {
      const bits = [];
      if (args.pattern) bits.push(`"${clip(args.pattern, 60)}"`);
      if (args.path) bits.push(`in ${basename(args.path)}`);
      return bits.length ? `Grep ${bits.join(' ')}` : 'Grep';
    }
    if (tool === 'Bash') {
      const cmd = args.command || args.cmd || args.script;
      return cmd ? `Bash ${clip(cmd, 120)}` : 'Bash';
    }
    if (tool === 'Task') {
      return args.description ? `Task ${clip(args.description, 100)}` : 'Task';
    }
    const firstKey = Object.keys(args)[0];
    if (firstKey) {
      return `${tool} ${clip(String(args[firstKey]), 100)}`;
    }
    return tool;
  }

  function summarizeContentBlocks(content) {
    if (!Array.isArray(content)) return '';
    const parts = [];
    content.forEach((block) => {
      if (!block || typeof block !== 'object') return;
      const blockType = String(block.type || '').trim();
      if (blockType === 'text' && block.text) {
        parts.push(clip(block.text));
        return;
      }
      if (blockType === 'tool_use') {
        parts.push(formatToolInput(block.name, block.input));
        return;
      }
      if (blockType === 'tool_result') {
        const body = typeof block.content === 'string'
          ? block.content
          : Array.isArray(block.content)
            ? block.content.filter((b) => b && b.type === 'text').map((b) => b.text).join('\n')
            : '';
        const size = body ? ` (${body.length} chars)` : '';
        parts.push(`tool result${size}${body ? `: ${clip(body, 120)}` : ''}`);
      }
    });
    return parts.filter(Boolean).join(' | ');
  }

  function summarizeStep(step) {
    if (!step || typeof step !== 'object') return 'step';
    const raw = step.raw && typeof step.raw === 'object' ? step.raw : step;
    const type = String(step.type || raw.type || 'event').trim() || 'event';

    if (type === 'assistant') {
      const msg = raw.message && typeof raw.message === 'object' ? raw.message : raw;
      const summary = summarizeContentBlocks(msg.content);
      if (summary) return summary;
      return 'assistant';
    }

    if (type === 'user') {
      const msg = raw.message && typeof raw.message === 'object' ? raw.message : raw;
      const summary = summarizeContentBlocks(msg.content);
      if (summary) return summary;
      const text = typeof msg.content === 'string' ? msg.content : raw.content;
      if (text) return clip(text);
      return 'user';
    }

    if (type === 'system') {
      const subtype = raw.subtype || raw.system_subtype;
      return subtype ? `system ${subtype}` : 'system';
    }

    if (type === 'tool_use') {
      return formatToolInput(raw.name, raw.input);
    }

    if (type === 'tool_result') {
      const body = typeof raw.content === 'string' ? raw.content : '';
      return body ? `tool result: ${clip(body, 160)}` : 'tool result';
    }

    if (type === 'result') {
      return clip(String(raw.result || 'result'), PREVIEW_MAX) || 'result';
    }

    return type;
  }

  function ensurePanel(correlationId, doc) {
    const anchor = resolveAnchor(doc);
    if (!anchor) return null;
    const panelId = `claude-live-${correlationId}`;
    let panel = (doc || document).getElementById(panelId);
    if (panel) return panel;
    panel = (doc || document).createElement('div');
    panel.id = panelId;
    panel.className = 'agent-live-trace claude-live-trace is-streaming';
    panel.dataset.correlationId = String(correlationId);
    const heading = (doc || document).createElement('div');
    heading.className = 'agent-live-trace__heading';
    heading.textContent = 'FCC harness (live)';
    panel.appendChild(heading);
    const steps = (doc || document).createElement('div');
    steps.className = 'agent-live-trace__steps';
    panel.appendChild(steps);
    anchor.appendChild(panel);
    return panel;
  }

  function formatHarnessHeading(stepCount, live) {
    const suffix = stepCount === 1 ? '' : 's';
    if (live) {
      return stepCount
        ? `FCC harness (live) · ${stepCount} step${suffix}`
        : 'FCC harness (live)';
    }
    return stepCount
      ? `FCC harness · ${stepCount} step${suffix}`
      : 'FCC harness';
  }

  function appendLiveClaudeStep(correlationId, step, doc) {
    if (!correlationId || !step) return;
    const list = _liveClaudeSteps.get(correlationId) || [];
    list.push(step);
    _liveClaudeSteps.set(correlationId, list);
    const panel = ensurePanel(correlationId, doc);
    if (!panel) return;
    const host = panel.querySelector('.agent-live-trace__steps');
    if (!host) return;
    const row = (doc || document).createElement('div');
    row.className = 'agent-live-trace__step';
    row.textContent = `#${list.length - 1} ${summarizeStep(step)}`;
    host.appendChild(row);
    host.scrollTop = host.scrollHeight;
    const heading = panel.querySelector('.agent-live-trace__heading');
    if (heading) {
      heading.textContent = formatHarnessHeading(list.length, true);
    }
    const anchor = resolveAnchor(doc);
    if (anchor && typeof anchor.scrollTop === 'number') {
      anchor.scrollTop = anchor.scrollHeight;
    }
  }

  function finalizeLiveClaudeTrace(correlationId, doc, beforeEl) {
    if (!correlationId) return null;
    const root = doc || (typeof document !== 'undefined' ? document : null);
    if (!root || typeof root.getElementById !== 'function') return null;
    const panelId = `claude-live-${correlationId}`;
    const panel = root.getElementById(panelId);
    if (!panel) return null;

    panel.classList.remove('is-streaming');
    panel.classList.add('is-complete');

    const list = _liveClaudeSteps.get(correlationId) || [];
    const heading = panel.querySelector('.agent-live-trace__heading');
    if (heading) {
      heading.textContent = formatHarnessHeading(list.length, false);
    }

    if (beforeEl && beforeEl.parentNode && panel.parentNode === beforeEl.parentNode) {
      beforeEl.parentNode.insertBefore(panel, beforeEl);
    }

    return panel;
  }

  global.appendLiveClaudeStep = appendLiveClaudeStep;
  global.finalizeLiveClaudeTrace = finalizeLiveClaudeTrace;
  global.OrionClaudeTrace = { appendLiveClaudeStep, finalizeLiveClaudeTrace, summarizeStep };
})(typeof window !== 'undefined' ? window : globalThis);
