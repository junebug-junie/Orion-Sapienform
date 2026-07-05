(function (global) {
  const _liveClaudeSteps = new Map();
  const LIVE_ANCHOR_ID = 'conversation';

  function resolveAnchor(doc) {
    const root = doc || (typeof document !== 'undefined' ? document : null);
    return root && root.getElementById ? root.getElementById(LIVE_ANCHOR_ID) : null;
  }

  function ensurePanel(correlationId, doc) {
    const anchor = resolveAnchor(doc);
    if (!anchor) return null;
    const panelId = `claude-live-${correlationId}`;
    let panel = (doc || document).getElementById(panelId);
    if (panel) return panel;
    panel = (doc || document).createElement('div');
    panel.id = panelId;
    panel.className = 'agent-live-trace claude-live-trace';
    const heading = (doc || document).createElement('div');
    heading.className = 'agent-live-trace__heading';
    heading.textContent = 'Claude harness (live)';
    panel.appendChild(heading);
    const steps = (doc || document).createElement('div');
    steps.className = 'agent-live-trace__steps';
    panel.appendChild(steps);
    anchor.appendChild(panel);
    return panel;
  }

  function summarizeStep(step) {
    if (!step || typeof step !== 'object') return 'step';
    const raw = step.raw && typeof step.raw === 'object' ? step.raw : step;
    const type = String(step.type || raw.type || 'event');
    if (type === 'assistant') {
      const msg = raw.message && raw.message.content;
      if (Array.isArray(msg)) {
        const text = msg.filter((b) => b && b.type === 'text').map((b) => b.text).join('');
        if (text) return text.slice(0, 240);
      }
    }
    if (type === 'result') return String(raw.result || 'result').slice(0, 240);
    return type;
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
  }

  global.appendLiveClaudeStep = appendLiveClaudeStep;
  global.OrionClaudeTrace = { appendLiveClaudeStep };
})(typeof window !== 'undefined' ? window : globalThis);
