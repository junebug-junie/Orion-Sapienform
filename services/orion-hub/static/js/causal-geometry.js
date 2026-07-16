(() => {
  const sections = {
    snapshot: document.getElementById('causalGeometrySnapshot'),
    history: document.getElementById('causalGeometryHistory'),
  };
  const proposalsContainer = document.getElementById('causalGeometryProposals');
  const sourceMeta = document.getElementById('causalGeometrySourceMeta');
  const actionError = document.getElementById('causalGeometryActionError');
  const refreshButton = document.getElementById('causalGeometryRefreshButton');

  async function fetchSection(path) {
    const res = await fetch(path, { headers: { 'Accept': 'application/json' } });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} for ${path}`);
    }
    return res.json();
  }

  function renderSection(target, payload) {
    if (!target) return;
    target.textContent = JSON.stringify(payload, null, 2);
  }

  function renderError(target, error) {
    if (!target) return;
    target.textContent = JSON.stringify({ degraded: true, error: String(error) }, null, 2);
  }

  function setActionError(error) {
    if (!actionError) return;
    if (!error) {
      actionError.classList.add('hidden');
      actionError.textContent = '';
      return;
    }
    actionError.classList.remove('hidden');
    actionError.textContent = `Action failed: ${String(error)}`;
  }

  async function postAction(path, body = {}) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body),
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = payload?.detail ? JSON.stringify(payload.detail) : `HTTP ${res.status}`;
      throw new Error(`${path}: ${detail}`);
    }
    return payload;
  }

  function renderProposalCard(proposal) {
    const card = document.createElement('div');
    card.className = 'rounded border border-gray-700 bg-gray-900 p-3 space-y-1';

    const title = document.createElement('div');
    title.className = 'text-sm font-semibold text-gray-100';
    title.textContent = proposal.proposal_id || '(unknown proposal)';
    card.appendChild(title);

    const meta = document.createElement('div');
    meta.className = 'text-[11px] text-gray-400';
    meta.textContent = `${proposal.mutation_class || ''} | ${proposal.target_surface || ''} | ${proposal.subject_ref || ''} | risk=${proposal.risk_tier || ''}`;
    card.appendChild(meta);

    if (proposal.rationale) {
      const rationale = document.createElement('div');
      rationale.className = 'text-xs text-gray-300';
      rationale.textContent = proposal.rationale;
      card.appendChild(rationale);
    }

    const actions = document.createElement('div');
    actions.className = 'flex gap-2 pt-1';

    const adoptButton = document.createElement('button');
    adoptButton.type = 'button';
    adoptButton.textContent = 'Adopt';
    adoptButton.className = 'px-2 py-1 text-xs rounded border border-emerald-700 bg-emerald-900/40 hover:bg-emerald-800/60';
    adoptButton.addEventListener('click', async () => {
      const operatorId = window.prompt('Operator id for adoption:');
      if (!operatorId) return;
      setActionError(null);
      try {
        await postAction(`/api/causal-geometry/proposals/${encodeURIComponent(proposal.proposal_id)}/adopt`, {
          operator_id: operatorId,
          rationale: '',
        });
        await refresh();
      } catch (error) {
        setActionError(error);
      }
    });
    actions.appendChild(adoptButton);

    const rejectButton = document.createElement('button');
    rejectButton.type = 'button';
    rejectButton.textContent = 'Reject';
    rejectButton.className = 'px-2 py-1 text-xs rounded border border-red-700 bg-red-900/40 hover:bg-red-800/60';
    rejectButton.addEventListener('click', async () => {
      const operatorId = window.prompt('Operator id for rejection:');
      if (!operatorId) return;
      setActionError(null);
      try {
        await postAction(`/api/causal-geometry/proposals/${encodeURIComponent(proposal.proposal_id)}/reject`, {
          operator_id: operatorId,
          reason: '',
        });
        await refresh();
      } catch (error) {
        setActionError(error);
      }
    });
    actions.appendChild(rejectButton);

    card.appendChild(actions);
    return card;
  }

  function renderProposals(payload) {
    if (!proposalsContainer) return;
    proposalsContainer.textContent = '';
    const items = Array.isArray(payload?.data) ? payload.data : [];
    if (items.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'text-xs text-gray-400';
      empty.textContent = 'No pending proposals.';
      proposalsContainer.appendChild(empty);
      return;
    }
    for (const proposal of items) {
      proposalsContainer.appendChild(renderProposalCard(proposal));
    }
  }

  function renderProposalsError(error) {
    if (!proposalsContainer) return;
    proposalsContainer.textContent = JSON.stringify({ degraded: true, error: String(error) }, null, 2);
  }

  async function refresh() {
    const startedAt = new Date().toISOString();
    const endpoints = [
      ['snapshot', '/api/causal-geometry/snapshot'],
      ['history', '/api/causal-geometry/history?limit=20'],
    ];

    const meta = [];
    for (const [key, path] of endpoints) {
      try {
        const payload = await fetchSection(path);
        renderSection(sections[key], payload.data ?? payload);
        const src = payload?.source || {};
        meta.push({ section: key, kind: src.kind || 'unknown', degraded: Boolean(src.degraded), error: src.error || null });
      } catch (error) {
        renderError(sections[key], error);
        meta.push({ section: key, kind: 'fallback', degraded: true, error: String(error) });
      }
    }

    try {
      const payload = await fetchSection('/api/causal-geometry/proposals?limit=50');
      renderProposals(payload);
      const src = payload?.source || {};
      meta.push({ section: 'proposals', kind: src.kind || 'unknown', degraded: Boolean(src.degraded), error: src.error || null });
    } catch (error) {
      renderProposalsError(error);
      meta.push({ section: 'proposals', kind: 'fallback', degraded: true, error: String(error) });
    }

    if (sourceMeta) {
      sourceMeta.textContent = `last_refresh=${startedAt} | ${meta.map((m) => `${m.section}:${m.kind}${m.degraded ? ':degraded' : ''}`).join(' | ')}`;
    }
  }

  if (refreshButton) {
    refreshButton.addEventListener('click', () => {
      setActionError(null);
      void refresh();
    });
  }

  void refresh();
})();
