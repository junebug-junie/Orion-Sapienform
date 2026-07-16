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

  function renderError(target, error) {
    if (!target) return;
    target.textContent = JSON.stringify({ degraded: true, error: String(error) }, null, 2);
  }

  function renderDegradedNotice(target, source, fallbackMessage) {
    if (!target) return;
    target.textContent = '';
    const message = document.createElement('div');
    message.className = 'text-xs text-gray-400';
    message.textContent = fallbackMessage;
    target.appendChild(message);

    const details = document.createElement('pre');
    details.className = 'text-[11px] text-gray-500 mt-2 whitespace-pre-wrap';
    details.textContent = JSON.stringify(source || {}, null, 2);
    target.appendChild(details);
  }

  function buildTable(headers) {
    const table = document.createElement('table');
    table.className = 'w-full text-left border-collapse';

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const header of headers) {
      const th = document.createElement('th');
      th.className = 'text-gray-400 border-b border-gray-700 py-1 pr-3 font-medium whitespace-nowrap';
      th.textContent = header;
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    table.appendChild(tbody);
    return { table, tbody };
  }

  function appendRow(tbody, values, rowClassName) {
    const row = document.createElement('tr');
    row.className = rowClassName;
    for (const value of values) {
      const td = document.createElement('td');
      td.className = 'py-1 pr-3 align-top';
      td.textContent = value === null || value === undefined || value === '' ? '—' : String(value);
      row.appendChild(td);
    }
    tbody.appendChild(row);
  }

  function renderEdgesTable(container, edges) {
    if (!container) return;
    container.textContent = '';
    if (!Array.isArray(edges) || edges.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'text-xs text-gray-400';
      empty.textContent = 'No edges in this snapshot.';
      container.appendChild(empty);
      return;
    }

    // Mirror scripts/causal_geometry_report.py's human report: strongest edges first
    // by absolute strength, so a weak-negative and weak-positive edge don't get buried
    // below a mid-magnitude one.
    const sorted = [...edges].sort((a, b) => Math.abs(Number(b?.strength) || 0) - Math.abs(Number(a?.strength) || 0));

    const { table, tbody } = buildTable(['source_id', 'target_id', 'lag_sec', 'strength', 'significance', 'n_samples']);
    for (const edge of sorted) {
      appendRow(
        tbody,
        [edge?.source_id, edge?.target_id, edge?.lag_sec, edge?.strength, edge?.significance, edge?.n_samples],
        'border-b border-gray-800 text-gray-300',
      );
    }
    container.appendChild(table);
  }

  function divergenceRowClassName(status, delta) {
    const absDelta = Math.abs(Number(delta) || 0);
    const base = 'border-b border-gray-800';
    if (status === 'both') {
      // A real observed-vs-designed edge with a meaningful drift is the thing an operator
      // should actually look at; keep small/no-delta "both" rows visually quiet.
      return absDelta > 0.05 ? `status-both ${base} text-amber-400` : `status-both ${base} text-gray-300`;
    }
    if (status === 'designed_only') {
      return `status-designed_only ${base} text-sky-400`;
    }
    if (status === 'observed_only') {
      return `status-observed_only ${base} text-purple-400`;
    }
    if (status === 'insufficient_data') {
      return `status-insufficient_data ${base} text-gray-500`;
    }
    return `${base} text-gray-300`;
  }

  function renderDivergenceTable(container, divergence) {
    if (!container) return;
    container.textContent = '';
    if (!Array.isArray(divergence) || divergence.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'text-xs text-gray-400';
      empty.textContent = 'No divergence entries in this snapshot.';
      container.appendChild(empty);
      return;
    }

    const { table, tbody } = buildTable(['source_id', 'target_id', 'status', 'observed', 'designed', 'delta']);
    for (const entry of divergence) {
      appendRow(
        tbody,
        [entry?.source_id, entry?.target_id, entry?.status, entry?.observed_strength, entry?.designed_weight, entry?.delta],
        divergenceRowClassName(entry?.status, entry?.delta),
      );
    }
    container.appendChild(table);
  }

  function renderSnapshot(target, payload) {
    if (!target) return;
    target.textContent = '';
    const source = payload?.source || {};
    const data = payload?.data;

    if (!data || source.degraded) {
      renderDegradedNotice(
        target,
        source,
        source.error ? `No snapshot available: ${source.error}` : 'No snapshot available.',
      );
      return;
    }

    const summary = document.createElement('div');
    summary.className = 'text-[11px] text-gray-400 mb-2';
    summary.textContent =
      `snapshot_id=${data.snapshot_id ?? '—'} | generated_at=${data.generated_at ?? '—'} | ` +
      `window=${data.window_start ?? '—'}..${data.window_end ?? '—'} | ` +
      `designed_topology_version=${data.designed_topology_version ?? '—'} | ` +
      `insufficient_data=${Boolean(data.insufficient_data)}`;
    target.appendChild(summary);

    const edgesHeading = document.createElement('div');
    edgesHeading.className = 'text-xs font-medium text-gray-300 mt-2 mb-1';
    edgesHeading.textContent = 'Edges (observed)';
    target.appendChild(edgesHeading);
    const edgesContainer = document.createElement('div');
    edgesContainer.className = 'overflow-auto';
    target.appendChild(edgesContainer);
    renderEdgesTable(edgesContainer, data.edges);

    const divergenceHeading = document.createElement('div');
    divergenceHeading.className = 'text-xs font-medium text-gray-300 mt-3 mb-1';
    divergenceHeading.textContent = 'Divergence (observed vs. designed)';
    target.appendChild(divergenceHeading);
    const divergenceContainer = document.createElement('div');
    divergenceContainer.className = 'overflow-auto';
    target.appendChild(divergenceContainer);
    renderDivergenceTable(divergenceContainer, data.divergence);

    if (Array.isArray(data.notes) && data.notes.length > 0) {
      const notesHeading = document.createElement('div');
      notesHeading.className = 'text-xs font-medium text-gray-300 mt-3 mb-1';
      notesHeading.textContent = 'Notes';
      target.appendChild(notesHeading);
      const notesList = document.createElement('ul');
      notesList.className = 'list-disc list-inside text-[11px] text-gray-400';
      for (const note of data.notes) {
        const li = document.createElement('li');
        li.textContent = note;
        notesList.appendChild(li);
      }
      target.appendChild(notesList);
    }
  }

  function renderHistory(target, payload) {
    if (!target) return;
    target.textContent = '';
    const source = payload?.source || {};
    const data = payload?.data;

    if (!Array.isArray(data) || source.degraded) {
      renderDegradedNotice(
        target,
        source,
        source.error ? `No snapshot history available: ${source.error}` : 'No snapshot history available.',
      );
      return;
    }

    if (data.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'text-xs text-gray-400';
      empty.textContent = 'No snapshots in history yet.';
      target.appendChild(empty);
      return;
    }

    // Most recent first: a timeline reads top-to-bottom as "now back to earlier".
    const sorted = [...data].sort((a, b) => String(b?.generated_at ?? '').localeCompare(String(a?.generated_at ?? '')));

    const { table, tbody } = buildTable(['snapshot_id', 'generated_at', 'insufficient_data', 'edge_count', 'divergence_count']);
    for (const snap of sorted) {
      const edgeCount = Array.isArray(snap?.edges) ? snap.edges.length : 0;
      const divergenceCount = Array.isArray(snap?.divergence) ? snap.divergence.length : 0;
      appendRow(
        tbody,
        [snap?.snapshot_id, snap?.generated_at, Boolean(snap?.insufficient_data), edgeCount, divergenceCount],
        'border-b border-gray-800 text-gray-300',
      );
    }
    target.appendChild(table);
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

    const sectionRenderers = {
      snapshot: renderSnapshot,
      history: renderHistory,
    };

    const meta = [];
    for (const [key, path] of endpoints) {
      try {
        const payload = await fetchSection(path);
        sectionRenderers[key](sections[key], payload);
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

  // Gentle polling: this is a nightly-cadence feature, not a live feed, so a slow interval
  // is enough to keep the tab reasonably fresh without hammering the endpoint. Pausing on
  // document hidden avoids polling background tabs; resuming does one immediate refresh so
  // the operator doesn't have to wait out a full interval after tabbing back in.
  const POLL_INTERVAL_MS = 45000;
  let pollTimer = null;

  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(() => {
      void refresh();
    }, POLL_INTERVAL_MS);
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
      stopPolling();
    } else {
      startPolling();
      void refresh();
    }
  });

  void refresh();
  startPolling();
})();
