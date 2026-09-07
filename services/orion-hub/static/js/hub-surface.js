(() => {
  const els = {
    error: document.getElementById('hubSurfaceError'),
    refresh: document.getElementById('hubSurfaceRefreshButton'),
    bridgeBranches: document.getElementById('bridgeBranches'),
    bridgeCallouts: document.getElementById('bridgeCallouts'),
    durableSwitchPill: document.getElementById('durableSwitchPill'),
    durableLifecycle: document.getElementById('durableLifecycle'),
    chartBridge: document.getElementById('chartBridge'),
    chartBridgeLegend: document.getElementById('chartBridgeLegend'),
    chartBridgeNote: document.getElementById('chartBridgeNote'),
    chartDurable: document.getElementById('chartDurable'),
    chartDurableLegend: document.getElementById('chartDurableLegend'),
    activityLog: document.getElementById('activityLog'),
    recentAttentionStalePill: document.getElementById('recentAttentionStalePill'),
    recentAttentionItems: document.getElementById('recentAttentionItems'),
  };

  // Only `top_down_override` needs a special label -- every other key is
  // shown verbatim so a value this map doesn't know about (a NEW enum member
  // added to VoluntaryOverrideAbsentReasonV1, orion/schemas/attention_frame.py)
  // still renders as itself instead of silently vanishing from the page.
  const BRANCH_LABELS = { top_down_override: 'top_down_override (fired)' };
  // Defect-class values per that schema's own comments: not a normal outcome,
  // should read as an alarm the moment it's non-zero, not blend into the gray
  // "just another branch" rows.
  const DEFECT_BRANCHES = new Set(['combiner_error', 'absence_unclassified']);
  const BRANCH_COLOR = {
    top_down_override: '#2dd4bf', // teal-400 -- the one branch that's the actual signal
  };
  const LANE_COLOR = {
    durable_run: '#4ade80', cortex_turn: '#2dd4bf', reverie: '#fbbf24',
    substrate_attention: '#9ca3af', curiosity: '#f87171',
  };

  function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s == null ? '' : String(s);
    return d.innerHTML;
  }

  async function fetchJson(path) {
    const res = await fetch(path, { headers: { Accept: 'application/json' } });
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${path}`);
    return res.json();
  }

  function showError(err) {
    if (!els.error) return;
    els.error.textContent = `Failed to load Hub Surface data: ${err}`;
    els.error.classList.remove('hidden');
  }

  // ---- decision 1: goal bridge -------------------------------------------

  function renderBridge(data) {
    const branches = data.branches || {};
    const baseline = data.baseline || {};
    // Every key the backend actually returned, not a fixed guess at what
    // exists -- VoluntaryOverrideAbsentReasonV1 has values this page has
    // never seen fire; if one starts firing it must show up here, not
    // vanish. top_down_override first (it's the one branch that's the
    // actual signal), defect-class values next (they're an alarm regardless
    // of size), everything else by descending live share.
    const present = Array.from(new Set([...Object.keys(branches), ...Object.keys(baseline)])).sort((a, b) => {
      if (a === 'top_down_override') return -1;
      if (b === 'top_down_override') return 1;
      const aDefect = DEFECT_BRANCHES.has(a), bDefect = DEFECT_BRANCHES.has(b);
      if (aDefect !== bDefect) return aDefect ? -1 : 1;
      return (branches[b] ?? 0) - (branches[a] ?? 0);
    });
    if (data.sample_count === 0) {
      els.bridgeBranches.innerHTML = `<div class="text-gray-500">No self-model rows in the last ${Math.round(data.window_minutes / 60)}h — nothing to summarize yet.</div>`;
      return;
    }
    const max = Math.max(...present.map((k) => Math.max(baseline[k] || 0, branches[k] || 0)), 1);
    els.bridgeBranches.innerHTML = present
      .map((k) => {
        const before = baseline[k];
        const after = branches[k] ?? 0;
        const isKey = k === 'top_down_override';
        const isDefect = DEFECT_BRANCHES.has(k);
        const color = isDefect ? '#f87171' : (BRANCH_COLOR[k] || '#9ca3af');
        const labelClass = isKey ? 'text-teal-300' : (isDefect ? 'text-red-300' : 'text-gray-300');
        return `
        <div class="grid gap-2 py-2 border-b border-gray-800 last:border-0" style="grid-template-columns: 1fr 160px">
          <div>
            <div class="font-mono text-[12px] ${labelClass}">${isKey ? '★ ' : ''}${isDefect ? '⚠ ' : ''}${escapeHtml(BRANCH_LABELS[k] || k)}</div>
            <div class="relative h-2 bg-gray-800 rounded mt-1 overflow-hidden">
              ${before != null ? `<div class="absolute inset-y-0 left-0 bg-gray-600 opacity-40 rounded" style="width:${(before / max) * 100}%"></div>` : ''}
              <div class="absolute inset-y-0 left-0 rounded" style="width:${(after / max) * 100}%; background:${color}; opacity:.9"></div>
            </div>
          </div>
          <div class="text-right text-[11px] text-gray-400 self-center">
            ${before != null ? `${before}% <span class="text-gray-600">&rarr;</span> ` : ''}<span class="text-gray-200 font-mono">${after}%</span>
          </div>
        </div>`;
      })
      .join('');
    els.bridgeCallouts.innerHTML = `
      <div class="text-[11px] text-teal-200 bg-teal-900/20 border border-teal-800/60 rounded p-2 mt-3">
        <b>What actually changed:</b> only the ★ branch is Orion's attention moving because of
        this mechanism — everything else shifted as a side effect of that branch growing.
      </div>
      <div class="text-[11px] text-gray-400 bg-gray-950 border border-gray-800 rounded p-2 mt-2">
        <b class="text-gray-300">Structurally impossible, not just rare:</b> the override can only
        ever fire for 5 domains (biometrics, execution, chat, route, bus_synaptic). A competing
        loop outside those five is invisible to this decision entirely — most of the remaining
        "no override" share, by design, not a gap this mechanism is meant to close.
      </div>
      <div class="text-[11px] text-gray-500 mt-2">sample: ${data.sample_count} self-model rows, last ${Math.round(data.window_minutes / 60)}h${data.malformed_row_count ? ` (${data.malformed_row_count} unparseable, excluded)` : ''}</div>
    `;
  }

  // ---- decision 2: durable runs -------------------------------------------

  function renderDurable(data) {
    const pill = els.durableSwitchPill;
    if (data.kickoff_via_cortex) {
      pill.className = 'text-[11px] font-mono px-2 py-1 rounded border whitespace-nowrap flex-shrink-0 border-green-700 bg-green-900/30 text-green-300';
      pill.textContent = 'HUB_CURIOSITY_KICKOFF_VIA_CORTEX = true';
    } else {
      pill.className = 'text-[11px] font-mono px-2 py-1 rounded border whitespace-nowrap flex-shrink-0 border-red-700 bg-red-900/30 text-red-300';
      pill.textContent = 'HUB_CURIOSITY_KICKOFF_VIA_CORTEX = false — direct path active, none of this runs';
    }

    const b = data.baseline || {};
    let html = `
      <div class="grid grid-cols-2 gap-3 mb-3">
        <div class="rounded p-3 border border-red-800 bg-red-950/30">
          <div class="text-[10px] uppercase tracking-wide text-red-300 font-semibold mb-1">Before this shipped</div>
          <div class="text-sm font-medium">${b.started ?? '?'} started, ${b.finished ?? '?'} finished</div>
          <div class="text-[11px] text-gray-400 mt-1">${escapeHtml(b.note || '')}</div>
        </div>
        <div class="rounded p-3 border border-green-800 bg-green-950/30">
          <div class="text-[10px] uppercase tracking-wide text-green-300 font-semibold mb-1">Now (last ${Math.round(data.window_minutes / 60)}h)</div>
          <div class="text-sm font-medium">${data.completed_runs} finished, ${data.abandoned_runs} abandoned</div>
          <div class="text-[11px] text-gray-400 mt-1">restarts and failures are survivable, not fatal</div>
        </div>
      </div>
      <div class="flex flex-wrap gap-2 mb-3">
        ${chip('completed', data.completed_runs, '#4ade80')}
        ${chip('resumed (' + data.resumed_runs + ' run' + (data.resumed_runs === 1 ? '' : 's') + ')', data.resumed_events, '#2dd4bf')}
        ${chip('running now', data.running_runs, '#2dd4bf')}
        ${chip('abandoned', data.abandoned_runs, '#6b7280', data.abandoned_runs === 0)}
      </div>
    `;

    if (data.example) {
      html += renderExample(data.example);
    } else {
      html += `<div class="text-[11px] text-gray-500">No run has resumed yet — nothing to trace through.</div>`;
    }
    html += `<div class="text-[11px] text-gray-500 bg-gray-950 border border-gray-800 rounded p-2 mt-3">
      <b class="text-gray-300">Structural boundary:</b> a checkpoint older than 24h is abandoned
      outright, never resumed into a different day's material — the designed edge of "durable",
      not a bug surfacing.
    </div>`;
    els.durableLifecycle.innerHTML = html;
  }

  function chip(label, n, color, dim) {
    return `<div class="flex items-center gap-2 px-3 py-1.5 rounded border border-gray-800 bg-gray-950 text-[12px] ${dim ? 'opacity-50' : ''}">
      <span class="w-1.5 h-1.5 rounded-full" style="background:${color}"></span>${escapeHtml(label)}
      <span class="font-mono font-semibold text-gray-200">${n}</span>
    </div>`;
  }

  function renderExample(example) {
    const steps = example.steps || [];
    const attempts = steps.filter((s) => s.node === 'harness_turn').length;
    const finish = steps.find((s) => s.node === 'finish' && s.status === 'completed');
    const detail = (finish && finish.detail) || {};
    const nodeOrder = ['harness_turn', 'read_turn_result', 'publish_attention_row', 'journal', 'finish'];
    const latestByNode = {};
    steps.forEach((s) => { latestByNode[s.node] = s; });

    const stepsHtml = nodeOrder
      .map((node) => {
        const s = latestByNode[node];
        const done = s && (s.status === 'completed' || (node !== 'finish' && s.status !== 'failed'));
        const label = s
          ? (s.status === 'failed' ? 'retry' : s.status)
          : 'pending';
        const ring = s && s.status === 'failed' ? 'border-amber-500 border-dashed bg-amber-900/20 text-amber-300'
          : done ? 'border-green-600 bg-green-900/20 text-green-300'
          : 'border-gray-700 text-gray-500';
        return `<div class="flex flex-col items-center gap-1 flex-1 min-w-[90px]">
          <div class="w-8 h-8 rounded-full border-2 flex items-center justify-center text-[11px] font-mono ${ring}">
            ${done ? '&#10003;' : (s ? '&#8635;' : '')}
          </div>
          <div class="text-[10.5px] font-semibold text-gray-300 text-center">${escapeHtml(node)}</div>
          <div class="text-[10px] text-gray-600">${label}</div>
        </div>`;
      })
      .join('<div class="h-px bg-gray-800 flex-1 self-start mt-4"></div>');

    return `
      <div class="rounded border border-gray-800 bg-gray-950 p-3 mb-3">
        <div class="text-[10px] uppercase tracking-wide text-gray-500 mb-2">Concrete instance — most-retried run, all-time (not scoped to the window above)</div>
        <div class="font-mono text-[12px] text-teal-300 mb-2">${escapeHtml(example.run_id)}</div>
        <div class="flex items-start gap-0 overflow-x-auto pb-1">${stepsHtml}</div>
        <p class="text-[11.5px] text-gray-400 mt-3 leading-relaxed">
          harness_turn failed and retried <b class="text-gray-200">${attempts}</b> time${attempts === 1 ? '' : 's'} before finishing.
          ${detail.reach_out === false ? 'Did <b class="text-gray-200">not</b> reach out to Juniper — the outcome stayed inside Orion\'s own self-model.' : ''}
          ${detail.reach_out === true ? '<b class="text-gray-200">Reached out</b> to Juniper as a result.' : ''}
        </p>
        ${detail.finding_text ? `<p class="text-[11.5px] text-gray-500 mt-2 italic">"${escapeHtml(String(detail.finding_text).slice(0, 260))}${String(detail.finding_text).length > 260 ? '…' : ''}"</p>` : ''}
      </div>
    `;
  }

  // ---- live: recent-attention ambient cue ---------------------------------

  function renderRecentAttention(data) {
    const pill = els.recentAttentionStalePill;
    const items = data.items || [];
    if (data.stale) {
      pill.className = 'text-[11px] font-mono px-2 py-1 rounded border whitespace-nowrap flex-shrink-0 border-gray-700 bg-gray-950 text-gray-400';
      pill.textContent = 'stale — nothing recent to notice';
    } else {
      pill.className = 'text-[11px] font-mono px-2 py-1 rounded border whitespace-nowrap flex-shrink-0 border-green-700 bg-green-900/30 text-green-300';
      pill.textContent = 'fresh';
    }
    if (!items.length) {
      els.recentAttentionItems.innerHTML = '<div class="text-gray-600">Nothing in this window — quiet, not broken.</div>';
      return;
    }
    els.recentAttentionItems.innerHTML = items
      .map((it) => {
        const color = LANE_COLOR[it.process] || '#9ca3af';
        return `<div class="flex items-start gap-3 py-1.5 border-b border-gray-800 last:border-0">
          <span class="font-mono text-[10px] px-1.5 py-0.5 rounded flex-shrink-0" style="background:${color}22; color:${color}">${escapeHtml(it.process)}</span>
          <div class="flex-1">
            <div class="text-[12.5px] text-gray-300">${escapeHtml(it.narrative)}</div>
            <div class="text-[11px] text-gray-600">${escapeHtml(it.age_label)}</div>
          </div>
        </div>`;
      })
      .join('');
  }

  // ---- charts ---------------------------------------------------------

  function svgLine(x1, y1, x2, y2, extra) {
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${extra || ''}/>`;
  }

  function renderBridgeTrend(data) {
    const days = data.days || [];
    const baseline = data.baseline || {};
    const W = 640, H = 220, mL = 36, mR = 16, mT = 14, mB = 30;
    const plotW = W - mL - mR, plotH = H - mT - mB;
    const series = [
      { key: 'top_down_override', color: '#2dd4bf', keyBranch: true },
      { key: 'goal_matched_no_loop', color: '#9ca3af' },
    ];
    if (!days.length) {
      els.chartBridge.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="#6b7280" font-size="12">Not enough days of data yet</text>`;
      els.chartBridgeNote.textContent = '';
      return;
    }
    const points = [{ day: 'baseline', ...baseline }, ...days];
    const yMax = 50;
    const x = (i) => mL + (points.length > 1 ? (i / (points.length - 1)) * plotW : 0);
    const y = (v) => mT + plotH - (Math.min(v, yMax) / yMax) * plotH;

    let svg = '';
    [0, 10, 20, 30, 40, 50].forEach((v) => {
      svg += svgLine(mL, y(v), W - mR, y(v), 'stroke="#1f2937" stroke-width="1"');
      svg += `<text x="${mL - 6}" y="${y(v) + 3}" text-anchor="end" fill="#6b7280" font-size="10" font-family="monospace">${v}%</text>`;
    });
    svg += svgLine(mL, mT, mL, H - mB, 'stroke="#374151"');
    svg += svgLine(mL, H - mB, W - mR, H - mB, 'stroke="#374151"');
    points.forEach((p, i) => {
      svg += `<text x="${x(i)}" y="${H - mB + 18}" text-anchor="middle" fill="#6b7280" font-size="9.5" font-family="monospace">${escapeHtml(p.day === 'baseline' ? 'before' : p.day.slice(5))}</text>`;
    });

    series.forEach((s) => {
      const vals = points.map((p) => (p.day === 'baseline' ? p[s.key] : (p.branches || {})[s.key]) ?? 0);
      const pts = vals.map((v, i) => `${x(i)},${y(v)}`).join(' ');
      svg += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="${s.keyBranch ? 2.5 : 2}"/>`;
      vals.forEach((v, i) => {
        svg += `<circle cx="${x(i)}" cy="${y(v)}" r="${s.keyBranch ? 4 : 3}" fill="${s.color}"/>`;
        svg += `<text x="${x(i)}" y="${y(v) - 8}" text-anchor="middle" fill="${s.color}" font-size="10" font-family="monospace">${v}%</text>`;
      });
    });
    els.chartBridge.innerHTML = svg;
    els.chartBridgeLegend.innerHTML = series
      .map((s) => `<div class="flex items-center gap-1.5"><span class="inline-block w-2.5 h-0.5" style="background:${s.color}"></span>${s.keyBranch ? '★ ' : ''}${escapeHtml(BRANCH_LABELS[s.key])}</div>`)
      .join('');

    const last = days[days.length - 1];
    const lastNoLoop = last ? (last.branches || {}).goal_matched_no_loop : null;
    if (days.length >= 2 && lastNoLoop != null) {
      els.chartBridgeNote.innerHTML = `<b>Worth watching, not yet a verdict:</b> <span class="font-mono">goal_matched_no_loop</span> is at ${lastNoLoop}% today, vs a ${baseline.goal_matched_no_loop}% baseline — a proper multi-day read is what tells you whether this is holding or drifting back.`;
    } else {
      els.chartBridgeNote.innerHTML = `<b>Not enough days yet</b> to say whether this is holding — check back once a few more daily reads have accumulated.`;
    }
  }

  function renderDurableTrend(data) {
    const series = data.series || [];
    const milestones = data.milestones || [];
    const W = 640, H = 180, mL = 30, mR = 16, mT = 14, mB = 34;
    const plotW = W - mL - mR, plotH = H - mT - mB;
    if (!series.length) {
      els.chartDurable.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="#6b7280" font-size="12">No completed runs yet</text>`;
      els.chartDurableLegend.innerHTML = '';
      return;
    }
    const yMax = Math.max(...series.map((s) => s.cumulative_completed), 1);
    const x = (i) => mL + (series.length > 1 ? (i / (series.length - 1)) * plotW : plotW / 2);
    const y = (v) => mT + plotH - (v / yMax) * plotH;

    let svg = '';
    for (let v = 0; v <= yMax; v++) {
      svg += svgLine(mL, y(v), W - mR, y(v), 'stroke="#1f2937"');
      svg += `<text x="${mL - 6}" y="${y(v) + 3}" text-anchor="end" fill="#6b7280" font-size="10" font-family="monospace">${v}</text>`;
    }
    svg += svgLine(mL, mT, mL, H - mB, 'stroke="#374151"');
    svg += svgLine(mL, H - mB, W - mR, H - mB, 'stroke="#374151"');

    let d = `M ${x(0)} ${y(series[0].cumulative_completed)}`;
    for (let i = 1; i < series.length; i++) {
      d += ` L ${x(i)} ${y(series[i - 1].cumulative_completed)} L ${x(i)} ${y(series[i].cumulative_completed)}`;
    }
    svg += `<path d="${d}" fill="none" stroke="#4ade80" stroke-width="2.5"/>`;
    series.forEach((s, i) => {
      svg += `<circle cx="${x(i)}" cy="${y(s.cumulative_completed)}" r="4" fill="#4ade80"/>`;
      svg += `<text x="${x(i)}" y="${H - mB + 18}" text-anchor="middle" fill="#6b7280" font-size="9.5" font-family="monospace">${escapeHtml(s.day.slice(5))}</text>`;
    });
    milestones.forEach((m) => {
      const idx = series.findIndex((s) => s.day === m.date);
      const px = idx >= 0 ? x(idx) : null;
      if (px == null) return;
      svg += svgLine(px, mT, px, H - mB, 'stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4 3"');
    });
    els.chartDurable.innerHTML = svg;
    els.chartDurableLegend.innerHTML = `
      <div class="flex items-center gap-1.5"><span class="inline-block w-2.5 h-0.5" style="background:#4ade80"></span>cumulative completed runs</div>
      ${milestones.map((m) => `<div class="flex items-center gap-1.5"><span class="inline-block w-2.5 border-t-2 border-dashed" style="border-color:#f59e0b"></span>${escapeHtml(m.date)}: ${escapeHtml(m.label)}</div>`).join('')}
    `;
  }

  // ---- activity log -----------------------------------------------------

  function renderActivity(data) {
    const rows = data.rows || [];
    if (!rows.length) {
      els.activityLog.innerHTML = '<div class="text-gray-600">No activity recorded yet.</div>';
      return;
    }
    els.activityLog.innerHTML = rows
      .map((r) => {
        const color = LANE_COLOR[r.process] || '#9ca3af';
        const time = r.generated_at ? r.generated_at.slice(11, 19) : '?';
        return `<div class="grid gap-3 py-2 border-b border-gray-800 last:border-0 items-baseline" style="grid-template-columns: 64px 120px 1fr">
          <div class="font-mono text-[11px] text-gray-600">${escapeHtml(time)}</div>
          <div><span class="font-mono text-[10px] px-1.5 py-0.5 rounded" style="background:${color}22; color:${color}">${escapeHtml(r.process)}</span></div>
          <div class="text-[12.5px] text-gray-300">${escapeHtml(r.reason_narrative || r.attention_reason || '')}</div>
        </div>`;
      })
      .join('');
  }

  // ---- orchestration ------------------------------------------------------

  async function loadAll() {
    els.error.classList.add('hidden');
    try {
      const [bridge, bridgeTrend, durable, durableTrend, recentAttention, activity] = await Promise.all([
        fetchJson('/api/hub-surface/bridge'),
        fetchJson('/api/hub-surface/bridge/trend'),
        fetchJson('/api/hub-surface/durable-runs'),
        fetchJson('/api/hub-surface/durable-runs/trend'),
        fetchJson('/api/hub-surface/recent-attention'),
        fetchJson('/api/hub-surface/activity?limit=40'),
      ]);
      renderBridge(bridge);
      renderBridgeTrend(bridgeTrend);
      renderDurable(durable);
      renderDurableTrend(durableTrend);
      renderRecentAttention(recentAttention);
      renderActivity(activity);
    } catch (err) {
      showError(err);
    }
  }

  els.refresh && els.refresh.addEventListener('click', loadAll);
  loadAll();
})();
