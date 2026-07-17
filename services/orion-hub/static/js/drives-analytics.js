(() => {
  // Standalone bundle for the Hub Drives Analytics page. This file is loaded ONLY by
  // /drives-analytics (see templates/drives-analytics.html) -- never by the Hub shell
  // directly, and it must never reach into the shell's global namespace.
  const DRIVE_KEYS = ['coherence', 'continuity', 'capability', 'relational', 'predictive', 'autonomy'];
  const ALLOWED_HOURS = [1, 6, 24, 168];
  const COLOR_MODES = ['combined', 'align', 'funnel'];
  const STALE_AFTER_MS = 5 * 60 * 1000; // 5 minutes, mirrors Hub STALE_AFTER_SEC.
  const POLL_INTERVAL_MS = 30000;

  // Tooltip copy: single source of truth for the (i) popover on every card. Each entry
  // covers three required angles for an operator: what it is, why it's built this way,
  // and how to read it / what failure modes look like. Keep this in sync with the Hub
  // README section on the Drives tab.
  const TOOLTIP_COPY = {
    gauges: {
      definition: 'Six thermometer gauges, one per DriveEngine drive (coherence, continuity, capability, relational, predictive, autonomy), filled to the latest audited pressure 0-1.',
      design: 'Color is a goal-alignment regime, not raw magnitude -- a pinned-high gauge with no matching goal is a warning, not a win. Funnel mode swaps to a neutral fill with a gate-verdict outline so pressure vs pipeline health are never conflated.',
      reading: 'Active badge = past the DriveEngine activation threshold. Outlined gauge = current dominant drive. A staleness chip means the audit rail has not produced a tick in over 5 minutes -- treat every color on this card as suspect until that clears.',
    },
    contributors: {
      definition: 'Per-drive contributor breakdown from tick_attribution (which tensions fed which drive) and tension_kinds, toggled between Live (latest tick only) and Window (aggregated across the selected hours).',
      design: 'Attribution was previously dropped at the SQL sink; Window aggregation only sums rows where tick_attribution is not null, and always reports null_attribution_row_count next to the real total so a thin post-migration window never reads as a false zero.',
      reading: 'Live shows what is driving pressure right now; Window shows the recent pattern. If null_attribution_row_count is high relative to attributed rows, the window mostly predates the tick_attribution migration -- read the aggregate as partial, not authoritative.',
    },
    kpi: {
      definition: 'A KPI strip: audit tick rate, active-drive count, co-activation fraction, top-dominant-drive share, the drive-rail gate verdict, and the age of the last audit.',
      design: 'The gate verdict here (gate_verdict_drive_only) is drive-only math from measure_autonomy_gate.py; it can never be the full offline GO because that also needs resource_pressure, which this Hub surface does not compute. GO_DRIVE_ONLY is labeled "drive economy OK (partial)", never bare "GO".',
      reading: 'Rising co-activation and a spread dominant share suggest a healthy multi-drive economy; a single drive owning the dominant share for most of the window (SATURATED) means the economy has collapsed to monoculture, regardless of how healthy pressures look individually.',
    },
    series: {
      definition: 'Dual time series: audit tick-rate over the window on top, six pressure sparklines underneath sharing the same time axis.',
      design: 'Points are downsampled to at most 240 per line server-side; after a stack rebuild the retained history can be much shorter than the requested window, so this card renders exactly the points that exist plus an explicit coverage banner -- it never fabricates points to fill the requested range.',
      reading: 'A flat or empty tick-rate line means the audit rail itself has gone quiet (check the producer, not the drives). Compare sparkline shapes against the coverage banner before reading trend -- a "trend" over 1 hour of thin post-rebuild history is not a trend yet.',
    },
    divergence: {
      definition: 'Side-by-side comparison of drive_state.v1 (the concept-store snapshot) against the latest drive_audits pressures, with a per-drive delta and the max absolute delta across all six drives.',
      design: 'CONCEPT_STORE_PATH has drifted to a host-local fallback default before (2026-07-13 incident class); this card raises a loud banner, not a quiet note, whenever the resolved path is that fallback default, since a silently-wrong path makes divergence numbers meaningless.',
      reading: 'Large deltas mean the concept store and the live audit rail disagree about drive state -- investigate before trusting either signal. autonomy_state_v2 is frozen/historical, never a live second signal; this card states that verbatim rather than implying a second live comparison exists.',
    },
    goals: {
      definition: 'Read-only list of active goals (artifact_id, drive_origin, proposal_status, goal_statement) plus funnel counts across the proposal pipeline.',
      design: 'This tab is orientation, not a mutation console -- unlike causal-geometry\'s adopt/reject buttons, this card has no action buttons of any kind. Goal-alignment coloring elsewhere on the page reads from this same data.',
      reading: 'If goals_available is false, the goal store could not be reached this refresh -- that shows as "goals unavailable" plainly, never as an invented empty list. A drive with sustained pressure and no matching goal here is the signal this whole page exists to surface.',
    },
    crosslinks: {
      definition: 'Plain navigation links to related Hub surfaces (Spark Cognitive EKG, Pressure Analytics, Causal Geometry) plus a raw-JSON debug panel of every payload this page fetched.',
      design: 'Pressure Analytics measures substrate mutation pressure, not the DriveEngine economy -- the two are easy to conflate by name alone, so this card states the distinction explicitly rather than just linking silently.',
      reading: 'Use the raw-JSON <details> panel when a rendered card looks wrong and you need to see the exact payload Hub received, before assuming the bug is in this page rather than upstream.',
    },
  };

  const els = {
    subjectSelect: document.getElementById('drivesSubjectSelect'),
    windowSelect: document.getElementById('drivesWindowSelect'),
    colorMode: document.getElementById('drivesColorMode'),
    refreshButton: document.getElementById('drivesRefreshButton'),
    autoRefresh: document.getElementById('drivesAutoRefresh'),
    lastLoaded: document.getElementById('drivesLastLoaded'),
    coverageBanner: document.getElementById('drivesCoverageBanner'),
    gaugesBody: document.getElementById('drivesGaugesBody'),
    contributorsBody: document.getElementById('drivesContributorsBody'),
    contributorsLiveButton: document.getElementById('drivesContributorsLiveButton'),
    contributorsWindowButton: document.getElementById('drivesContributorsWindowButton'),
    kpiBody: document.getElementById('drivesKpiBody'),
    seriesBody: document.getElementById('drivesSeriesBody'),
    divergenceBody: document.getElementById('drivesDivergenceBody'),
    goalsBody: document.getElementById('drivesGoalsBody'),
    crossLinksBody: document.getElementById('drivesCrossLinksBody'),
    tooltipPopover: document.getElementById('drivesTooltipPopover'),
  };

  function readState() {
    const params = new URLSearchParams(window.location.search);
    const rawHours = parseInt(params.get('window'), 10);
    const hours = ALLOWED_HOURS.includes(rawHours) ? rawHours : 24;
    const subject = params.get('subject') || 'orion';
    const rawColor = params.get('color');
    const color = COLOR_MODES.includes(rawColor) ? rawColor : 'combined';
    return { subject, hours, color, contributorsMode: 'live' };
  }

  const state = readState();
  let payloads = {};

  function writeUrl() {
    const params = new URLSearchParams();
    params.set('window', String(state.hours));
    params.set('subject', state.subject);
    params.set('color', state.color);
    const next = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState(null, '', next);
  }

  async function fetchSection(path) {
    const res = await fetch(path, { headers: { Accept: 'application/json' } });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} for ${path}`);
    }
    return res.json();
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function isStale(observedAtIso) {
    if (!observedAtIso) return true;
    const observed = new Date(observedAtIso).getTime();
    if (Number.isNaN(observed)) return true;
    return Date.now() - observed > STALE_AFTER_MS;
  }

  function fmtPct(fraction) {
    if (fraction === null || fraction === undefined || Number.isNaN(Number(fraction))) return '—';
    return `${(Number(fraction) * 100).toFixed(1)}%`;
  }

  function fmtAge(observedAtIso) {
    if (!observedAtIso) return 'unknown';
    const observed = new Date(observedAtIso).getTime();
    if (Number.isNaN(observed)) return 'unknown';
    const seconds = Math.max(0, Math.round((Date.now() - observed) / 1000));
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
    return `${(seconds / 3600).toFixed(1)}h ago`;
  }

  // -------------------------------------------------------------------------
  // Card 1: Six-drive gauges
  // -------------------------------------------------------------------------

  function gaugeColorClass(key, { colorMode, goalAlignment, kpis }) {
    if (colorMode === 'funnel') {
      return 'drive-color-neutral';
    }
    // 'align' and 'combined' both use goal-aligned per-drive coloring for the gauges
    // themselves (combined applies funnel/gate rules to the KPI strip instead).
    const perDrive = goalAlignment && goalAlignment.per_drive ? goalAlignment.per_drive[key] : null;
    const color = perDrive && perDrive.color_align ? perDrive.color_align : 'neutral';
    return `drive-color-${color}`;
  }

  function gaugeOutlineClass(colorMode, kpis) {
    if (colorMode !== 'funnel') return '';
    const verdict = kpis && kpis.gate_verdict_drive_only;
    if (verdict === 'GO_DRIVE_ONLY') return 'ring-2 ring-emerald-500';
    if (verdict === 'SATURATED') return 'ring-2 ring-red-500';
    if (verdict === 'NO-GO') return 'ring-2 ring-amber-500';
    return 'ring-2 ring-gray-600';
  }

  function renderGauges(container, { snapshot, goalAlignment, windowPayload, colorMode }) {
    if (!container) return;
    container.textContent = '';
    if (!snapshot || snapshot.degraded) {
      container.appendChild(
        el('div', 'text-xs text-gray-400', snapshot && snapshot.error ? `Gauges unavailable: ${snapshot.error}` : 'Gauges unavailable.'),
      );
      return;
    }
    const pressures = snapshot.drive_pressures || {};
    const activeDrives = new Set(snapshot.active_drives || []);
    const dominant = snapshot.dominant_drive;
    const stale = Boolean(snapshot.stale) || isStale(snapshot.observed_at);
    const kpis = windowPayload && !windowPayload.degraded ? windowPayload.kpis : null;

    for (const key of DRIVE_KEYS) {
      const pressure = Number(pressures[key] || 0);
      const gauge = el('div', 'drive-gauge');
      const label = el('div', 'text-[11px] text-gray-400', key);
      gauge.appendChild(label);

      const tube = el('div', `drive-gauge-tube ${key === dominant ? 'drive-gauge-dominant' : ''}`);
      const fill = el('div', `drive-gauge-fill ${gaugeColorClass(key, { colorMode, goalAlignment, kpis })} ${gaugeOutlineClass(colorMode, kpis)}`);
      fill.style.height = `${Math.max(0, Math.min(100, pressure * 100)).toFixed(1)}%`;
      tube.appendChild(fill);
      gauge.appendChild(tube);

      const value = el('div', 'text-[11px] text-gray-300', pressure.toFixed(2));
      gauge.appendChild(value);

      if (activeDrives.has(key)) {
        gauge.appendChild(el('div', 'text-[10px] text-emerald-400', 'active'));
      }
      if (key === dominant) {
        gauge.appendChild(el('div', 'text-[10px] text-sky-400', 'dominant'));
      }
      container.appendChild(gauge);
    }

    if (stale) {
      const chip = el(
        'div',
        'text-[11px] text-amber-400 border border-amber-900/60 bg-amber-950/30 rounded px-2 py-1 mt-1 w-full',
        `stale: last audit observed_at=${snapshot.observed_at || 'unknown'}`,
      );
      container.appendChild(chip);
    }
  }

  // -------------------------------------------------------------------------
  // Card 2: Contributors (Live vs Window)
  // -------------------------------------------------------------------------

  function renderContributorsBars(container, perDrive) {
    const maxVal = Math.max(0.0001, ...DRIVE_KEYS.map((k) => Math.abs(Number(perDrive[k] || 0))));
    for (const key of DRIVE_KEYS) {
      const value = Number(perDrive[key] || 0);
      const row = el('div', 'flex items-center gap-2 mb-1');
      row.appendChild(el('div', 'w-20 text-gray-400', key));
      const barTrack = el('div', 'flex-1 h-3 bg-gray-950 border border-gray-800 rounded overflow-hidden');
      const bar = el('div', 'h-full bg-sky-600');
      bar.style.width = `${Math.max(0, Math.min(100, (Math.abs(value) / maxVal) * 100)).toFixed(1)}%`;
      barTrack.appendChild(bar);
      row.appendChild(barTrack);
      row.appendChild(el('div', 'w-14 text-right text-gray-300', value.toFixed(3)));
      container.appendChild(row);
    }
  }

  function renderContributors(container, { snapshot, windowPayload, contributorsMode }) {
    if (!container) return;
    container.textContent = '';

    const modeLabel = el('div', 'text-[11px] text-gray-500 mb-2', contributorsMode === 'live' ? 'Live (latest tick)' : 'Window (aggregated over selected hours)');
    container.appendChild(modeLabel);

    if (contributorsMode === 'live') {
      if (!snapshot || snapshot.degraded) {
        container.appendChild(el('div', 'text-xs text-gray-400', 'Live contributors unavailable.'));
        return;
      }
      const attribution = snapshot.tick_attribution;
      if (!attribution || typeof attribution !== 'object') {
        container.appendChild(el('div', 'text-xs text-gray-400', 'No tick_attribution recorded for the latest audit (pre-migration or dropped tick).'));
        return;
      }
      renderContributorsBars(container, attribution);
      const kinds = Array.isArray(snapshot.tension_kinds) ? snapshot.tension_kinds : [];
      if (kinds.length > 0) {
        const kindsHeading = el('div', 'text-[11px] text-gray-400 mt-2', 'tension_kinds:');
        container.appendChild(kindsHeading);
        const list = el('ul', 'list-disc list-inside text-[11px] text-gray-400');
        for (const kind of kinds) {
          list.appendChild(el('li', null, kind));
        }
        container.appendChild(list);
      }
      return;
    }

    // Window mode.
    if (!windowPayload || windowPayload.degraded) {
      container.appendChild(el('div', 'text-xs text-gray-400', 'Window contributors unavailable.'));
      return;
    }
    const attribution = windowPayload.attribution || {};
    const perDrive = attribution.per_drive || {};
    renderContributorsBars(container, perDrive);
    const attributed = Number(attribution.attributed_row_count || 0);
    const nullCount = Number(attribution.null_attribution_row_count || 0);
    const total = attributed + nullCount;
    if (nullCount > 0) {
      container.appendChild(
        el(
          'div',
          'text-[11px] text-amber-400 mt-2',
          `${nullCount} of ${total} rows have no attribution recorded (pre-migration or dropped tick).`,
        ),
      );
    } else if (total > 0) {
      container.appendChild(el('div', 'text-[11px] text-gray-500 mt-2', `${attributed} of ${total} rows attributed.`));
    }
  }

  // -------------------------------------------------------------------------
  // Card 3: KPI strip
  // -------------------------------------------------------------------------

  function kpiChip(label, value, extraClass) {
    return el('div', `px-2 py-1 rounded border border-gray-700 bg-gray-950 ${extraClass || ''}`, `${label}: ${value}`);
  }

  function gateVerdictLabel(verdict) {
    // GO_DRIVE_ONLY is drive-rail-only math; full GO also needs resource_pressure, which
    // this Hub surface never computes. Never render the bare string "GO" for this value.
    if (verdict === 'GO_DRIVE_ONLY') return 'drive economy OK (partial)';
    if (verdict === 'SATURATED') return 'SATURATED';
    if (verdict === 'NO-GO') return 'NO-GO';
    if (verdict === 'UNMEASURABLE') return 'UNMEASURABLE';
    return verdict || 'unknown';
  }

  // Coloring model (spec: "KPI strip / goals card" column) -- align mode colors the gate
  // chip from per-drive goal-alignment severity; funnel and combined both color it from
  // the funnel/gate verdict math. Gauges themselves are handled separately in gaugeColorClass.
  function alignSeverityColor(goalAlignment) {
    const perDrive = (goalAlignment && goalAlignment.per_drive) || {};
    const rank = { red: 3, yellow: 2, green: 1, neutral: 0 };
    let worst = 'neutral';
    for (const key of DRIVE_KEYS) {
      const entry = perDrive[key];
      const color = entry && entry.color_align ? entry.color_align : 'neutral';
      if ((rank[color] ?? 0) > (rank[worst] ?? 0)) worst = color;
    }
    return worst;
  }

  function alignColorTextClass(color) {
    if (color === 'red') return 'text-red-400';
    if (color === 'yellow') return 'text-amber-400';
    if (color === 'green') return 'text-emerald-400';
    return 'text-gray-400';
  }

  function renderKpiStrip(container, { windowPayload, snapshot, seriesPayload, colorMode, goalAlignment }) {
    if (!container) return;
    container.textContent = '';
    if (!windowPayload || windowPayload.degraded) {
      container.appendChild(el('div', 'text-xs text-gray-400', windowPayload && windowPayload.error ? `KPIs unavailable: ${windowPayload.error}` : 'KPIs unavailable.'));
      return;
    }
    const kpis = windowPayload.kpis || {};
    const coverage = kpis.coverage || windowPayload.coverage || {};
    const coverageHours = Number(coverage.coverage_hours || 0);
    const tickRate = coverageHours > 0 ? (kpis.record_count / coverageHours).toFixed(2) : '—';

    container.appendChild(kpiChip('tick rate', `${tickRate}/h`));
    container.appendChild(kpiChip('active drives', snapshot && !snapshot.degraded ? snapshot.active_count : '—'));
    container.appendChild(kpiChip('co-activation', fmtPct(kpis.coactivation_frac)));
    container.appendChild(kpiChip('top dominant share', fmtPct(kpis.top_dominant_share)));

    const verdict = kpis.gate_verdict_drive_only;
    let verdictClass;
    if (colorMode === 'align') {
      // Align mode: strip chip reflects per-drive goal-alignment severity, not gate math.
      verdictClass = alignColorTextClass(alignSeverityColor(goalAlignment));
    } else {
      // funnel and combined modes both use funnel/gate-verdict rules for this chip.
      verdictClass = verdict === 'SATURATED' ? 'text-red-400' : verdict === 'GO_DRIVE_ONLY' ? 'text-emerald-400' : verdict === 'NO-GO' ? 'text-amber-400' : 'text-gray-400';
    }
    container.appendChild(kpiChip('gate verdict', gateVerdictLabel(verdict), verdictClass));

    const lastAge = snapshot && !snapshot.degraded ? fmtAge(snapshot.observed_at) : 'unknown';
    container.appendChild(kpiChip('last audit', lastAge));

    if (kpis.gate_note) {
      container.appendChild(el('div', 'w-full text-[11px] text-gray-500 mt-1', kpis.gate_note));
    }
  }

  // -------------------------------------------------------------------------
  // Card 4: Dual time series
  // -------------------------------------------------------------------------

  function buildSparklineSvg(points, { color, xMin, xMax }) {
    const svgNs = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNs, 'svg');
    svg.setAttribute('viewBox', '0 0 200 40');
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.classList.add('drives-sparkline');
    if (!points || points.length === 0) return svg;

    const values = points.map((p) => Number(p.v ?? p.count ?? 0));
    const maxV = Math.max(0.0001, ...values);
    const xSpan = Math.max(1, xMax - xMin);
    const coords = points.map((p) => {
      const t = new Date(p.t).getTime();
      const rawX = ((t - xMin) / xSpan) * 200;
      const x = Number.isFinite(rawX) ? rawX : 0;
      const v = Number(p.v ?? p.count ?? 0);
      const y = 40 - (v / maxV) * 38 - 1;
      return `${x.toFixed(1)},${Number.isFinite(y) ? y.toFixed(1) : 39}`;
    });
    const polyline = document.createElementNS(svgNs, 'polyline');
    polyline.setAttribute('points', coords.join(' '));
    polyline.setAttribute('fill', 'none');
    polyline.setAttribute('stroke', color || '#38bdf8');
    polyline.setAttribute('stroke-width', '1.5');
    svg.appendChild(polyline);
    return svg;
  }

  function renderSeries(container, seriesPayload) {
    if (!container) return;
    container.textContent = '';
    if (!seriesPayload || seriesPayload.degraded) {
      container.appendChild(el('div', 'text-xs text-gray-400', seriesPayload && seriesPayload.error ? `Series unavailable: ${seriesPayload.error}` : 'Series unavailable.'));
      return;
    }
    const coverage = seriesPayload.coverage || {};
    const requestedHours = Number(coverage.requested_hours || seriesPayload.hours || 0);
    const coverageHours = coverage.coverage_hours === null || coverage.coverage_hours === undefined ? null : Number(coverage.coverage_hours);
    if (coverageHours !== null && coverageHours < requestedHours) {
      container.appendChild(
        el(
          'div',
          'text-[11px] text-amber-400 border border-amber-900/60 bg-amber-950/30 rounded px-2 py-1 mb-2',
          `Thin history: only ${coverageHours.toFixed(1)}h of ${requestedHours}h requested is available (row_count=${coverage.row_count ?? 0}).${coverage.retention_note ? ' ' + coverage.retention_note : ''}`,
        ),
      );
    }

    const tickRate = Array.isArray(seriesPayload.tick_rate) ? seriesPayload.tick_rate : [];
    const allTimes = tickRate.map((p) => new Date(p.t).getTime()).filter((t) => Number.isFinite(t));
    for (const key of DRIVE_KEYS) {
      const points = (seriesPayload.pressures && seriesPayload.pressures[key]) || [];
      for (const p of points) {
        const t = new Date(p.t).getTime();
        if (Number.isFinite(t)) allTimes.push(t);
      }
    }
    const xMin = allTimes.length ? Math.min(...allTimes) : 0;
    const xMax = allTimes.length ? Math.max(...allTimes) : 1;

    container.appendChild(el('div', 'text-[11px] text-gray-400 mb-1', 'tick rate'));
    if (tickRate.length === 0) {
      container.appendChild(el('div', 'text-xs text-gray-500 mb-2', 'No tick-rate points in this window.'));
    } else {
      container.appendChild(buildSparklineSvg(tickRate, { color: '#38bdf8', xMin, xMax }));
    }

    const pressureGrid = el('div', 'grid grid-cols-2 md:grid-cols-3 gap-2 mt-3');
    for (const key of DRIVE_KEYS) {
      const cell = el('div', 'border border-gray-800 rounded p-2');
      cell.appendChild(el('div', 'text-[11px] text-gray-400', key));
      const points = (seriesPayload.pressures && seriesPayload.pressures[key]) || [];
      if (points.length === 0) {
        cell.appendChild(el('div', 'text-[11px] text-gray-600', 'no points'));
      } else {
        cell.appendChild(buildSparklineSvg(points, { color: '#a78bfa', xMin, xMax }));
      }
      pressureGrid.appendChild(cell);
    }
    container.appendChild(pressureGrid);
  }

  // -------------------------------------------------------------------------
  // Card 5: Divergence
  // -------------------------------------------------------------------------

  function renderDivergence(container, divergencePayload) {
    if (!container) return;
    container.textContent = '';
    if (!divergencePayload) {
      container.appendChild(el('div', 'text-xs text-gray-400', 'Divergence unavailable.'));
      return;
    }
    if (divergencePayload.store_path_is_fallback_default) {
      container.appendChild(
        el(
          'div',
          'text-xs text-red-300 border border-red-800 bg-red-950/40 rounded px-2 py-2 mb-2 font-semibold',
          `CONCEPT_STORE_PATH is unset; using the host-local fallback default (${divergencePayload.store_path}). Divergence numbers below may not reflect the intended concept store.`,
        ),
      );
    }

    const table = el('table', 'w-full text-left border-collapse');
    const thead = el('thead');
    const headRow = el('tr');
    for (const header of ['drive', 'drive_state', 'audit', 'delta']) {
      headRow.appendChild(el('th', 'text-gray-400 border-b border-gray-700 py-1 pr-3 font-medium', header));
    }
    thead.appendChild(headRow);
    table.appendChild(thead);
    const tbody = el('tbody');
    const driveState = divergencePayload.drive_state_pressures || {};
    const auditP = divergencePayload.audit_pressures || {};
    const deltas = divergencePayload.deltas || {};
    for (const key of DRIVE_KEYS) {
      const row = el('tr', 'border-b border-gray-800 text-gray-300');
      const dsVal = driveState[key];
      const auVal = auditP[key];
      const dVal = deltas[key];
      row.appendChild(el('td', 'py-1 pr-3', key));
      row.appendChild(el('td', 'py-1 pr-3', dsVal === null || dsVal === undefined ? '—' : Number(dsVal).toFixed(3)));
      row.appendChild(el('td', 'py-1 pr-3', auVal === null || auVal === undefined ? '—' : Number(auVal).toFixed(3)));
      row.appendChild(el('td', 'py-1 pr-3', dVal === null || dVal === undefined ? '—' : Number(dVal).toFixed(3)));
      tbody.appendChild(row);
    }
    table.appendChild(tbody);
    container.appendChild(table);

    container.appendChild(
      el('div', 'text-[11px] text-gray-400 mt-2', `max_abs_delta=${Number(divergencePayload.max_abs_delta || 0).toFixed(3)}`),
    );
    container.appendChild(
      el('div', 'text-[11px] text-gray-500 mt-1', divergencePayload.autonomy_state_v2_note || 'autonomy_state_v2_note: frozen/historical — not a live second signal'),
    );
    if (Array.isArray(divergencePayload.notes) && divergencePayload.notes.length > 0) {
      const notesList = el('ul', 'list-disc list-inside text-[11px] text-gray-500 mt-1');
      for (const note of divergencePayload.notes) {
        notesList.appendChild(el('li', null, note));
      }
      container.appendChild(notesList);
    }
  }

  // -------------------------------------------------------------------------
  // Card 6: Goals (strictly read-only)
  // -------------------------------------------------------------------------

  // Coloring model for goal cards (spec: "KPI strip / goals card" column): align mode
  // borders each goal by its drive_origin's per-drive alignment color; funnel/combined
  // both border by funnel/pipeline stage instead, since that column is funnel/gate-ruled
  // in both those modes.
  function goalBorderClass(goal, { colorMode, goalAlignmentPayload }) {
    if (colorMode === 'align') {
      const key = String(goal.drive_origin || '').trim().toLowerCase();
      const perDrive = (goalAlignmentPayload && goalAlignmentPayload.per_drive) || {};
      const entry = perDrive[key];
      const color = entry && entry.color_align ? entry.color_align : 'neutral';
      if (color === 'red') return 'border-red-700';
      if (color === 'yellow') return 'border-amber-700';
      if (color === 'green') return 'border-emerald-700';
      return 'border-gray-800';
    }
    const status = String(goal.proposal_status || '').trim().toLowerCase();
    if (status === 'completed' || status === 'active') return 'border-emerald-700';
    if (status === 'executing' || status === 'planned') return 'border-sky-700';
    if (status === 'archived') return 'border-gray-700';
    return 'border-gray-800';
  }

  function renderGoals(container, goalAlignmentPayload, colorMode) {
    if (!container) return;
    container.textContent = '';
    if (!goalAlignmentPayload) {
      container.appendChild(el('div', 'text-xs text-gray-400', 'Goals unavailable.'));
      return;
    }
    if (!goalAlignmentPayload.goals_available) {
      container.appendChild(el('div', 'text-xs text-gray-400', 'goals unavailable'));
      return;
    }
    const goals = Array.isArray(goalAlignmentPayload.active_goals) ? goalAlignmentPayload.active_goals : [];
    if (goals.length === 0) {
      container.appendChild(el('div', 'text-xs text-gray-500 mb-2', 'No active goals.'));
    } else {
      for (const goal of goals) {
        const card = el('div', `border rounded p-2 mb-2 ${goalBorderClass(goal, { colorMode, goalAlignmentPayload })}`);
        card.appendChild(el('div', 'text-gray-200 font-medium', goal.goal_statement || '(no statement)'));
        card.appendChild(
          el(
            'div',
            'text-[11px] text-gray-400',
            `artifact_id=${goal.artifact_id || '—'} | drive_origin=${goal.drive_origin || '—'} | proposal_status=${goal.proposal_status || '—'}`,
          ),
        );
        container.appendChild(card);
      }
    }
    // No action buttons on this card -- strictly read-only, unlike causal-geometry's
    // adopt/reject controls, which do not apply to this orientation-only surface.
    const funnel = goalAlignmentPayload.funnel || {};
    const funnelRow = el('div', 'flex flex-wrap gap-2 mt-2');
    for (const stage of ['proposed', 'active', 'planned', 'executing', 'completed', 'archived']) {
      funnelRow.appendChild(kpiChip(stage, funnel[stage] ?? 0));
    }
    container.appendChild(funnelRow);
  }

  // -------------------------------------------------------------------------
  // Card 7: Cross-links
  // -------------------------------------------------------------------------

  function renderCrossLinks(container, allPayloads) {
    if (!container) return;
    container.textContent = '';
    const linksRow = el('div', 'flex flex-wrap gap-3 mb-2');
    const links = [
      { href: '/spark/ui', label: 'Spark Cognitive EKG' },
      { href: '/#pressure', label: 'Hub Pressure Analytics' },
      { href: '/causal-geometry', label: 'Causal Geometry' },
    ];
    for (const link of links) {
      const a = el('a', 'px-2 py-1 rounded border border-gray-700 bg-gray-950 text-sky-400 hover:bg-gray-800', link.label);
      a.href = link.href;
      linksRow.appendChild(a);
    }
    container.appendChild(linksRow);
    container.appendChild(
      el(
        'div',
        'text-[11px] text-gray-500 mb-2',
        'Note: Hub Pressure Analytics measures substrate mutation pressure. It is not the same thing as the DriveEngine economy shown on this page.',
      ),
    );

    const details = el('details', 'text-[11px]');
    const summary = el('summary', 'text-gray-400 cursor-pointer', 'Raw fetched payloads (debug)');
    details.appendChild(summary);
    const pre = el('pre', 'text-gray-500 whitespace-pre-wrap mt-2');
    pre.textContent = JSON.stringify(allPayloads || {}, null, 2);
    details.appendChild(pre);
    container.appendChild(details);
  }

  // -------------------------------------------------------------------------
  // Controls: subject select, coverage banner
  // -------------------------------------------------------------------------

  function renderSubjectSelect(selectEl, subjectsPayload, currentSubject) {
    if (!selectEl) return;
    const subjects = subjectsPayload && Array.isArray(subjectsPayload.subjects) ? subjectsPayload.subjects : [];
    selectEl.textContent = '';
    const seen = new Set();
    for (const entry of subjects) {
      const opt = document.createElement('option');
      opt.value = entry.subject;
      const oldest = entry.oldest_ts ? new Date(entry.oldest_ts).toISOString().slice(0, 10) : 'n/a';
      const newest = entry.newest_ts ? new Date(entry.newest_ts).toISOString().slice(0, 10) : 'n/a';
      opt.textContent = `${entry.subject} (${entry.row_count} rows, ${oldest}..${newest})`;
      selectEl.appendChild(opt);
      seen.add(entry.subject);
    }
    if (!seen.has(currentSubject)) {
      const opt = document.createElement('option');
      opt.value = currentSubject;
      opt.textContent = currentSubject;
      selectEl.appendChild(opt);
    }
    selectEl.value = currentSubject;
  }

  function renderCoverageBanner(bannerEl, { windowPayload, seriesPayload, degradedAny }) {
    if (!bannerEl) return;
    const messages = [];
    for (const [name, payload] of [['window', windowPayload], ['series', seriesPayload]]) {
      if (!payload || payload.degraded) continue;
      const coverage = payload.coverage || (payload.kpis && payload.kpis.coverage) || {};
      const requested = Number(coverage.requested_hours || payload.hours || 0);
      const covered = coverage.coverage_hours === null || coverage.coverage_hours === undefined ? null : Number(coverage.coverage_hours);
      if (covered !== null && covered < requested) {
        messages.push(`${name}: only ${covered.toFixed(1)}h of ${requested}h covered`);
      }
    }
    if (degradedAny) {
      messages.push('one or more Drives Analytics endpoints are degraded this refresh');
    }
    if (messages.length === 0) {
      bannerEl.classList.add('hidden');
      bannerEl.textContent = '';
      return;
    }
    bannerEl.classList.remove('hidden');
    bannerEl.textContent = messages.join(' | ');
  }

  // -------------------------------------------------------------------------
  // Tooltip popover
  // -------------------------------------------------------------------------

  function showTooltip(cardId, anchorEl) {
    const copy = TOOLTIP_COPY[cardId];
    if (!copy || !els.tooltipPopover) return;
    els.tooltipPopover.textContent = '';
    els.tooltipPopover.appendChild(el('div', 'font-semibold text-gray-100 mb-1', 'What it is'));
    els.tooltipPopover.appendChild(el('div', 'mb-2', copy.definition));
    els.tooltipPopover.appendChild(el('div', 'font-semibold text-gray-100 mb-1', 'Why it is designed this way'));
    els.tooltipPopover.appendChild(el('div', 'mb-2', copy.design));
    els.tooltipPopover.appendChild(el('div', 'font-semibold text-gray-100 mb-1', 'How to read it'));
    els.tooltipPopover.appendChild(el('div', null, copy.reading));
    els.tooltipPopover.classList.remove('hidden');
    if (anchorEl) {
      const rect = anchorEl.getBoundingClientRect();
      els.tooltipPopover.style.top = `${window.scrollY + rect.bottom + 6}px`;
      els.tooltipPopover.style.left = `${window.scrollX + Math.max(0, rect.left - 200)}px`;
    }
  }

  function hideTooltip() {
    if (els.tooltipPopover) els.tooltipPopover.classList.add('hidden');
  }

  document.querySelectorAll('.drives-info-button').forEach((button) => {
    button.addEventListener('click', (event) => {
      event.stopPropagation();
      const cardId = button.getAttribute('data-card');
      if (els.tooltipPopover && !els.tooltipPopover.classList.contains('hidden')) {
        hideTooltip();
        return;
      }
      showTooltip(cardId, button);
    });
  });
  document.addEventListener('click', hideTooltip);

  // -------------------------------------------------------------------------
  // Fetch + render orchestration
  // -------------------------------------------------------------------------

  async function fetchAll() {
    const subject = encodeURIComponent(state.subject);
    const hours = state.hours;
    const endpoints = {
      subjects: '/api/drives-analytics/subjects',
      snapshot: `/api/drives-analytics/snapshot?subject=${subject}`,
      window: `/api/drives-analytics/window?subject=${subject}&hours=${hours}`,
      series: `/api/drives-analytics/series?subject=${subject}&hours=${hours}`,
      goalAlignment: `/api/drives-analytics/goal-alignment?subject=${subject}`,
      divergence: `/api/drives-analytics/divergence?subject=${subject}`,
    };
    const entries = Object.entries(endpoints);
    const results = await Promise.allSettled(entries.map(([, path]) => fetchSection(path)));
    const out = {};
    let degradedAny = false;
    results.forEach((result, idx) => {
      const [key] = entries[idx];
      if (result.status === 'fulfilled') {
        out[key] = result.value;
        if (result.value && result.value.degraded) degradedAny = true;
      } else {
        out[key] = { degraded: true, error: String(result.reason) };
        degradedAny = true;
      }
    });
    out._degradedAny = degradedAny;
    return out;
  }

  function renderAll() {
    renderGauges(els.gaugesBody, {
      snapshot: payloads.snapshot,
      goalAlignment: payloads.goalAlignment,
      windowPayload: payloads.window,
      colorMode: state.color,
    });
    renderContributors(els.contributorsBody, {
      snapshot: payloads.snapshot,
      windowPayload: payloads.window,
      contributorsMode: state.contributorsMode,
    });
    renderKpiStrip(els.kpiBody, {
      windowPayload: payloads.window,
      snapshot: payloads.snapshot,
      seriesPayload: payloads.series,
      colorMode: state.color,
      goalAlignment: payloads.goalAlignment,
    });
    renderSeries(els.seriesBody, payloads.series);
    renderDivergence(els.divergenceBody, payloads.divergence);
    renderGoals(els.goalsBody, payloads.goalAlignment, state.color);
    renderCrossLinks(els.crossLinksBody, payloads);
    renderSubjectSelect(els.subjectSelect, payloads.subjects, state.subject);
    renderCoverageBanner(els.coverageBanner, {
      windowPayload: payloads.window,
      seriesPayload: payloads.series,
      degradedAny: payloads._degradedAny,
    });
    if (els.lastLoaded) {
      els.lastLoaded.textContent = new Date().toISOString();
    }
    if (els.windowSelect) els.windowSelect.value = String(state.hours);
    if (els.colorMode) els.colorMode.value = state.color;
    if (els.contributorsLiveButton && els.contributorsWindowButton) {
      const liveActive = state.contributorsMode === 'live';
      els.contributorsLiveButton.classList.toggle('bg-gray-800', liveActive);
      els.contributorsWindowButton.classList.toggle('bg-gray-800', !liveActive);
    }
  }

  async function refreshAll() {
    try {
      payloads = await fetchAll();
    } catch (error) {
      // Hard fetch-error fallback only -- every card above already has a real DOM-builder
      // path for degraded/empty payloads; this branch only fires if fetchAll itself throws.
      payloads = { _degradedAny: true, _fetchError: String(error) };
    }
    renderAll();
    writeUrl();
  }

  // -------------------------------------------------------------------------
  // Controls wiring
  // -------------------------------------------------------------------------

  if (els.subjectSelect) {
    els.subjectSelect.addEventListener('change', () => {
      state.subject = els.subjectSelect.value || 'orion';
      void refreshAll();
    });
  }
  if (els.windowSelect) {
    els.windowSelect.addEventListener('change', () => {
      const value = parseInt(els.windowSelect.value, 10);
      state.hours = ALLOWED_HOURS.includes(value) ? value : 24;
      void refreshAll();
    });
  }
  if (els.colorMode) {
    els.colorMode.addEventListener('change', () => {
      const value = els.colorMode.value;
      state.color = COLOR_MODES.includes(value) ? value : 'combined';
      renderAll();
      writeUrl();
    });
  }
  if (els.refreshButton) {
    els.refreshButton.addEventListener('click', () => {
      void refreshAll();
    });
  }
  if (els.contributorsLiveButton) {
    els.contributorsLiveButton.addEventListener('click', () => {
      state.contributorsMode = 'live';
      renderAll();
    });
  }
  if (els.contributorsWindowButton) {
    els.contributorsWindowButton.addEventListener('click', () => {
      state.contributorsMode = 'window';
      renderAll();
    });
  }

  // -------------------------------------------------------------------------
  // Polling: gentle interval, pause on hidden tab, immediate refresh on resume.
  // -------------------------------------------------------------------------

  let pollTimer = null;

  function startPolling() {
    if (pollTimer) return;
    if (els.autoRefresh && !els.autoRefresh.checked) return;
    pollTimer = setInterval(() => {
      void refreshAll();
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
      void refreshAll();
    }
  });

  if (els.autoRefresh) {
    els.autoRefresh.addEventListener('change', () => {
      if (els.autoRefresh.checked) {
        startPolling();
      } else {
        stopPolling();
      }
    });
  }

  void refreshAll();
  startPolling();
})();
