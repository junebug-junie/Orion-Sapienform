(() => {
  const sections = {
    runtimeStatus: document.getElementById('substrateRuntimeStatus'),
    debugResult: document.getElementById('substrateDebugRunResult'),
    diagnosisSummary: document.getElementById('substrateDiagnosisSummary'),
    overview: document.getElementById('substrateOverview'),
    hotspots: document.getElementById('substrateHotspots'),
    queue: document.getElementById('substrateReviewQueue'),
    executions: document.getElementById('substrateReviewExecutions'),
    telemetry: document.getElementById('substrateTelemetrySummary'),
    calibration: document.getElementById('substrateCalibration'),
    policyComparison: document.getElementById('substratePolicyComparison'),
  };
  const sourceMeta = document.getElementById('substrateSourceMeta');
  const actionError = document.getElementById('substrateActionError');
  const refreshButton = document.getElementById('substrateRefreshButton');
  const bootstrapButton = document.getElementById('substrateBootstrapButton');
  const executeOnceButton = document.getElementById('substrateExecuteOnceButton');
  const debugRunButton = document.getElementById('substrateDebugRunButton');

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

  function setButtonBusy(button, label, busy) {
    if (!button) return;
    if (!button.dataset.defaultLabel) {
      button.dataset.defaultLabel = button.textContent || '';
    }
    button.disabled = busy;
    button.textContent = busy ? `${label}…` : button.dataset.defaultLabel;
    button.classList.toggle('opacity-60', busy);
    button.classList.toggle('cursor-not-allowed', busy);
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

  function renderDiagnosis(payload) {
    if (!sections.diagnosisSummary) return;
    const diagnosis = payload?.diagnosis || {};
    const severity = diagnosis.severity || 'unknown';
    const summary = diagnosis.summary || 'No diagnosis summary available.';
    const categories = Array.isArray(diagnosis.categories) ? diagnosis.categories : [];
    const severityClass =
      severity === 'degraded'
        ? 'border-red-700 bg-red-950/40 text-red-200'
        : severity === 'warning'
          ? 'border-amber-700 bg-amber-950/40 text-amber-200'
          : 'border-emerald-700 bg-emerald-950/40 text-emerald-200';
    sections.diagnosisSummary.className = `text-sm rounded border p-3 ${severityClass}`;
    sections.diagnosisSummary.textContent = `${summary}${categories.length ? `\nCategories: ${categories.join(' | ')}` : ''}`;
  }

  async function refresh() {
    const startedAt = new Date().toISOString();
    const endpoints = [
      ['runtimeStatus', '/api/substrate/review-runtime/status'],
      ['overview', '/api/substrate/overview?limit=10'],
      ['hotspots', '/api/substrate/hotspots?limit=20'],
      ['queue', '/api/substrate/review-queue?limit=50'],
      ['executions', '/api/substrate/review-executions?limit=50'],
      ['telemetry', '/api/substrate/telemetry-summary?limit=200'],
      ['calibration', '/api/substrate/calibration?limit=20'],
      ['policyComparison', '/api/substrate/policy-comparison?pair_mode=baseline_vs_active&sample_limit=300'],
    ];

    const meta = [];
    for (const [key, path] of endpoints) {
      try {
        const payload = await fetchSection(path);
        renderSection(sections[key], payload.data ?? payload);
        const src = payload.source || {};
        meta.push({ section: key, kind: src.kind || 'unknown', degraded: Boolean(src.degraded), error: src.error || null });
      } catch (error) {
        renderError(sections[key], error);
        meta.push({ section: key, kind: 'fallback', degraded: true, error: String(error) });
      }
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

  if (bootstrapButton) {
    bootstrapButton.addEventListener('click', async () => {
      setActionError(null);
      setButtonBusy(bootstrapButton, 'Bootstrapping', true);
      try {
        await postAction('/api/substrate/review-runtime/bootstrap', {});
        await refresh();
      } catch (error) {
        setActionError(error);
      } finally {
        setButtonBusy(bootstrapButton, 'Bootstrapping', false);
      }
    });
  }

  if (executeOnceButton) {
    executeOnceButton.addEventListener('click', async () => {
      setActionError(null);
      setButtonBusy(executeOnceButton, 'Executing', true);
      try {
        await postAction('/api/substrate/review-runtime/execute-once', {});
        await refresh();
      } catch (error) {
        setActionError(error);
      } finally {
        setButtonBusy(executeOnceButton, 'Executing', false);
      }
    });
  }

  if (debugRunButton) {
    debugRunButton.addEventListener('click', async () => {
      setActionError(null);
      setButtonBusy(debugRunButton, 'Running Debug Pass', true);
      try {
        const payload = await postAction('/api/substrate/review-runtime/debug-run', {});
        renderSection(sections.debugResult, payload);
        renderDiagnosis(payload);
        await refresh();
      } catch (error) {
        setActionError(error);
        renderError(sections.debugResult, error);
      } finally {
        setButtonBusy(debugRunButton, 'Running Debug Pass', false);
      }
    });
  }

  void refresh();
})();
