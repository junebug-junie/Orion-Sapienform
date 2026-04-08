(() => {
  const sections = {
    overview: document.getElementById('substrateOverview'),
    hotspots: document.getElementById('substrateHotspots'),
    queue: document.getElementById('substrateReviewQueue'),
    executions: document.getElementById('substrateReviewExecutions'),
    telemetry: document.getElementById('substrateTelemetrySummary'),
    calibration: document.getElementById('substrateCalibration'),
    policyComparison: document.getElementById('substratePolicyComparison'),
  };
  const sourceMeta = document.getElementById('substrateSourceMeta');
  const refreshButton = document.getElementById('substrateRefreshButton');

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

  async function refresh() {
    const startedAt = new Date().toISOString();
    const endpoints = [
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
      void refresh();
    });
  }

  void refresh();
})();
