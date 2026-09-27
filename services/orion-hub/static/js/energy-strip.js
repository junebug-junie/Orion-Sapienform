(function () {
  "use strict";

  const LATEST_URL = "/api/energy/latest";
  const DAILY_URL = "/api/energy/usage/daily?days=14";
  const POLL_MS = 60000;
  const BAR_MAX_PX = 60;
  let timer = null;

  function isNum(v) {
    return v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v));
  }

  // Unknown money is "unknown" -- rendering it as $0.00 would say the house was free.
  function fmtUsd(v) {
    if (!isNum(v)) return "unknown";
    const n = Number(v);
    return (n < 0 ? "-$" : "$") + Math.abs(n).toFixed(2);
  }

  function fmtRate(v) {
    return isNum(v) ? "$" + Number(v).toFixed(4) + "/kWh" : "unknown";
  }

  function pressureLabel(p) {
    return (
      { over_forecast: "over RMP forecast", near_forecast: "near RMP forecast", normal: "under RMP forecast" }[p] ||
      "unknown"
    );
  }

  function pressureLine(stakes) {
    const s = stakes || {};
    const label = pressureLabel(s.pressure);
    return s.pressure_reason ? label + " (" + s.pressure_reason + ")" : label;
  }

  function importerLabel(imp) {
    if (!imp) return "importer: no status yet";
    return "importer: " + imp.state + (imp.reason ? " (" + imp.reason + ")" : "");
  }

  function reconcileLine(kind, r) {
    if (!r) return kind === "actual" ? "No closed bill reconciled yet." : "No RMP forecast yet.";
    const label = (kind === "actual" ? "Last bill " : "RMP forecast ") + r.billing_period_start;
    if (r.reconcile_gap) return label + ": Orion can't price this period yet (" + r.reconcile_gap + ")";
    return label + ": Orion " + fmtUsd(r.orion_total_usd) + " vs RMP " + fmtUsd(r.utility_total_usd) + " (diff " + fmtUsd(r.delta_usd) + ")";
  }

  function barHeights(points, maxPx) {
    const values = points.map(function (p) { return isNum(p.kwh) ? Number(p.kwh) : 0; });
    const max = values.reduce(function (a, b) { return Math.max(a, b); }, 0);
    return values.map(function (v) { return max > 0 ? Math.round((v / max) * maxPx) : 0; });
  }

  function byId(id) {
    return typeof document !== "undefined" ? document.getElementById(id) : null;
  }

  function setText(id, text) {
    const el = byId(id);
    if (el) el.textContent = text;
  }

  function renderLatest(body) {
    const s = (body && body.stakes) || {};
    const importerText = body && body.error
      ? "energy data unavailable (" + body.error + ")"
      : importerLabel(body && body.importer);
    setText("energyImporterState", importerText);
    setText("energyCycleToDate", fmtUsd(s.cycle_to_date_total_usd));
    setText("energyProjected", fmtUsd(s.orion_projected_total_usd));
    setText("energyForecast", fmtUsd(s.forecast_total_usd));
    setText("energyMarginal", fmtRate(s.marginal_usd_per_kwh));
    setText("energyPressure", pressureLine(s));
    const rec = (body && body.reconcile) || {};
    setText("energyReconcileActual", reconcileLine("actual", rec.actual));
    setText("energyReconcileForecast", reconcileLine("forecast", rec.forecast));
  }

  function renderDaily(body) {
    const el = byId("energyDailyBars");
    if (!el) return;
    const points = (body && body.points) || [];
    const heights = barHeights(points, BAR_MAX_PX);
    el.replaceChildren();
    points.forEach(function (p, i) {
      const bar = document.createElement("div");
      bar.style.height = heights[i] + "px";
      bar.style.width = "8px";
      bar.style.background = "currentColor";
      bar.title = p.day + ": " + Number(p.kwh).toFixed(1) + " kWh (" + p.hours + " h)";
      el.appendChild(bar);
    });
  }

  async function poll() {
    try {
      const [latest, daily] = await Promise.all([
        fetch(LATEST_URL).then(function (r) { return r.json(); }),
        fetch(DAILY_URL).then(function (r) { return r.json(); }),
      ]);
      renderLatest(latest);
      renderDaily(daily);
    } catch (err) {
      setText("energyImporterState", "energy API unavailable");
    }
  }

  function activate() {
    if (timer) return;
    poll();
    timer = setInterval(function () {
      if (typeof document !== "undefined" && document.hidden) return;
      poll();
    }, POLL_MS);
  }

  function deactivate() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }

  const api = {
    activate: activate,
    deactivate: deactivate,
    refresh: poll,
    fmtUsd: fmtUsd,
    fmtRate: fmtRate,
    pressureLabel: pressureLabel,
    pressureLine: pressureLine,
    importerLabel: importerLabel,
    reconcileLine: reconcileLine,
    barHeights: barHeights,
    renderLatest: renderLatest,
    renderDaily: renderDaily,
  };

  if (typeof window !== "undefined") {
    window.OrionEnergyStrip = api;
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})();
