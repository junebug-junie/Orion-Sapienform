// Pure history adapter: timestamps, fixed scaling and explicit interpolation gaps.
(function (root) {
  'use strict';
  const WINDOWS = Object.freeze({'24h': 86400, '3d': 259200, '7d': 604800});
  const DURATION_SEC = 60;
  const GAP_OPTIONS = Object.freeze([90, 300, 900, 1800, 3600]);
  const KEYS = Object.freeze({magnetic: 'magnetic_ut', lidar: 'lidar_mm'});
  const finite = v => typeof v === 'number' && Number.isFinite(v);
  function parseTime(value) {
    if (typeof value !== 'string') return NaN;
    return Date.parse(/(?:Z|[+-]\d\d:\d\d)$/i.test(value) ? value : value + 'Z');
  }
  function prepare(payload, windowEnd) {
    if (payload?.ok !== true || !Object.hasOwn(WINDOWS, payload.window) || !finite(windowEnd)) return null;
    const series = {}, stats = {};
    const end = windowEnd, start = end - WINDOWS[payload.window] * 1000;
    let count = 0;
    for (const key of Object.values(KEYS)) {
      const raw = Array.isArray(payload.series?.[key]) ? payload.series[key] : [];
      const byTime = new Map();
      let invalid = 0;
      for (const point of raw) {
        const t = parseTime(point?.t);
        if (!Number.isFinite(t)) { invalid++; continue; }
        if (t < start || t > end) continue;
        const v = finite(point.v) && point.v >= 0 ? point.v : null;
        if (v === null) invalid++;
        // Conflicting duplicate timestamps are unavailable, never an arbitrary winner.
        byTime.set(t, byTime.has(t) && byTime.get(t) !== v ? null : v);
      }
      const points = [...byTime].sort((a, b) => a[0] - b[0]).map(([t, v]) => Object.freeze({t, v}));
      series[key] = Object.freeze(points);
      count += points.length;
      const gaps = points.slice(1).map((p, i) => (p.t - points[i].t) / 1000).sort((a, b) => a - b);
      const rawCount = payload.stats?.[key]?.n_raw;
      stats[key] = Object.freeze({returned: raw.length, valid: points.filter(p => p.v !== null).length, invalid,
        nRaw: Number.isInteger(rawCount) && rawCount >= 0 ? rawCount : null,
        downsampled: Number.isInteger(rawCount) ? rawCount > raw.length : null,
        minSpacingSec: gaps[0] ?? null, medianSpacingSec: gaps[Math.floor(gaps.length / 2)] ?? null,
        maxSpacingSec: gaps[gaps.length - 1] ?? null});
    }
    if (!count) return null;
    return Object.freeze({version: 1, window: payload.window, start, end, boundsSource: 'browser request start (approximate)',
      durationSec: DURATION_SEC, compression: WINDOWS[payload.window] / DURATION_SEC,
      grainSec: finite(payload.grain_sec) && payload.grain_sec > 0 ? payload.grain_sec : null,
      series: Object.freeze(series), stats: Object.freeze(stats)});
  }
  function interpolate(points, at, gapSec) {
    if (!Number.isFinite(at) || !finite(gapSec) || gapSec <= 0 || !points?.length) return null;
    // Lower bound by source timestamp; no shared array index across channels.
    let lo = 0, hi = points.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (points[mid].t < at) lo = mid + 1; else hi = mid; }
    const right = points[lo], left = points[lo - 1];
    if (right?.t === at) return right.v === null ? null : {value: right.v, from: at, to: at};
    if (!left || !right || left.v === null || right.v === null || right.t - left.t > gapSec * 1000) return null;
    return {value: left.v + (right.v - left.v) * (at - left.t) / (right.t - left.t), from: left.t, to: right.t};
  }
  function frame(history, playbackSec, gapSec, preset) {
    const validTime = finite(playbackSec) && history && playbackSec >= 0 && playbackSec <= history.durationSec;
    const at = validTime ? history.start + playbackSec * history.compression * 1000 : null;
    return {version: 1, mode: 'history', sourceTime: at === null ? null : new Date(at).toISOString(),
      playbackSec, compression: history?.compression ?? null, gapSec,
      readings: preset.voices.map(def => {
        const key = KEYS[def.voice];
        const sample = at === null || !key ? null : interpolate(history.series[key], at, gapSec);
        return {key: def.key, unit: def.unit, value: sample?.value ?? null, status: sample ? 'ok' : 'missing',
          sourceId: key ? 'biometrics-summary:' + key : null,
          // Interpolated source position is separate from the actual observation bracket.
          observedAt: sample ? new Date(sample.from).toISOString() : null,
          bracketEnd: sample ? new Date(sample.to).toISOString() : null,
          reason: !key ? 'aggregated vibration omitted — no reconstructed impacts' : sample ? 'aggregated history level' : 'history gap / unavailable'};
      })};
  }
  const api = {WINDOWS, DURATION_SEC, GAP_OPTIONS, prepare, interpolate, frame};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.OrionCabinetHistory = api;
})(typeof window === 'undefined' ? globalThis : window);
