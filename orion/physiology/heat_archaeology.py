"""Heat archaeology: where did a node's watts go, and since when.

Pure and I/O-free: no clock, no database, no bus, no numpy (plain-Python least squares; the
design matrix is at most ~11 columns by ~720 hourly rows). The Postgres reads live in
``orion/introspect/power_attribution.py``; orion-hub's ``power_attribution`` responder runs
them and passes plain rows here. Orion reaches it through the orion-introspect MCP tool from a
unified chat turn and from a curiosity run alike: one implementation, two callers.

The method is the one that found the 2026-10-14 circe step by hand (circe ~790 W -> ~1100 W):

1. **Steps.** Binary segmentation of hourly wall watts (``orion_energy_hourly``): split where
   the two-mean model removes the most squared error, keep the split only if both sides are at
   least ``STEP_MIN_SEGMENT_H`` long and the means differ by ``STEP_MIN_DELTA_W``. Repeat up to
   ``STEP_MAX``. Each step is decomposed by the node's own hourly GPU/CPU watts
   (``orion_biometrics_cluster.measurements_by_node``); "rest" is wall minus both (fans, PSU
   loss, disks, RAM). Segment lengths count hourly rows, not clock hours: with missing hours a
   "24 h" side can span longer. The bar is in autocorrelation-corrected standard errors
   (``lag1``/``ar1_inflation``), because hourly watts wander.
2. **Workloads.** Per-hour GPU-pool busy-hours per holder (``gpu_pool_events`` end rows,
   ``held_ms`` spread over the hours it covered). Durable-run holds are reservations, not
   load: their compute is the child leases they start, so they are reported, not fitted.
3. **Regime.** A CPU-load regime indicator (``load_15m`` above an Otsu threshold) when the
   node's load is clearly two-moded. On 2026-10-14 this was a crash loop worth ~144 W that no
   lease explained.
4. **Fit.** Ordinary least squares of hourly wall watts (and GPU watts on the pool node) on
   busy-hours + regime + intercept. Each step's delta is split as coefficient x (mean after -
   mean before) per term; whatever is left is ``unexplained``.
5. **Shifts the leases cannot see.** Steps in ``load_15m`` and CPU watts are reported on their
   own, pointing at host processes and container restart counts.

Everything is correlational. An always-on marker is confounded with anything else that started
the same hour. Results carry ``verified=False`` and say UNVERIFIED until an intervention (stop
the suspect, watch the watts) confirms them.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Mapping, Sequence

from orion.schemas.heat_archaeology import (
    EvidenceRefV1,
    FitTermV1,
    NodeHeatArchaeologyV1,
    PowerStepV1,
    RegimeShiftV1,
    StepAttributionV1,
    WorkloadFitV1,
)

# GPU-pool cards are circe's: every `granted` row's detail.grant.served_by is circe-worker-*
# (measured 2026-10-23 over 10 days: gpu0-gpu3, six workers, no other host).
POOL_NODE = "circe"

STEP_MIN_SEGMENT_H = 24
STEP_MIN_DELTA_W = 60.0
# Separation in autocorrelation-corrected standard errors (see lag1). Picked on flat AR(1)
# series with no step, 200 draws each (binary segmentation searches every split, so the bar
# is above the one-split 2.0): live-like rho 0.6 (circe hourly residuals 2026-10-23: 0.45-0.62)
# false-alarms 1.5-2.5%; extreme rho 0.97 ~9%. 5.0 also missed the real, hand-found
# 2026-10-14 circe step (t 4.2 in a 14-day window), so 4.0.
STEP_MIN_T = 4.0
STEP_MAX = 4
# Lag-1 autocorrelation is clipped here so the inflation factor stays finite (<= 199).
RHO_MAX = 0.99
LOAD_STEP_MIN_DELTA = 2.0
# A CPU-load shift this close to a wall step is the same event; it is not attributed twice.
LOAD_SHIFT_SAME_AS_STEP = timedelta(hours=6)
CPU_STEP_MIN_DELTA_W = 20.0

# Lease groups fitted individually; the rest are summed into OTHER_LEASES.
FIT_MAX_GROUPS = 8
FIT_MIN_GROUP_BUSY_H = 1.0
FIT_MIN_HOURS = 48
OTHER_LEASES = "other leases"

# Otsu split of hourly load_15m: kept only when the two modes explain this share of the
# variance and each holds at least REGIME_MIN_HOURS hours. A unimodal sample still "separates":
# Otsu's between/total variance is 2/pi ~ 0.64 for a normal, 0.75 for a uniform, ~0.66 for an
# exponential (checked on 2000 draws each), so the bar sits above all three. Live 2026-10-23:
# circe 0.97 (idle ~1 vs crash loop ~13), athena 0.61 (one mode, heavy tail), hecate 0.21.
REGIME_MIN_SEPARATION = 0.85
REGIME_MIN_HOURS = 6

# A contribution smaller than this is left out of the sentence (still in the data).
SUMMARY_MIN_CONTRIB_W = 5.0
SUMMARY_MAX_TERMS = 4
# A term is named in the sentence only when its coefficient is this many standard errors from 0.
SUMMARY_MIN_T = 2.0

CAVEATS = (
    "Correlational: a fit over hourly averages, not a measurement of cause. UNVERIFIED until an "
    "intervention (stop the suspect, watch the watts) confirms it.",
    "An always-on marker is confounded with anything else that started the same hour.",
    "Durable-run holds are reservations; their compute appears as the child leases they start.",
    "Standard errors are corrected for hour-to-hour wander (lag-1 autocorrelation), not for "
    "every pattern in it; a term named as credible can still be a coincidence.",
    "Busy-hours count lease-hours, not cards; a lease still open at the window's end counts only "
    "if granted in its last 6 hours.",
    "A CPU-load regime or shift is invisible to GPU leases: check host processes and container "
    "restart counts (docker inspect -f '{{.RestartCount}}').",
)

HOLDER_LABELS = {"http:anthropic": "harness", "http:openai": "ai-town"}
RESERVATION_PREFIX = "durable-runs:"


@dataclass(frozen=True)
class HourReading:
    """One node's hourly averages. ``None`` = no reading that hour, never 0."""

    hour: datetime
    node: str
    wall_w: float | None = None
    gpu_w: float | None = None
    cpu_w: float | None = None
    load_15m: float | None = None


@dataclass(frozen=True)
class PoolBusy:
    """Busy-hours one GPU-pool group held inside one hour (0..cards)."""

    hour: datetime
    group: str
    busy_h: float
    reservation: bool = False


def holder_group(holder: str, workflow: str | None) -> tuple[str, bool]:
    """(group name, is_reservation) for a gpu_pool_events holder."""
    if holder.startswith(RESERVATION_PREFIX):
        return f"run:{workflow or '?'}", True
    return HOLDER_LABELS.get(holder, holder), False


# --------------------------------------------------------------------------- statistics


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _var(xs: Sequence[float]) -> float:
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def lag1(xs: Sequence[float]) -> float:
    """Lag-1 autocorrelation of a centred series, clipped to [0, RHO_MAX].

    Hourly watts wander: a busy hour is followed by a busy hour. Treating each hour as an
    independent draw makes every standard error too small (review 2026-10-23: AR(1) noise at
    rho 0.97 produced a "step" in 62 of 100 flat series and a "credible" unrelated term in
    3 of 4 fits). Negative values are clipped to 0: they would only shrink the errors.
    """
    if len(xs) < 3:
        return 0.0
    m = _mean(xs)
    c = [x - m for x in xs]
    den = sum(v * v for v in c)
    if den <= 0:
        return 0.0
    num = sum(a * b for a, b in zip(c, c[1:]))
    return min(max(num / den, 0.0), RHO_MAX)


def ar1_inflation(rho_a: float, rho_b: float = 1.0) -> float:
    """Variance inflation for a mean (rho_b=1) or a regression slope under AR(1) errors and an
    AR(1) regressor: (1 + ra*rb) / (1 - ra*rb) (Bartlett 1935; the standard effective-sample
    correction)."""
    r = min(rho_a * rho_b, RHO_MAX)
    return (1.0 + r) / (1.0 - r)


def find_steps(
    values: Sequence[float],
    *,
    min_seg: int = STEP_MIN_SEGMENT_H,
    min_delta: float = STEP_MIN_DELTA_W,
    min_t: float = STEP_MIN_T,
    max_steps: int = STEP_MAX,
) -> list[int]:
    """Indices where a new level starts (binary segmentation on the mean). Sorted."""
    n = len(values)
    pre = [0.0]
    pre2 = [0.0]
    for v in values:
        pre.append(pre[-1] + v)
        pre2.append(pre2[-1] + v * v)

    def sse(a: int, b: int) -> float:
        k = b - a
        s = pre[b] - pre[a]
        return (pre2[b] - pre2[a]) - s * s / k

    def best_split(a: int, b: int) -> tuple[float, int] | None:
        if b - a < 2 * min_seg:
            return None
        whole = sse(a, b)
        best: tuple[float, int] | None = None
        for k in range(a + min_seg, b - min_seg + 1):
            gain = whole - sse(a, k) - sse(k, b)
            if best is None or gain > best[0]:
                best = (gain, k)
        if best is None:
            return None
        k = best[1]
        n1, n2 = k - a, b - k
        m1 = (pre[k] - pre[a]) / n1
        m2 = (pre[b] - pre[k]) / n2
        v1 = sse(a, k) / n1
        v2 = sse(k, b) / n2
        # Pooled lag-1 autocorrelation of the residuals inside each side (the pair across the
        # split is left out): the hours are not independent, so the plain SE is inflated.
        resid = [(values[i] - m1, values[i + 1] - m1) for i in range(a, k - 1)]
        resid += [(values[i] - m2, values[i + 1] - m2) for i in range(k, b - 1)]
        den = sum(x * x for x, _ in resid)
        rho = min(max(sum(x * y for x, y in resid) / den, 0.0), RHO_MAX) if den > 0 else 0.0
        se = math.sqrt((v1 / n1 + v2 / n2) * ar1_inflation(rho))
        t = abs(m2 - m1) / se if se > 0 else math.inf
        if abs(m2 - m1) < min_delta or t < min_t:
            return None
        return best

    splits: list[int] = []
    segments = [(0, n)]
    while len(splits) < max_steps:
        candidates = [(cand, seg) for seg in segments if (cand := best_split(*seg)) is not None]
        if not candidates:
            break
        (gain, k), (a, b) = max(candidates, key=lambda c: c[0][0])
        splits.append(k)
        segments.remove((a, b))
        segments += [(a, k), (k, b)]
    return sorted(splits)


def _segments(n: int, splits: Sequence[int]) -> list[tuple[int, int]]:
    edges = [0, *splits, n]
    return list(zip(edges[:-1], edges[1:]))


def otsu_threshold(values: Sequence[float]) -> tuple[float, float] | None:
    """(threshold, separation) of the best two-class split; separation = between/total variance."""
    xs = sorted(values)
    n = len(xs)
    if n < 2 * REGIME_MIN_HOURS:
        return None
    total_var = _var(xs)
    if total_var <= 0:
        return None
    total = sum(xs)
    best: tuple[float, float] | None = None
    run = sum(xs[: REGIME_MIN_HOURS - 1])
    # Class 0 is xs[:i]; i stops where class 1 still holds REGIME_MIN_HOURS values.
    for i in range(REGIME_MIN_HOURS, n - REGIME_MIN_HOURS + 1):
        run += xs[i - 1]
        if xs[i - 1] == xs[i]:
            continue
        w0, w1 = i / n, (n - i) / n
        m0, m1 = run / i, (total - run) / (n - i)
        sep = w0 * w1 * (m0 - m1) ** 2 / total_var
        if best is None or sep > best[1]:
            best = ((xs[i - 1] + xs[i]) / 2.0, sep)
    return best


def _solve(a: list[list[float]], b: list[list[float]]) -> list[list[float]] | None:
    """Gauss-Jordan with partial pivoting: returns a^-1 b, or None when a is singular."""
    p = len(a)
    m = [row[:] + bb[:] for row, bb in zip(a, b)]
    scale = max((abs(a[i][i]) for i in range(p)), default=1.0) or 1.0
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-9 * scale:
            return None
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        m[col] = [x / pv for x in m[col]]
        for r in range(p):
            if r != col and m[r][col] != 0.0:
                f = m[r][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [row[p:] for row in m]


@dataclass(frozen=True)
class OlsResult:
    names: list[str]
    coef: list[float]
    se: list[float | None]
    r2: float
    residual_sd: float
    dropped: list[str]


def ols(columns: Mapping[str, Sequence[float]], y: Sequence[float]) -> OlsResult | None:
    """y ~ columns + intercept. Constant and collinear columns are dropped and named.

    Solved on centred, unit-variance columns (so a collinear pair shows up as a ~0 pivot on a
    well-scaled matrix), then mapped back to watts per original unit.
    """
    n = len(y)
    dropped: list[str] = []
    names = []
    for name, col in columns.items():
        if _var(col) <= 1e-12:
            dropped.append(name)
        else:
            names.append(name)
    stats = {nm: (_mean(columns[nm]), math.sqrt(_var(columns[nm]))) for nm in names}
    while True:
        cols = [[(v - stats[nm][0]) / stats[nm][1] for v in columns[nm]] for nm in names] + [[1.0] * n]
        p = len(cols)
        if n <= p:
            return None
        xtx = [[sum(ci[k] * cj[k] for k in range(n)) for cj in cols] for ci in cols]
        identity = [[1.0 if i == j else 0.0 for j in range(p)] for i in range(p)]
        inv = _solve(xtx, identity)
        if inv is not None:
            break
        # Drop the latest-listed column of the collinear set: earlier columns are the heavier
        # workloads (callers list by total busy-hours), so the lighter alias goes.
        dropped.append(names.pop(_collinear_victim(xtx, len(names))))
        stats = {nm: stats[nm] for nm in names}
    xty = [sum(ci[k] * y[k] for k in range(n)) for ci in cols]
    beta = [sum(inv[i][j] * xty[j] for j in range(p)) for i in range(p)]
    fitted = [sum(beta[j] * cols[j][k] for j in range(p)) for k in range(n)]
    rss = sum((yk - fk) ** 2 for yk, fk in zip(y, fitted))
    ym = _mean(y)
    tss = sum((yk - ym) ** 2 for yk in y)
    dof = n - p
    sigma2 = rss / dof if dof > 0 else 0.0
    # Back to original units: b_j = beta_j / sd_j; intercept = beta_0 - sum(b_j * mean_j).
    q = len(names)
    scale = [1.0 / stats[nm][1] for nm in names]
    coef = [beta[j] * scale[j] for j in range(q)]
    c = [-stats[nm][0] / stats[nm][1] for nm in names]
    intercept = beta[q] + sum(cj * beta[j] for j, cj in enumerate(c))
    # Autocorrelation-corrected SEs: residuals and regressors both wander hour to hour, so each
    # term's variance is inflated by Bartlett's AR(1) factor for its own column.
    rho_e = lag1([yk - fk for yk, fk in zip(y, fitted)])
    se = [
        math.sqrt(max(sigma2 * inv[j][j], 0.0) * ar1_inflation(rho_e, lag1(cols[j]))) * scale[j]
        for j in range(q)
    ]
    # Var(intercept) = Var(b0 + sum c_j b_j) from the full covariance sigma2 * inv.
    w = [*c, 1.0]
    var_i = sigma2 * sum(w[i] * w[j] * inv[i][j] for i in range(p) for j in range(p)) * ar1_inflation(rho_e)
    return OlsResult(
        names=[*names, "intercept"], coef=[*coef, intercept], se=[*se, math.sqrt(max(var_i, 0.0))],
        r2=1.0 - rss / tss if tss > 0 else 0.0,
        residual_sd=math.sqrt(sigma2), dropped=dropped,
    )


def _collinear_victim(xtx: list[list[float]], q: int) -> int:
    """Index of the first column (in order) that is a linear combination of the ones before it."""
    for k in range(1, q + 1):
        sub = [row[:k] for row in xtx[:k]]
        ident = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
        if _solve(sub, ident) is None:
            return k - 1
    return q - 1


# --------------------------------------------------------------------------- analysis


def _r(x: float, nd: int = 1) -> float:
    return round(float(x), nd)


def _fmt_w(x: float) -> str:
    return f"{round(x) + 0:+d} W"  # + 0 turns a rounded -0 into 0


def _fmt_hour(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M UTC")


def _comp_mean(rows: Sequence[HourReading], attr: str) -> float | None:
    vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
    return _mean(vals) if vals else None


def _rest(r: HourReading) -> float | None:
    if r.wall_w is None or r.gpu_w is None or r.cpu_w is None:
        return None
    return r.wall_w - r.gpu_w - r.cpu_w


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else _r(b - a)


def _series_shifts(rows: Sequence[HourReading], attr: str, min_delta: float) -> list[RegimeShiftV1]:
    pts = [(r.hour, getattr(r, attr)) for r in rows if getattr(r, attr) is not None]
    if len(pts) < 2 * STEP_MIN_SEGMENT_H:
        return []
    vals = [v for _, v in pts]
    splits = find_steps(vals, min_delta=min_delta)
    segs = _segments(len(vals), splits)
    out = []
    for i, k in enumerate(splits):
        (a, _), (_, b) = segs[i], segs[i + 1]
        before, after = _mean(vals[a:k]), _mean(vals[k:b])
        out.append(RegimeShiftV1(
            series="load_15m" if attr == "load_15m" else "cpu_w",
            at=pts[k][0], before=_r(before, 2), after=_r(after, 2), delta=_r(after - before, 2),
            segment_start=pts[a][0], segment_end=pts[b - 1][0] + timedelta(hours=1),
        ))
    return out


def _lease_matrix(
    hours: Sequence[datetime], busy: Iterable[PoolBusy],
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """(load columns aligned to ``hours``, reservation mean busy-h per hour)."""
    idx = {h: i for i, h in enumerate(hours)}
    load: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(hours))
    reserve: dict[str, float] = defaultdict(float)
    for b in busy:
        i = idx.get(b.hour)
        if i is None:
            continue
        if b.reservation:
            reserve[b.group] += b.busy_h
        else:
            load[b.group][i] += b.busy_h
    totals = sorted(((sum(col), g) for g, col in load.items()), reverse=True)
    keep = [g for tot, g in totals[:FIT_MAX_GROUPS] if tot >= FIT_MIN_GROUP_BUSY_H]
    cols = {g: load[g] for g in keep}
    rest = [g for _, g in totals if g not in keep]
    if rest:
        other = [sum(load[g][i] for g in rest) for i in range(len(hours))]
        if sum(other) >= FIT_MIN_GROUP_BUSY_H:
            cols[OTHER_LEASES] = other
    n = max(len(hours), 1)
    res = {g: _r(v / n, 3) for g, v in sorted(reserve.items(), key=lambda kv: -kv[1])[:5]}
    return cols, res


def _fit(
    target: str, y: Sequence[float], columns: Mapping[str, Sequence[float]], regime_name: str | None,
) -> tuple[WorkloadFitV1, OlsResult] | None:
    res = ols(columns, y)
    if res is None:
        return None
    terms = []
    for name, c, s in zip(res.names, res.coef, res.se):
        kind = "intercept" if name == "intercept" else ("regime" if name == regime_name else "lease")
        level = 1.0 if name == "intercept" else _mean(columns[name])
        terms.append(FitTermV1(
            name=name, kind=kind, watts_per_unit=_r(c), se=None if s is None else _r(s),
            mean_level=_r(level, 3),
        ))
    return WorkloadFitV1(
        target=target, n_hours=len(y), r2=_r(res.r2, 3), residual_sd_w=_r(res.residual_sd),
        terms=terms, dropped=res.dropped,
    ), res


def analyze_node(
    node: str,
    readings: Sequence[HourReading],
    busy: Iterable[PoolBusy] = (),
    *,
    window_start: datetime,
    window_end: datetime,
    pool_node: str = POOL_NODE,
) -> NodeHeatArchaeologyV1:
    rows = sorted((r for r in readings if r.node == node), key=lambda r: r.hour)
    wall_rows = [r for r in rows if r.wall_w is not None]
    caveats = list(CAVEATS)
    evidence: list[EvidenceRefV1] = []
    base = dict(node=node, window_start=window_start, window_end=window_end, hours_with_wall=len(wall_rows))
    if not wall_rows:
        return NodeHeatArchaeologyV1(
            **base, summary=f"{node}: no wall-power reading in this window, so nothing can be attributed.",
            caveats=caveats,
        )

    daily: dict[str, list[float]] = defaultdict(list)
    for r in wall_rows:
        daily[r.hour.strftime("%Y-%m-%d")].append(r.wall_w)
    daily_wall = [(d, _r(_mean(v), 0)) for d, v in sorted(daily.items())]
    mean_wall = _mean([r.wall_w for r in wall_rows])

    # 1. Steps on wall watts.
    walls = [r.wall_w for r in wall_rows]
    splits = find_steps(walls)
    segs = _segments(len(walls), splits)
    steps: list[PowerStepV1] = []
    step_segs: list[tuple[list[HourReading], list[HourReading]]] = []
    for i, k in enumerate(splits):
        (a, _), (_, b) = segs[i], segs[i + 1]
        before, after = wall_rows[a:k], wall_rows[k:b]
        bw, aw = _mean(walls[a:k]), _mean(walls[k:b])
        se = math.sqrt(_var(walls[a:k]) / len(before) + _var(walls[k:b]) / len(after))
        rb = [x for x in (_rest(r) for r in before) if x is not None]
        ra = [x for x in (_rest(r) for r in after) if x is not None]
        step = PowerStepV1(
            at=wall_rows[k].hour, before_w=_r(bw, 0), after_w=_r(aw, 0), delta_w=_r(aw - bw, 0),
            before_hours=len(before), after_hours=len(after),
            t_stat=_r((aw - bw) / se if se > 0 else 0.0, 1),
            gpu_delta_w=_delta(_comp_mean(before, "gpu_w"), _comp_mean(after, "gpu_w")),
            cpu_delta_w=_delta(_comp_mean(before, "cpu_w"), _comp_mean(after, "cpu_w")),
            rest_delta_w=_delta(_mean(rb) if rb else None, _mean(ra) if ra else None),
            load_15m_before=None if (lb := _comp_mean(before, "load_15m")) is None else _r(lb, 2),
            load_15m_after=None if (la := _comp_mean(after, "load_15m")) is None else _r(la, 2),
        )
        steps.append(step)
        step_segs.append((before, after))
        # The step's own fields are the values; evidence names the rows they came from.
        evidence.append(EvidenceRefV1(
            claim=f"wall power step at {_fmt_hour(step.at)}", query="energy_hourly",
            values={"from": _fmt_hour(before[0].hour), "to": _fmt_hour(after[-1].hour),
                    "hours": len(before) + len(after)},
        ))
        if step.gpu_delta_w is not None:
            evidence.append(EvidenceRefV1(
                claim=f"GPU/CPU/rest split of the {_fmt_hour(step.at)} step", query="biometrics_hourly",
            ))

    # 5 (first, so the fit can be read across them). Shifts the leases cannot see.
    shifts = _series_shifts(rows, "load_15m", LOAD_STEP_MIN_DELTA) + _series_shifts(rows, "cpu_w", CPU_STEP_MIN_DELTA_W)
    shifts.sort(key=lambda s: s.at)
    for s in shifts:
        evidence.append(EvidenceRefV1(
            claim=f"{s.series} shift at {_fmt_hour(s.at)}", query="biometrics_hourly",
            values={"before": s.before, "after": s.after},
        ))

    # 2-4. Fit wall (and GPU on the pool node) on workloads + regime + intercept.
    fit_rows = [r for r in wall_rows if r.load_15m is not None]
    fit = gpu_fit = None
    fit_note = None
    attributions: list[StepAttributionV1] = []
    reservations: dict[str, float] = {}
    threshold = None
    if len(fit_rows) < FIT_MIN_HOURS:
        fit_note = (f"no fit: {len(fit_rows)} hours have both wall watts and a CPU-load reading, "
                    f"{FIT_MIN_HOURS} needed")
    else:
        hours = [r.hour for r in fit_rows]
        # The regime column goes first: if a lease is an exact alias of it, the lease (listed
        # later) is the one dropped and named in fit.dropped, not the regime.
        columns: dict[str, list[float]] = {}
        regime_name = None
        split = otsu_threshold([r.load_15m for r in fit_rows])
        if split is not None and split[1] >= REGIME_MIN_SEPARATION:
            threshold = _r(split[0], 2)
            ind = [1.0 if r.load_15m > split[0] else 0.0 for r in fit_rows]
            if REGIME_MIN_HOURS <= sum(ind) <= len(ind) - REGIME_MIN_HOURS:
                regime_name = f"CPU-load regime (load_15m > {threshold:g})"
                columns[regime_name] = ind
        if node == pool_node:
            leases, reservations = _lease_matrix(hours, busy)
            columns.update(leases)
        fitted = _fit("wall_w", [r.wall_w for r in fit_rows], columns, regime_name) if columns else None
        if fitted is None:
            fit_note = ("no fit: no workload busy-hours on this machine and its CPU load is not "
                        "two-moded, so there is nothing to attribute watts to")
        else:
            fit, res = fitted
            evidence.append(EvidenceRefV1(
                claim="wall-watts fit on workload busy-hours", query="pool_busy_hours",
                values={"r2": fit.r2, "n_hours": fit.n_hours, "residual_sd_w": fit.residual_sd_w,
                        "terms": len(fit.terms)},
            ))
            coef = {k: v for k, v in zip(res.names, res.coef) if k != "intercept"}

            def attribute(at: datetime, lo: datetime, hi: datetime, anchor: str) -> StepAttributionV1 | None:
                bi = [i for i, h in enumerate(hours) if lo <= h < at]
                ai = [i for i, h in enumerate(hours) if at <= h < hi]
                if not bi or not ai:
                    return None
                bw = _mean([fit_rows[i].wall_w for i in bi])
                aw = _mean([fit_rows[i].wall_w for i in ai])
                contrib = {}
                for name, c in coef.items():
                    col = columns[name]
                    v = c * (_mean([col[i] for i in ai]) - _mean([col[i] for i in bi]))
                    if abs(v) >= 1.0:  # sub-watt shares are noise; they stay in unexplained's sum
                        contrib[name] = _r(v, 0)
                return StepAttributionV1(
                    step_at=at, anchor=anchor, before_w=_r(bw, 0), after_w=_r(aw, 0),
                    delta_w=_r(aw - bw, 0), contributions_w=contrib,
                    unexplained_w=_r(aw - bw - sum(contrib.values()), 0),
                )

            for step, (before, after) in zip(steps, step_segs):
                got = attribute(step.at, before[0].hour, after[-1].hour + timedelta(hours=1), "wall_step")
                if got is not None:
                    attributions.append(got)
            for sh in shifts:
                if sh.series != "load_15m" or sh.segment_start is None or sh.segment_end is None:
                    continue
                if any(abs(st.at - sh.at) <= LOAD_SHIFT_SAME_AS_STEP for st in steps):
                    continue  # the wall step already tells this story
                got = attribute(sh.at, sh.segment_start, sh.segment_end, "load_shift")
                if got is not None:
                    attributions.append(got)
        gpu_rows = [r for r in fit_rows if r.gpu_w is not None]
        if fit is not None and node == pool_node and len(gpu_rows) == len(fit_rows):
            gfit = _fit("gpu_w", [r.gpu_w for r in fit_rows], columns, regime_name)
            # Fit quality only: the wall fit's terms are the attribution; per-term GPU watts
            # would double the payload for a cross-check.
            gpu_fit = gfit[0].model_copy(update={"terms": []}) if gfit is not None else None

    result = NodeHeatArchaeologyV1(
        **base, mean_wall_w=_r(mean_wall, 0), daily_wall_w=daily_wall, steps=steps, fit=fit,
        gpu_fit=gpu_fit, fit_note=fit_note, attributions=attributions, regime_shifts=shifts, reservations=reservations,
        load_regime_threshold=threshold, evidence=evidence, summary="", caveats=caveats,
    )
    return result.model_copy(update={"summary": summarize(result)})


def _largest_step(result: NodeHeatArchaeologyV1) -> PowerStepV1 | None:
    return max(result.steps, key=lambda s: abs(s.delta_w), default=None)


def _credible(fit: WorkloadFitV1) -> set[str]:
    """Terms whose coefficient is at least SUMMARY_MIN_T standard errors from zero."""
    return {t.name for t in fit.terms if t.se is not None and t.se > 0 and abs(t.watts_per_unit) >= SUMMARY_MIN_T * t.se}


def _attribution_bits(attr: StepAttributionV1, credible: set[str]) -> str:
    """The named parts always add up to attr.delta_w: named + other terms + unexplained."""
    ranked = sorted(attr.contributions_w.items(), key=lambda kv: -abs(kv[1]))
    shown = [(k, v) for k, v in ranked if k in credible and abs(v) >= SUMMARY_MIN_CONTRIB_W][:SUMMARY_MAX_TERMS]
    named = {k for k, _ in shown}
    other = sum(v for k, v in ranked if k not in named)
    bits = []
    for name, v in shown:
        if name.startswith("CPU-load regime"):
            bits.append(f"{name} {_fmt_w(v)}, which no GPU lease explains (check host processes and "
                        "container restarts)")
        else:
            bits.append(f"{name} {_fmt_w(v)}")
    if abs(other) >= SUMMARY_MIN_CONTRIB_W:
        bits.append(f"other terms, too small or too noisy to name, {_fmt_w(other)}")
    bits.append(f"unexplained {_fmt_w(attr.unexplained_w)}")
    return "; ".join(bits)


def _step_sentence(step: PowerStepV1, attr: StepAttributionV1 | None, credible: set[str]) -> str:
    comp = [f"{label} {_fmt_w(v)}" for label, v in (
        ("GPU", step.gpu_delta_w), ("CPU", step.cpu_delta_w), ("rest", step.rest_delta_w)) if v is not None]
    head = (
        f"{_fmt_w(step.delta_w)} around {_fmt_hour(step.at)} ({step.before_w:.0f} -> {step.after_w:.0f} W"
        + (f"; {', '.join(comp)}" if comp else "") + ")"
    )
    if attr is None:
        return head + "."
    lead = "" if abs(attr.delta_w - step.delta_w) < 1 else f" over the fitted hours {_fmt_w(attr.delta_w)}"
    return head + lead + ": " + _attribution_bits(attr, credible) + "."


def _shift_sentence(attr: StepAttributionV1, shift: RegimeShiftV1 | None, credible: set[str]) -> str:
    load = f"CPU load {shift.before:g} -> {shift.after:g} " if shift is not None else "CPU load shifted "
    return (
        f"{load}at {_fmt_hour(attr.step_at)}; across it wall power went {attr.before_w:.0f} -> "
        f"{attr.after_w:.0f} W ({_fmt_w(attr.delta_w)}): " + _attribution_bits(attr, credible) + "."
    )


def summarize(result: NodeHeatArchaeologyV1) -> str:
    """Plain English, changes in time order. Every number in it is in the structured result."""
    node = result.node
    if result.mean_wall_w is None:
        return f"{node}: no wall-power reading in this window, so nothing can be attributed."
    parts: list[str] = []
    credible = _credible(result.fit) if result.fit is not None else set()
    shift_attrs = [a for a in result.attributions if a.anchor == "load_shift"]
    if not result.steps:
        parts.append(
            f"{node}: no wall-power step of {STEP_MIN_DELTA_W:.0f} W or more stands out from the "
            f"hour-to-hour wander in this window; wall power averaged {result.mean_wall_w:.0f} W "
            f"over {result.hours_with_wall} h."
        )
    else:
        last = result.steps[-1]
        parts.append(
            f"{node}: about {last.after_w:.0f} W since around {_fmt_hour(last.at)}; "
            f"{len(result.steps)} step(s) in this window."
        )
    if result.fit is not None:
        parts.append(f"Fit of hourly wall watts on workloads: R2 {result.fit.r2:.2f} over {result.fit.n_hours} h.")
    elif result.fit_note:
        parts.append(result.fit_note[0].upper() + result.fit_note[1:] + ".")
    by_at = {a.step_at: a for a in result.attributions if a.anchor == "wall_step"}
    parts += [_step_sentence(s, by_at.get(s.at), credible) for s in result.steps]
    shifts_by_at = {s.at: s for s in result.regime_shifts if s.series == "load_15m"}
    parts += [_shift_sentence(a, shifts_by_at.get(a.step_at), credible) for a in shift_attrs]
    if result.fit is not None:
        floor = next((t.watts_per_unit for t in result.fit.terms if t.kind == "intercept"), None)
        if floor is not None:
            parts.append(f"Floor with no listed workload running: ~{floor:.0f} W.")
    told = {a.step_at for a in shift_attrs}
    load_shifts = [s for s in result.regime_shifts if s.series == "load_15m" and s.at not in told]
    if load_shifts:
        parts.append("CPU load (15 min) shifts: " + ", ".join(
            f"{s.before:g} -> {s.after:g} at {_fmt_hour(s.at)}" for s in load_shifts) + ".")
    if result.steps or result.fit is not None:
        parts.append("Correlational; UNVERIFIED until stopping the suspect drops the watts.")
    return " ".join(parts)


def analyze(
    readings: Sequence[HourReading],
    busy: Sequence[PoolBusy] = (),
    *,
    window_start: datetime,
    window_end: datetime,
    nodes: Sequence[str] | None = None,
    pool_node: str = POOL_NODE,
) -> list[NodeHeatArchaeologyV1]:
    """One result per node, largest absolute step first (nodes with no step last)."""
    names = list(nodes) if nodes is not None else sorted({r.node for r in readings})
    out = [
        analyze_node(n, readings, busy, window_start=window_start, window_end=window_end, pool_node=pool_node)
        for n in names
    ]
    return sorted(out, key=lambda r: -abs(_largest_step(r).delta_w) if r.steps else 0.0)
