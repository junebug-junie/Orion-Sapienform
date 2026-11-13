"""Family A: input hygiene, baseline rungs, and joint (whitened) surprise. Pure numpy/pandas.

Spec: docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md, "A. Neural forecaster".
Question: "what will my body and room read next?", every raw reading at 5 and 15 min ahead.

``app/family_a_live.py`` runs these in the service's select -> issue -> score loop (slice 2);
``evals/run_family_a_baselines_eval.py`` runs them on live history for the ladder and calibration.

1. **Hygiene gate** (``screen_series``, ``gate_named_inputs``). Refuse, never degrade. Each series must
   be present, vary (no value over 85 % of buckets, at least 50 distinct values) and return to rest
   (touches its central band on most days; daily medians not a monotone drift). A series resting on its
   floor most of the time is event-like: an input, never a target. Missing stays NaN plus a mask
   channel; it is never filled with 0.0 in real units.
2. **Standardisation** from training rows only (``fit_standardizer``), stored in the manifest.
3. **Rungs**, all predicting the standardised change z[t+k] - z[t] (direct, one fit per lead):
   ``persistence`` (change 0), ``ar`` (one ridge per series on its own last 6 buckets), ``ridge_var``
   (one ridge per target on lags 0,1,2,3,6,12 of every input + its missing-mask at the same lags + time
   of day). ``fit_horizon`` fits them; ``HorizonFit.predict`` serves them.
4. **Split**: the repo's ``block_purged_temporal_split`` (orion/mood_arc/fit_encoder.py) over origin
   indices, purged by the longest lag + lead. Seeds move the block edges.
5. **Significance**: per-origin joint loss (mean squared standardised error over targets), ridge vs the
   better baseline per target, tested with the overlap-aware paired test (``backtest.paired_gap``) AND a
   2-hour moving-block bootstrap whose blocks never straddle two held-out stretches. A lower average
   alone is not a win. ``backtest`` screens inputs on the training rows only.
6. **Joint surprise**: residuals whitened by their own covariance (Mahalanobis d^2, chi-square tail on
   the observed subset), then a conformal p against a held calibration set, so p is uniform when the
   model is calibrated. surprisal = -ln p; calm point = E[-ln p] for a discrete uniform p (``calm_point``)
   -> excess rests at 0 by theory. ``ADAPTIVE`` calibration scales d^2 by the hour-before difficulty
   (``score_joint``). Stored per bucket by the live loop; no node, no bus event, no attention input yet.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# --- config --------------------------------------------------------------------------------------

RAW_TABLES = ("biometrics", "cluster", "cooling", "office_env", "closet_env")
# Hint channels are never inputs (spec: the mood-arc encoder's main failure). Matched on id and key.
HINT_PATTERN = re.compile(r"pressures?$|prediction_error|action_warrant|surprise|salience|hint|warrant|headroom|"
                          r"composite|anomaly|_score")
_ID = re.compile(r"^[a-z0-9_-]+(\.[a-z0-9_]+)+$")
_KEY = re.compile(r"^[a-z0-9_]{1,40}$")
_NODE = re.compile(r"^[a-z0-9_-]{1,40}$")
_DUR = re.compile(r"^\s*(\d+)\s*min\s*$")


class FamilyAConfigError(ValueError):
    pass


class HygieneRefused(RuntimeError):
    """A named input failed the gate at run time: the run refuses instead of training on it."""

    def __init__(self, refusals: Mapping[str, Sequence[str]]):
        self.refusals = {k: tuple(v) for k, v in refusals.items()}
        super().__init__("family A hygiene gate refused: " +
                         "; ".join(f"{k}: {','.join(v)}" for k, v in sorted(self.refusals.items())))


@dataclass(frozen=True)
class Hygiene:
    min_present: float
    max_mode_share: float
    min_distinct: int
    event_mode_share: float
    min_rest_day_share: float
    max_drift_rho: float
    max_stale_min: int
    max_duplicate_corr: float


@dataclass(frozen=True)
class InputSource:
    table: str
    node: str | None = None
    key: str | None = None
    role: str | None = None


@dataclass(frozen=True)
class InputSpec:
    id: str
    unit: str
    source: InputSource


@dataclass(frozen=True)
class FamilyAConfig:
    grain_min: int
    horizons_min: tuple[int, ...]
    hygiene: Hygiene
    inputs: tuple[InputSpec, ...]


def _minutes(v: Any, where: str) -> int:
    m = _DUR.match(str(v))
    if not m or int(m.group(1)) <= 0:
        raise FamilyAConfigError(f"{where}: bad duration {v!r} (want '<n>min')")
    return int(m.group(1))


def _source(raw: Mapping, iid: str) -> InputSource:
    table = raw.get("table")
    if table not in RAW_TABLES:
        raise FamilyAConfigError(f"{iid}: source table {table!r} is not a raw-reading table {RAW_TABLES}")
    if table == "cooling":
        if not raw.get("role") or not _KEY.match(str(raw["role"])) or HINT_PATTERN.search(str(raw["role"])):
            raise FamilyAConfigError(f"{iid}: cooling source needs a [a-z0-9_] role (not a hint channel)")
        return InputSource(table=table, role=str(raw["role"]))
    node, key = raw.get("node"), raw.get("key")
    if not node or not _NODE.match(str(node)) or not key or not _KEY.match(str(key)):
        raise FamilyAConfigError(f"{iid}: {table} source needs a node and a [a-z0-9_] key")
    if HINT_PATTERN.search(str(key)):
        raise FamilyAConfigError(f"{iid}: key {key!r} names a hint channel, not a raw reading")
    return InputSource(table=table, node=str(node), key=str(key))


def parse_config(doc: Mapping) -> FamilyAConfig:
    if (doc or {}).get("version") != 1:
        raise FamilyAConfigError("family A config version must be 1")
    grain = _minutes(doc.get("grain", "5min"), "grain")
    horizons = tuple(sorted(_minutes(h, "horizons") for h in doc.get("horizons") or ()))
    if not horizons or any(h % grain for h in horizons):
        raise FamilyAConfigError("horizons must be non-empty multiples of the grain")
    h = doc.get("hygiene") or {}
    try:
        hyg = Hygiene(min_present=float(h["min_present"]), max_mode_share=float(h["max_mode_share"]),
                      min_distinct=int(h["min_distinct"]), event_mode_share=float(h["event_mode_share"]),
                      min_rest_day_share=float(h["min_rest_day_share"]), max_drift_rho=float(h["max_drift_rho"]),
                      max_stale_min=int(h["max_stale_min"]), max_duplicate_corr=float(h["max_duplicate_corr"]))
    except KeyError as exc:
        raise FamilyAConfigError(f"hygiene.{exc.args[0]} is required (thresholds are fixed in config)") from exc
    inputs, seen = [], set()
    sources: dict[InputSource, str] = {}
    for raw in doc.get("inputs") or ():
        iid = str(raw.get("id", ""))
        if not _ID.match(iid):
            raise FamilyAConfigError(f"bad input id {iid!r}")
        if HINT_PATTERN.search(iid):
            raise FamilyAConfigError(f"{iid}: names a hint channel, not a raw reading")
        if iid in seen:
            raise FamilyAConfigError(f"duplicate input id {iid}")
        seen.add(iid)
        src = _source(raw.get("source") or {}, iid)
        if src in sources:
            raise FamilyAConfigError(f"{iid}: same source as {sources[src]} (one reading, two names)")
        sources[src] = iid
        inputs.append(InputSpec(id=iid, unit=str(raw.get("unit", "")), source=src))
    if not inputs:
        raise FamilyAConfigError("no inputs")
    return FamilyAConfig(grain_min=grain, horizons_min=horizons, hygiene=hyg, inputs=tuple(inputs))


# --- hygiene -------------------------------------------------------------------------------------

TARGET, COVARIATE, REFUSED = "target", "covariate", "refused"


@dataclass(frozen=True)
class SeriesVerdict:
    id: str
    status: str
    reasons: tuple[str, ...]
    stats: dict = field(default_factory=dict)


def _spearman(a: np.ndarray) -> float:
    """Spearman rho of ``a`` against its own index (ties averaged)."""
    from scipy import stats

    if len(a) < 3 or np.all(a == a[0]):
        return 0.0
    return float(stats.spearmanr(np.arange(len(a)), a).statistic)


def screen_series(values: pd.Series, hyg: Hygiene, series_id: str = "") -> SeriesVerdict:
    """Hygiene verdict for one series of grain buckets (DatetimeIndex, NaN = no reading)."""
    v = values.to_numpy(dtype=float)
    present = np.isfinite(v)
    share = float(present.mean()) if len(v) else 0.0
    stats: dict[str, Any] = {"present_share": round(share, 4), "n_buckets": int(len(v))}
    if share < hyg.min_present or present.sum() < 2:
        return SeriesVerdict(series_id, REFUSED, ("absent",), stats)
    x = v[present]
    rounded = np.array([float(f"{a:.6g}") for a in x])
    uniq, counts = np.unique(rounded, return_counts=True)
    mode_i = int(np.argmax(counts))
    mode_share = float(counts[mode_i] / len(x))
    stats.update(distinct=int(len(uniq)), mode_share=round(mode_share, 4), mode=float(uniq[mode_i]),
                 std=float(np.std(x)))
    reasons: list[str] = []
    if len(uniq) < hyg.min_distinct:
        reasons.append("few_distinct")
    if mode_share > hyg.max_mode_share:
        reasons.append("one_value")
    # Can return to rest: most whole days touch the window's central 80 % band, and the daily medians
    # are not a one-way drift (a decay-to-zero or a counter fails here even while it keeps moving).
    lo, hi = np.quantile(x, [0.10, 0.90])
    stats["band"] = [float(lo), float(hi)]
    stats["min"] = float(x.min())
    full_days, rest_share, rho = _rest_shape(values, lo, hi)
    stats.update(n_days=len(full_days), rest_day_share=round(rest_share, 4), drift_rho=round(rho, 4))
    if not full_days or rest_share < hyg.min_rest_day_share:
        reasons.append("no_rest")
    if len(full_days) >= 5 and abs(rho) >= hyg.max_drift_rho:
        reasons.append("drift")
    if reasons:
        return SeriesVerdict(series_id, REFUSED, tuple(reasons), stats)
    if uniq[mode_i] <= float(x.min()) and mode_share > hyg.event_mode_share:
        return SeriesVerdict(series_id, COVARIATE, ("event_like",), stats)
    return SeriesVerdict(series_id, TARGET, (), stats)


def _grain_min(index: pd.DatetimeIndex) -> float:
    return (index[1] - index[0]).total_seconds() / 60.0 if len(index) > 1 else 5.0


def _rest_shape(values: pd.Series, lo: float, hi: float) -> tuple[list, float, float]:
    """(whole days, share of them touching [lo, hi], Spearman rho of their medians vs time).
    A whole day holds readings in at least half of a full day's buckets (not of the window's share of
    that day: a window edge 20 min into a day is not a day)."""
    s = values[np.isfinite(values.to_numpy(dtype=float))]
    per_day = s.groupby(s.index.floor("1D"))
    need = 0.5 * 1440.0 / _grain_min(values.index)
    days = [d for d, g in per_day if len(g) >= need]
    touched = [d for d in days if bool(((per_day.get_group(d) >= lo) & (per_day.get_group(d) <= hi)).any())]
    share = len(touched) / len(days) if days else 0.0
    medians = np.array([float(per_day.get_group(d).median()) for d in days])
    rho = _spearman(medians) if len(medians) >= 5 else 0.0
    return days, share, rho


# The full screen counts distinct values and whole days, both of which scale with the window: a 24 h
# scoring window of a 1 C-resolution GPU temperature honestly holds < 50 distinct 5-min means. Windows
# shorter than this are gated by the run-time checks below instead (see gate_named_inputs).
FULL_SCREEN_MIN_DAYS = 7


def screen_frame(frame: pd.DataFrame, hyg: Hygiene) -> dict[str, SeriesVerdict]:
    """Per-series verdicts, then the independence check: a passing series whose bucket means correlate
    with an earlier-listed passing series above ``max_duplicate_corr`` is the same reading twice (the
    CCS811's eCO2 is computed from its TVOC: corr 0.99999, 2026-11-13) and is refused as a duplicate."""
    out = {c: screen_series(frame[c], hyg, c) for c in frame.columns}
    kept: list[str] = []
    for c in frame.columns:
        v = out[c]
        if v.status == REFUSED:
            continue
        dup = next((k for k in kept if abs(float(np.nan_to_num(frame[c].corr(frame[k], min_periods=50))))
                    > hyg.max_duplicate_corr), None)
        if dup is not None:
            out[c] = SeriesVerdict(c, REFUSED, (f"duplicate_of:{dup}",), v.stats)
        else:
            kept.append(c)
    return out


def stale_inputs(last_seen: Mapping[str, datetime | None], now: datetime, max_stale_min: int) -> list[str]:
    out = []
    for iid, ts in last_seen.items():
        if ts is None or (now - ts).total_seconds() > max_stale_min * 60:
            out.append(iid)
    return sorted(out)


def _short_window_reasons(values: pd.Series, hyg: Hygiene, band: Sequence[float] | None,
                          status: str = TARGET, floor: float | None = None) -> list[str]:
    """Run-time checks for a scoring window shorter than FULL_SCREEN_MIN_DAYS: present, not stuck on one
    value, and still inside the regime it was trained on (touches the TRAINING central 80 % band).

    An event-like covariate whose most common value is its training floor (``floor``) may sit there all
    day -- a quiet day is its rest state, not a stuck sensor -- so ``one_value`` and the band check do
    not apply to it; ``absent`` and stale still do."""
    v = values.to_numpy(dtype=float)
    present = np.isfinite(v)
    if not len(v) or present.mean() < hyg.min_present or present.sum() < 2:
        return ["absent"]
    x = v[present]
    _u, counts = np.unique(np.array([float(f"{a:.6g}") for a in x]), return_counts=True)
    out = []
    if status == COVARIATE and floor is not None and float(np.min(x)) >= floor - 1e-9 and \
            float(_u[int(np.argmax(counts))]) <= floor + 1e-9:
        return out
    if counts.max() / len(x) > hyg.max_mode_share:
        out.append("one_value")
    if band is not None and not bool(((x >= band[0]) & (x <= band[1])).any()):
        out.append("left_trained_band")
    return out


def gate_named_inputs(named: Mapping[str, str], frame: pd.DataFrame, hyg: Hygiene,
                      last_seen: Mapping[str, datetime | None] | None = None, now: datetime | None = None,
                      bands: Mapping[str, Sequence[float]] | None = None,
                      floors: Mapping[str, float] | None = None) -> None:
    """Refuse (raise ``HygieneRefused``) unless every named input still passes, in the same role.

    ``named`` maps input id -> status in the manifest (target / covariate). Windows of at least
    FULL_SCREEN_MIN_DAYS get the full screen: an input now refused, or a named target that turned
    event-like, refuses the run. Shorter (scoring) windows get ``_short_window_reasons`` against the
    training band in ``bands`` (manifest ``stats.band``; ``floors`` = ``stats.min`` for event-like
    covariates). Any input missing from the frame, or stale,
    refuses either way.
    """
    span_days = (len(frame) * _grain_min(frame.index)) / 1440.0
    refusals: dict[str, list[str]] = {}
    for iid, status in named.items():
        if iid not in frame.columns:
            refusals[iid] = ["not_loaded"]
            continue
        if span_days < FULL_SCREEN_MIN_DAYS:
            reasons = _short_window_reasons(frame[iid], hyg, (bands or {}).get(iid), status,
                                            (floors or {}).get(iid))
            if reasons:
                refusals[iid] = reasons
            continue
        verdict = screen_series(frame[iid], hyg, iid)
        if verdict.status == REFUSED:
            refusals[iid] = list(verdict.reasons)
        elif status == TARGET and verdict.status != TARGET:
            refusals[iid] = ["role_changed:" + verdict.status]
    if last_seen is not None and now is not None:
        for iid in stale_inputs({k: last_seen.get(k) for k in named}, now, hyg.max_stale_min):
            refusals.setdefault(iid, []).append("stale")
    if refusals:
        raise HygieneRefused(refusals)


# --- standardisation -----------------------------------------------------------------------------

STD_FLOOR = 1e-9


def fit_standardizer(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-series mean/std from TRAINING rows only (NaN ignored). Stored in the manifest."""
    out = {}
    for c in train.columns:
        x = train[c].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 2:
            raise ValueError(f"{c}: fewer than 2 training readings, cannot standardise")
        out[c] = {"mean": float(x.mean()), "std": float(max(x.std(), STD_FLOOR))}
    return out


def standardize(frame: pd.DataFrame, stats: Mapping[str, Mapping[str, float]]) -> np.ndarray:
    """(n, m) standardised values, NaN where no reading (never 0.0-filled here)."""
    cols = list(frame.columns)
    mu = np.array([stats[c]["mean"] for c in cols])
    sd = np.array([stats[c]["std"] for c in cols])
    return (frame.to_numpy(dtype=float) - mu) / sd


# --- split ---------------------------------------------------------------------------------------

VAR_LAGS = (0, 1, 2, 3, 6, 12)
AR_LAGS = 6
HELD_OUT_FRAC = 0.2
N_BLOCKS = 5


def purge_rows(horizon_steps: int) -> int:
    """Embargo between train and held-out origins: the longest feature lag (VAR 12 or AR 5, summed here
    -- more than needed, on purpose) plus the lead."""
    return max(VAR_LAGS) + max(AR_LAGS - 1, 0) + horizon_steps


def blocked_split(n: int, horizon_steps: int, seed: int, n_blocks: int = N_BLOCKS,
                  held_out_frac: float = HELD_OUT_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, held_idx) origin indices via the repo's block_purged_temporal_split.

    The seed trims a random 0..(block/2) rows off the front, which moves every block edge, so two seeds
    test different held-out stretches of the same history.
    """
    from orion.mood_arc.fit_encoder import block_purged_temporal_split

    rng = np.random.default_rng(seed)
    offset = int(rng.integers(0, max(1, n // n_blocks // 2))) if seed else 0
    idx = np.arange(offset, n)
    train, held = block_purged_temporal_split([np.array([i]) for i in idx], held_out_frac=held_out_frac,
                                              purge_gap_windows=purge_rows(horizon_steps), n_blocks=n_blocks)
    return train.ravel().astype(int), held.ravel().astype(int)


def inner_split(train_idx: np.ndarray, horizon_steps: int, n_blocks: int = N_BLOCKS) -> tuple[np.ndarray, np.ndarray]:
    """Split TRAIN origins again (same method) into fit rows and validation rows (alpha choice + conformal
    calibration). Validation is spread over every block, like the held-out set it stands in for."""
    from orion.mood_arc.fit_encoder import block_purged_temporal_split

    fit, val = block_purged_temporal_split([np.array([i]) for i in train_idx], held_out_frac=0.25,
                                           purge_gap_windows=purge_rows(horizon_steps), n_blocks=n_blocks)
    return fit.ravel().astype(int), val.ravel().astype(int)


# --- features + ridge ----------------------------------------------------------------------------

# Up to 1e5 so the validation choice can shrink the cross-series terms to ~0 (ridge_var -> AR).
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)


def _lagged(a: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    if lag == 0:
        return a.copy()
    out[lag:] = a[:-lag]
    return out


def var_features(z_all: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    """(n, f): every input's standardised value at lags VAR_LAGS (missing -> 0 = the training mean), then
    every input's missing-mask at the same lags (1 = no reading at that lag), then time of day. So a
    reading missing 30 min ago is flagged at that lag, not read as an ordinary mean value (slice 1 had a
    lag-0 mask only). Rows inside the first max(VAR_LAGS) have no lag history: those lags read as
    missing (value 0, mask 1)."""
    filled = np.where(np.isfinite(z_all), z_all, 0.0)
    cols = [_lagged(filled, lag) for lag in VAR_LAGS]
    cols = [np.where(np.isfinite(c), c, 0.0) for c in cols]
    mask = (~np.isfinite(z_all)).astype(float)
    masks = [_lagged(mask, lag) for lag in VAR_LAGS]
    masks = [np.where(np.isfinite(c), c, 1.0) for c in masks]
    minutes = np.asarray(index.hour * 60 + index.minute, dtype=float)
    phase = 2 * np.pi * minutes / 1440.0
    return np.column_stack(cols + masks + [np.sin(phase), np.cos(phase)])


def ar_features(z: np.ndarray) -> np.ndarray:
    """(n, AR_LAGS) own lags 0..AR_LAGS-1; NaN where any is missing (AR rows need full history)."""
    return np.column_stack([_lagged(z, lag) for lag in range(AR_LAGS)])


def deltas(z: np.ndarray, k: int) -> np.ndarray:
    """z[t+k] - z[t] per row (NaN where either is missing or t+k is past the end)."""
    out = np.full_like(z, np.nan)
    out[:-k] = z[k:] - z[:-k]
    return out


@dataclass
class RidgeFit:
    beta: np.ndarray
    intercept: float
    alpha: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta + self.intercept


class GramCache:
    """The eigendecomposition of a row subset's centred Gram matrix, shared by every target fitted on
    the same rows. Most readings are present on the same rows, so ridge_var's ~500-column Gram is built
    once per row set instead of once per target (the expensive part of a fit). Exact: same numbers."""

    def __init__(self, X: np.ndarray, max_entries: int = 4):
        self.X = X
        self.max_entries = max_entries
        self._c: dict[bytes, tuple] = {}

    def get(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = np.asarray(rows, dtype=np.int64).tobytes()
        hit = self._c.get(key)
        if hit is None:
            Xr = self.X[rows]
            xm = Xr.mean(axis=0)
            Xc = Xr - xm
            w, V = np.linalg.eigh(Xc.T @ Xc)
            hit = (xm, Xc, np.clip(w, 0.0, None), V)
            if len(self._c) >= self.max_entries:
                self._c.pop(next(iter(self._c)))
            self._c[key] = hit
        return hit


def ridge_path(X: np.ndarray, y: np.ndarray, alphas: Sequence[float] = ALPHAS,
               gram: tuple | None = None) -> list[RidgeFit]:
    """Closed-form ridge for several alphas (centered; intercept unpenalised). ``gram`` = a precomputed
    (x mean, centred X, eigenvalues, eigenvectors) of these same rows (``GramCache.get``)."""
    ym = float(y.mean())
    yc = y - ym
    if gram is None:
        xm = X.mean(axis=0)
        Xc = X - xm
        w, V = np.linalg.eigh(Xc.T @ Xc)
        w = np.clip(w, 0.0, None)
    else:
        xm, Xc, w, V = gram
    proj = V.T @ (Xc.T @ yc)
    fits = []
    for a in alphas:
        beta = V @ (proj / (w + a))
        fits.append(RidgeFit(beta=beta, intercept=ym - float(xm @ beta), alpha=float(a)))
    return fits


def fit_ridge_selected(X: np.ndarray, y: np.ndarray, sub_rows: np.ndarray, sel_rows: np.ndarray,
                       fit_rows: np.ndarray, cache: GramCache | None = None) -> RidgeFit:
    """Choose alpha by fitting on ``sub_rows`` and scoring on ``sel_rows`` (both inside ``fit_rows``), then
    refit on all of ``fit_rows`` with that alpha. Rows outside ``fit_rows`` (conformal calibration,
    held-out) never influence the choice. ``cache`` (built on this same X) shares Gram matrices."""
    alpha = ALPHAS[len(ALPHAS) // 2]
    if len(sub_rows) >= 2 and len(sel_rows):
        path = ridge_path(X[sub_rows], y[sub_rows], gram=cache.get(sub_rows) if cache else None)
        losses = [float(np.mean((f.predict(X[sel_rows]) - y[sel_rows]) ** 2)) for f in path]
        alpha = path[int(np.argmin(losses))].alpha
    return ridge_path(X[fit_rows], y[fit_rows], alphas=(alpha,), gram=cache.get(fit_rows) if cache else None)[0]


# --- fit + backtest ------------------------------------------------------------------------------

MIN_FIT_ROWS = 288   # one day of origins per target
RUNGS = ("ridge_var", "ar")   # the rungs that can serve; persistence is the floor they are tested against


@dataclass
class TargetFit:
    ar: RidgeFit
    rv: RidgeFit


@dataclass
class HorizonFit:
    """One lead's fitted rungs: train-only standardisation, AR + ridge_var per target, and the row sets
    they came from. ``predict`` works on any frame laid out like the training one (same input columns,
    a full grain grid), so the live loop calls it on the last day of readings."""
    horizon_steps: int
    targets: list[str]
    inputs: list[str]
    standardizer: dict[str, dict[str, float]]
    fits: dict[str, TargetFit]
    alphas: dict[str, dict[str, float]]
    refused_targets: dict[str, str]
    fit_rows: np.ndarray
    val_rows: np.ndarray

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Per rung: (n, m) predicted standardised change z[t+k] - z[t] per row x target, NaN where the
        target cannot be forecast (an AR lag missing, or the target refused at fit time). Also
        ``z`` (n, m): the targets' standardised values, and ``y`` (n, m): the real change (NaN past
        the end)."""
        z_all = standardize(frame[self.inputs], self.standardizer)
        X = var_features(z_all, frame.index)
        col = {c: i for i, c in enumerate(self.inputs)}
        n, m = len(frame), len(self.targets)
        out = {r: np.full((n, m), np.nan) for r in RUNGS}
        out["z"] = np.full((n, m), np.nan)
        out["y"] = np.full((n, m), np.nan)
        for j, t in enumerate(self.targets):
            z = z_all[:, col[t]]
            out["z"][:, j] = z
            out["y"][:, j] = deltas(z, self.horizon_steps)
            tf = self.fits.get(t)
            if tf is None:
                continue
            A = ar_features(z)
            ok = np.isfinite(A).all(axis=1)
            ar_pred = np.full(n, np.nan)
            ar_pred[ok] = tf.ar.predict(A[ok])
            out["ar"][:, j] = ar_pred
            out["ridge_var"][ok, j] = ar_pred[ok] + tf.rv.predict(X[ok])
        return out


def fit_horizon(frame: pd.DataFrame, targets: Sequence[str], inputs: Sequence[str], horizon_steps: int,
                train_idx: np.ndarray) -> HorizonFit:
    """Fit AR and ridge_var for every target at one lead on TRAIN origins only.

    Standardisation uses the training origins' rows. Train origins are split again (purged) into fit
    rows (the model) and validation rows (conformal calibration only); the fit rows are split once
    more to choose each ridge penalty, so the calibration rows choose nothing."""
    train_idx = np.sort(np.asarray(train_idx, dtype=int))
    stats = fit_standardizer(frame.iloc[train_idx][list(inputs)])
    z_all = standardize(frame[list(inputs)], stats)
    X = var_features(z_all, frame.index)
    fit_rows, val_rows = inner_split(train_idx, horizon_steps)
    sub_rows, sel_rows = inner_split(fit_rows, horizon_steps)
    n = len(frame)
    col = {c: i for i, c in enumerate(inputs)}
    cache = GramCache(X)
    fits: dict[str, TargetFit] = {}
    alphas: dict[str, dict[str, float]] = {}
    refused: dict[str, str] = {}
    for t in targets:
        z = z_all[:, col[t]]
        y = deltas(z, horizon_steps)
        A = ar_features(z)
        ok_ar = np.isfinite(A).all(axis=1) & np.isfinite(y)
        f_ar = fit_rows[ok_ar[fit_rows]]
        s_ar, e_ar = sub_rows[ok_ar[sub_rows]], sel_rows[ok_ar[sel_rows]]
        if len(f_ar) < MIN_FIT_ROWS:
            refused[t] = f"fit_rows={len(f_ar)} < {MIN_FIT_ROWS}"
            continue
        ar = fit_ridge_selected(A, y, s_ar, e_ar, f_ar)
        # ridge_var is nested on AR: the AR forecast plus a ridge on every input's lags fitted to what AR
        # missed. At a large alpha it IS the AR forecast, so the rung above can only win by what the
        # other readings add -- the ladder's "keep a rung only if it beats the one below".
        ar_pred = np.full(n, np.nan)
        ar_pred[ok_ar] = ar.predict(A[ok_ar])
        rv = fit_ridge_selected(X, y - np.where(ok_ar, ar_pred, 0.0), s_ar, e_ar, f_ar, cache=cache)
        fits[t] = TargetFit(ar=ar, rv=rv)
        alphas[t] = {"ar": ar.alpha, "ridge_var": rv.alpha}
    return HorizonFit(horizon_steps=horizon_steps, targets=list(targets), inputs=list(inputs), standardizer=stats,
                      fits=fits, alphas=alphas, refused_targets=refused, fit_rows=fit_rows, val_rows=val_rows)


@dataclass
class HorizonRun:
    horizon_min: int
    seed: int
    targets: list[str]
    held: np.ndarray                  # held-out origin indices
    sq: dict[str, np.ndarray]         # model -> (len(held), m) squared standardised error, NaN = not scored
    resid_val: np.ndarray             # (len(val), m) ridge residuals on validation rows (conformal calibration)
    resid_fit: np.ndarray             # (len(fit), m) ridge residuals on fit rows (covariance)
    resid_held: np.ndarray            # (len(held), m) ridge residuals on held-out rows
    alphas: dict[str, dict[str, float]]
    refused_targets: dict[str, str] = field(default_factory=dict)
    # (n, m) residuals of each serving rung at EVERY row the rung could forecast and the truth exists
    # (in-sample on fit rows). The adaptive difficulty reads the hour before each origin from here.
    resid_all: dict[str, np.ndarray] = field(default_factory=dict)
    fit_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    val_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    horizon_steps: int = 1


def run_horizon(frame: pd.DataFrame, targets: Sequence[str], inputs: Sequence[str], horizon_steps: int,
                grain_min: int, seed: int, train_idx: np.ndarray | None = None,
                held_idx: np.ndarray | None = None) -> HorizonRun:
    """Fit persistence / AR / ridge_var for every target at one lead and score them on held-out origins.

    Standardisation uses the training origins' rows only. ``train_idx``/``held_idx`` override the blocked
    split (the live rest check passes "everything before the last day" / "the last day").
    """
    n = len(frame)
    if train_idx is None or held_idx is None:
        train_idx, held_idx = blocked_split(n, horizon_steps, seed)
    hf = fit_horizon(frame, targets, inputs, horizon_steps, train_idx)
    pr = hf.predict(frame)
    y = pr["y"]
    m = len(targets)
    resid_all = {r: y - pr[r] for r in RUNGS}
    rv_res = resid_all["ridge_var"]
    # Common scoring set: every rung can forecast this origin and the outcome is real.
    ok = np.isfinite(rv_res) & np.isfinite(pr["ar"])
    sq = {name: np.full((len(held_idx), m), np.nan) for name in ("persistence", "ar", "ridge_var")}
    h_ok = ok[held_idx]
    yh = y[held_idx]
    sq["persistence"][h_ok] = (yh ** 2)[h_ok]
    sq["ar"][h_ok] = ((pr["ar"][held_idx] - yh) ** 2)[h_ok]
    sq["ridge_var"][h_ok] = (rv_res[held_idx] ** 2)[h_ok]
    resid_held = np.where(h_ok, rv_res[held_idx], np.nan)
    return HorizonRun(horizon_min=horizon_steps * grain_min, seed=seed, targets=list(targets), held=held_idx, sq=sq,
                      resid_val=rv_res[hf.val_rows], resid_fit=rv_res[hf.fit_rows], resid_held=resid_held,
                      alphas=hf.alphas, refused_targets=hf.refused_targets, resid_all=resid_all,
                      fit_rows=hf.fit_rows, val_rows=hf.val_rows, horizon_steps=horizon_steps)


def backtest(frame: pd.DataFrame, hyg: Hygiene, horizon_steps: int, grain_min: int,
             seed: int) -> tuple[HorizonRun | None, dict[str, SeriesVerdict]]:
    """One seed of the backtest with the input screen on TRAINING rows only.

    Slice 1 screened the whole window, so which readings became targets depended on the held-out rows'
    shape. Here the held-out rows are dropped before the screen (shape, not predictability, but still a
    decision the held-out rows must not touch). Returns (None, verdicts) when no input passes as a target."""
    train_idx, held_idx = blocked_split(len(frame), horizon_steps, seed)
    verdicts = screen_frame(frame.iloc[np.sort(train_idx)], hyg)
    targets = [k for k, v in verdicts.items() if v.status == TARGET]
    covs = [k for k, v in verdicts.items() if v.status == COVARIATE]
    if not targets:
        return None, verdicts
    return run_horizon(frame, targets, targets + covs, horizon_steps, grain_min, seed,
                       train_idx=train_idx, held_idx=held_idx), verdicts


# --- significance --------------------------------------------------------------------------------

BOOT_N = 2000
# Moving-block length for the bootstrap, in minutes. Held-out origins come in ~5 contiguous stretches of
# ~12 h each at 14 days; a one-day block would span stretches and leave ~3 blocks per resample.
BOOT_BLOCK_MIN = 120


def segments_of(positions: np.ndarray) -> np.ndarray:
    """Segment id per element of a sorted position array: a new segment wherever the next position is
    not the neighbour of the last (a held-out stretch ends, or a missing hour breaks a run)."""
    positions = np.asarray(positions)
    if not len(positions):
        return np.zeros(0, dtype=int)
    return np.concatenate([[0], np.cumsum(np.diff(positions) != 1)]).astype(int)


def bootstrap_blocks(segments: np.ndarray, block: int) -> list[np.ndarray]:
    """Candidate blocks (row positions): every window of ``block`` rows inside one segment, or a segment
    shorter than that whole. No candidate holds rows of two segments."""
    seg = np.asarray(segments)
    cands: list[np.ndarray] = []
    for sid in np.unique(seg):
        pos = np.flatnonzero(seg == sid)
        if len(pos) <= block:
            cands.append(pos)
        else:
            cands.extend(pos[i:i + block] for i in range(len(pos) - block + 1))
    return cands


def block_bootstrap_p(d: np.ndarray, block: int, seed: int, n_boot: int = BOOT_N,
                      segments: np.ndarray | None = None) -> tuple[float, float, float]:
    """One-sided p that mean(d) < 0, from a moving-block bootstrap of the centred series (the null
    "true mean is 0"), plus the 95 % percentile interval of the mean. NaN rows dropped first.

    ``segments`` (same length as ``d``): a block never straddles two segments (two held-out stretches
    hours apart are not neighbours; slice 1 let a block join them). A segment shorter than ``block``
    contributes itself whole as one candidate block."""
    d = np.asarray(d, dtype=float)
    seg = np.zeros(len(d), dtype=int) if segments is None else np.asarray(segments)
    keep = np.isfinite(d)
    d, seg = d[keep], seg[keep]
    n = len(d)
    if n < 2 * block or n < 10:
        return float("nan"), float("nan"), float("nan")
    cands = bootstrap_blocks(seg, block)
    rng = np.random.default_rng(seed)
    obs = float(d.mean())
    means = np.empty(n_boot)
    for b in range(n_boot):
        tot, cnt = 0.0, 0
        while cnt < n:
            c = cands[int(rng.integers(0, len(cands)))]
            take = c[:n - cnt]
            tot += float(d[take].sum())
            cnt += len(take)
        means[b] = tot / cnt
    null = means - obs                                  # bootstrap distribution under mean 0
    p = float((1 + np.sum(null <= obs)) / (n_boot + 1))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return p, float(lo), float(hi)


def joint_loss(sq: np.ndarray) -> np.ndarray:
    """Per origin: mean squared standardised error over the targets scored there (NaN if none)."""
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(sq).sum(axis=1)
        tot = np.nansum(sq, axis=1)
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def best_baseline(run: HorizonRun) -> tuple[np.ndarray, dict[str, str]]:
    """Per target, the baseline (persistence or AR) with the lower held-out loss -- chosen on the
    held-out set itself, which favours the baseline (a conservative bar for ridge)."""
    out = np.full_like(run.sq["persistence"], np.nan)
    pick = {}
    for j, t in enumerate(run.targets):
        p, a = run.sq["persistence"][:, j], run.sq["ar"][:, j]
        use_ar = np.nanmean(a) < np.nanmean(p) if np.isfinite(a).any() else False
        out[:, j] = a if use_ar else p
        pick[t] = "ar" if use_ar else "persistence"
    return out, pick


def compare(run: HorizonRun, candidate: str, reference: np.ndarray, grain_min: int, seed: int) -> dict:
    """Paired comparison on per-origin joint loss: overlap-aware t (backtest.paired_gap) and a 2-hour
    moving-block bootstrap. Both must say p < 0.05 for a win."""
    from app.backtest import overlap_lag, paired_gap
    from scipy import stats

    lc = joint_loss(run.sq[candidate])
    lr = joint_loss(reference)
    both = np.isfinite(lc) & np.isfinite(lr)
    seg = segments_of(np.asarray(run.held))[both]
    lc, lr = lc[both], lr[both]
    d = lc - lr
    mean, se, n = paired_gap(lc, lr, overlap_lag(run.horizon_min, grain_min))
    p_t = None if mean is None else (float(stats.t.cdf(mean / se, df=n - 1)) if se else (0.0 if mean < 0 else 1.0))
    p_b, lo, hi = block_bootstrap_p(d, block=max(1, BOOT_BLOCK_MIN // grain_min), seed=seed, segments=seg)
    ratio = float(lc.mean() / lr.mean()) if len(lr) and lr.mean() > 0 else float("nan")
    return {"n": int(n), "loss_candidate": float(lc.mean()) if len(lc) else None,
            "loss_reference": float(lr.mean()) if len(lr) else None, "ratio": ratio,
            "gap": mean, "gap_ci95": [lo, hi], "p_paired_t": p_t, "p_block_bootstrap": p_b,
            "wins": bool(p_t is not None and p_t < 0.05 and np.isfinite(p_b) and p_b < 0.05)}


def per_target_table(run: HorizonRun, reference: np.ndarray, pick: Mapping[str, str], grain_min: int) -> list[dict]:
    """Per target: ridge / best-baseline loss ratio and a Holm-adjusted one-sided paired p."""
    from app.backtest import overlap_lag, paired_win_p
    from app.live import holm

    raw, rows = {}, []
    for j, t in enumerate(run.targets):
        c, r = run.sq["ridge_var"][:, j], reference[:, j]
        both = np.isfinite(c) & np.isfinite(r)
        if both.sum() < 10:
            continue
        p, _n = paired_win_p(c[both], r[both], overlap_lag(run.horizon_min, grain_min))
        raw[t] = 1.0 if p is None else p
        rows.append({"target": t, "baseline": pick[t], "n": int(both.sum()),
                     "ratio": float(c[both].mean() / max(r[both].mean(), 1e-12))})
    adj = holm(raw) if raw else {}
    for row in rows:
        row["p_holm"] = adj.get(row["target"])
    return rows


# --- joint surprise ------------------------------------------------------------------------------

SHRINK = 0.10


def calm_point(n_cal: int) -> float:
    """E[-ln p] for a conformal p uniform on {1, ..., n+1} / (n+1): ln(n+1) - ln((n+1)!)/(n+1).
    -> 1.0 as n grows (the continuous E[-ln U])."""
    m = n_cal + 1
    return math.log(m) - math.lgamma(m + 1) / m


EIG_FLOOR = 1e-3   # x mean variance: the smallest eigenvalue the whitening may divide by


def shrunk_covariance(resid: np.ndarray, shrink: float = SHRINK) -> tuple[np.ndarray, np.ndarray]:
    """(cov, usable): pairwise-complete residual covariance, shrunk toward its diagonal, then
    eigen-clipped so it is positive definite. Pairwise-complete estimates under blocky missingness are
    not PSD on their own (min eigenvalue -0.20 seen in review), and a negative direction would make d^2
    negative and read as calm. ``usable`` marks targets with a finite, positive variance from >= 10 fit
    rows; the rest are left out of the joint score (never given a tiny variance that explodes d^2)."""
    df = pd.DataFrame(resid)
    cov = df.cov(min_periods=10).to_numpy()
    diag = np.diag(cov)
    usable = np.isfinite(diag) & (diag > 0)
    c = cov[np.ix_(usable, usable)]
    c = np.where(np.isfinite(c), c, 0.0)
    c = (1 - shrink) * c + shrink * np.diag(np.diag(c))
    out = np.full_like(cov, np.nan)
    if usable.any():
        w, V = np.linalg.eigh(c)
        w = np.maximum(w, EIG_FLOOR * float(np.mean(np.diag(c))))
        out[np.ix_(usable, usable)] = (V * w) @ V.T
    return out, usable


def mahalanobis_tail(resid: np.ndarray, cov: np.ndarray, min_observed: int) -> tuple[np.ndarray, np.ndarray, list]:
    """Per row: d^2 on the observed targets, its chi-square tail probability (dimension = observed
    count), and per-target shares of d^2 (attribution hint). Rows with < min_observed targets -> NaN."""
    from scipy import stats

    n = len(resid)
    d2 = np.full(n, np.nan)
    tail = np.full(n, np.nan)
    shares: list = [None] * n
    for i in range(n):
        o = np.isfinite(resid[i])
        if o.sum() < min_observed:
            continue
        r = resid[i, o]
        w = np.linalg.solve(cov[np.ix_(o, o)], r)
        q = float(r @ w)
        d2[i] = q
        tail[i] = float(stats.chi2.sf(q, df=int(o.sum())))
        s = np.full(resid.shape[1], np.nan)
        s[o] = (r * w) / q if q > 0 else 0.0
        shares[i] = s
    return d2, tail, shares


def conformal_p(tail: np.ndarray, cal_tail: np.ndarray) -> np.ndarray:
    """p = (1 + #{calibration tails <= this tail}) / (n_cal + 1): uniform when calibration and test are
    exchangeable, whatever the residuals' real shape (heavy tails included)."""
    cal = np.sort(cal_tail[np.isfinite(cal_tail)])
    out = np.full(len(tail), np.nan)
    ok = np.isfinite(tail)
    out[ok] = (1 + np.searchsorted(cal, tail[ok], side="right")) / (len(cal) + 1)
    return out


# Adaptive calibration (the over-calm fix; same method as app/adaptive.py, normalized split conformal,
# Lei et al. 2018): each origin's d^2 is divided by how big the joint misses were in the hour before it
# -- the mean d^2 / dof of the rows whose TARGET fell in that hour, all known at issue time -- and the
# conformal p is taken against calibration rows scaled the same way. One covariance gives calm and busy
# hours the same yardstick, so a calm day reads below 0 (slice 1: 68-72 % of hours at rest vs ~55 %);
# a locally scaled covariance (s * Sigma) judges each hour against its own recent difficulty.
DIFFICULTY_ROWS = 12          # one hour of 5-min targets
MIN_DIFFICULTY_N = 6          # scored rows inside that hour, else no difficulty (adaptive p is NaN)
DIFF_FLOOR_Q = 0.05           # floor: this quantile of the calibration difficulties (and >= 0.1 x median)
STATIC, ADAPTIVE = "static", "adaptive"


@dataclass
class JointCal:
    """Everything the joint score needs from a fit, small enough to store with the model run: the
    whitening covariance (zero outside ``usable``), the "half the usable targets observed" bar, and the
    calibration tails for the static and adaptive p."""
    cov: np.ndarray
    usable: np.ndarray
    min_obs: int
    cal_tail: np.ndarray
    cal_tail_adaptive: np.ndarray
    diff_floor: float | None

    def to_json(self) -> dict:
        def r6(a) -> list:
            return [float(f"{v:.6g}") for v in np.asarray(a, dtype=float).ravel()]
        return {"cov": [r6(row) for row in self.cov], "usable": [bool(u) for u in self.usable],
                "min_obs": int(self.min_obs), "cal_tail": r6(self.cal_tail),
                "cal_tail_adaptive": r6(self.cal_tail_adaptive), "diff_floor": self.diff_floor}

    @classmethod
    def from_json(cls, d: Mapping) -> "JointCal":
        return cls(cov=np.asarray(d["cov"], dtype=float), usable=np.asarray(d["usable"], dtype=bool),
                   min_obs=int(d["min_obs"]), cal_tail=np.asarray(d["cal_tail"], dtype=float),
                   cal_tail_adaptive=np.asarray(d["cal_tail_adaptive"], dtype=float),
                   diff_floor=None if d.get("diff_floor") is None else float(d["diff_floor"]))


def joint_d2(resid: np.ndarray, cal: JointCal) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """(d2, dof, static chi-square tail, shares) per row on the usable targets observed there; rows with
    fewer than ``min_obs`` observed usable targets are NaN (dof 0)."""
    keep = np.where(cal.usable, 1.0, np.nan)
    r = np.asarray(resid, dtype=float) * keep
    d2, tail, shares = mahalanobis_tail(r, cal.cov, cal.min_obs)
    dof = np.where(np.isfinite(d2), np.isfinite(r).sum(axis=1), 0).astype(int)
    return d2, dof, tail, shares


def difficulty_rows(ratio: np.ndarray, horizon_steps: int, window: int = DIFFICULTY_ROWS,
                    min_n: int = MIN_DIFFICULTY_N) -> np.ndarray:
    """Per origin row i: mean of ``ratio`` (d^2 / dof) over the origins whose target fell in the hour up
    to origin i, i.e. rows [i - k - window + 1, i - k]. NaN where fewer than ``min_n`` are finite. Rows
    are a full grain grid, so a gap is NaN rows, never neighbours."""
    rolled = pd.Series(np.asarray(ratio, dtype=float)).rolling(window, min_periods=min_n).mean().to_numpy()
    out = np.full(len(rolled), np.nan)
    k = int(horizon_steps)
    if k < len(rolled):
        out[k:] = rolled[:len(rolled) - k]
    return out


def _floor_of(difficulties: np.ndarray) -> float | None:
    d = difficulties[np.isfinite(difficulties) & (difficulties > 0)]
    if not len(d):
        return None
    return max(float(np.quantile(d, DIFF_FLOOR_Q)), 0.1 * float(np.median(d)), 1e-9)


def adaptive_tail(d2: np.ndarray, dof: np.ndarray, difficulty: np.ndarray, floor: float | None) -> np.ndarray:
    """chi-square tail of d^2 / s on dof, s = the hour's difficulty floored (a flat hour must not blow a
    score up). NaN where there is no difficulty."""
    from scipy import stats

    out = np.full(len(d2), np.nan)
    if floor is None:
        return out
    ok = np.isfinite(d2) & np.isfinite(difficulty) & (dof > 0)
    s = np.maximum(difficulty[ok], floor)
    out[ok] = stats.chi2.sf(d2[ok] / s, df=dof[ok])
    return out


def joint_calibration(resid_fit: np.ndarray, resid_val: np.ndarray, val_difficulty: np.ndarray | None = None,
                      min_observed_frac: float = 0.5) -> JointCal:
    """Covariance from fit-row residuals; static calibration tails from validation rows; adaptive tails
    from the same validation rows scaled by ``val_difficulty`` (their own hour-before difficulty)."""
    cov, usable = shrunk_covariance(resid_fit)
    m = int(usable.sum())   # refused targets (all-NaN columns) and unusable variances do not count
    cov = np.where(np.isfinite(cov), cov, 0.0)
    cal = JointCal(cov=cov, usable=usable, min_obs=max(1, int(math.ceil(min_observed_frac * m))),
                   cal_tail=np.zeros(0), cal_tail_adaptive=np.zeros(0), diff_floor=None)
    d2, dof, tail, _ = joint_d2(resid_val, cal)
    cal.cal_tail = tail[np.isfinite(tail)]
    if val_difficulty is not None:
        cal.diff_floor = _floor_of(np.asarray(val_difficulty, dtype=float))
        at = adaptive_tail(d2, dof, np.asarray(val_difficulty, dtype=float), cal.diff_floor)
        cal.cal_tail_adaptive = at[np.isfinite(at)]
    return cal


@dataclass
class JointScore:
    d2: np.ndarray
    dof: np.ndarray
    p_static: np.ndarray
    p_adaptive: np.ndarray
    shares: list


def score_joint(resid: np.ndarray, cal: JointCal, difficulty: np.ndarray | None = None) -> JointScore:
    d2, dof, tail, shares = joint_d2(resid, cal)
    p_static = conformal_p(tail, cal.cal_tail)
    p_adaptive = np.full(len(d2), np.nan)
    if difficulty is not None and len(cal.cal_tail_adaptive):
        p_adaptive = conformal_p(adaptive_tail(d2, dof, np.asarray(difficulty, dtype=float), cal.diff_floor),
                                 cal.cal_tail_adaptive)
    return JointScore(d2=d2, dof=dof, p_static=p_static, p_adaptive=p_adaptive, shares=shares)


@dataclass
class JointSurprise:
    excess: np.ndarray        # per held-out origin (NaN = not scored)
    p: np.ndarray
    shares: list
    n_cal: int
    calm: float


def run_difficulty(run: HorizonRun, cal: JointCal, rung: str = "ridge_var") -> np.ndarray:
    """(n,) hour-before difficulty per row from the rung's residuals at every row (``resid_all``)."""
    d2, dof, _t, _s = joint_d2(run.resid_all[rung], cal)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(dof > 0, d2 / np.maximum(dof, 1), np.nan)
    return difficulty_rows(ratio, run.horizon_steps)


def joint_surprise(run: HorizonRun, min_observed_frac: float = 0.5, calibration: str = STATIC) -> JointSurprise:
    cal = joint_calibration(run.resid_fit, run.resid_val, min_observed_frac=min_observed_frac)
    if calibration == ADAPTIVE:
        diff = run_difficulty(run, cal)
        cal = joint_calibration(run.resid_fit, run.resid_val, diff[run.val_rows], min_observed_frac)
        sc = score_joint(run.resid_held, cal, diff[run.held])
        p, n_cal = sc.p_adaptive, len(cal.cal_tail_adaptive)
    else:
        sc = score_joint(run.resid_held, cal)
        p, n_cal = sc.p_static, len(cal.cal_tail)
    calm = calm_point(n_cal)
    return JointSurprise(excess=-np.log(p) - calm, p=p, shares=sc.shares, n_cal=n_cal, calm=calm)


def rest_summary(excess: np.ndarray, index: pd.DatetimeIndex, seed: int = 0, grain_min: int | None = None) -> dict:
    """Does the joint excess sit at 0 and come back to it?

    - mean excess with a 2-hour moving-block bootstrap 95 % interval (calm: contains 0);
    - share of hours whose mean excess is <= 0 ("at rest"), beside the same share for a perfectly
      calibrated forecaster (``null_share_hours_at_rest``: iid uniform p, same counts per hour, same calm
      point; ~0.55 because -ln p is right-skewed);
    - longest run of hours above +0.5 (a stuck-high score never comes back).
    """
    s = pd.Series(excess, index=index).dropna()
    if s.empty:
        return {"n": 0}
    groups = s.groupby(s.index.floor("1h"))
    hourly = groups.mean()
    g = int(grain_min or _grain_min(index))
    pos = ((s.index - s.index[0]).total_seconds() // (60 * g)).astype(int).to_numpy()
    _p, lo, hi = block_bootstrap_p(s.to_numpy(), block=max(1, BOOT_BLOCK_MIN // g), seed=seed, n_boot=1000,
                                   segments=segments_of(pos))
    run, longest = 0, 0
    for v in hourly.to_numpy():
        run = run + 1 if v > 0.5 else 0
        longest = max(longest, run)
    rng = np.random.default_rng(seed)
    counts = groups.size().to_numpy()
    sims = [np.mean([(-np.log(rng.uniform(size=c))).mean() - 1.0 <= 0 for c in counts]) for _ in range(200)]
    return {"n": int(len(s)), "mean_excess": float(s.mean()), "mean_ci95": [lo, hi],
            "hours": int(len(hourly)),
            "share_hours_at_rest": float(np.mean(hourly.to_numpy() <= 0)),
            "null_share_hours_at_rest": float(np.mean(sims)),
            "longest_hours_above_0_5": int(longest),
            "hourly_min": float(hourly.min()), "hourly_max": float(hourly.max())}


def mean_shares(js: JointSurprise, targets: Sequence[str]) -> dict[str, float]:
    """Average share of d^2 each reading carries over the scored origins (sums to ~1; shares can be
    negative where a residual pair cancels). Independence evidence: how much of the joint score the
    readings another signal already scores (cabinet / office temperature) carry."""
    rows = [s for s in js.shares if s is not None]
    if not rows:
        return {}
    m = np.nanmean(np.vstack(rows), axis=0)
    return {t: float(v) for t, v in zip(targets, m) if np.isfinite(v)}


def top_spikes(js: JointSurprise, index: pd.DatetimeIndex, targets: Sequence[str], k: int = 5) -> list[dict]:
    """The k largest joint misses with the three readings carrying most of each (a hint, not a cause)."""
    order = np.argsort(np.nan_to_num(js.excess, nan=-np.inf))[::-1][:k]
    out = []
    for i in order:
        if not np.isfinite(js.excess[i]) or js.shares[i] is None:
            continue
        sh = js.shares[i]
        top = np.argsort(np.nan_to_num(sh, nan=-np.inf))[::-1][:3]
        out.append({"ts": index[i].isoformat(), "excess": float(js.excess[i]), "p": float(js.p[i]),
                    "carried_by": [{"target": targets[t], "share": float(sh[t])} for t in top if np.isfinite(sh[t])]})
    return out


# --- manifest ------------------------------------------------------------------------------------

MANIFEST_VERSION = 1


def build_manifest(config: FamilyAConfig, verdicts: Mapping[str, SeriesVerdict],
                   standardizer: Mapping[str, Mapping[str, float]], window: tuple[datetime, datetime],
                   train_end: datetime, extra: Mapping | None = None) -> dict:
    """The record a later trained rung loads: named inputs with their role, train-only standardisation,
    the fixed hygiene thresholds, and every refusal with its reason."""
    specs = {s.id: s for s in config.inputs}
    inputs = {}
    for iid, v in verdicts.items():
        row: dict[str, Any] = {"status": v.status, "reasons": list(v.reasons), "stats": v.stats,
                               "unit": specs[iid].unit, "source": specs[iid].source.__dict__}
        if iid in standardizer:
            row["standardize"] = dict(standardizer[iid])
        inputs[iid] = row
    return {"manifest_version": MANIFEST_VERSION, "family": "A", "grain_min": config.grain_min,
            "horizons_min": list(config.horizons_min), "hygiene": config.hygiene.__dict__,
            "window_start": window[0].isoformat(), "window_end": window[1].isoformat(),
            "train_end": train_end.isoformat(), "inputs": inputs, **(dict(extra) if extra else {})}


def named_from_manifest(manifest: Mapping) -> dict[str, str]:
    return {k: v["status"] for k, v in manifest["inputs"].items() if v["status"] in (TARGET, COVARIATE)}
