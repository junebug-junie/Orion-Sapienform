"""Family A, slice 1: input hygiene, baseline rungs, and joint (whitened) surprise. Pure numpy/pandas.

Spec: docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md, "A. Neural forecaster".
Question: "what will my body and room read next?", every raw reading at 5 and 15 min ahead.

Nothing here runs in the live service yet. ``evals/run_family_a_baselines_eval.py`` loads raw readings
(``app/family_a_load.py``) and calls these functions. A later patch moves the gate and the winning rung
into the select -> issue -> score loop.

1. **Hygiene gate** (``screen_series``, ``gate_named_inputs``). Refuse, never degrade. Each series must
   be present, vary (no value over 85 % of buckets, at least 50 distinct values) and return to rest
   (touches its central band on most days; daily medians not a monotone drift). A series resting on its
   floor most of the time is event-like: an input, never a target. Missing stays NaN plus a mask
   channel; it is never filled with 0.0 in real units.
2. **Standardisation** from training rows only (``fit_standardizer``), stored in the manifest.
3. **Rungs**, all predicting the standardised change z[t+k] - z[t] (direct, one fit per lead):
   ``persistence`` (change 0), ``ar`` (one ridge per series on its own last 6 buckets), ``ridge_var``
   (one ridge per target on lags 0,1,2,3,6,12 of every input + mask + time of day).
4. **Split**: the repo's ``block_purged_temporal_split`` (orion/mood_arc/fit_encoder.py) over origin
   indices, purged by the longest lag + lead. Seeds move the block edges.
5. **Significance**: per-origin joint loss (mean squared standardised error over targets), ridge vs the
   better baseline per target, tested with the overlap-aware paired test (``backtest.paired_gap``) AND a
   one-day moving-block bootstrap. A lower average alone is not a win.
6. **Joint surprise**: residuals whitened by their own covariance (Mahalanobis d^2, chi-square tail on
   the observed subset), then a conformal p against a held calibration set, so p is uniform when the
   model is calibrated. surprisal = -ln p; calm point = E[-ln p] for a discrete uniform p (``calm_point``)
   -> excess rests at 0 by theory. An eval output only: no node, no bus event, no attention input.
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
HINT_PATTERN = re.compile(r"pressures?$|prediction_error|action_warrant|surprise|salience|_hint\b|warrant")
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
        if not raw.get("role") or not _KEY.match(str(raw["role"])):
            raise FamilyAConfigError(f"{iid}: cooling source needs a [a-z0-9_] role")
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
    for raw in doc.get("inputs") or ():
        iid = str(raw.get("id", ""))
        if not _ID.match(iid):
            raise FamilyAConfigError(f"bad input id {iid!r}")
        if HINT_PATTERN.search(iid):
            raise FamilyAConfigError(f"{iid}: names a hint channel, not a raw reading")
        if iid in seen:
            raise FamilyAConfigError(f"duplicate input id {iid}")
        seen.add(iid)
        inputs.append(InputSpec(id=iid, unit=str(raw.get("unit", "")), source=_source(raw.get("source") or {}, iid)))
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
        dup = next((k for k in kept if abs(float(frame[c].corr(frame[k], min_periods=50) or 0.0))
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


def _short_window_reasons(values: pd.Series, hyg: Hygiene, band: Sequence[float] | None) -> list[str]:
    """Run-time checks for a scoring window shorter than FULL_SCREEN_MIN_DAYS: present, not stuck on one
    value, and still inside the regime it was trained on (touches the TRAINING central 80 % band)."""
    v = values.to_numpy(dtype=float)
    present = np.isfinite(v)
    if not len(v) or present.mean() < hyg.min_present or present.sum() < 2:
        return ["absent"]
    x = v[present]
    _u, counts = np.unique(np.array([float(f"{a:.6g}") for a in x]), return_counts=True)
    out = []
    if counts.max() / len(x) > hyg.max_mode_share:
        out.append("one_value")
    if band is not None and not bool(((x >= band[0]) & (x <= band[1])).any()):
        out.append("left_trained_band")
    return out


def gate_named_inputs(named: Mapping[str, str], frame: pd.DataFrame, hyg: Hygiene,
                      last_seen: Mapping[str, datetime | None] | None = None, now: datetime | None = None,
                      bands: Mapping[str, Sequence[float]] | None = None) -> None:
    """Refuse (raise ``HygieneRefused``) unless every named input still passes, in the same role.

    ``named`` maps input id -> status in the manifest (target / covariate). Windows of at least
    FULL_SCREEN_MIN_DAYS get the full screen: an input now refused, or a named target that turned
    event-like, refuses the run. Shorter (scoring) windows get ``_short_window_reasons`` against the
    training band in ``bands`` (manifest ``stats.band``). Any input missing from the frame, or stale,
    refuses either way.
    """
    span_days = (len(frame) * _grain_min(frame.index)) / 1440.0
    refusals: dict[str, list[str]] = {}
    for iid, status in named.items():
        if iid not in frame.columns:
            refusals[iid] = ["not_loaded"]
            continue
        if span_days < FULL_SCREEN_MIN_DAYS:
            reasons = _short_window_reasons(frame[iid], hyg, (bands or {}).get(iid))
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
    """Embargo between train and held-out origins: the longest feature lag plus the lead."""
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
    """(n, f): every input's standardised value at lags VAR_LAGS (missing -> 0 = the training mean, with
    the lag-0 mask channel saying it was missing), the mask, and time of day. Rows inside the first
    max(VAR_LAGS) have no lag history: their lag columns read as missing (0) too."""
    filled = np.where(np.isfinite(z_all), z_all, 0.0)
    cols = [_lagged(filled, lag) for lag in VAR_LAGS]
    cols = [np.where(np.isfinite(c), c, 0.0) for c in cols]
    mask = (~np.isfinite(z_all)).astype(float)
    minutes = np.asarray(index.hour * 60 + index.minute, dtype=float)
    phase = 2 * np.pi * minutes / 1440.0
    return np.column_stack(cols + [mask, np.sin(phase), np.cos(phase)])


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


def ridge_path(X: np.ndarray, y: np.ndarray, alphas: Sequence[float] = ALPHAS) -> list[RidgeFit]:
    """Closed-form ridge for several alphas (centered; intercept unpenalised)."""
    xm, ym = X.mean(axis=0), float(y.mean())
    Xc, yc = X - xm, y - ym
    w, V = np.linalg.eigh(Xc.T @ Xc)
    w = np.clip(w, 0.0, None)
    proj = V.T @ (Xc.T @ yc)
    fits = []
    for a in alphas:
        beta = V @ (proj / (w + a))
        fits.append(RidgeFit(beta=beta, intercept=ym - float(xm @ beta), alpha=float(a)))
    return fits


def fit_ridge_selected(X: np.ndarray, y: np.ndarray, fit_rows: np.ndarray, val_rows: np.ndarray) -> RidgeFit:
    """Fit on ``fit_rows`` for every alpha; keep the alpha with the lowest validation squared error."""
    path = ridge_path(X[fit_rows], y[fit_rows])
    if len(val_rows) == 0:
        return path[len(path) // 2]
    losses = [float(np.mean((f.predict(X[val_rows]) - y[val_rows]) ** 2)) for f in path]
    return path[int(np.argmin(losses))]


# --- backtest ------------------------------------------------------------------------------------

MIN_FIT_ROWS = 288   # one day of origins per target


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
    stats = fit_standardizer(frame.iloc[np.sort(train_idx)][list(inputs)])
    z_all = standardize(frame[list(inputs)], stats)
    X = var_features(z_all, frame.index)
    fit_rows, val_rows = inner_split(np.sort(train_idx), horizon_steps)
    k = horizon_steps
    m = len(targets)
    sq = {name: np.full((len(held_idx), m), np.nan) for name in ("persistence", "ar", "ridge_var")}
    resid_val = np.full((len(val_rows), m), np.nan)
    resid_fit = np.full((len(fit_rows), m), np.nan)
    resid_held = np.full((len(held_idx), m), np.nan)
    alphas: dict[str, dict[str, float]] = {}
    refused: dict[str, str] = {}
    col = {c: i for i, c in enumerate(inputs)}
    for j, t in enumerate(targets):
        z = z_all[:, col[t]]
        y = deltas(z, k)
        A = ar_features(z)
        ok_ar = np.isfinite(A).all(axis=1) & np.isfinite(y)
        f_ar, v_ar = fit_rows[ok_ar[fit_rows]], val_rows[ok_ar[val_rows]]
        if len(f_ar) < MIN_FIT_ROWS:
            refused[t] = f"fit_rows={len(f_ar)} < {MIN_FIT_ROWS}"
            continue
        ar = fit_ridge_selected(A, y, f_ar, v_ar)
        # ridge_var is nested on AR: the AR forecast plus a ridge on every input's lags fitted to what AR
        # missed. At a large alpha it IS the AR forecast, so the rung above can only win by what the
        # other readings add -- the ladder's "keep a rung only if it beats the one below".
        ar_pred = np.full(n, np.nan)
        ar_pred[ok_ar] = ar.predict(A[ok_ar])
        rv = fit_ridge_selected(X, y - np.where(ok_ar, ar_pred, 0.0), f_ar, v_ar)
        alphas[t] = {"ar": ar.alpha, "ridge_var": rv.alpha}

        def rv_pred(rows: np.ndarray) -> np.ndarray:
            return ar_pred[rows] + rv.predict(X[rows])

        # Common scoring set: every rung can forecast this origin and the outcome is real.
        h_ok = ok_ar[held_idx]
        hrows = held_idx[h_ok]
        truth = y[hrows]
        sq["persistence"][h_ok, j] = truth ** 2
        sq["ar"][h_ok, j] = (ar_pred[hrows] - truth) ** 2
        r = truth - rv_pred(hrows)
        sq["ridge_var"][h_ok, j] = r ** 2
        resid_held[h_ok, j] = r
        vm = ok_ar[val_rows]
        resid_val[vm, j] = y[val_rows[vm]] - rv_pred(val_rows[vm])
        fm = ok_ar[fit_rows]
        resid_fit[fm, j] = y[fit_rows[fm]] - rv_pred(fit_rows[fm])
    return HorizonRun(horizon_min=k * grain_min, seed=seed, targets=list(targets), held=held_idx, sq=sq,
                      resid_val=resid_val, resid_fit=resid_fit, resid_held=resid_held, alphas=alphas,
                      refused_targets=refused)


# --- significance --------------------------------------------------------------------------------

BOOT_N = 2000
# Moving-block length for the bootstrap, in minutes. Held-out origins come in ~5 contiguous stretches of
# ~12 h each at 14 days; a one-day block would span stretches and leave ~3 blocks per resample.
BOOT_BLOCK_MIN = 120


def block_bootstrap_p(d: np.ndarray, block: int, seed: int, n_boot: int = BOOT_N) -> tuple[float, float, float]:
    """One-sided p that mean(d) < 0, from a moving-block bootstrap of the centred series (the null
    "true mean is 0"), plus the 95 % percentile interval of the mean. NaN rows dropped first."""
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2 * block or n < 10:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    obs = float(d.mean())
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
    lc, lr = lc[both], lr[both]
    d = lc - lr
    mean, se, n = paired_gap(lc, lr, overlap_lag(run.horizon_min, grain_min))
    p_t = None if mean is None else (float(stats.t.cdf(mean / se, df=n - 1)) if se else (0.0 if mean < 0 else 1.0))
    p_b, lo, hi = block_bootstrap_p(d, block=max(1, BOOT_BLOCK_MIN // grain_min), seed=seed)
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


def shrunk_covariance(resid: np.ndarray, shrink: float = SHRINK) -> np.ndarray:
    """Pairwise-complete residual covariance, shrunk toward its diagonal so it inverts."""
    df = pd.DataFrame(resid)
    cov = df.cov(min_periods=10).to_numpy()
    d = np.diag(np.diag(cov))
    out = (1 - shrink) * cov + shrink * d
    return np.where(np.isfinite(out), out, 0.0) + np.eye(len(out)) * 1e-9


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


@dataclass
class JointSurprise:
    excess: np.ndarray        # per held-out origin (NaN = not scored)
    p: np.ndarray
    shares: list
    n_cal: int
    calm: float


def joint_surprise(run: HorizonRun, min_observed_frac: float = 0.5) -> JointSurprise:
    m = len(run.targets)
    min_obs = max(1, int(math.ceil(min_observed_frac * m)))
    cov = shrunk_covariance(run.resid_fit)
    _d2c, cal_tail, _ = mahalanobis_tail(run.resid_val, cov, min_obs)
    _d2, tail, shares = mahalanobis_tail(run.resid_held, cov, min_obs)
    n_cal = int(np.isfinite(cal_tail).sum())
    p = conformal_p(tail, cal_tail)
    calm = calm_point(n_cal)
    return JointSurprise(excess=-np.log(p) - calm, p=p, shares=shares, n_cal=n_cal, calm=calm)


def rest_summary(excess: np.ndarray, index: pd.DatetimeIndex, seed: int = 0) -> dict:
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
    _p, lo, hi = block_bootstrap_p(s.to_numpy(), block=max(1, BOOT_BLOCK_MIN // int(_grain_min(index))),
                                   seed=seed, n_boot=1000)
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
