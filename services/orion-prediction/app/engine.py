"""The forecast loop for every target series: select -> issue -> score -> monitor.

- select: backtest the whole zoo, gate per horizon, store one run row per model plus the two
  stitched blends (champion, baseline). Every ``select_interval_sec``, or at start when the last
  stored selection is older than that or no longer matches the registry.
- issue: when a new grain bucket completes, fit the served models on the training view, stitch
  the champion and baseline paths, attach conformal quantiles and exceedance probabilities, store
  the horizon rows for both roles, publish the champion.
- score: the reducer. Every stored row whose target bucket is complete gets the real bucket value
  (real readings only, no interpolation) or ``score_status='missing'``.
- monitor: live coverage, champion vs baseline, staleness, failed or degenerate output -> alert set
  in /health and /live, logged when it changes.

Horizon labels: ``origin_ts`` is the start of the last complete bucket; the step for horizon h
targets the bucket starting at ``origin_ts + h``. At issue time (one settle after the origin bucket
ends) the 5-min step is the bucket already in progress.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd

from app.backtest import (
    COVERAGE_BAND,
    GATE_VERSION,
    QUANTILES,
    Selection,
    exceedance_probability,
    paired_win_p,
    q_key,
    quantile_paths,
    run_backtest,
    select_champion,
    step_owner,
    stitch,
)
from app.events import adopt_after, with_events
from app.grading import grade
from app.frame import covariate_series, floor_ts, mask_windows, to_grain, train_ready, training_slice
from app.models import build_models, split_seasonalities
from app.series import Registry, Seasonality, SeriesSpec
from orion.schemas.prediction import (
    PREDICTION_FORECAST_CHANNEL,
    PREDICTION_FORECAST_KIND,
    PredictionExceedanceV1,
    PredictionForecastV1,
    PredictionStepV1,
)

logger = logging.getLogger("orion-prediction.engine")

LIVE_WINDOW = timedelta(days=7)
MIN_LIVE_COVERAGE_N = 200
MIN_INDEPENDENT = 20   # non-overlapping live forecasts a horizon needs before it can be judged
# A live loss counts only if the champion is this much worse AND the loss is significant.
WORSE_MARGIN = 1.05
STALE_GRAINS = 3


class InputStale(RuntimeError):
    pass


class DegenerateForecast(ValueError):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _clean(a: np.ndarray) -> list:
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in np.asarray(a, dtype=float)]


def _from_json(rows: list) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float)


@dataclass(frozen=True)
class Served:
    """Everything issuing needs from one selection. Replaced as a whole, never field by field, so a
    concurrent reader (the HTTP guard) can never pair one selection's assignment with another's
    residuals."""
    selection: Selection
    selection_id: str
    selected_at: datetime
    run_ids: dict[str, str]            # champion / baseline -> run_id
    residuals: dict[str, np.ndarray]   # champion / baseline -> stitched residual paths
    usable: tuple[Seasonality, ...]
    refused: tuple[str, ...]
    train_from: datetime | None = None  # the effective training start this selection was built on
    events_key: str | None = None       # fingerprint of the change events it was built with


@dataclass
class SeriesState:
    served: Served | None = None
    next_select_at: datetime | None = None
    alerts: dict[str, str] = field(default_factory=dict)
    published_alert_codes: frozenset = frozenset()
    last_issue_error: tuple[str, str] | None = None   # (alert code, detail)
    last_select_error: str | None = None
    last_select_seconds: float | None = None
    last_tick_error: str | None = None
    last_input_ts: datetime | None = None
    events: dict = field(default_factory=dict)   # with_events summary, for /health


class Engine:
    def __init__(self, registry: Registry, store: Any, *, backtest_windows: int, backtest_step_min: int,
                 select_interval_sec: float, settle_sec: float, publish_enabled: bool,
                 publish: Callable[[str, Any], Awaitable[None]] | None = None, source: Any = None,
                 clock: Callable[[], datetime] = utcnow, weather: dict | None = None):
        self.registry = registry
        self.weather = weather            # {"lat", "lon", "refresh_sec"} or None (weather off)
        self.weather_state: dict = {}
        self.store = store
        self.backtest_windows = backtest_windows
        self.backtest_step_min = backtest_step_min
        self.select_interval = timedelta(seconds=select_interval_sec)
        self.settle = timedelta(seconds=settle_sec)
        self.publish_enabled = publish_enabled
        self.publish = publish
        self.source = source
        self.clock = clock
        self.state: dict[str, SeriesState] = {s.series_id: SeriesState() for s in registry.targets()}

    # --- change events --------------------------------------------------------------------
    def change_events(self) -> list[dict]:
        """All live change events. A failed read falls back to the last good list (and the error is
        surfaced as an alert) -- a hiccup on this table must not stop forecasting."""
        loader = getattr(self.store, "change_events", None)
        if loader is None:
            return []
        try:
            self._events = loader()
            self._events_error = None
        except Exception as exc:  # noqa: BLE001
            self._events_error = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("prediction_change_events_unreadable err=%s (using %d cached)", self._events_error,
                           len(getattr(self, "_events", [])))
        return list(getattr(self, "_events", []))

    def effective(self, spec: SeriesSpec, now: datetime) -> SeriesSpec:
        """The registry spec with change events folded in (exclusions, adopted/pending breaks)."""
        events = self.change_events()
        eff, summary = with_events(spec, events, now,
                                   adopt_after(self.backtest_windows, self.backtest_step_min, spec))
        if getattr(self, "_events_error", None):
            summary["error"] = self._events_error
        self.state.setdefault(spec.series_id, SeriesState()).events = summary
        return eff

    # --- data -----------------------------------------------------------------------------
    def history_span(self, spec: SeriesSpec) -> timedelta:
        backtest = self.backtest_windows * self.backtest_step_min + max(spec.horizons_min)
        return timedelta(minutes=spec.train_window_min + backtest + spec.grain_min)

    def latest_complete_bucket(self, spec: SeriesSpec, now: datetime) -> pd.Timestamp:
        return pd.Timestamp(floor_ts(now - self.settle, spec.grain_min)) - pd.Timedelta(minutes=spec.grain_min)

    def stale_after(self, spec: SeriesSpec) -> timedelta:
        return timedelta(minutes=STALE_GRAINS * spec.grain_min) + self.settle

    def input_fresh(self, spec: SeriesSpec, now: datetime) -> bool:
        ts = self.state[spec.series_id].last_input_ts
        return ts is not None and now - ts <= self.stale_after(spec)

    def _points(self, spec: SeriesSpec, now: datetime, span: timedelta | None):
        end = (self.latest_complete_bucket(spec, now) + pd.Timedelta(minutes=spec.grain_min)).to_pydatetime()
        start = end - (span or self.history_span(spec))
        pts = self.store.points(spec, start, end)
        if pts:
            self.state.setdefault(spec.series_id, SeriesState()).last_input_ts = max(t for t, _ in pts)
        return pts, start, end

    def load_series(self, spec: SeriesSpec, now: datetime, span: timedelta | None = None) -> pd.Series:
        pts, start, end = self._points(spec, now, span)
        return to_grain(pts, spec.grain_min, spec.agg, spec.max_fill_min, start=start, end=end)

    def load_covariates(self, spec: SeriesSpec, index: pd.DatetimeIndex) -> pd.DataFrame | None:
        """spec.covariates on ``index``: real buckets only (weather spreads its hourly values), each
        covariate's derived rules applied (app.frame.covariate_series), and the target's exclusions
        (incl. '*' change events) masked. None when the series declares none or the index is empty."""
        if not spec.covariates or len(index) == 0:
            return None
        start = index[0].to_pydatetime()
        end = (index[-1] + pd.Timedelta(minutes=spec.grain_min)).to_pydatetime()
        cols = {cid: mask_windows(covariate_series(self.store, self.registry.get(cid), start, end, index), spec.exclude)
                for cid in spec.covariates}
        return pd.DataFrame(cols, index=index)

    # --- weather --------------------------------------------------------------------------
    def weather_due(self, now: datetime) -> bool:
        return self.weather is not None and (self.weather_state.get("next_at") is None
                                             or now >= self.weather_state["next_at"])

    def refresh_weather(self, now: datetime) -> dict:
        """Hourly: the shared Open-Meteo client -> prediction_weather_hourly. A failure is recorded and
        retried at the next interval. Only observed hours are read as covariates, so a dead feed leaves
        the proxy without new values: once its newest hour is older than the hold + lag, direct_arx's
        freshness check refuses and those horizons fall back to the baseline."""
        from app.weather import refresh

        cfg = self.weather
        try:
            backfill = min((s.train_from for s in self.registry.targets() if s.train_from), default=None)
            self.weather_state = {**refresh(self.store, cfg["lat"], cfg["lon"], now, backfill),
                                  "next_at": now + timedelta(seconds=cfg["refresh_sec"])}
        except Exception as exc:  # noqa: BLE001
            self.weather_state = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300],
                                  "next_at": now + timedelta(seconds=cfg["refresh_sec"])}
            logger.warning("prediction_weather_refresh_failed err=%s", self.weather_state["error"])
        return self.weather_state

    # --- select ---------------------------------------------------------------------------
    def restore(self, spec: SeriesSpec) -> bool:
        """Re-adopt the last stored selection -- only if it still fits the registry and the zoo."""
        rows = self.store.latest_selection(spec.series_id)
        blends = {r["kind"]: r for r in rows if r["kind"] in ("champion_blend", "baseline_blend")}
        if set(blends) != {"champion_blend", "baseline_blend"}:
            return False
        champ, base = blends["champion_blend"], blends["baseline_blend"]
        residuals = {"champion": _from_json(champ["residual_paths"]), "baseline": _from_json(base["residual_paths"])}
        assignment = {int(k): v for k, v in champ["assignment"].items()}
        base_assignment = {int(k): v for k, v in base["assignment"].items()}
        usable = tuple(s for s in spec.seasonalities if s.name in champ["usable_seasonalities"])
        zoo = set(build_models(spec, list(usable)))
        fits = (champ["grain_min"] == spec.grain_min
                and (champ.get("metrics") or {}).get("covariates", []) == list(spec.covariates)
                and (champ.get("metrics") or {}).get("gate_version") == GATE_VERSION
                and all(r.ndim == 2 and r.shape[1] == spec.max_horizon_steps for r in residuals.values())
                and set(assignment) == set(spec.horizons_min) == set(base_assignment)
                and set(assignment.values()) | set(base_assignment.values()) <= zoo)
        if not fits:
            logger.warning("prediction_restore_skipped series=%s reason=stored selection no longer fits the registry",
                           spec.series_id)
            return False
        st = self.state[spec.series_id]
        st.served = Served(
            selection=Selection(assignment, base_assignment, bool(champ["served_baseline"]),
                                {r["model"]: r["metrics"] for r in rows if r["kind"] == "model"}),
            selection_id=str(champ["selection_id"]), selected_at=champ["fitted_at"],
            run_ids={"champion": str(champ["run_id"]), "baseline": str(base["run_id"])},
            residuals=residuals, usable=usable, refused=tuple(champ["refused_seasonalities"]),
            train_from=_parse_ts((champ.get("metrics") or {}).get("train_from")),
            events_key=(champ.get("metrics") or {}).get("events_key"))
        st.next_select_at = champ["fitted_at"] + self.select_interval
        return True

    def select(self, spec: SeriesSpec, now: datetime) -> Selection:
        st = self.state[spec.series_id]
        t0 = time.monotonic()
        pts, start, end = self._points(spec, now, None)
        series = to_grain(pts, spec.grain_min, spec.agg, spec.max_fill_min, start=start, end=end)
        # Residuals are measured against real readings only, as live scoring is.
        actuals = mask_windows(to_grain(pts, spec.grain_min, spec.agg, 0, start=start, end=end), spec.exclude)
        train = training_slice(series, spec)
        usable, refused = split_seasonalities(spec, len(train))
        names = list(build_models(spec, usable))
        step = max(1, self.backtest_step_min // spec.grain_min)
        covariates = self.load_covariates(spec, series.index)
        results = {name: run_backtest(series, spec, name, lambda u, n=name: build_models(spec, u)[n],
                                      self.backtest_windows, step, actuals=actuals, covariates=covariates)
                   for name in names}
        sel = select_champion(results, spec, self.backtest_step_min)
        h = spec.max_horizon_steps
        res = {n: r.residuals for n, r in results.items()}
        champ_res = stitch(res, step_owner(sel.assignment, spec.grain_min, h))
        base_res = stitch(res, step_owner(sel.baseline_assignment, spec.grain_min, h))
        selection_id = str(uuid.uuid4())
        common = {"selection_id": selection_id, "series_id": spec.series_id, "fitted_at": now,
                  "train_start": train.index[0].to_pydatetime() if len(train) else None,
                  "train_end": train.index[-1].to_pydatetime() if len(train) else None,
                  "n_obs": len(train), "grain_min": spec.grain_min,
                  "usable_seasonalities": [s.name for s in usable], "refused_seasonalities": refused}
        rows = [{**common, "run_id": str(uuid.uuid4()), "kind": "model", "model": name, "n_windows": r.n_windows,
                 "metrics": sel.metrics[name], "assignment": {}, "served_baseline": False,
                 "backtest_seconds": r.fit_seconds, "residual_paths": _clean(r.residuals), "errors": r.errors}
                for name, r in results.items()]
        blend_s = sum(r.fit_seconds for r in results.values())
        champ_id, base_id = str(uuid.uuid4()), str(uuid.uuid4())
        for run_id, kind, assignment, served_baseline, paths in (
                (champ_id, "champion_blend", sel.assignment, sel.served_baseline, champ_res),
                (base_id, "baseline_blend", sel.baseline_assignment, True, base_res)):
            rows.append({**common, "run_id": run_id, "kind": kind, "model": kind,
                         "n_windows": int(np.any(np.isfinite(paths), axis=1).sum()),
                         "metrics": {"covariates": list(spec.covariates), "gate_version": GATE_VERSION,
                                     "train_from": spec.train_from.isoformat() if spec.train_from else None,
                                     "events_key": st.events.get("fingerprint")},
                         "assignment": {str(k): v for k, v in assignment.items()},
                         "served_baseline": served_baseline, "backtest_seconds": blend_s,
                         "residual_paths": _clean(paths), "errors": []})
        self.store.insert_runs(rows)
        st.served = Served(selection=sel, selection_id=selection_id, selected_at=now,
                           run_ids={"champion": champ_id, "baseline": base_id},
                           residuals={"champion": champ_res, "baseline": base_res},
                           usable=tuple(usable), refused=tuple(refused), train_from=spec.train_from,
                           events_key=st.events.get("fingerprint"))
        st.next_select_at = now + self.select_interval
        st.last_select_seconds = time.monotonic() - t0
        st.last_select_error = None
        logger.info("prediction_selected series=%s assignment=%s served_baseline=%s refused=%s seconds=%.1f",
                    spec.series_id, sel.assignment, sel.served_baseline, refused, st.last_select_seconds)
        return sel

    def selection_due(self, spec: SeriesSpec, now: datetime) -> bool:
        """``spec`` is the effective spec: a change event that moved the training start (a break
        adopted) makes the served selection stale at once."""
        st = self.state[spec.series_id]
        backing_off = st.last_select_error is not None and st.next_select_at is not None and now < st.next_select_at
        stale = st.served is not None and (st.served.train_from != spec.train_from
                                           or st.served.events_key != st.events.get("fingerprint"))
        if stale and not backing_off:
            return True
        return st.next_select_at is None or now >= st.next_select_at

    # --- issue ----------------------------------------------------------------------------
    def build_forecast(self, spec: SeriesSpec, series: pd.Series, origin: pd.Timestamp, now: datetime) -> dict:
        """Champion + baseline rows and the publishable payload. Refuses stale input outright: a
        forecast from a sensor that stopped is the last value extended, not a forecast."""
        st = self.state[spec.series_id]
        served = st.served
        if served is None:
            raise RuntimeError("no selection yet")
        if not self.input_fresh(spec, now):
            raise InputStale(f"last reading {st.last_input_ts.isoformat() if st.last_input_ts else 'never'}")
        sel = served.selection
        train = training_slice(series, spec, end=origin + pd.Timedelta(minutes=spec.grain_min))
        y, filled = train_ready(train)
        if filled > 0.10:
            raise RuntimeError(f"training view is {filled:.0%} gaps")
        h = spec.max_horizon_steps
        zoo = build_models(spec, list(served.usable))
        needed = sel.models_needed()
        X = None
        if any(zoo[n].uses_covariates for n in needed):
            cov = self.load_covariates(spec, train.index)
            X = cov.to_numpy(dtype=float) if cov is not None else None
        # Each model on its own: one dead covariate must not take down horizons (or the baseline)
        # that never use it. Baseline models use no covariates; if one of them fails, nothing is served.
        paths, failed = {}, {}
        for name in needed:
            try:
                paths[name] = zoo[name].forecast(y, h, train.index, X)
            except Exception as exc:  # noqa: BLE001 -- that model's steps fall back to the baseline below
                failed[name] = f"{type(exc).__name__}: {exc}"[:200]
        base_owners = step_owner(sel.baseline_assignment, spec.grain_min, h)
        if any(o in failed for o in base_owners):
            raise RuntimeError(f"baseline model failed: {failed}")
        fallback_steps = 0
        out = {"rows": [], "payload": None}
        for role, assignment in (("champion", sel.assignment), ("baseline", sel.baseline_assignment)):
            owners = step_owner(assignment, spec.grain_min, h)
            residuals = served.residuals[role]
            if role == "champion" and failed:
                # Steps owned by a failed model are served by that step's baseline model, with the
                # baseline's residual column, so point and interval stay matched.
                swap = np.array([o in failed for o in owners])
                fallback_steps = int(sum(swap))
                owners = [b if s else o for o, b, s in zip(owners, base_owners, swap)]
                residuals = residuals.copy()
                residuals[:, swap] = served.residuals["baseline"][:, swap]
            point = stitch(paths, owners)
            qs = quantile_paths(point, residuals)
            _check_not_degenerate(point, qs, spec)
            steps = []
            for hm in spec.horizons_min:
                j = spec.steps(hm) - 1
                quant = {k: float(v[j]) for k, v in qs.items()}
                target = origin + pd.Timedelta(minutes=hm)
                out["rows"].append({"run_id": served.run_ids[role], "series_id": spec.series_id,
                                    "model": f"{role}_blend", "role": role, "issued_at": now,
                                    "origin_ts": origin.to_pydatetime(), "target_ts": target.to_pydatetime(),
                                    "horizon_min": hm, "point": float(point[j]), "quantiles": quant})
                steps.append(PredictionStepV1(target_ts=target.to_pydatetime(), horizon_min=hm,
                                              point=float(point[j]), quantiles=quant))
            if role == "champion":
                exceed = []
                for t in spec.thresholds:
                    p, n = exceedance_probability(point, residuals, t.value, spec.steps(t.window_min))
                    exceed.append(PredictionExceedanceV1(name=t.name, threshold=t.value, window_min=t.window_min,
                                                         probability=p, n_paths=n))
                served_by = {hm: owners[spec.steps(hm) - 1] for hm in sorted(sel.assignment)}
                out["payload"] = PredictionForecastV1(
                    series_id=spec.series_id,
                    model="+".join(f"{m}@{hm}" for hm, m in served_by.items()),
                    served_baseline=sel.served_baseline, run_id=served.run_ids[role], issued_at=now,
                    origin_ts=origin.to_pydatetime(), grain_min=spec.grain_min, steps=steps, exceedance=exceed)
        out["failed_models"] = failed
        out["fallback_steps"] = fallback_steps
        return out

    async def issue(self, spec: SeriesSpec, now: datetime, series: pd.Series, sync) -> bool:
        st = self.state[spec.series_id]
        origin = self.latest_complete_bucket(spec, now)
        last = await sync(self.store.last_origin, spec.series_id)
        if last is not None and pd.Timestamp(last) >= origin:
            return False
        try:
            built = await sync(self.build_forecast, spec, series, origin, now)
        except InputStale:
            st.last_issue_error = None   # alerts_for reports input_stale itself
            return False
        except DegenerateForecast as exc:
            st.last_issue_error = ("degenerate_forecast", str(exc)[:300])
            logger.error("prediction_issue_refused series=%s err=%s", spec.series_id, exc)
            return False
        except Exception as exc:  # noqa: BLE001 -- recorded as an alert, retried next bucket
            st.last_issue_error = ("issue_failed", f"{type(exc).__name__}: {exc}"[:300])
            logger.error("prediction_issue_failed series=%s err=%s", spec.series_id, st.last_issue_error[1])
            return False
        p = built["payload"]
        exc_rows = [{"series_id": spec.series_id, "origin_ts": p.origin_ts, "issued_at": now, "name": e.name,
                     "threshold": e.threshold, "window_start": p.origin_ts + timedelta(minutes=spec.grain_min),
                     "window_end": p.origin_ts + timedelta(minutes=spec.grain_min + e.window_min),
                     "probability": e.probability, "n_paths": e.n_paths, "served_baseline": p.served_baseline}
                    for e in p.exceedance]
        await sync(self.store.insert_issue, built["rows"], exc_rows)
        st.last_issue_error = None
        if built["failed_models"]:
            st.last_issue_error = ("model_fallback", f"{built['fallback_steps']} step(s) served by the baseline; "
                                                     f"failed: {built['failed_models']}"[:300])
            logger.warning("prediction_model_fallback series=%s %s", spec.series_id, st.last_issue_error[1])
        if self.publish_enabled and self.publish is not None:
            from orion.core.bus.bus_schemas import BaseEnvelope

            try:
                await self.publish(PREDICTION_FORECAST_CHANNEL, BaseEnvelope(
                    kind=PREDICTION_FORECAST_KIND, source=self.source, payload=built["payload"].model_dump(mode="json")))
            except Exception as exc:  # noqa: BLE001 -- the stored row is the record; next bucket re-publishes
                logger.error("prediction_publish_failed series=%s err=%s", spec.series_id, exc)
        return True

    # --- score ----------------------------------------------------------------------------
    def score(self, spec: SeriesSpec, now: datetime) -> int:
        cutoff = self.latest_complete_bucket(spec, now) + pd.Timedelta(minutes=spec.grain_min)
        due = self.store.due_for_scoring(spec.series_id, cutoff.to_pydatetime())
        if not due:
            return 0
        lo = min(r["target_ts"] for r in due)
        pts = self.store.points(spec, lo, cutoff.to_pydatetime())
        actual = to_grain(pts, spec.grain_min, spec.agg, 0, start=lo, end=cutoff.to_pydatetime())
        scores = []
        for r in due:
            v = actual.get(pd.Timestamp(r["target_ts"]))
            base = {"forecast_id": r["forecast_id"], "scored_at": now}
            if v is None or not np.isfinite(v):
                scores.append({**base, "score_status": "missing", "actual": None, "abs_error": None,
                               "in_80": None, "in_95": None})
                continue
            q = r["quantiles"]
            scores.append({**base, "score_status": "scored", "actual": float(v), "abs_error": abs(float(v) - r["point"]),
                           "in_80": q[q_key(0.1)] <= v <= q[q_key(0.9)],
                           "in_95": q[q_key(0.025)] <= v <= q[q_key(0.975)]})
        self.store.write_scores(scores)
        return len(scores)

    def score_exceedances(self, spec: SeriesSpec, now: datetime) -> int:
        """Outcome of each stored threshold probability once its window has passed (plus settle):
        hit_raw = any single reading >= threshold, hit_mean = any grain-bucket mean >= threshold."""
        cutoff = self.latest_complete_bucket(spec, now) + pd.Timedelta(minutes=spec.grain_min)
        due = self.store.due_exceedances(spec.series_id, cutoff.to_pydatetime())
        if not due:
            return 0
        lo, hi = min(r["window_start"] for r in due), max(r["window_end"] for r in due)
        pts = [(t, v) for t, v in self.store.points(spec, lo, hi) if v is not None and np.isfinite(v)]
        raw = pd.Series([v for _, v in pts], index=pd.DatetimeIndex([t for t, _ in pts]), dtype=float).sort_index() \
            if pts else pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        means = to_grain(pts, spec.grain_min, spec.agg, 0, start=lo, end=hi)
        out = []
        for r in due:
            a, b = pd.Timestamp(r["window_start"]), pd.Timestamp(r["window_end"])
            w = raw[(raw.index >= a) & (raw.index < b)]
            m = means[(means.index >= a) & (means.index < b)]
            n_buckets = int(round((b - a) / pd.Timedelta(minutes=spec.grain_min)))
            observed = int(m.notna().sum())
            complete = observed == n_buckets
            # A crossing seen in partial data is real; "no crossing" needs the whole window observed --
            # a sensor that stopped after 2 minutes must not read as a calm 30 minutes.
            raw_hit = bool((w >= r["threshold"]).any()) if len(w) else False
            mean_hit = bool((m.dropna() >= r["threshold"]).any())
            out.append({"exceedance_id": r["exceedance_id"], "scored_at": now, "n_readings": int(len(w)),
                        "n_buckets_observed": observed,
                        "hit_raw": True if raw_hit else (False if complete else None),
                        "hit_mean": True if mean_hit else (False if complete else None),
                        "max_raw": float(w.max()) if len(w) else None})
        self.store.write_exceedance_outcomes(out)
        return len(out)

    def exceedance_grades(self, spec: SeriesSpec, now: datetime) -> dict:
        rows = self.store.graded_exceedances(spec.series_id, now - LIVE_WINDOW)
        out = {}
        for t in spec.thresholds:
            span = timedelta(minutes=t.window_min)
            # Same name, same threshold, same window: a changed config must not mix old rows in.
            mine = [r for r in rows if r["name"] == t.name and r["threshold"] == t.value
                    and r["window_end"] - r["window_start"] == span]
            out[t.name] = {
                "threshold": t.value, "window_min": t.window_min,
                "vs_single_reading": grade([r["probability"] for r in mine if r["hit_raw"] is not None],
                                           [r["hit_raw"] for r in mine if r["hit_raw"] is not None]),
                "vs_bucket_mean": grade([r["probability"] for r in mine if r["hit_mean"] is not None],
                                        [r["hit_mean"] for r in mine if r["hit_mean"] is not None]),
            }
        return out

    # --- monitor --------------------------------------------------------------------------
    def live_metrics(self, spec: SeriesSpec, now: datetime) -> dict:
        rows = self.store.live_scores(spec.series_id, now - LIVE_WINDOW)
        out: dict = {"window_days": LIVE_WINDOW.days, "horizons": {}}
        champ = [r for r in rows if r["role"] == "champion"]
        base = {(r["origin_ts"], r["horizon_min"]): r["abs_error"] for r in rows if r["role"] == "baseline"}

        def rel(pairs):
            if not pairs:
                return None
            b = float(np.mean([b for _, b in pairs]))
            return float(np.mean([a for a, _ in pairs])) / b if b > 0 else None

        for hm in spec.horizons_min:
            c = [r for r in champ if r["horizon_min"] == hm]
            pairs = [(r["abs_error"], base[(r["origin_ts"], hm)]) for r in c if (r["origin_ts"], hm) in base]
            out["horizons"][str(hm)] = {
                "n": len(c),
                "mae": float(np.mean([r["abs_error"] for r in c])) if c else None,
                "coverage80": float(np.mean([r["in_80"] for r in c])) if c else None,
                "coverage95": float(np.mean([r["in_95"] for r in c])) if c else None,
                "rel_mae_vs_baseline": rel(pairs), "n_pairs": len(pairs),
            }
        out["n"] = len(champ)
        out["coverage80"] = float(np.mean([r["in_80"] for r in champ])) if champ else None
        # Champion vs baseline, per horizon, ONLY where they forecast something different (a horizon
        # served by the baseline itself is a tie by construction -- 2026-10-20 the alert read "1.00x"
        # with 4 of 5 horizons served by naive). Forecasts issued every grain overlap for h/grain
        # steps, so only every k-th origin (k = h/grain) is an independent sample; rows arrive ordered
        # by (horizon, origin). A horizon is judged only with >= MIN_INDEPENDENT such samples.
        base_pt = {(r["origin_ts"], r["horizon_min"]): r["point"] for r in rows if r["role"] == "baseline"}
        out["losing_horizons"] = []
        for hm in spec.horizons_min:
            diff = [(r["abs_error"], base[k]) for r in champ if r["horizon_min"] == hm
                    and (k := (r["origin_ts"], hm)) in base and abs(r["point"] - base_pt[k]) > 1e-9]
            step = max(1, hm // spec.grain_min)
            thin = diff[::step]
            h = out["horizons"][str(hm)]
            h.update(n_differing=len(diff), n_independent=len(thin), rel_mae_differing=rel(diff),
                     p_baseline_better=None)
            if len(thin) >= MIN_INDEPENDENT:
                c_err = np.array([a for a, _ in thin]); b_err = np.array([b for _, b in thin])
                h["p_baseline_better"], _ = paired_win_p(b_err, c_err, lag=1)
                if (h["rel_mae_differing"] or 0) >= WORSE_MARGIN and h["p_baseline_better"] < 0.05:
                    out["losing_horizons"].append(hm)
        return out

    def alerts_for(self, spec: SeriesSpec, now: datetime, live: dict, last_issued: datetime | None) -> dict[str, str]:
        st = self.state[spec.series_id]
        alerts: dict[str, str] = {}
        stale = self.stale_after(spec)
        if not self.input_fresh(spec, now):
            alerts["input_stale"] = f"last reading {st.last_input_ts.isoformat() if st.last_input_ts else 'never'}"
        elif st.served is not None and (last_issued is None or now - last_issued > stale):
            alerts["forecast_stale"] = f"last forecast {last_issued.isoformat() if last_issued else 'never'}"
        if st.last_issue_error:
            alerts[st.last_issue_error[0]] = st.last_issue_error[1]
        if st.last_select_error:
            alerts["selection_failed"] = st.last_select_error
        if st.last_tick_error:
            alerts["tick_failed"] = st.last_tick_error
        if st.events.get("error"):
            alerts["change_events_unreadable"] = st.events["error"]
        c = live.get("coverage80")
        if c is not None and live.get("n", 0) >= MIN_LIVE_COVERAGE_N and not (COVERAGE_BAND[0] <= c <= COVERAGE_BAND[1]):
            alerts["coverage_out_of_band"] = f"live 80% coverage {c:.2f} over {live['n']} scores"
        losing = live.get("losing_horizons") or []
        if losing:
            alerts["worse_than_baseline"] = "; ".join(
                f"{hm} min: {live['horizons'][str(hm)]['rel_mae_differing']:.2f}x the baseline over "
                f"{live['horizons'][str(hm)]['n_independent']} independent forecasts "
                f"(p={live['horizons'][str(hm)]['p_baseline_better']:.3f})" for hm in losing)
        return alerts

    async def monitor(self, spec: SeriesSpec, now: datetime, sync) -> dict[str, str]:
        st = self.state[spec.series_id]
        live = await sync(self.live_metrics, spec, now)
        last_issued = await sync(self.store.last_issued, spec.series_id)
        st.alerts = self.alerts_for(spec, now, live, last_issued)
        codes = frozenset(st.alerts)
        if codes != st.published_alert_codes:
            logger.warning("prediction_alerts_changed series=%s active=%s", spec.series_id, st.alerts)
            st.published_alert_codes = codes
        return st.alerts

    # --- tick -----------------------------------------------------------------------------
    async def tick_series(self, spec: SeriesSpec, now: datetime, sync) -> None:
        st = self.state[spec.series_id]
        spec = await sync(self.effective, spec, now)
        if st.served is None and st.next_select_at is None:
            await sync(self.restore, spec)
        if self.selection_due(spec, now):
            try:
                await sync(self.select, spec, now)
            except Exception as exc:  # noqa: BLE001 -- keep serving the previous selection, if any
                st.last_select_error = f"{type(exc).__name__}: {exc}"[:300]
                st.next_select_at = now + self.select_interval   # back off a full interval
                logger.exception("prediction_select_failed series=%s", spec.series_id)
        if st.served is not None:
            series = await sync(self.load_series, spec, now, timedelta(minutes=spec.train_window_min + spec.grain_min))
            await self.issue(spec, now, series, sync)
        await sync(self.score, spec, now)
        await sync(self.score_exceedances, spec, now)

    async def tick(self, sync) -> None:
        now = self.clock()
        for spec in self.registry.targets():
            st = self.state[spec.series_id]
            try:
                await self.tick_series(spec, now, sync)
                st.last_tick_error = None
            except Exception as exc:  # noqa: BLE001 -- one series' failure must not stop the others
                st.last_tick_error = f"{type(exc).__name__}: {exc}"[:300]
                logger.exception("prediction_series_tick_failed series=%s", spec.series_id)
            try:
                await self.monitor(spec, now, sync)
            except Exception:  # noqa: BLE001
                logger.exception("prediction_monitor_failed series=%s", spec.series_id)
        # After the series loop: a slow weather call (up to ~25 s) must not delay issuing or scoring.
        if self.weather_due(now):
            await sync(self.refresh_weather, now)


def _check_not_degenerate(point: np.ndarray, qs: dict[str, np.ndarray], spec: SeriesSpec) -> None:
    """Refuse output that only looks like a forecast: non-finite values, missing or zero-width intervals."""
    if not np.all(np.isfinite(point)):
        raise DegenerateForecast("non-finite point forecast")
    idx = [spec.steps(hm) - 1 for hm in spec.horizons_min]
    for k in [q_key(q) for q in QUANTILES]:
        if not np.all(np.isfinite(qs[k][idx])):
            raise DegenerateForecast(f"quantile {k} unknown at a served horizon (too few backtest residual paths)")
    lo, hi = qs[q_key(0.1)][idx], qs[q_key(0.9)][idx]
    if np.all(hi - lo <= 0):
        raise DegenerateForecast("zero-width 80% interval at every horizon")
