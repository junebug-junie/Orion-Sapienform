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
    QUANTILES,
    Selection,
    exceedance_probability,
    q_key,
    quantile_paths,
    run_backtest,
    select_champion,
    step_owner,
    stitch,
)
from app.frame import floor_ts, mask_windows, to_grain, train_ready, training_slice
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
MIN_LIVE_PAIRS = 100
STALE_GRAINS = 3


class InputStale(RuntimeError):
    pass


class DegenerateForecast(ValueError):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


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


class Engine:
    def __init__(self, registry: Registry, store: Any, *, backtest_windows: int, backtest_step_min: int,
                 select_interval_sec: float, settle_sec: float, publish_enabled: bool,
                 publish: Callable[[str, Any], Awaitable[None]] | None = None, source: Any = None,
                 clock: Callable[[], datetime] = utcnow):
        self.registry = registry
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
            residuals=residuals, usable=usable, refused=tuple(champ["refused_seasonalities"]))
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
        results = {name: run_backtest(series, spec, name, lambda u, n=name: build_models(spec, u)[n],
                                      self.backtest_windows, step, actuals=actuals) for name in names}
        sel = select_champion(results, spec)
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
                         "n_windows": int(np.any(np.isfinite(paths), axis=1).sum()), "metrics": {},
                         "assignment": {str(k): v for k, v in assignment.items()},
                         "served_baseline": served_baseline, "backtest_seconds": blend_s,
                         "residual_paths": _clean(paths), "errors": []})
        self.store.insert_runs(rows)
        st.served = Served(selection=sel, selection_id=selection_id, selected_at=now,
                           run_ids={"champion": champ_id, "baseline": base_id},
                           residuals={"champion": champ_res, "baseline": base_res},
                           usable=tuple(usable), refused=tuple(refused))
        st.next_select_at = now + self.select_interval
        st.last_select_seconds = time.monotonic() - t0
        st.last_select_error = None
        logger.info("prediction_selected series=%s assignment=%s served_baseline=%s refused=%s seconds=%.1f",
                    spec.series_id, sel.assignment, sel.served_baseline, refused, st.last_select_seconds)
        return sel

    def selection_due(self, spec: SeriesSpec, now: datetime) -> bool:
        nxt = self.state[spec.series_id].next_select_at
        return nxt is None or now >= nxt

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
        paths = {name: zoo[name].forecast(y, h, train.index) for name in sel.models_needed()}
        out = {"rows": [], "payload": None}
        for role, assignment in (("champion", sel.assignment), ("baseline", sel.baseline_assignment)):
            point = stitch(paths, step_owner(assignment, spec.grain_min, h))
            qs = quantile_paths(point, served.residuals[role])
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
                    p, n = exceedance_probability(point, served.residuals[role], t.value, spec.steps(t.window_min))
                    exceed.append(PredictionExceedanceV1(name=t.name, threshold=t.value, window_min=t.window_min,
                                                         probability=p, n_paths=n))
                out["payload"] = PredictionForecastV1(
                    series_id=spec.series_id,
                    model="+".join(f"{m}@{hm}" for hm, m in sorted(sel.assignment.items())),
                    served_baseline=sel.served_baseline, run_id=served.run_ids[role], issued_at=now,
                    origin_ts=origin.to_pydatetime(), grain_min=spec.grain_min, steps=steps, exceedance=exceed)
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
        await sync(self.store.insert_forecasts, built["rows"])
        st.last_issue_error = None
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
        recent = [r for r in champ if r["origin_ts"] >= now - timedelta(hours=24)]
        pairs24 = [(r["abs_error"], base[(r["origin_ts"], r["horizon_min"])]) for r in recent
                   if (r["origin_ts"], r["horizon_min"]) in base]
        out["rel_mae_vs_baseline_24h"] = rel(pairs24)
        out["n_pairs_24h"] = len(pairs24)
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
        c = live.get("coverage80")
        if c is not None and live.get("n", 0) >= MIN_LIVE_COVERAGE_N and not (COVERAGE_BAND[0] <= c <= COVERAGE_BAND[1]):
            alerts["coverage_out_of_band"] = f"live 80% coverage {c:.2f} over {live['n']} scores"
        rel = live.get("rel_mae_vs_baseline_24h")
        if (st.served is not None and not st.served.selection.served_baseline and rel is not None
                and live.get("n_pairs_24h", 0) >= MIN_LIVE_PAIRS and rel >= 1.0):
            alerts["worse_than_baseline"] = f"24h MAE {rel:.2f}x the baseline over {live['n_pairs_24h']} pairs"
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
