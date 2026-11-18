"""Family A in the live loop: every raw body and room reading forecast at once, 5 and 15 min ahead.

Spec: docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md, "A. Neural forecaster".
Slice 1 (PR #3267) measured the gate and the rungs offline. This runs them in the service's own
select -> issue -> score loop, as one multivariate series, on the service's own tables:

- **select** (at start, then every ``select_interval``): load 14 days of 5-min bucket means, screen every
  input (``family_a.screen_frame``), backtest ``ridge_var`` against the better baseline per target on
  2 seeds (inputs screened on each seed's training rows only), and serve ``ridge_var`` at a horizon only
  when it wins on every seed (paired t AND block bootstrap, p < 0.05); else serve ``ar``. Then refit the
  served rung on the whole window, keeping validation rows apart for the conformal calibration. One
  ``prediction_model_run`` row per rung (kind ``model``) plus the served ``champion_blend`` row, whose
  ``metrics`` hold everything scoring needs later (targets, standardisation, covariance, calibration
  tails), so rows issued before a restart are scored with the run that issued them.
- **issue** (each completed 5-min bucket): the hygiene gate runs on the last 24 h of every named input
  (present, inside its training band, fresh, and not flatter than its flattest training day unless resting
  at its floor -- a frozen sensor; a flat reading at rest is not refused). A refusal stores one ``refused`` joint row per
  horizon with the reasons and logs them; nothing is forecast (refuse, never degrade). Otherwise each
  forecastable reading gets a ``prediction_forecast`` row (``series_id = 'family_a:<id>'``, role
  champion, 80/95 % ranges from its validation residuals), the other rung a shadow row, and each horizon
  one ``prediction_joint_surprise`` row (status ``issued``) carrying the hour-before difficulty known
  now. A horizon whose target bucket has already completed is skipped, never back-filled.
- **score** (each tick, before issuing): every due reading row gets its real bucket mean or ``missing``;
  each joint row gets d^2 on the readings that arrived (whitened by the run's covariance), the static
  and the adaptive conformal p, and excess = -ln p - calm point. Too few readings -> ``missing``.

**Joint groups** (#3575 / #3580): beside the blended joint row, each horizon gets one joint row per group
(``family_a_body``, ``family_a_room``; every input's ``group`` in config/prediction/family_a.v1.yaml), scored
the same way on the group's own readings with the group's own covariance, calibration tails and hour-before
difficulty (all fit at select time on the same residuals, stored under ``metrics[h]["groups"]``). A group
whose calibration pool is under ``MIN_CAL`` is left out of that run (logged in ``groups_skipped``), never
scored on a thin pool. Scored group rows with the adaptive p and a target in the last ``EVENT_MAX_AGE`` are
queued as ``prediction.joint_surprise.v1`` events (``take_pending_events``; app/main.py publishes them);
substrate-runtime folds them into ``node:prediction.body_joint`` / ``node:prediction.room_joint``, the
successor of ``node:substrate.biometrics``. The blended row is still stored (its calm check, #3309) and is
never published.
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from app import family_a as fa
from app.adaptive import MAX_HORIZON, adaptive_offsets, pool_since
from app.backtest import QUANTILES, q_key, residual_quantiles
from app.frame import floor_ts

logger = logging.getLogger("orion-prediction.family_a")

FAMILY = "family_a"
PREFIX = FAMILY + ":"
TRAIN_DAYS = 14
GATE_HOURS = 24
SELECT_SEEDS = (0, 1)
# Which conformal p is served as ``p`` / ``excess`` (both are always stored). Chosen on held-out data,
# see the slice 2 PR report and evals/run_family_a_baselines_eval.py.
SERVED_CALIBRATION = fa.ADAPTIVE
RETENTION = timedelta(days=35)      # reading rows (forecast + shadow); joint rows are kept
PRUNE_EVERY = timedelta(hours=24)
RETRY_SELECT = timedelta(minutes=30)
INTERVAL_SOURCE = "validation"      # per-reading ranges: the served rung's validation residuals
# Per-reading ranges from the reading's own recent live misses, scaled by its last hour (app/adaptive.py,
# #3310). The validation ranges are one width for calm and busy hours: replay 11-14..16, athena.load_1m
# was covered 0.64 of the time at 80 % (chassis watts 0.64, board temp 0.68) while PM2.5 read 0.97.
ADAPTIVE_SOURCE = "adaptive"
K10, K90 = QUANTILES.index(0.1), QUANTILES.index(0.9)
TOP_SHARES = 5
BLAS_THREADS = 4
DUE_LIMIT = 2000                    # joint rows scored per pass (~3.5 days of backlog per pass)
# Joint groups (#3575 / #3580): group -> its family value in prediction_joint_surprise.
GROUP_FAMILIES: dict[str, str] = {g: f"{FAMILY}_{g}" for g in fa.GROUPS}
GROUP_OF_FAMILY: dict[str, str] = {v: k for k, v in GROUP_FAMILIES.items()}
# A scored group row becomes an event only while its target is this recent: a backlog scored after an
# outage is the record (in the table), never published as now. Same as the substrate node's window.
EVENT_MAX_AGE = timedelta(minutes=30)
PENDING_EVENTS_MAX = 64             # bus down: keep the newest, drop the rest (the table is the record)

Loader = Callable[[datetime, datetime], "tuple[pd.DataFrame, dict[str, datetime | None]]"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def limit_blas(threads: int = BLAS_THREADS) -> Any:
    """Hold OpenBLAS to ``threads`` for the whole process. Called ONCE at service start (and by the
    replay eval), never per tick: OpenBLAS has one process-wide setting, and switching it while another
    thread is inside a BLAS call is not documented as safe. Why: OpenBLAS defaults to one thread per
    core (64 here); on the loaded host (load ~150 on 96 cores, 2026-11-13) one 42-reading fit took
    > 120 s that way vs 1.5 s on 2 threads. The single-series engine's numpy calls share the limit."""
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=threads, user_api="blas")


def _finite(obj: Any) -> Any:
    """JSON-safe copy: NaN / inf -> None (the store refuses NaN), numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_finite(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    return obj


@dataclass
class GroupCal:
    """One joint group's scoring yardstick: its targets (a subset of the run's) and its own calibration."""
    targets: list[str]
    cal: fa.JointCal


@dataclass
class ServedHorizon:
    fit: fa.HorizonFit
    rung: str
    cal: fa.JointCal
    q_offsets: np.ndarray            # (len(QUANTILES), m) standardised residual offsets, validation rows
    groups: dict[str, GroupCal] = field(default_factory=dict)


@dataclass
class Served:
    run_id: str
    selection_id: str
    fitted_at: datetime
    named: dict[str, str]
    bands: dict[str, list]
    floors: dict[str, float]
    horizons: dict[int, ServedHorizon]
    flat_ceilings: dict[str, float | None] = field(default_factory=dict)


@dataclass
class RunInfo:
    """What scoring needs from the run that issued a row (read back from its stored metrics)."""
    targets: list[str]
    std: dict[str, float]
    cal: fa.JointCal
    groups: dict[str, GroupCal] = field(default_factory=dict)


class SelectRefused(RuntimeError):
    pass


@dataclass
class State:
    last_select: dict = field(default_factory=dict)
    last_issue: dict = field(default_factory=dict)
    last_score: dict = field(default_factory=dict)
    counts: dict = field(default_factory=lambda: {"issued": 0, "refused": 0, "late_horizons": 0,
                                                  "scored": 0, "missing": 0, "shadow_failed": 0,
                                                  "group_scored": 0, "group_missing": 0,
                                                  "events_queued": 0, "events_dropped_full": 0})
    errors: dict = field(default_factory=dict)


class FamilyALive:
    def __init__(self, store: Any, config: fa.FamilyAConfig, loader: Loader, *, settle_sec: float = 60.0,
                 select_interval_sec: float = 21600.0, clock: Callable[[], datetime] = utcnow,
                 seeds: tuple[int, ...] = SELECT_SEEDS, train_days: float = TRAIN_DAYS,
                 adaptive_intervals: bool = True, events_enabled: bool = True):
        self.store = store
        self.cfg = config
        self.loader = loader
        self.grain = timedelta(minutes=config.grain_min)
        self.settle = timedelta(seconds=settle_sec)
        self.select_interval = timedelta(seconds=select_interval_sec)
        self.clock = clock
        self.seeds = tuple(seeds)
        self.train_days = train_days
        self.adaptive_intervals = adaptive_intervals
        self.due_limit = DUE_LIMIT
        self.served: Served | None = None
        self.next_select: datetime | None = None
        self.last_origin: datetime | None = None
        self._origin_loaded = False
        self.next_prune: datetime | None = None
        self._runs: dict[tuple[str, int], RunInfo | None] = {}
        self.state = State()
        self.events_enabled = events_enabled
        self._pending_events: list[dict] = []

    # --- clock ------------------------------------------------------------------------------------
    def latest_complete(self, now: datetime) -> datetime:
        """Start of the last grain bucket that is complete (settled) at ``now``."""
        return floor_ts(now - self.settle, self.cfg.grain_min) - self.grain

    # --- tick -------------------------------------------------------------------------------------
    def tick(self, now: datetime | None = None) -> dict:
        """score -> issue -> select (if due) -> prune (daily). One failing stage never stops the others."""
        now = now or self.clock()
        out: dict[str, Any] = {}
        for name, fn in (("score", self.score), ("issue", self.issue)):
            try:
                out[name] = fn(now)
                self.state.errors.pop(name, None)
            except Exception as exc:  # noqa: BLE001 -- one bad stage must not stop the loop
                self.state.errors[name] = f"{type(exc).__name__}: {exc}"[:300]
                logger.exception("family_a_%s_failed", name)
        if self.next_select is None or now >= self.next_select:
            try:
                out["select"] = self.select(now)
                self.state.errors.pop("select", None)
                self.next_select = now + self.select_interval
            except Exception as exc:  # noqa: BLE001
                self.state.errors["select"] = f"{type(exc).__name__}: {exc}"[:300]
                self.state.last_select = {"at": now, "ok": False, "error": self.state.errors["select"]}
                self.next_select = now + RETRY_SELECT
                if isinstance(exc, SelectRefused):
                    logger.warning("family_a_select_refused %s", exc)
                else:
                    logger.exception("family_a_select_failed")
        if self.next_prune is None or now >= self.next_prune:
            try:
                out["pruned"] = self.store.prune_family(PREFIX, now - RETENTION)
                self.next_prune = now + PRUNE_EVERY
            except Exception as exc:  # noqa: BLE001
                self.state.errors["prune"] = f"{type(exc).__name__}: {exc}"[:300]
                self.next_prune = now + RETRY_SELECT
        return out

    # --- select -----------------------------------------------------------------------------------
    def select(self, now: datetime) -> dict:
        t0 = time.time()
        g = self.cfg.grain_min
        hyg = self.cfg.hygiene
        end = self.latest_complete(now) + self.grain
        start = end - timedelta(days=self.train_days)
        frame, _last_seen = self.loader(start, end)
        verdicts = fa.screen_frame(frame, hyg)
        targets = [k for k, v in verdicts.items() if v.status == fa.TARGET]
        covs = [k for k, v in verdicts.items() if v.status == fa.COVARIATE]
        refused = {k: list(v.reasons) for k, v in verdicts.items() if v.status == fa.REFUSED}
        if not targets:
            raise SelectRefused(f"no input passed as a target: {refused}")
        inputs = targets + covs
        n = len(frame)
        backtest: dict[int, dict] = {}
        horizons: dict[int, ServedHorizon] = {}
        groups_skipped: dict[str, dict[str, str]] = {}
        for h in self.cfg.horizons_min:
            k = h // g
            seeds = []
            for seed in self.seeds:
                run, _v = fa.backtest(frame, hyg, k, g, seed)
                if run is None:
                    seeds.append({"seed": seed, "wins": False, "reason": "no_targets_on_training_rows"})
                    continue
                base, _pick = fa.best_baseline(run)
                cmp_ = fa.compare(run, "ridge_var", base, g, seed)
                ar_cmp = fa.compare(run, "ar", run.sq["persistence"], g, seed)
                seeds.append({"seed": seed, "wins": cmp_["wins"], "ridge_vs_best_baseline": cmp_,
                              "ar_vs_persistence": ar_cmp, "n_targets": len(run.targets)})
            rung = "ridge_var" if seeds and all(s["wins"] for s in seeds) else "ar"
            hf = fa.fit_horizon(frame, targets, inputs, k, np.arange(n - k))
            pr = hf.predict(frame)
            resid = pr["y"] - pr[rung]
            cal = _calibrate(resid, hf, k)
            if len(cal.cal_tail) < fa.MIN_CAL:
                raise SelectRefused(f"{h} min: calibration pool {len(cal.cal_tail)} rows < {fa.MIN_CAL}")
            groups, skipped = self._group_calibrations(hf, resid, k)
            if skipped:
                groups_skipped[str(h)] = skipped
                logger.warning("family_a_groups_skipped h=%d %s", h, skipped)
            horizons[h] = ServedHorizon(fit=hf, rung=rung, cal=cal, q_offsets=residual_quantiles(resid[hf.val_rows]),
                                        groups=groups)
            backtest[h] = {"rung": rung, "seeds": seeds}
        selection_id, run_id = str(uuid.uuid4()), str(uuid.uuid4())
        first = horizons[self.cfg.horizons_min[0]].fit
        manifest = fa.build_manifest(self.cfg, verdicts, first.standardizer, (start, end), end,
                                     extra={"rung": {str(h): s.rung for h, s in horizons.items()}})
        seconds = time.time() - t0
        common = {"selection_id": selection_id, "series_id": FAMILY, "fitted_at": now, "train_start": start,
                  "train_end": end, "n_obs": n, "grain_min": g, "usable_seasonalities": [],
                  "refused_seasonalities": [], "n_windows": len(self.seeds), "backtest_seconds": seconds,
                  "residual_paths": [], "errors": _finite([{"input": k, "reasons": v} for k, v in refused.items()])}
        rows = []
        for model in fa.RUNGS:
            rows.append({**common, "run_id": str(uuid.uuid4()), "kind": "model", "model": model,
                         "metrics": _finite({str(h): {"seeds": b["seeds"], "served": b["rung"] == model}
                                             for h, b in backtest.items()}),
                         "assignment": {}, "served_baseline": False})
        champ_metrics: dict[str, Any] = {"manifest": manifest, "calibration_served": SERVED_CALIBRATION}
        for h, s in horizons.items():
            champ_metrics[str(h)] = {
                "rung": s.rung, "targets": s.fit.targets, "refused_targets": s.fit.refused_targets,
                "standardizer": {t: s.fit.standardizer[t] for t in s.fit.targets},
                "alphas": s.fit.alphas, "joint": s.cal.to_json(),
                "n_fit_rows": int(len(s.fit.fit_rows)), "n_val_rows": int(len(s.fit.val_rows)),
                "backtest": backtest[h]["seeds"],
                "groups": {g: {"targets": gc.targets, "joint": gc.cal.to_json()} for g, gc in s.groups.items()},
                "groups_skipped": groups_skipped.get(str(h), {})}
        rows.append({**common, "run_id": run_id, "kind": "champion_blend", "model": "family_a_blend",
                     "metrics": _finite(champ_metrics),
                     "assignment": {str(h): s.rung for h, s in horizons.items()},
                     "served_baseline": any(s.rung != "ridge_var" for s in horizons.values())})
        self.store.insert_runs(rows)
        self.served = Served(
            run_id=run_id, selection_id=selection_id, fitted_at=now,
            named={k: verdicts[k].status for k in inputs},
            bands={k: verdicts[k].stats["band"] for k in inputs},
            # the rest floor (normal days only), never the window minimum: a probe frozen at a new low
            # must not become its own "floor" and pass as rest
            floors={k: verdicts[k].stats.get("rest_floor") for k in inputs}, horizons=horizons,
            flat_ceilings={k: verdicts[k].stats.get("max_day_mode_share") for k in inputs})
        for h, s in horizons.items():
            self._runs[(run_id, h)] = RunInfo(targets=s.fit.targets,
                                              std={t: s.fit.standardizer[t]["std"] for t in s.fit.targets},
                                              cal=s.cal, groups=dict(s.groups))
        summary = {"at": now, "ok": True, "run_id": run_id, "seconds": round(seconds, 1),
                   "window": [start, end], "n_targets": len(targets), "n_covariates": len(covs),
                   "refused_inputs": refused, "rung": {str(h): s.rung for h, s in horizons.items()},
                   "fit_refused_targets": {str(h): s.fit.refused_targets for h, s in horizons.items()
                                           if s.fit.refused_targets},
                   "groups": {str(h): {g: len(gc.targets) for g, gc in s.groups.items()} for h, s in horizons.items()},
                   "groups_skipped": groups_skipped}
        self.state.last_select = summary
        logger.info("family_a_selected run_id=%s rung=%s targets=%d covariates=%d refused=%s seconds=%.1f",
                    run_id, summary["rung"], len(targets), len(covs), sorted(refused), seconds)
        return summary

    def _group_calibrations(self, hf: fa.HorizonFit, resid: np.ndarray, k: int) -> tuple[dict[str, GroupCal], dict[str, str]]:
        """Per joint group: the group's own calibration on its own residual columns (same fit / validation
        rows as the blend). Marginal of the residuals, so a group's d^2 never borrows another group's misses.
        A group with no target this run, or a calibration pool under MIN_CAL, is skipped with the reason."""
        groups: dict[str, GroupCal] = {}
        skipped: dict[str, str] = {}
        for g in fa.GROUPS:
            cols = [j for j, t in enumerate(hf.targets) if self.cfg.group_of(t) == g]
            if not cols:
                skipped[g] = "no_targets"
                continue
            cal = _calibrate(resid[:, cols], hf, k)
            if len(cal.cal_tail) < fa.MIN_CAL or len(cal.cal_tail_adaptive) < fa.MIN_CAL:
                skipped[g] = (f"calibration pool static {len(cal.cal_tail)} / adaptive "
                              f"{len(cal.cal_tail_adaptive)} rows < {fa.MIN_CAL}")
                continue
            groups[g] = GroupCal(targets=[hf.targets[j] for j in cols], cal=cal)
        return groups, skipped

    # --- issue ------------------------------------------------------------------------------------
    def current_difficulty(self, origin: datetime, family: str = FAMILY) -> dict[int, float | None]:
        """Per horizon: mean d^2 / dof of the scored joint rows (of ``family``: the blend or one group) whose
        target fell in the hour up to ``origin`` (rows (origin - 1 h, origin]), None under MIN_DIFFICULTY_N --
        the backtest's ``difficulty_rows`` on stored rows."""
        window = self.grain * fa.DIFFICULTY_ROWS
        rows = self.store.family_recent_scored(family, origin - window)
        out: dict[int, float | None] = {}
        for h in self.cfg.horizons_min:
            ratios = [float(r["d2"]) / int(r["n_observed"]) for r in rows
                      if int(r["horizon_min"]) == h and r.get("d2") is not None and r.get("n_observed")
                      and origin - window < r["target_ts"] <= origin]
            out[h] = float(np.mean(ratios)) if len(ratios) >= fa.MIN_DIFFICULTY_N else None
        return out

    def adaptive_ranges(self, origin: datetime, served: Served) -> dict[tuple[str, int], np.ndarray]:
        """(reading, horizon) -> residual offsets at QUANTILES, from that reading's own scored champion rows
        in the POOL before ``origin`` (the rung serving each horizon now; app/adaptive.adaptive_offsets). A
        reading or horizon with too thin a pool or no recent difficulty is absent and keeps its validation
        range; a failure here keeps every validation range (never costs the forecast)."""
        if not self.adaptive_intervals:
            return {}
        try:
            readings = sorted({t for s in served.horizons.values() for t in s.fit.targets})
            rows = self.store.family_champion_pool([PREFIX + t for t in readings], pool_since(origin),
                                                   int(MAX_HORIZON.total_seconds() // 60))
            by: dict[str, list[dict]] = {}
            for r in rows:
                by.setdefault(r["series_id"], []).append(r)
            owners = {int(h): s.rung for h, s in served.horizons.items()}
            out: dict[tuple[str, int], np.ndarray] = {}
            for sid, rs in by.items():
                offs, _why = adaptive_offsets(rs, pd.Timestamp(origin), sorted(owners), self.cfg.grain_min, owners=owners)
                for h, o in offs.items():
                    out[(sid[len(PREFIX):], int(h))] = o
            self.state.errors.pop("adaptive", None)
            return out
        except Exception as exc:  # noqa: BLE001 -- the validation range is a valid forecast
            self.state.errors["adaptive"] = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("family_a_adaptive_failed %s: %s", type(exc).__name__, str(exc)[:200])
            return {}

    def issue(self, now: datetime) -> dict:
        served = self.served
        if served is None:
            return {"skipped": "no_selection"}
        origin = self.latest_complete(now)
        if not self._origin_loaded:
            self.last_origin = self.store.family_last_origin(FAMILY)
            self._origin_loaded = True
        if self.last_origin is not None and origin <= self.last_origin:
            return {"skipped": "already_issued"}
        end = origin + self.grain
        frame, last_seen = self.loader(end - timedelta(hours=GATE_HOURS), end)
        base = {"family": FAMILY, "run_id": served.run_id, "issued_at": now, "origin_ts": origin}
        try:
            fa.gate_named_inputs(served.named, frame, self.cfg.hygiene, last_seen, now,
                                 bands=served.bands, floors=served.floors, flat_ceilings=served.flat_ceilings)
        except fa.HygieneRefused as exc:
            joints = [{**base, "family": fam, "horizon_min": h, "target_ts": origin + timedelta(minutes=h),
                       "status": "refused", "refusal": exc.refusals, "n_forecast": 0, "difficulty": None}
                      for h, sh in served.horizons.items()
                      for fam in (FAMILY, *(GROUP_FAMILIES[g] for g in sh.groups))]
            self.store.insert_family_issue([], joints)
            self.last_origin = origin
            self.state.counts["refused"] += 1
            self.state.last_issue = {"at": now, "origin": origin, "ok": False, "refused": exc.refusals}
            logger.warning("family_a_issue_refused origin=%s reasons=%s", origin.isoformat(), exc.refusals)
            return {"refused": exc.refusals}
        diffs = self.current_difficulty(origin)
        group_diffs = {g: self.current_difficulty(origin, GROUP_FAMILIES[g])
                       for g in sorted({g for sh in served.horizons.values() for g in sh.groups})}
        adapt = self.adaptive_ranges(origin, served)
        n_adaptive = 0
        forecasts, shadows, joints = [], [], []
        n_by_h: dict[int, int] = {}
        for h, sh in served.horizons.items():
            target = origin + timedelta(minutes=h)
            if now >= target + self.grain:   # the target bucket already completed: never back-fill
                self.state.counts["late_horizons"] += 1
                continue
            pr = sh.fit.predict(frame)
            i = len(frame) - 1
            other = "ar" if sh.rung == "ridge_var" else "ridge_var"
            row = {**base, "horizon_min": h, "target_ts": target}
            n_f = 0
            forecast_targets: set[str] = set()
            for j, t in enumerate(sh.fit.targets):
                zt, dz = pr["z"][i, j], pr[sh.rung][i, j]
                offs = sh.q_offsets[:, j]
                live = adapt.get((t, int(h)))
                # adaptive only with a real width: >= 80 % identical scaled misses (a near-constant reading)
                # gives equal 0.1 / 0.9 offsets, and family C refuses a zero-width range
                if live is not None and not (np.isfinite(live).all() and live[K90] > live[K10]):
                    live = None
                if not (np.isfinite(zt) and np.isfinite(dz) and (live is not None or np.isfinite(offs).all())):
                    continue
                st = sh.fit.standardizer[t]
                mu, sd = st["mean"], st["std"]
                point = float(mu + sd * (zt + dz))
                if live is not None:
                    quantiles = {q_key(q): point + float(live[k]) for k, q in enumerate(QUANTILES)}
                    source = ADAPTIVE_SOURCE
                    n_adaptive += 1
                else:
                    quantiles = {q_key(q): float(mu + sd * (zt + dz + offs[k])) for k, q in enumerate(QUANTILES)}
                    source = INTERVAL_SOURCE
                forecasts.append({**row, "series_id": PREFIX + t, "model": sh.rung, "role": "champion",
                                  "point": point, "interval_source": source, "quantiles": quantiles})
                n_f += 1
                forecast_targets.add(t)
                dz2 = pr[other][i, j]
                if np.isfinite(dz2):
                    shadows.append({**row, "series_id": PREFIX + t, "model": other,
                                    "point": float(mu + sd * (zt + dz2))})
            n_by_h[h] = n_f
            joints.append({**row, "status": "issued", "refusal": None, "n_forecast": n_f, "difficulty": diffs.get(h)})
            for g, gc in sh.groups.items():
                joints.append({**row, "family": GROUP_FAMILIES[g], "status": "issued", "refusal": None,
                               "n_forecast": len(forecast_targets.intersection(gc.targets)),
                               "difficulty": group_diffs[g].get(h)})
        if joints:
            self.store.insert_family_issue(forecasts, joints)
            self.state.counts["issued"] += 1
        self.last_origin = origin
        if shadows:
            try:
                self.store.insert_family_shadow(shadows)
            except Exception as exc:  # noqa: BLE001 -- the shadow never costs the served issue
                self.state.counts["shadow_failed"] += 1
                self.state.errors["shadow"] = f"{type(exc).__name__}: {exc}"[:300]
        self.state.last_issue = {"at": now, "origin": origin, "ok": True, "n_forecast": n_by_h,
                                 "n_adaptive": n_adaptive, "difficulty": diffs}
        return {"origin": origin, "n_forecast": n_by_h}

    # --- score ------------------------------------------------------------------------------------
    def run_info(self, run_id: Any, horizon_min: int) -> RunInfo | None:
        key = (str(run_id), int(horizon_min))
        if key not in self._runs:
            row = self.store.run_metrics(run_id)
            m = ((row or {}).get("metrics") or {}).get(str(horizon_min))
            self._runs[key] = None if not m else RunInfo(
                targets=list(m["targets"]), std={t: float(v["std"]) for t, v in m["standardizer"].items()},
                cal=fa.JointCal.from_json(m["joint"]),
                groups={g: GroupCal(targets=list(v["targets"]), cal=fa.JointCal.from_json(v["joint"]))
                        for g, v in (m.get("groups") or {}).items()})
        return self._runs[key]

    def score(self, now: datetime) -> dict:
        cutoff = self.latest_complete(now) + self.grain
        joints, readings = self.store.family_due(FAMILY, PREFIX, cutoff, limit=self.due_limit)
        # Group rows score in the same pass as the blended row of their bucket, from the same readings (they
        # are always issued together). A group row whose blended row is not taken this pass (a backlog past
        # the limit) waits with it: taking it alone would score its readings without the blended row.
        blend_keys = {(str(j["run_id"]), j["origin_ts"], int(j["horizon_min"])) for j in joints}
        groups = [j for j in self.store.family_due_joints(list(GROUP_FAMILIES.values()),
                                                         sorted({j["origin_ts"] for j in joints}))
                  if (str(j["run_id"]), j["origin_ts"], int(j["horizon_min"])) in blend_keys]
        joints = [{**j, "family": FAMILY} for j in joints] + groups
        # Only the readings of the joint rows taken this pass: a joint row scores from readings scored in
        # the same pass, so a reading scored without its joint (a backlog past the joint limit) would
        # leave that joint with nothing to score from later. The rest wait for the next pass.
        keys = {(str(j["run_id"]), j["origin_ts"], int(j["horizon_min"])) for j in joints}
        readings = [r for r in readings if (str(r["run_id"]), r["origin_ts"], int(r["horizon_min"])) in keys]
        shadows = self.store.family_due_shadow(PREFIX, cutoff)
        if not (joints or readings or shadows):
            return {"scored": 0}
        lo = min(r["target_ts"] for r in [*joints, *readings, *shadows])
        frame, _ = self.loader(floor_ts(lo, self.cfg.grain_min), cutoff)

        def actual(ts: datetime, series_id: str) -> float:
            col = series_id[len(PREFIX):]
            t = pd.Timestamp(ts)
            if col not in frame.columns or t not in frame.index:
                return float("nan")
            return float(frame.at[t, col])

        r_scores, by_key = [], {}
        for r in readings:
            v = actual(r["target_ts"], r["series_id"])
            base = {"forecast_id": r["forecast_id"], "scored_at": now}
            if not np.isfinite(v):
                r_scores.append({**base, "score_status": "missing", "actual": None, "abs_error": None,
                                 "in_80": None, "in_95": None})
                continue
            q = r["quantiles"]
            r_scores.append({**base, "score_status": "scored", "actual": v, "abs_error": abs(v - float(r["point"])),
                             "in_80": q[q_key(0.1)] <= v <= q[q_key(0.9)],
                             "in_95": q[q_key(0.025)] <= v <= q[q_key(0.975)]})
            key = (str(r["run_id"]), r["origin_ts"], int(r["horizon_min"]))
            by_key.setdefault(key, []).append((r["series_id"][len(PREFIX):], v, float(r["point"])))
        j_scores, events = [], []
        n_group_ok = n_group = 0
        for jr in joints:
            h = int(jr["horizon_min"])
            info = self.run_info(jr["run_id"], h)
            got = by_key.get((str(jr["run_id"]), jr["origin_ts"], h), [])
            group = GROUP_OF_FAMILY.get(jr["family"])
            if info is None:
                sc_row = {"joint_id": jr["joint_id"], "scored_at": now, "n_observed": len(got),
                          "status": "missing", "calibration": "no_run"}
            elif group is not None and group not in info.groups:
                sc_row = {"joint_id": jr["joint_id"], "scored_at": now, "n_observed": len(got),
                          "status": "missing", "calibration": "no_group"}
            else:
                gc = info.groups[group] if group is not None else GroupCal(targets=info.targets, cal=info.cal)
                sc_row = _score_joint_row(jr["joint_id"], gc, info.std, got, jr.get("difficulty"), now)
            j_scores.append(sc_row)
            if group is None:
                continue
            n_group += 1
            n_group_ok += sc_row["status"] == "scored"
            if (sc_row["status"] == "scored" and sc_row["calibration"] == fa.ADAPTIVE
                    and jr["target_ts"] >= now - EVENT_MAX_AGE):
                events.append(_event(jr, group, sc_row, len(info.groups[group].cal.cal_tail_adaptive)))
        self.store.write_family_scores(r_scores, j_scores)
        self._queue_events(events)
        s_scores = []
        for s in shadows:
            v = actual(s["target_ts"], s["series_id"])
            s_scores.append({"shadow_id": s["shadow_id"], "scored_at": now,
                             "actual": v if np.isfinite(v) else None,
                             "abs_error": abs(v - float(s["point"])) if np.isfinite(v) else None})
        if s_scores:
            self.store.write_family_shadow_scores(s_scores)
        n_ok = sum(1 for s in j_scores if s["status"] == "scored") - n_group_ok
        n_blend = len(j_scores) - n_group
        self.state.counts["scored"] += n_ok
        self.state.counts["missing"] += n_blend - n_ok
        self.state.counts["group_scored"] += n_group_ok
        self.state.counts["group_missing"] += n_group - n_group_ok
        self.state.last_score = {"at": now, "readings": len(r_scores), "joint": n_blend, "joint_scored": n_ok,
                                 "group": n_group, "group_scored": n_group_ok, "events": len(events),
                                 "shadow": len(s_scores)}
        return {"scored": len(r_scores), "joint": n_blend, "joint_scored": n_ok, "group_scored": n_group_ok,
                "events": len(events)}

    # --- events -----------------------------------------------------------------------------------
    def _queue_events(self, events: list[dict]) -> None:
        if not (self.events_enabled and events):
            return
        self._pending_events.extend(events)
        self.state.counts["events_queued"] += len(events)
        over = len(self._pending_events) - PENDING_EVENTS_MAX
        if over > 0:   # keep the newest: an old score published late would read as now
            self._pending_events = self._pending_events[over:]
            self.state.counts["events_dropped_full"] += over

    def take_pending_events(self) -> list[dict]:
        """The ``prediction.joint_surprise.v1`` payloads queued by the last score passes, oldest first;
        clears the queue. Called by app/main.py after each tick (same task, never concurrent with it)."""
        out, self._pending_events = self._pending_events, []
        return out

    # --- health -----------------------------------------------------------------------------------
    def health(self) -> dict:
        s = self.served
        return _finite({
            "served": None if s is None else {"run_id": s.run_id, "fitted_at": s.fitted_at.isoformat(),
                                              "rung": {str(h): x.rung for h, x in s.horizons.items()},
                                              "n_named_inputs": len(s.named)},
            "calibration_served": SERVED_CALIBRATION,
            "next_select": self.next_select.isoformat() if self.next_select else None,
            "last_select": _iso(self.state.last_select), "last_issue": _iso(self.state.last_issue),
            "last_score": _iso(self.state.last_score), "counts": dict(self.state.counts),
            "groups": None if s is None else {str(h): sorted(x.groups) for h, x in s.horizons.items()},
            "events_enabled": self.events_enabled, "events_pending": len(self._pending_events),
            "errors": dict(self.state.errors)})


def _calibrate(resid: np.ndarray, hf: fa.HorizonFit, k: int) -> fa.JointCal:
    """Static then adaptive calibration of one joint (the blend or one group's columns): covariance on the
    fit rows, tails on the validation rows, each validation row scaled by its own hour-before difficulty.
    Returned exactly as stored (6 significant digits), so a row scores the same before and after a restart."""
    cal0 = fa.joint_calibration(resid[hf.fit_rows], resid[hf.val_rows])
    d2, dof, _t, _s = fa.joint_d2(resid, cal0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(dof > 0, d2 / np.maximum(dof, 1), np.nan)
    diff = fa.difficulty_rows(ratio, k)
    cal = fa.joint_calibration(resid[hf.fit_rows], resid[hf.val_rows], diff[hf.val_rows])
    return fa.JointCal.from_json(cal.to_json())


def _score_joint_row(joint_id: Any, gc: GroupCal, std: Mapping[str, float], got: list, difficulty: Any,
                     now: datetime) -> dict:
    """Score one joint row (blend or group) from the readings that arrived: d^2 on the joint's own targets,
    static and adaptive conformal p, excess = -ln p - calm point, top shares. Too few readings -> missing."""
    base = {"joint_id": joint_id, "scored_at": now, "n_observed": len(got)}
    col = {t: j for j, t in enumerate(gc.targets)}
    r = np.full((1, len(gc.targets)), np.nan)
    n_in = 0
    for t, v, point in got:
        if t in col and std.get(t):
            r[0, col[t]] = (v - point) / std[t]
            n_in += 1
    base["n_observed"] = n_in
    sc = fa.score_joint(r, gc.cal, np.array([np.nan if difficulty is None else float(difficulty)]))
    if not np.isfinite(sc.d2[0]):
        return {**base, "status": "missing", "calibration": "too_few_readings"}
    p_s, p_a = float(sc.p_static[0]), float(sc.p_adaptive[0])
    if not np.isfinite(p_s):
        return {**base, "status": "missing", "calibration": "no_calibration"}
    if SERVED_CALIBRATION == fa.ADAPTIVE and np.isfinite(p_a):
        p, calib, n_cal = p_a, fa.ADAPTIVE, len(gc.cal.cal_tail_adaptive)
    else:
        p, n_cal = p_s, len(gc.cal.cal_tail)
        calib = fa.STATIC if SERVED_CALIBRATION == fa.STATIC else "static:no_difficulty"
    calm = fa.calm_point(n_cal)
    shares = sc.shares[0]
    top = np.argsort(np.nan_to_num(shares, nan=-np.inf))[::-1][:TOP_SHARES]
    return {**base, "status": "scored", "n_observed": int(sc.dof[0]), "d2": float(sc.d2[0]),
            "p_static": p_s, "p_adaptive": p_a if np.isfinite(p_a) else None, "p": p,
            "calibration": calib, "calm_point": calm, "excess": float(-math.log(p) - calm),
            "top_shares": {gc.targets[t]: round(float(shares[t]), 4) for t in top if np.isfinite(shares[t])}}


def _event(jr: Mapping, group: str, sc: Mapping, n_cal: int) -> dict:
    """``PredictionJointSurpriseV1`` payload (orion/schemas/prediction.py) for one scored group row."""
    return {"joint_id": str(jr["joint_id"]), "family": FAMILY, "group": group, "run_id": str(jr["run_id"]),
            "horizon_min": int(jr["horizon_min"]), "origin_ts": jr["origin_ts"], "target_ts": jr["target_ts"],
            "scored_at": sc["scored_at"], "calibration": sc["calibration"], "n_observed": int(sc["n_observed"]),
            "n_cal": int(n_cal), "d2": float(sc["d2"]), "p": float(sc["p"]), "calm_point": float(sc["calm_point"]),
            "excess": float(sc["excess"]), "top_shares": dict(sc.get("top_shares") or {})}


def _iso(d: Mapping) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, (list, tuple)):
            out[k] = [x.isoformat() if isinstance(x, datetime) else x for x in v]
        elif isinstance(v, dict):
            out[k] = {str(a): b for a, b in v.items()}
        else:
            out[k] = v
    return out


# --- in-memory store: the replay dry run (evals) and the tests -----------------------------------------

class MemoryStore:
    """The PostgresStore methods FamilyALive calls, in memory. Same semantics: unique keys skip
    duplicates, scoring updates only unscored rows."""

    def __init__(self) -> None:
        self.runs: list[dict] = []
        self.forecasts: list[dict] = []
        self.shadows: list[dict] = []
        self.joints: list[dict] = []
        self._ids = 0

    def _id(self) -> int:
        self._ids += 1
        return self._ids

    def insert_runs(self, rows: list[dict]) -> None:
        self.runs.extend(dict(r) for r in rows)

    def run_metrics(self, run_id: Any) -> dict | None:
        return next((r for r in self.runs if str(r["run_id"]) == str(run_id)), None)

    def family_last_origin(self, family: str) -> datetime | None:
        o = [j["origin_ts"] for j in self.joints if j["family"] == family]
        return max(o) if o else None

    def insert_family_issue(self, forecast_rows: list[dict], joint_rows: list[dict]) -> None:
        keys = {(f["series_id"], f["role"], f["origin_ts"], f["horizon_min"]) for f in self.forecasts}
        for r in forecast_rows:
            if (r["series_id"], r["role"], r["origin_ts"], r["horizon_min"]) not in keys:
                self.forecasts.append({**r, "forecast_id": self._id(), "scored_at": None})
        jkeys = {(j["family"], j["origin_ts"], j["horizon_min"]) for j in self.joints}
        for r in joint_rows:
            if (r["family"], r["origin_ts"], r["horizon_min"]) not in jkeys:
                self.joints.append({**r, "joint_id": self._id(), "scored_at": None})

    def insert_family_shadow(self, rows: list[dict]) -> None:
        self.shadows.extend({**r, "shadow_id": self._id(), "scored_at": None} for r in rows)

    def family_due(self, family: str, prefix: str, before: datetime, limit: int = 2000) -> tuple[list, list]:
        j = [r for r in self.joints if r["family"] == family and r["status"] == "issued" and r["target_ts"] < before]
        f = [r for r in self.forecasts if r["series_id"].startswith(prefix) and r["scored_at"] is None
             and r["target_ts"] < before]
        return j[:limit], f[:limit * 100]

    def family_due_joints(self, families: list[str], origins: list[datetime]) -> list[dict]:
        fams, ors = set(families), set(origins)
        return [r for r in self.joints if r["family"] in fams and r["status"] == "issued" and r["origin_ts"] in ors]

    def family_due_shadow(self, prefix: str, before: datetime, limit: int = 200000) -> list[dict]:
        return [r for r in self.shadows if r["series_id"].startswith(prefix) and r["scored_at"] is None
                and r["target_ts"] < before][:limit]

    def write_family_scores(self, reading_scores: list[dict], joint_scores: list[dict]) -> None:
        f = {r["forecast_id"]: r for r in self.forecasts}
        for s in reading_scores:
            if f[s["forecast_id"]]["scored_at"] is None:
                f[s["forecast_id"]].update(s)
        j = {r["joint_id"]: r for r in self.joints}
        for s in joint_scores:
            if j[s["joint_id"]]["status"] == "issued":
                j[s["joint_id"]].update(s)

    def write_family_shadow_scores(self, rows: list[dict]) -> None:
        sh = {r["shadow_id"]: r for r in self.shadows}
        for s in rows:
            sh[s["shadow_id"]].update(s)

    def family_champion_pool(self, series_ids: list[str], since: datetime, max_horizon_min: int) -> list[dict]:
        ids = set(series_ids)
        return [{**r, "owner": r["model"]} for r in self.forecasts
                if r["series_id"] in ids and r.get("role") == "champion"
                and r.get("score_status") == "scored" and r["target_ts"] >= since
                and int(r["horizon_min"]) <= max_horizon_min]

    def family_recent_scored(self, family: str, since: datetime) -> list[dict]:
        return [r for r in self.joints if r["family"] == family and r["status"] == "scored" and r["target_ts"] > since]

    def family_joint_rows(self, family: str, since: datetime) -> list[dict]:
        return sorted((r for r in self.joints if r["family"] == family and r["target_ts"] >= since),
                      key=lambda r: (r["horizon_min"], r["target_ts"]))

    def prune_family(self, prefix: str, before: datetime) -> int:
        n0 = len(self.forecasts) + len(self.shadows)
        self.forecasts = [r for r in self.forecasts if not (r["series_id"].startswith(prefix) and r["target_ts"] < before)]
        self.shadows = [r for r in self.shadows if not (r["series_id"].startswith(prefix) and r["target_ts"] < before)]
        return n0 - len(self.forecasts) - len(self.shadows)


def frame_loader(frame: pd.DataFrame, grain_min: int) -> Loader:
    """A loader over an already-loaded frame (the replay dry run): rows [start, end), and last_seen as of
    ``end`` -- exactly what ``family_a_load.load_frame`` returns from the database for that range."""
    def load(start: datetime, end: datetime) -> tuple[pd.DataFrame, dict]:
        idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(minutes=grain_min),
                            freq=f"{grain_min}min", tz="UTC")
        sub = frame.reindex(idx)
        last = {}
        for c in sub.columns:
            ok = sub[c].dropna()
            last[c] = ok.index[-1].to_pydatetime() + timedelta(minutes=grain_min) if len(ok) else None
        return sub, last
    return load
