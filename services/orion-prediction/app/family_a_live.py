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
  (present, not stuck, inside its training band, fresh). A refusal stores one ``refused`` joint row per
  horizon with the reasons and logs them; nothing is forecast (refuse, never degrade). Otherwise each
  forecastable reading gets a ``prediction_forecast`` row (``series_id = 'family_a:<id>'``, role
  champion, 80/95 % ranges from its validation residuals), the other rung a shadow row, and each horizon
  one ``prediction_joint_surprise`` row (status ``issued``) carrying the hour-before difficulty known
  now. A horizon whose target bucket has already completed is skipped, never back-filled.
- **score** (each tick, before issuing): every due reading row gets its real bucket mean or ``missing``;
  each joint row gets d^2 on the readings that arrived (whitened by the run's covariance), the static
  and the adaptive conformal p, and excess = -ln p - calm point. Too few readings -> ``missing``.

No substrate node, no bus event, no consumer: that is the next patch, gated on the 7-day calm check
(``evals/run_family_a_calm_live_eval.py``).
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
TOP_SHARES = 5
BLAS_THREADS = 4

Loader = Callable[[datetime, datetime], "tuple[pd.DataFrame, dict[str, datetime | None]]"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
class ServedHorizon:
    fit: fa.HorizonFit
    rung: str
    cal: fa.JointCal
    q_offsets: np.ndarray            # (len(QUANTILES), m) standardised residual offsets, validation rows


@dataclass
class Served:
    run_id: str
    selection_id: str
    fitted_at: datetime
    named: dict[str, str]
    bands: dict[str, list]
    floors: dict[str, float]
    horizons: dict[int, ServedHorizon]


@dataclass
class RunInfo:
    """What scoring needs from the run that issued a row (read back from its stored metrics)."""
    targets: list[str]
    std: dict[str, float]
    cal: fa.JointCal


class SelectRefused(RuntimeError):
    pass


@dataclass
class State:
    last_select: dict = field(default_factory=dict)
    last_issue: dict = field(default_factory=dict)
    last_score: dict = field(default_factory=dict)
    counts: dict = field(default_factory=lambda: {"issued": 0, "refused": 0, "late_horizons": 0,
                                                  "scored": 0, "missing": 0, "shadow_failed": 0})
    errors: dict = field(default_factory=dict)


class FamilyALive:
    def __init__(self, store: Any, config: fa.FamilyAConfig, loader: Loader, *, settle_sec: float = 60.0,
                 select_interval_sec: float = 21600.0, clock: Callable[[], datetime] = utcnow,
                 seeds: tuple[int, ...] = SELECT_SEEDS, train_days: float = TRAIN_DAYS):
        self.store = store
        self.cfg = config
        self.loader = loader
        self.grain = timedelta(minutes=config.grain_min)
        self.settle = timedelta(seconds=settle_sec)
        self.select_interval = timedelta(seconds=select_interval_sec)
        self.clock = clock
        self.seeds = tuple(seeds)
        self.train_days = train_days
        self.served: Served | None = None
        self.next_select: datetime | None = None
        self.last_origin: datetime | None = None
        self._origin_loaded = False
        self.next_prune: datetime | None = None
        self._runs: dict[tuple[str, int], RunInfo | None] = {}
        self.state = State()

    # --- clock ------------------------------------------------------------------------------------
    def latest_complete(self, now: datetime) -> datetime:
        """Start of the last grain bucket that is complete (settled) at ``now``."""
        return floor_ts(now - self.settle, self.cfg.grain_min) - self.grain

    # --- tick -------------------------------------------------------------------------------------
    def tick(self, now: datetime | None = None) -> dict:
        """score -> issue -> select (if due) -> prune (daily). One failing stage never stops the others.

        BLAS is held to BLAS_THREADS for the tick: OpenBLAS defaults to one thread per core (64 here),
        and on a loaded host (load ~150 on 96 cores, 2026-11-13) one 42-reading fit took > 120 s that
        way vs 1.5 s on 2 threads. The limit is process-wide while it holds (OpenBLAS has one setting),
        so the single-series loop's numpy calls in that window run on BLAS_THREADS too."""
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=BLAS_THREADS, user_api="blas"):
            return self._tick(now or self.clock())

    def _tick(self, now: datetime) -> dict:
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
            cal0 = fa.joint_calibration(resid[hf.fit_rows], resid[hf.val_rows])
            d2, dof, _t, _s = fa.joint_d2(resid, cal0)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(dof > 0, d2 / np.maximum(dof, 1), np.nan)
            diff = fa.difficulty_rows(ratio, k)
            cal = fa.joint_calibration(resid[hf.fit_rows], resid[hf.val_rows], diff[hf.val_rows])
            horizons[h] = ServedHorizon(fit=hf, rung=rung, cal=cal, q_offsets=residual_quantiles(resid[hf.val_rows]))
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
                "backtest": backtest[h]["seeds"]}
        rows.append({**common, "run_id": run_id, "kind": "champion_blend", "model": "family_a_blend",
                     "metrics": _finite(champ_metrics),
                     "assignment": {str(h): s.rung for h, s in horizons.items()},
                     "served_baseline": any(s.rung != "ridge_var" for s in horizons.values())})
        self.store.insert_runs(rows)
        self.served = Served(
            run_id=run_id, selection_id=selection_id, fitted_at=now,
            named={k: verdicts[k].status for k in inputs},
            bands={k: verdicts[k].stats["band"] for k in inputs},
            floors={k: verdicts[k].stats["min"] for k in inputs}, horizons=horizons)
        for h, s in horizons.items():
            self._runs[(run_id, h)] = RunInfo(targets=s.fit.targets,
                                              std={t: s.fit.standardizer[t]["std"] for t in s.fit.targets},
                                              cal=s.cal)
        summary = {"at": now, "ok": True, "run_id": run_id, "seconds": round(seconds, 1),
                   "window": [start, end], "n_targets": len(targets), "n_covariates": len(covs),
                   "refused_inputs": refused, "rung": {str(h): s.rung for h, s in horizons.items()},
                   "fit_refused_targets": {str(h): s.fit.refused_targets for h, s in horizons.items()
                                           if s.fit.refused_targets}}
        self.state.last_select = summary
        logger.info("family_a_selected run_id=%s rung=%s targets=%d covariates=%d refused=%s seconds=%.1f",
                    run_id, summary["rung"], len(targets), len(covs), sorted(refused), seconds)
        return summary

    # --- issue ------------------------------------------------------------------------------------
    def current_difficulty(self, origin: datetime) -> dict[int, float | None]:
        """Per horizon: mean d^2 / dof of the scored joint rows whose target fell in the hour up to
        ``origin`` (rows (origin - 1 h, origin]), None under MIN_DIFFICULTY_N -- the backtest's
        ``difficulty_rows`` on stored rows."""
        window = self.grain * fa.DIFFICULTY_ROWS
        rows = self.store.family_recent_scored(FAMILY, origin - window)
        out: dict[int, float | None] = {}
        for h in self.cfg.horizons_min:
            ratios = [float(r["d2"]) / int(r["n_observed"]) for r in rows
                      if int(r["horizon_min"]) == h and r.get("d2") is not None and r.get("n_observed")
                      and origin - window < r["target_ts"] <= origin]
            out[h] = float(np.mean(ratios)) if len(ratios) >= fa.MIN_DIFFICULTY_N else None
        return out

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
                                 bands=served.bands, floors=served.floors)
        except fa.HygieneRefused as exc:
            joints = [{**base, "horizon_min": h, "target_ts": origin + timedelta(minutes=h), "status": "refused",
                       "refusal": exc.refusals, "n_forecast": 0, "difficulty": None}
                      for h in served.horizons]
            self.store.insert_family_issue([], joints)
            self.last_origin = origin
            self.state.counts["refused"] += 1
            self.state.last_issue = {"at": now, "origin": origin, "ok": False, "refused": exc.refusals}
            logger.warning("family_a_issue_refused origin=%s reasons=%s", origin.isoformat(), exc.refusals)
            return {"refused": exc.refusals}
        diffs = self.current_difficulty(origin)
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
            for j, t in enumerate(sh.fit.targets):
                zt, dz = pr["z"][i, j], pr[sh.rung][i, j]
                offs = sh.q_offsets[:, j]
                if not (np.isfinite(zt) and np.isfinite(dz) and np.isfinite(offs).all()):
                    continue
                st = sh.fit.standardizer[t]
                mu, sd = st["mean"], st["std"]
                forecasts.append({**row, "series_id": PREFIX + t, "model": sh.rung, "role": "champion",
                                  "point": float(mu + sd * (zt + dz)), "interval_source": INTERVAL_SOURCE,
                                  "quantiles": {q_key(q): float(mu + sd * (zt + dz + offs[k]))
                                                for k, q in enumerate(QUANTILES)}})
                n_f += 1
                dz2 = pr[other][i, j]
                if np.isfinite(dz2):
                    shadows.append({**row, "series_id": PREFIX + t, "model": other,
                                    "point": float(mu + sd * (zt + dz2))})
            n_by_h[h] = n_f
            joints.append({**row, "status": "issued", "refusal": None, "n_forecast": n_f, "difficulty": diffs.get(h)})
        self.store.insert_family_issue(forecasts, joints)
        self.last_origin = origin
        self.state.counts["issued"] += 1
        if shadows:
            try:
                self.store.insert_family_shadow(shadows)
            except Exception as exc:  # noqa: BLE001 -- the shadow never costs the served issue
                self.state.counts["shadow_failed"] += 1
                self.state.errors["shadow"] = f"{type(exc).__name__}: {exc}"[:300]
        self.state.last_issue = {"at": now, "origin": origin, "ok": True, "n_forecast": n_by_h,
                                 "difficulty": diffs}
        return {"origin": origin, "n_forecast": n_by_h}

    # --- score ------------------------------------------------------------------------------------
    def run_info(self, run_id: Any, horizon_min: int) -> RunInfo | None:
        key = (str(run_id), int(horizon_min))
        if key not in self._runs:
            row = self.store.run_metrics(run_id)
            m = ((row or {}).get("metrics") or {}).get(str(horizon_min))
            self._runs[key] = None if not m else RunInfo(
                targets=list(m["targets"]), std={t: float(v["std"]) for t, v in m["standardizer"].items()},
                cal=fa.JointCal.from_json(m["joint"]))
        return self._runs[key]

    def score(self, now: datetime) -> dict:
        cutoff = self.latest_complete(now) + self.grain
        joints, readings = self.store.family_due(FAMILY, PREFIX, cutoff)
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
        j_scores = []
        for jr in joints:
            h = int(jr["horizon_min"])
            info = self.run_info(jr["run_id"], h)
            got = by_key.get((str(jr["run_id"]), jr["origin_ts"], h), [])
            base = {"joint_id": jr["joint_id"], "scored_at": now, "n_observed": len(got)}
            if info is None:
                j_scores.append({**base, "status": "missing", "calibration": "no_run"})
                continue
            col = {t: j for j, t in enumerate(info.targets)}
            r = np.full((1, len(info.targets)), np.nan)
            for t, v, point in got:
                if t in col and info.std.get(t):
                    r[0, col[t]] = (v - point) / info.std[t]
            diff = jr.get("difficulty")
            sc = fa.score_joint(r, info.cal, np.array([np.nan if diff is None else float(diff)]))
            if not np.isfinite(sc.d2[0]):
                j_scores.append({**base, "status": "missing", "calibration": "too_few_readings"})
                continue
            p_s, p_a = float(sc.p_static[0]), float(sc.p_adaptive[0])
            if SERVED_CALIBRATION == fa.ADAPTIVE and np.isfinite(p_a):
                p, calib, n_cal = p_a, fa.ADAPTIVE, len(info.cal.cal_tail_adaptive)
            else:
                p, n_cal = p_s, len(info.cal.cal_tail)
                calib = fa.STATIC if SERVED_CALIBRATION == fa.STATIC else "static:no_difficulty"
            calm = fa.calm_point(n_cal)
            shares = sc.shares[0]
            top = np.argsort(np.nan_to_num(shares, nan=-np.inf))[::-1][:TOP_SHARES]
            j_scores.append({**base, "status": "scored", "n_observed": int(sc.dof[0]), "d2": float(sc.d2[0]),
                             "p_static": p_s, "p_adaptive": p_a if np.isfinite(p_a) else None, "p": p,
                             "calibration": calib, "calm_point": calm, "excess": float(-math.log(p) - calm),
                             "top_shares": {info.targets[t]: round(float(shares[t]), 4) for t in top
                                            if np.isfinite(shares[t])}})
        self.store.write_family_scores(r_scores, j_scores)
        s_scores = []
        for s in shadows:
            v = actual(s["target_ts"], s["series_id"])
            s_scores.append({"shadow_id": s["shadow_id"], "scored_at": now,
                             "actual": v if np.isfinite(v) else None,
                             "abs_error": abs(v - float(s["point"])) if np.isfinite(v) else None})
        if s_scores:
            self.store.write_family_shadow_scores(s_scores)
        n_ok = sum(1 for s in j_scores if s["status"] == "scored")
        self.state.counts["scored"] += n_ok
        self.state.counts["missing"] += len(j_scores) - n_ok
        self.state.last_score = {"at": now, "readings": len(r_scores), "joint": len(j_scores), "joint_scored": n_ok,
                                 "shadow": len(s_scores)}
        return {"scored": len(r_scores), "joint": len(j_scores), "joint_scored": n_ok}

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
            "errors": dict(self.state.errors)})


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
