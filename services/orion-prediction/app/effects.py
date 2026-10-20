"""Causal endpoints' orchestration: data loading + storage around app/causal.py.

``best_effect(action)`` is the Orion-facing answer to "if X happens, how much does the target
move?": the newest experimental estimate if one exists, else the pooled observational estimates,
else nothing plus the advice to run a switchback experiment. ``claim_kind`` always says which.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.causal import (
    CausalError,
    analyze_switchback,
    design_switchback,
    intervention_effect,
    scenario,
    time_of_day_profile,
)
from app.engine import Engine, InputStale
from app.events import breaks_between
from app.frame import mask_windows, to_grain, training_slice
from app.models import build_models, split_seasonalities
from app.series import SeriesSpec

ITS_MODEL = "mstl_ets"
PROFILE_DAYS = 14


class Effects:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.store = engine.store

    def _spec(self, series_id: str, role: str | None = None) -> SeriesSpec:
        spec = self.engine.registry.get(series_id)
        if spec is None or (role and spec.role != role):
            raise CausalError(f"unknown {role or ''} series {series_id}".replace("  ", " "))
        return spec

    def _series(self, spec: SeriesSpec, start: datetime, end: datetime) -> pd.Series:
        """Regular view with the spec's exclusions (registry + change events) masked out."""
        return mask_windows(to_grain(self.store.points(spec, start, end), spec.grain_min, spec.agg,
                                     spec.max_fill_min, start=start, end=end), spec.exclude)

    # --- level 1 --------------------------------------------------------------------------
    def scenario(self, series_id: str, covariate_id: str, path: list[float], days: float = 7.0) -> dict:
        spec = self.engine.effective(self._spec(series_id, "target"), self.engine.clock())
        if covariate_id not in spec.covariates:
            raise CausalError(f"{covariate_id} is not a declared covariate of {series_id}")
        cov = self._spec(covariate_id)
        now = self.engine.clock()
        end = self.engine.latest_complete_bucket(spec, now) + pd.Timedelta(minutes=spec.grain_min)
        start = end.to_pydatetime() - timedelta(days=days)
        if spec.train_from and start < spec.train_from:
            start = spec.train_from
        y = training_slice(self._series(spec, start, end.to_pydatetime()), spec)
        x = self._series(cov, start, end.to_pydatetime()).reindex(y.index)
        res = scenario(y, x, covariate_id, path, spec.grain_min)
        return {**res.__dict__, "series_id": series_id, "unit": spec.unit, "covariate_unit": cov.unit}

    # --- level 2 --------------------------------------------------------------------------
    def log_intervention(self, action: str, started_at: datetime, ended_at: datetime, source: str,
                         notes: str = "", arm: str | None = None, experiment_id: str | None = None) -> dict:
        if ended_at <= started_at:
            raise CausalError("ended_at must be after started_at")
        return self.store.insert_intervention({"action": action, "started_at": started_at, "ended_at": ended_at,
                                               "source": source, "notes": notes, "arm": arm,
                                               "experiment_id": experiment_id})

    def estimate_intervention(self, intervention_id: str, series_id: str, pre_days: float = 3.0,
                              lag_min: int = 30) -> dict:
        spec = self._spec(series_id, "target")
        iv = self.store.intervention(intervention_id)
        if iv is None:
            raise CausalError("unknown intervention")
        if iv.get("experiment_id"):
            raise CausalError("this intervention is an experiment block; analyze the experiment instead")
        start, end = iv["started_at"], iv["ended_at"]
        now = self.engine.clock()
        if end + timedelta(minutes=lag_min) > now:
            raise CausalError("the intervention window (plus lag) has not finished yet")
        span_start = start - timedelta(days=pre_days * 4)
        stop = end + timedelta(minutes=lag_min + spec.grain_min)
        crossed = breaks_between(self.engine.change_events(), series_id, span_start, stop)
        if crossed:
            raise CausalError(f"a change-event break at {crossed[-1].isoformat()} falls inside the "
                              "intervention's history window; its counterfactual would straddle two regimes")
        spec = self.engine.effective(spec, now)
        series = self._series(spec, span_start, stop)
        usable, _ = split_seasonalities(spec, int(pre_days * 1440 // spec.grain_min))
        model = build_models(spec, usable)[ITS_MODEL]
        busy = [(r["started_at"], r["ended_at"] + timedelta(minutes=lag_min)) for r in self.store.interventions(limit=500)
                if str(r["intervention_id"]) != str(intervention_id)]
        est = intervention_effect(series, model, start, end, spec.grain_min, pre_days=pre_days, lag_min=lag_min,
                                  busy=busy)
        return self.store.insert_effect({"action": iv["action"], "series_id": series_id, "claim_kind": est.claim_kind,
                                         "intervention_id": str(intervention_id), "effect": est.effect,
                                         "ci_lo": est.ci_lo, "ci_hi": est.ci_hi, "p_value": est.p_value,
                                         "n": est.n, "details": est.details})

    # --- level 3 --------------------------------------------------------------------------
    def guard(self, spec: SeriesSpec) -> dict:
        t = spec.threshold("experiment_guard")
        if t is None:
            return {"safe": False, "reason": f"{spec.series_id} declares no experiment_guard threshold"}
        st = self.engine.state.get(spec.series_id)
        if st is None or st.served is None:
            return {"safe": False, "reason": "no current selection"}
        now = self.engine.clock()
        spec = self.engine.effective(spec, now)
        series = self.engine.load_series(spec, now, timedelta(minutes=spec.train_window_min + spec.grain_min))
        origin = self.engine.latest_complete_bucket(spec, now)
        try:
            built = self.engine.build_forecast(spec, series, origin, now)
        except InputStale as exc:
            return {"safe": False, "reason": f"input stale ({exc})", "checked_at": now.isoformat()}
        except Exception as exc:  # noqa: BLE001 -- no forecast means no permission
            return {"safe": False, "reason": f"no forecast: {type(exc).__name__}: {exc}"[:300],
                    "checked_at": now.isoformat()}
        ex = next((e for e in built["payload"].exceedance if e.name == "experiment_guard"), None)
        p = ex.probability if ex else None
        safe = p is not None and p < (t.max_probability if t.max_probability is not None else 0.0)
        return {"safe": safe, "probability": p, "threshold": t.value, "window_min": t.window_min,
                "max_probability": t.max_probability, "checked_at": now.isoformat(),
                "reason": "ok" if safe else ("probability unknown" if p is None else "too likely to overheat")}

    def design_experiment(self, action: str, series_id: str, start_at: datetime, block_min: int, n_blocks: int,
                          burn_in_min: int, seed: int | None = None) -> dict:
        spec = self._spec(series_id, "target")
        if burn_in_min >= block_min:
            raise CausalError("burn_in_min must be shorter than block_min")
        guard = self.guard(spec)
        design = design_switchback(start_at, block_min, n_blocks, seed)
        row = {"experiment_id": str(uuid.uuid4()), "action": action, "series_id": series_id,
               "created_at": self.engine.clock(), "burn_in_min": burn_in_min, "design": design, "guard": guard}
        self.store.insert_experiment(row)
        return {**row, "status": "designed",
                "note": ("No dispatcher runs this design. Whoever performs the action logs each block via "
                         "POST /interventions with experiment_id and arm, then calls /experiments/{id}/analyze. "
                         "Re-check GET /experiments/guard before every ON block.")}

    def analyze_experiment(self, experiment_id: str) -> dict:
        exp = self.store.experiment(experiment_id)
        if exp is None:
            raise CausalError("unknown experiment")
        spec = self._spec(exp["series_id"], "target")
        logged = [r for r in self.store.interventions(experiment_id=experiment_id, limit=1000) if r["arm"]]
        if not logged:
            raise CausalError("no logged blocks for this experiment")
        blocks = self._match_design(exp, logged)
        first = min(pd.Timestamp(b["start"]) for b in blocks).to_pydatetime()
        last = max(pd.Timestamp(b["end"]) for b in blocks).to_pydatetime()
        if last > self.engine.clock():
            raise CausalError("the experiment has blocks that have not finished")
        spec = self.engine.effective(spec, self.engine.clock())
        hist_start = first - timedelta(days=PROFILE_DAYS)
        before = breaks_between(self.engine.change_events(), spec.series_id, hist_start, first)
        if before:   # the time-of-day profile must come from the regime the experiment ran in
            hist_start = before[-1]
        history = self._series(spec, hist_start, first)
        profile = time_of_day_profile(history.dropna(), spec.grain_min) if history.notna().any() else None
        series = self._series(spec, first, last)
        est = analyze_switchback(series, blocks, exp["burn_in_min"], spec.grain_min, profile)
        if est.effect is not None:
            self.store.set_experiment_status(experiment_id, "analyzed")
        return self.store.insert_effect({"action": exp["action"], "series_id": spec.series_id,
                                         "claim_kind": est.claim_kind, "experiment_id": experiment_id,
                                         "effect": est.effect, "ci_lo": est.ci_lo, "ci_hi": est.ci_hi,
                                         "p_value": est.p_value, "n": est.n, "details": est.details})

    @staticmethod
    def _match_design(exp: dict, logged: list[dict], tolerance_min: float = 2.0) -> list[dict]:
        """Only blocks run as randomised count as an experiment: each logged block must be a design
        block (start and end within tolerance, same arm) for the experiment's own action."""
        design = exp["design"]["blocks"]
        tol = pd.Timedelta(minutes=tolerance_min)
        used, out, bad = set(), [], []
        for r in logged:
            if r["action"] != exp["action"]:
                bad.append(f"{r['intervention_id']}: action {r['action']} != {exp['action']}")
                continue
            s0, s1 = pd.Timestamp(r["started_at"]), pd.Timestamp(r["ended_at"])
            hit = next((b for b in design if b["index"] not in used
                        and abs(pd.Timestamp(b["start"]) - s0) <= tol and abs(pd.Timestamp(b["end"]) - s1) <= tol), None)
            if hit is None:
                bad.append(f"{r['intervention_id']}: {s0.isoformat()} matches no design block")
            elif hit["arm"] != r["arm"]:
                bad.append(f"{r['intervention_id']}: logged arm {r['arm']} but the design says {hit['arm']}")
            else:
                used.add(hit["index"])
                out.append({"index": hit["index"], "arm": hit["arm"], "start": s0.isoformat(), "end": s1.isoformat()})
        if bad:
            raise CausalError("logged blocks do not follow the randomised design: " + "; ".join(bad[:5]))
        return sorted(out, key=lambda b: b["index"])

    # --- the Orion-facing answer --------------------------------------------------------------
    def best_effect(self, action: str, series_id: str) -> dict:
        spec = self._spec(series_id, "target")
        rows = [r for r in self.store.effects(action, series_id) if r["effect"] is not None]
        exp = [r for r in rows if r["claim_kind"] == "experimental_effect"]
        if exp:
            r = exp[0]
            return {"action": action, "series_id": series_id, "unit": spec.unit, "claim_kind": "experimental_effect",
                    "effect": r["effect"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p_value": r["p_value"],
                    "n": r["n"], "estimate_ids": [str(r["estimate_id"])],
                    "summary": f"randomised experiment: {action} changes {series_id} by {r['effect']:+.2f} {spec.unit}"}
        # Newest estimate per intervention only: re-estimating one occurrence must not count it twice.
        newest: dict[str, dict] = {}
        for r in rows:   # store.effects is newest-first
            if r["claim_kind"] == "observational_effect" and r["ci_lo"] is not None:
                newest.setdefault(str(r["intervention_id"]), r)
        obs = list(newest.values())
        if obs:
            # Inverse-variance pool, with each estimate's SE read off its placebo 95% interval.
            se = np.array([max((r["ci_hi"] - r["ci_lo"]) / (2 * 1.96), 1e-9) for r in obs])
            w = 1 / se ** 2
            eff = float(np.sum(w * np.array([r["effect"] for r in obs])) / np.sum(w))
            pse = float(np.sqrt(1 / np.sum(w)))
            return {"action": action, "series_id": series_id, "unit": spec.unit, "claim_kind": "observational_effect",
                    "effect": eff, "ci_lo": eff - 1.96 * pse, "ci_hi": eff + 1.96 * pse, "p_value": None,
                    "n": len(obs), "estimate_ids": [str(r["estimate_id"]) for r in obs],
                    "summary": (f"{len(obs)} logged occurrence(s): {action} went with {eff:+.2f} {spec.unit}; "
                                "confounded by anything that changed at the same time. Run a switchback "
                                "experiment for a causal answer.")}
        return {"action": action, "series_id": series_id, "unit": spec.unit, "claim_kind": "none", "effect": None,
                "ci_lo": None, "ci_hi": None, "p_value": None, "n": 0, "estimate_ids": [],
                "summary": (f"no estimate for {action}: log occurrences via POST /interventions, or design a "
                            "switchback experiment via POST /experiments/design")}
