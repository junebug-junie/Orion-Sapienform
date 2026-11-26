"""Live parity: do the substrate surprise folds, fed the rows the producers stored, reproduce the node values
the live substrate actually wrote? Read-only.

Shared-bet-primitives Patch A (PR #4013) moved the fold math into orion/prediction/surprisal.py. Its golden
tests rebuild bus events from stored rows; this eval anchors that rebuild to the real bus: for every stored
substrate_node_prediction_error_history value of node:prediction.room / .body / .room_joint /
.body_joint, replay the stored forecast scores (room/body, through the producer's own event gate) and joint
rows (n_cal recovered from the stored calm point) through the fold as of that time, and compare.

Agreement is not 100 % even with unchanged code: the live windows are in memory, keyed on arrival time,
and reset on restart. Measured 2026-11-26 over the 24 h before the rebuild: room/body 525/549 (95.6 %),
joint 541/561 (96.4 %), identical on the pre-refactor tree. The post-rebuild check: run with
--since <rebuild time> over >= 12 h after the orion-substrate-runtime rebuild; --check fails below
--min-share (0.9).

    python services/orion-prediction/evals/run_surprise_node_live_parity_eval.py --hours 24
    python services/orion-prediction/evals/run_surprise_node_live_parity_eval.py --since 2026-11-26T06:00Z --check
"""
import argparse, json, subprocess, sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "services" / "orion-prediction"), str(REPO)]
ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=24.0)
ap.add_argument("--since", type=str, default=None, help="ISO time; overrides --hours (e.g. the rebuild time)")
ap.add_argument("--check", action="store_true")
ap.add_argument("--min-share", type=float, default=0.9)
A = ap.parse_args()
from app.surprise import CALM_SURPRISAL, conformal_p_value, event_gate, excess


def _load(name):
    """orion/substrate/<name>.py by path: the orion.substrate package __init__ pulls in the graph store
    (requests, redis), which this image does not ship. The fold modules themselves are stdlib +
    orion.prediction.surprisal (same pattern as run_surprise_node_replay_eval.py)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, REPO / "orion" / "substrate" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


PredictionSurpriseState = _load("prediction_surprise").PredictionSurpriseState

def q(sql):
    out = subprocess.run(["docker","exec","orion-athena-sql-db","psql","-U","postgres","-d","conjourney","-Atc",
        f"BEGIN READ ONLY; SELECT coalesce(json_agg(t),'[]') FROM ({sql}) t; COMMIT;"],capture_output=True,text=True,check=True).stdout
    return json.loads("\n".join(l for l in out.splitlines() if l not in ("BEGIN","COMMIT")))
ts = lambda s: datetime.fromisoformat(s)
SINCE = (f"'{datetime.fromisoformat(A.since.replace('Z', '+00:00')).isoformat()}'::timestamptz" if A.since
         else f"now() - interval '{A.hours} hours'")
rows = q(f"""SELECT f.forecast_id, f.series_id, f.role, f.horizon_min, f.origin_ts, f.target_ts, f.scored_at, f.score_status,
 f.actual, f.quantiles, f.interval_source, r.assignment ->> f.horizon_min::text AS served_model
 FROM prediction_forecast f LEFT JOIN prediction_model_run r ON r.run_id = f.run_id
 WHERE f.series_id IN ('office.temp_c','athena.cabinet_temp_c') AND f.scored_at > {SINCE} - interval '1 hour'
 ORDER BY f.scored_at, f.forecast_id""")
hist = q(f"""SELECT node_id, observed_at, value FROM substrate_node_prediction_error_history
 WHERE node_id IN ('node:prediction.room','node:prediction.body') AND observed_at > {SINCE} ORDER BY observed_at""")
events = []
for r in rows:
    ok, _ = event_gate(r, {"score_status": r["score_status"], "actual": r["actual"]})
    if not ok: continue
    p, _ = conformal_p_value(float(r["actual"]), r["quantiles"])
    events.append({"series_id": r["series_id"], "horizon_min": r["horizon_min"], "origin_ts": ts(r["origin_ts"]),
                   "target_ts": ts(r["target_ts"]), "excess": excess(p), "calm_surprisal": CALM_SURPRISAL,
                   "forecast_id": str(r["forecast_id"]), "scored_at": ts(r["scored_at"])})
st = PredictionSurpriseState(); i = 0; n = match = 0; worst = 0.0
for h in hist:
    t = ts(h["observed_at"])
    while i < len(events) and events[i]["scored_at"] <= t:
        st.add(events[i], now=events[i]["scored_at"]); i += 1
    rd = st.reading(h["node_id"], now=t)
    if rd is None: continue
    n += 1; d = abs(round(rd.value, 6) - h["value"]); worst = max(worst, d); match += d <= 1e-6
res = {"room_body": {"history_rows": len(hist), "compared": n, "match_1e-6": match, "share": match / n if n else None, "max_abs_diff": worst}}


from orion.prediction.tests.surprisal_golden import n_cal_from_calm_point
JointSurpriseState = _load("prediction_joint_surprise").JointSurpriseState
jr = q(f"""SELECT joint_id, family, horizon_min, origin_ts, target_ts, scored_at, excess, calm_point, top_shares
 FROM prediction_joint_surprise WHERE family IN ('family_a_body','family_a_room') AND status='scored'
 AND excess IS NOT NULL AND scored_at > {SINCE} - interval '1 hour' ORDER BY scored_at, joint_id""")
jh = q(f"""SELECT node_id, observed_at, value FROM substrate_node_prediction_error_history
 WHERE node_id IN ('node:prediction.room_joint','node:prediction.body_joint') AND observed_at > {SINCE} ORDER BY observed_at""")
ncal = [n_cal_from_calm_point(r["calm_point"]) for r in jr]
ev = [{**r, "group": r["family"].removeprefix("family_a_"), "origin_ts": ts(r["origin_ts"]), "target_ts": ts(r["target_ts"]),
       "n_cal": n, "scored_at": ts(r["scored_at"])} for r, n in zip(jr, ncal)]
st = JointSurpriseState(); i = n = match = 0; worst = 0.0
for h in jh:
    t = ts(h["observed_at"])
    while i < len(ev) and ev[i]["scored_at"] <= t:
        st.add(ev[i], now=ev[i]["scored_at"]); i += 1
    rd = st.reading(h["node_id"], now=t)
    if rd is None: continue
    n += 1; d = abs(round(rd.value, 6) - h["value"]); worst = max(worst, d); match += d <= 1e-6
res["joint"] = {"history_rows": len(jh), "compared": n, "match_1e-6": match, "share": match / n if n else None,
                "max_abs_diff": worst, "n_cal_range": [min(ncal), max(ncal)] if ncal else None}
print(json.dumps(res, indent=1))
if A.check:
    bad = [k for k, v in res.items() if not v["compared"] or v["share"] < A.min_share]
    print("CHECK FAIL " + ", ".join(bad) if bad else "CHECK PASS")
    sys.exit(1 if bad else 0)
