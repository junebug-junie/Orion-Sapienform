"""Golden fixture for the shared surprise library (Patch A of
docs/superpowers/specs/2026-11-25-shared-bet-primitives-design.md).

Patch A moves surprise math into ``orion/prediction/surprisal.py`` and ``orion/prediction/grading.py``. It
must change no live number. This module holds one ``compute_*`` function per side. Each runs the
code that is first on ``sys.path`` over the same inputs, so:

- ``build`` ran ONCE against the pre-refactor tree (main at cf5889fdd, ``git archive``) and wrote the
  outputs as ``expected`` into ``fixtures/surprisal_golden_2026-11-26.json.gz``;
- the golden tests run the same ``compute_*`` against the current tree and compare to 1e-9.

Inputs are live stored rows (read-only ``psql``, 2026-11-26): room/body forecast scores
(``prediction_forecast``), joint surprise rows (``prediction_joint_surprise``), chat turn self-bets
(``prediction_turn_self_bet``), walkway rhythm expectations (``vision_percept_expectation``) and the
reachable forecasts (``prediction_reachable_forecast``). Ids are hashed; no text is kept.

Re-building is only for a new patch that is ALLOWED to move a number. Usage (repo root):

    python orion/prediction/tests/surprisal_golden.py dump  > /tmp/x/inputs.json        # live, read-only
    python orion/prediction/tests/surprisal_golden.py build --code-root <pre-change tree> \\
        --inputs /tmp/x/inputs.json --out orion/prediction/tests/fixtures/surprisal_golden_2026-11-26.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "surprisal_golden_2026-11-26.json.gz"
TOL = 1e-9

# --- live dump (read-only) ---------------------------------------------------------------------------

_SQL = {
    "room_body": """
        SELECT * FROM (
          SELECT series_id, horizon_min, origin_ts, target_ts, scored_at, actual, quantiles,
                 row_number() OVER (PARTITION BY series_id ORDER BY scored_at DESC, forecast_id DESC) AS rn
          FROM prediction_forecast
          WHERE series_id IN ('office.temp_c', 'athena.cabinet_temp_c') AND role = 'champion'
            AND interval_source = 'adaptive' AND score_status = 'scored' AND actual IS NOT NULL
            AND scored_at > now() - interval '3 days') t
        WHERE rn <= 500 ORDER BY scored_at, series_id, horizon_min, origin_ts""",
    "joint": """
        SELECT * FROM (
          SELECT joint_id, family, horizon_min, origin_ts, target_ts, scored_at, excess, calm_point, top_shares,
                 row_number() OVER (PARTITION BY family ORDER BY scored_at DESC, joint_id DESC) AS rn
          FROM prediction_joint_surprise
          WHERE family IN ('family_a_body', 'family_a_room') AND status = 'scored' AND excess IS NOT NULL
            AND scored_at > now() - interval '3 days') t
        WHERE rn <= 500 ORDER BY scored_at, joint_id""",
    "turn": """
        SELECT correlation_id, faculty, caller, placed_at, settled_at, status, stated_p, hit
        FROM prediction_turn_self_bet
        WHERE caller = 'chat' AND faculty IN ('turn.self.asks_question', 'turn.self.more_steps_than_usual')
          AND placed_at > now() - interval '21 days'
        ORDER BY placed_at, bet_id""",
    "street": """
        SELECT expectation_id, confidence, status, scored_at
        FROM vision_percept_expectation
        WHERE stream_id = 'walkway' AND status IN ('met', 'missed') AND confidence IS NOT NULL
          AND scored_at > now() - interval '10 days'
        ORDER BY scored_at, expectation_id""",
    "reachable": """
        SELECT p, p_state, p_baseline, happened FROM prediction_reachable_forecast
        WHERE happened IS NOT NULL ORDER BY origin""",
}


def _h(x: Any) -> str:
    return hashlib.sha256(str(x).encode()).hexdigest()[:12]


def _psql_json(sql: str) -> list[dict]:
    out = subprocess.run(
        ["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney", "-Atc",
         f"BEGIN READ ONLY; SELECT coalesce(json_agg(t), '[]') FROM ({sql}) t; COMMIT;"],
        check=True, capture_output=True, text=True).stdout
    lines = out.splitlines()   # BEGIN / the json_agg (several lines) / COMMIT
    return json.loads("\n".join(ln for ln in lines if ln not in ("BEGIN", "COMMIT")))


def dump() -> dict:
    rb = [{k: r[k] for k in ("series_id", "horizon_min", "origin_ts", "target_ts", "scored_at", "actual",
                              "quantiles")} for r in _psql_json(_SQL["room_body"])]
    joint = [{"joint_id": _h(r["joint_id"]), "group": r["family"].removeprefix("family_a_"),
              **{k: r[k] for k in ("horizon_min", "origin_ts", "target_ts", "scored_at", "excess", "calm_point",
                                   "top_shares")}} for r in _psql_json(_SQL["joint"])]
    turn = [{**{k: r[k] for k in ("faculty", "caller", "placed_at", "settled_at", "status", "stated_p", "hit")},
             "correlation_id": _h(r["correlation_id"])} for r in _psql_json(_SQL["turn"])]
    street = [{**{k: r[k] for k in ("confidence", "status", "scored_at")}, "expectation_id": _h(r["expectation_id"])}
              for r in _psql_json(_SQL["street"])]
    return {"dumped_at": datetime.now(timezone.utc).isoformat(), "room_body": rb, "joint": joint, "turn": turn,
            "street": street, "reachable": _psql_json(_SQL["reachable"])}


# --- shared helpers ----------------------------------------------------------------------------------


def ts(s: str) -> datetime:
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def n_cal_from_calm_point(calm_point: float) -> int:
    """The calibration-pool size whose calm point (family_a.calm_point: ln m - lgamma(m + 1) / m, m = n + 1)
    is the stored one. The joint event carries n_cal; the table stores only its calm point."""
    for n in range(1, 50000):
        m = n + 1
        if abs(math.log(m) - math.lgamma(m + 1) / m - calm_point) < 1e-13:
            return n
    raise ValueError(f"no n_cal for calm_point {calm_point!r}")


def _reading(r: Any) -> list[float] | None:
    return None if r is None else [r.mean_excess, r.z, r.value, r.n]


# --- substrate side (stdlib + orion.substrate) ----------------------------------------------------------


def compute_substrate(inp: dict, *, pre_change: bool = False) -> dict:
    from orion.substrate import calm_band as cb
    from orion.substrate import curiosity_seed_bar as csb
    from orion.substrate import prediction_joint_surprise as pj
    from orion.substrate import prediction_street_rhythm as psr
    from orion.substrate import prediction_surprise as ps
    from orion.substrate import prediction_turn_surprise as pt

    out: dict[str, Any] = {}

    st = ps.PredictionSurpriseState()
    rows = []
    for e in inp["room_body_events"]:
        ev = {**e, "origin_ts": ts(e["origin_ts"]), "target_ts": ts(e["target_ts"])}
        now = ts(e["scored_at"])
        node, verdict = st.add(ev, now=now)
        rows.append([node, verdict, _reading(st.reading(node, now=now)) if verdict == "accepted" else None])
    out["room_body"] = rows

    st = pj.JointSurpriseState()
    rows = []
    for e in inp["joint"]:
        ev = {**e, "origin_ts": ts(e["origin_ts"]), "target_ts": ts(e["target_ts"]),
              "n_cal": n_cal_from_calm_point(e["calm_point"])}
        now = ts(e["scored_at"])
        node, verdict = st.add(ev, now=now)
        r = st.reading(node, now=now) if verdict == "accepted" else None
        rows.append([node, verdict, _reading(r), None if r is None else r.top_reading])
    out["joint"] = rows

    st = pt.TurnSurpriseState()
    rows = []
    for e in inp["turn_events"]:
        ev = {**e, "settled_at": ts(e["settled_at"])}
        now = ts(e["settled_at"])
        node, verdict = st.add(ev, now=now)
        rows.append([node, verdict, _reading(st.reading(node, now=now)) if verdict == "accepted" else None])
    out["chat_turn"] = rows

    st = psr.StreetRhythmState()
    rows = []
    samples = []
    for e in inp["street_events"]:
        ev = {**e, "scored_at": ts(e["scored_at"])}
        now = ts(e["scored_at"])
        node, verdict = st.add(ev, now=now)
        rows.append([node, verdict, _reading(st.reading(now=now)) if verdict == "accepted" else None])
        samples.append(psr.RhythmSample(expectation_id=e["expectation_id"], scored_at=ev["scored_at"],
                                        excess=e["excess"], calm_var=e["calm_var"], happened=e["happened"],
                                        p=e["p"]))
    out["street_rhythm"] = rows
    out["street_rhythm_replay"] = [[t.isoformat(), v] for t, v in psr.replay_node_values(samples)]

    # Constants and the two calm variances.
    out["z_full_scale"] = [ps.Z_FULL_SCALE, pj.Z_FULL_SCALE, pt.Z_FULL_SCALE, psr.Z_FULL_SCALE, cb.Z_FULL_SCALE,
                           csb.Z_FULL_SCALE]
    out["calm_band_p_floor"] = cb.P_FLOOR
    out["calm_band_calm_excess_variance"] = [cb.calm_excess_variance(p) for p in (0.01, 0.05, 0.1, 0.3)]
    out["calm_band_z_tail"] = [[rule, z, cb.z_tail(rule, z), cb.z_tail(rule, z, strict=True)]
                               for rule in (cb.RULE_GAMMA, cb.RULE_FLOORED, cb.RULE_CHAT_BETS, cb.RULE_CANTELLI)
                               for z in (-0.5, 0.0, 0.7, 1.65, 2.45, 3.0)]
    out["calm_band_bet_window_atoms"] = [cb.bet_window_atoms(qs) for qs in ((0.1,), (0.1, 0.25), (0.02, 0.7, 0.5))]
    out["seed_bars"] = dict(sorted(csb.CURIOSITY_SEED_BAR_BY_NODE_ID.items()))
    origins = [datetime(2026, 11, 11, 8, m, s, tzinfo=timezone.utc) for m in range(0, 60, 5) for s in (0, 3)]
    out["independent_origin"] = [[o.isoformat(), h, ps.independent_origin(o, h), pj.independent_origin(o, h)]
                                 for o in origins for h in (5, 15)]
    # The moved calm variances, read from their home on each side of the change (``pre_change``: build only).
    if pre_change:
        joint_var = pj.calm_excess_variance
        out["room_body_calm_var"] = ps.calm_excess_variance(0.95)   # took the calm surprisal
    else:
        from orion.prediction.surprisal import calm_excess_variance, calm_excess_variance_discrete as joint_var
        out["room_body_calm_var"] = calm_excess_variance(1.0 - 0.95)  # takes the floor
    out["joint_calm_excess_variance"] = [[n, joint_var(n)] for n in (1, 2, 10, 101, 1009, 5000)]
    return out


# --- service side (orion-prediction app.*, numpy/scipy) -------------------------------------------------


def _log_loss(mod: Any, p: float, y: bool, pre_change: bool) -> float:
    """Before the change ``mod.log_loss`` took (p, y) and clipped with the module's own P_CLIP; after it,
    the shared ``log_loss(p, y, clip)`` takes the clip. Chosen explicitly by ``pre_change`` (set only by
    ``build``), never by inspecting the signature, so a later signature change fails the test loudly."""
    return mod.log_loss(p, y) if pre_change else mod.log_loss(p, y, mod.P_CLIP)


def compute_service(inp: dict, *, pre_change: bool = False, separate: bool = True) -> dict:
    import app.action_bets as ab
    import app.reachable as rc
    import app.street_rhythm as sr
    import app.surprise as su
    import app.turn_surprise as tsu
    from orion.curiosity.value import entropy_nats
    from orion.prediction.grading import grade
    from orion.self_monitor.record import wilson as record_wilson

    out: dict[str, Any] = {}
    # Room / body: conformal p -> surprisal / excess per stored score (the event's excess).
    out["room_body_excess"] = []
    for r in inp["room_body"]:
        p, cens = su.conformal_p_value(float(r["actual"]), r["quantiles"])
        out["room_body_excess"].append([p, cens, su.surprisal(p), su.excess(p)])
    out["surprise_grid"] = [[p, su.surprisal(p), su.excess(p)] for p in (0.0, 0.01, 0.05, 0.051, 0.2, 0.5, 1.0, 1.5)]
    out["surprise_constants"] = [su.P_FLOOR, su.CALM_SURPRISAL]

    # Turn bets: the live scoring path (held-out recalibration -> bet terms -> turn z).
    rows = [{**r, "placed_at": ts(r["placed_at"]), "settled_at": ts(r["settled_at"]) if r["settled_at"] else None}
            for r in inp["turn"]]
    hist = tsu.CalibrationHistory(rows)
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["correlation_id"]].append(r)
    turns = []
    for cid, trs in by.items():
        res, reason = tsu.score_turn(trs, hist)
        if res is None:
            turns.append([cid, reason, None])
            continue
        turns.append([cid, reason, [res.excess, res.calm_var, res.z, tsu.node_value(res.z),
                                    [[b.faculty, b.stated_p, b.calibrated_p, b.happened, b.surprisal,
                                      b.expected_surprisal, b.excess, b.calm_var] for b in res.bets]]])
    out["turns"] = turns
    grid = [0.0, 0.005, 0.01, 0.02, 0.1, 0.25, 0.5, 0.7, 0.98, 0.99, 0.995, 1.0]
    out["turn_bet_terms_grid"] = [[q, y, *(lambda b: [b.calibrated_p, b.surprisal, b.expected_surprisal, b.excess,
                                                     b.calm_var])(tsu.bet_terms("f", q, q, y))]
                                  for q in grid for y in (True, False)]
    out["turn_log_loss_grid"] = [[q, y, _log_loss(tsu, q, y, pre_change)] for q in grid for y in (True, False)]
    out["turn_node_value"] = [[z, tsu.node_value(z)] for z in (-1.0, 0.0, 0.5, 1.5, 2.999, 3.0, 7.0)]

    # Street: raw stated p terms per live row.
    out["street_terms"] = [[r["expectation_id"], *(lambda t: [t.p, t.happened, t.surprisal, t.calm_surprisal,
                                                             t.excess, t.calm_var])(
        sr.outcome_terms(float(r["confidence"]), r["status"] == "met"))] for r in inp["street"]]
    out["street_terms_grid"] = [[q, y, *(lambda t: [t.p, t.surprisal, t.calm_surprisal, t.excess, t.calm_var])(
        sr.outcome_terms(q, y))] for q in grid for y in (True, False)]

    # Reachable: log loss per row and model, the calibration bins.
    rr = [dict(r) for r in inp["reachable"]]
    out["reachable_log_loss"] = [[_log_loss(rc, r[k], r["happened"], pre_change) for k in ("p", "p_state", "p_baseline")]
                                 for r in rr]
    out["reachable_log_loss_grid"] = [[q, y, _log_loss(rc, q, y, pre_change)] for q in grid + [1e-5, 1 - 1e-5]
                                      for y in (True, False)]
    out["reachable_calibration"] = {k: rc.calibration(rr, k) for k in ("p", "p_state", "p_baseline")}
    edge = [{"p": v, "happened": h} for v, h in ((1.0, True), (0.0, False), (0.02, True), (0.4, False))]
    out["reachable_calibration_edges"] = rc.calibration(edge)
    out["grade_reachable"] = grade([r["p"] for r in rr], [r["happened"] for r in rr])
    out["grade_edges"] = grade([0.0, 0.05, 0.999, 1.0, 1.0], [False, True, True, True, False])

    # Binary entropy (action_bets, curiosity) and Wilson (self-monitor, promise eval).
    out["bernoulli_entropy"] = [[p, ab.bernoulli_entropy(p)] for p in [-0.1] + grid + [1.1]]
    out["beta_eig"] = [[a, b, ab.beta_eig(a, b)] for a, b in ((1, 1), (2, 8), (30, 3), (0.5, 0.5))]
    out["entropy_nats"] = [[p, entropy_nats(p)] for p in [None, -0.1, 1.7] + grid]
    # n_eff is a fractional Kish count live (record.py reduce_faculty); <= 0 reads (0, 1).
    out["record_wilson"] = [[s, n, *record_wilson(s, n)] for s, n in (
        (0.0, 0), (0.5, -1.0), (0.5, 0.37), (0.5, 4.2), (0.9, 30.0), (1.0, 7.5), (0.0, 2.718),
        (0.3712, 12.8317), (0.8125, 163.402), (0.07, 1.0001))]
    if separate:
        out["promise_wilson"] = compute_promise_wilson()
    return out


# Kept apart: the promise eval needs jinja2. The service golden skips this key; the promise eval's own test
# pins these values as literals (test_promise_ledger_replay_eval.py), so that lane imports no golden code.
SEPARATE_SERVICE_KEYS = ("promise_wilson",)


def compute_promise_wilson() -> list:
    from orion.harness.evals.promise_ledger_replay_eval import wilson as promise_wilson

    return [[k, n, promise_wilson(k, n)] for k, n in ((0, 0), (21, 29), (5, 5), (0, 9))]


# --- build ---------------------------------------------------------------------------------------------


def _events(raw: dict, code_root: Path) -> dict:
    """The bus events the substrate folds see, made from the raw rows by the PRE-change producer code."""
    sys.path[:0] = [str(code_root / "services" / "orion-prediction"), str(code_root)]
    import app.street_rhythm as sr
    import app.surprise as su
    import app.turn_surprise as tsu

    rb = []
    for r in raw["room_body"]:
        p, _ = su.conformal_p_value(float(r["actual"]), r["quantiles"])
        rb.append({"series_id": r["series_id"], "horizon_min": r["horizon_min"], "origin_ts": r["origin_ts"],
                   "target_ts": r["target_ts"], "scored_at": r["scored_at"], "excess": su.excess(p),
                   "calm_surprisal": su.CALM_SURPRISAL, "forecast_id": _h((r["series_id"], r["origin_ts"],
                                                                         r["horizon_min"]))})
    rows = [{**r, "placed_at": ts(r["placed_at"]), "settled_at": ts(r["settled_at"]) if r["settled_at"] else None}
            for r in raw["turn"]]
    hist = tsu.CalibrationHistory(rows)
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["correlation_id"]].append(r)
    te = []
    for cid, trs in by.items():
        res, _ = tsu.score_turn(trs, hist)
        settled = [r["settled_at"] for r in trs if r["settled_at"] is not None]
        if res is None or not settled:
            continue
        te.append({"caller": "chat", "correlation_id": cid, "settled_at": max(settled).isoformat(),
                   "excess": res.excess, "calm_var": res.calm_var, "bets": [b.faculty for b in res.bets]})
    te.sort(key=lambda e: e["settled_at"])
    se = []
    for r in raw["street"]:
        t = sr.outcome_terms(float(r["confidence"]), r["status"] == "met")
        se.append({"stream_id": "walkway", "expectation_id": r["expectation_id"], "scored_at": r["scored_at"],
                   "excess": t.excess, "calm_var": t.calm_var, "happened": t.happened, "p": t.p})
    return {"room_body_events": rb, "turn_events": te, "street_events": se}


def build(code_root: Path, inputs: Path, out: Path) -> None:
    raw = json.loads(inputs.read_text())
    inp = {**raw, **_events(raw, code_root)}
    expected = {"substrate": compute_substrate(inp, pre_change=True), "service": compute_service(inp, pre_change=True)}
    for mod in list(sys.modules):  # nothing imported from the pre-change tree may leak into a later run
        if mod == "app" or mod.startswith(("app.", "orion")):
            del sys.modules[mod]
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps({"inputs": inp, "expected": expected, "code_root_commit": "cf5889fdd"},
                      separators=(",", ":"), default=str) + "\n"
    out.write_bytes(gzip.compress(blob.encode(), mtime=0))


def load(path: Path = FIXTURE) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


# --- comparison ----------------------------------------------------------------------------------------


def diffs(got: Any, want: Any, path: str = "", tol: float = TOL) -> list[str]:
    """Every place ``got`` differs from ``want`` (floats to ``tol``, absolute; everything else exact)."""
    got = json.loads(json.dumps(got, default=str))
    return _diffs(got, want, path, tol)


def _diffs(got: Any, want: Any, path: str, tol: float) -> list[str]:
    if isinstance(want, bool) or want is None or isinstance(want, str):
        return [] if got == want else [f"{path}: {got!r} != {want!r}"]
    if isinstance(want, (int, float)):
        if isinstance(got, bool) or not isinstance(got, (int, float)):
            return [f"{path}: {got!r} != {want!r}"]
        if math.isnan(want) or math.isinf(want):
            return [] if str(got) == str(want) else [f"{path}: {got!r} != {want!r}"]
        return [] if abs(got - want) <= tol else [f"{path}: {got!r} != {want!r} (|d|={abs(got - want):.3g})"]
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            return [f"{path}: length/type {type(got).__name__} {len(got) if isinstance(got, list) else ''} "
                    f"!= {len(want)}"]
        return [d for i, (g, w) in enumerate(zip(got, want)) for d in _diffs(g, w, f"{path}[{i}]", tol)]
    if isinstance(want, dict):
        if not isinstance(got, dict) or set(got) != set(want):
            return [f"{path}: keys {sorted(got) if isinstance(got, dict) else got!r} != {sorted(want)}"]
        return [d for k in want for d in _diffs(got[k], want[k], f"{path}.{k}", tol)]
    raise TypeError(f"{path}: unexpected {type(want).__name__}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dump")
    b = sub.add_parser("build")
    b.add_argument("--code-root", type=Path, required=True)
    b.add_argument("--inputs", type=Path, required=True)
    b.add_argument("--out", type=Path, default=FIXTURE)
    args = ap.parse_args()
    if args.cmd == "dump":
        json.dump(dump(), sys.stdout, default=str)
        return 0
    build(args.code_root.resolve(), args.inputs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
