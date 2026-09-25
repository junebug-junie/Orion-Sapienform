#!/usr/bin/env python3
"""Replay route/chat prediction error, old definition vs new, over the live projections.

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \\
        "select projection_json from substrate_route_arbitration_projection" > route.json
    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \\
        "select projection_json from substrate_chat_session_projection" > chat.json
    python scripts/analysis/replay_route_chat_prediction_error_definitions.py \\
        --route-json route.json --chat-json chat.json

Why the projections are enough: both reducers stamp the tick's clock as
``last_updated_at`` on every run/turn they write, and live runs/turns are
single-shot creates. Sorting by ``last_updated_at`` and grouping equal stamps
therefore recovers the tick sequence, and each tick's ``prev``/``curr`` pair is
"everything before this stamp" / "that plus this stamp's group". Route evicts
runs older than 24h (``ROUTE_ARBITRATION_MAX_AGE_SEC``); the replay applies the
same window to ``prev``, but runs evicted before the dump are gone, so the
first 24h of the route history has a smaller denominator than live did. That
only flatters v1 (smaller denominator = larger number), never v2.

v1 is a frozen copy of the pre-2026-09-25 functions, kept here so the
comparison survives the live code moving on. v2 is imported from the live
module. ``chat v2+touched`` is an analysis-only variant (not shipped) showing
what the same dilution fix route received would do to chat.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from orion.bus.ewma import compute_ewma_update  # noqa: E402
from orion.schemas.chat_projection import ChatSessionProjectionV1, ChatTurnStateV1  # noqa: E402
from orion.schemas.route_projection import (  # noqa: E402
    RouteArbitrationProjectionV1,
    RouteArbitrationRunStateV1,
)
from orion.substrate import prediction_error as pe  # noqa: E402

ROUTE_FIELDS = ("lane", "lane_reason", "output_mode", "mind_requested")
ROUTE_MAX_AGE = timedelta(seconds=86400)
# Candidate A's node-target EWMA (orion/attention/field_attention/candidate_precision_weighted.py).
try:
    from orion.attention.field_attention.candidate_precision_weighted import (  # noqa: E402
        NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA as NODE_ALPHA,
        NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE as NODE_MIN_VAR,
    )
except Exception:  # pragma: no cover - analysis fallback
    NODE_ALPHA, NODE_MIN_VAR = 0.2, 1e-5


# --- frozen v1 -----------------------------------------------------------------


def route_v1(prev: RouteArbitrationProjectionV1, curr: RouteArbitrationProjectionV1) -> float:
    scores: list[float] = []
    fallback = pe._latest_run(prev.runs)
    for trace_id, curr_run in curr.runs.items():
        prev_run = prev.runs.get(trace_id) or fallback
        if prev_run is None:
            continue
        scores.append(
            sum(1.0 if getattr(prev_run, f) != getattr(curr_run, f) else 0.0 for f in ROUTE_FIELDS)
            / len(ROUTE_FIELDS)
        )
    return sum(scores) / len(scores) if scores else 0.0


def _chat_hints_v1(turn: ChatTurnStateV1) -> dict[str, float]:
    return {
        "conversation_load": min(1.0, turn.word_count / 150.0),
        "repair_pressure": turn.repair_pressure_level,
        "topic_coherence": max(0.0, 1.0 - turn.repair_pressure_level),
    }


def _chat_raw(prev, curr, *, keys, hints, touched_only: bool) -> float | None:
    deltas: list[float] = []
    fallback = pe._latest_run(prev.turns)
    items = pe._touched_runs(prev.turns, curr.turns) if touched_only else curr.turns.items()
    for turn_id, curr_turn in items:
        prev_turn = prev.turns.get(turn_id) or fallback
        if prev_turn is None:
            continue
        ph, ch = hints(prev_turn), hints(curr_turn)
        deltas.extend(abs(ch.get(k, 0.0) - ph.get(k, 0.0)) for k in keys)
    return sum(deltas) / len(deltas) if deltas else None


class _Ewma:
    def __init__(self, alpha: float, min_var: float, sat: float = 3.0) -> None:
        self.alpha, self.min_var, self.sat = alpha, min_var, sat
        self.ewma = self.var = 0.0
        self.n = 0

    def score(self, value: float) -> float:
        u = compute_ewma_update(
            prev_ewma=self.ewma, prev_variance=self.var, prev_count=self.n,
            value=value, alpha=self.alpha, min_variance=self.min_var,
        )
        self.ewma, self.var, self.n = u.ewma, u.variance, self.n + 1
        return 0.0 if u.zscore is None else min(1.0, max(0.0, u.zscore) / self.sat)

    def fold(self, value: float) -> None:
        self.score(value)


# --- replay --------------------------------------------------------------------


def _ticks(items: dict, stamp) -> list[tuple[datetime, list[str]]]:
    groups: dict[datetime, list[str]] = {}
    for key, obj in items.items():
        groups.setdefault(stamp(obj), []).append(key)
    return sorted(groups.items())


def replay_route(raw: dict) -> dict[str, list[float]]:
    runs = {k: RouteArbitrationRunStateV1.model_validate(v) for k, v in raw["runs"].items()}
    ordered = sorted(runs.items(), key=lambda kv: kv[1].last_updated_at)
    out = {"v1": [], "v2": []}
    for stamp, keys in _ticks(runs, lambda r: r.last_updated_at):
        prev_runs = {
            k: r for k, r in ordered
            if r.last_updated_at < stamp and r.last_updated_at >= stamp - ROUTE_MAX_AGE
        }
        curr_runs = dict(prev_runs)
        curr_runs.update({k: runs[k] for k in keys})
        mk = lambda rs: RouteArbitrationProjectionV1(  # noqa: E731
            projection_id="replay", generated_at=stamp, runs=rs
        )
        prev, curr = mk(prev_runs), mk(curr_runs)
        out["v1"].append(route_v1(prev, curr))
        out["v2"].append(pe.route_prediction_error(prev, curr))
    return out


def replay_chat(raw: dict) -> dict[str, list[float]]:
    turns = {k: ChatTurnStateV1.model_validate(v) for k, v in raw["turns"].items()}
    variants = {
        "v1": (("conversation_load", "repair_pressure", "topic_coherence"), _chat_hints_v1, False),
        "v2": (("conversation_load", "repair_pressure"), pe.compute_chat_pressure_hints, False),
        "v2+touched": (("conversation_load", "repair_pressure"), pe.compute_chat_pressure_hints, True),
    }
    ewmas = {
        name: _Ewma(pe._CHAT_PREDICTION_ERROR_EWMA_ALPHA, pe._CHAT_PREDICTION_ERROR_MIN_VARIANCE)
        for name in variants
    }
    out: dict[str, list[float]] = {name: [] for name in variants}
    out.update({f"{name}:raw": [] for name in variants})
    floor_bound = {name: 0 for name in variants}
    # Switchover probe for the projection's OWN EWMA baseline (ChatSessionProjectionV1
    # .prediction_error_baseline_*): at the midpoint, v2 continues from v1's baseline
    # ("carried") -- what deploying without a reset does -- vs v2's own history.
    carried: _Ewma | None = None
    switch_at = len({t.last_updated_at for t in turns.values()}) // 2
    tick_index = 0
    prev_turns: dict[str, ChatTurnStateV1] = {}
    for stamp, keys in _ticks(turns, lambda t: t.last_updated_at):
        curr_turns = dict(prev_turns)
        curr_turns.update({k: turns[k] for k in keys})
        mk = lambda ts: ChatSessionProjectionV1(  # noqa: E731
            projection_id="replay", generated_at=stamp, turns=ts
        )
        prev, curr = mk(prev_turns), mk(curr_turns)
        for name, (keys_, hints, touched) in variants.items():
            rawv = _chat_raw(prev, curr, keys=keys_, hints=hints, touched_only=touched)
            if rawv is None:
                continue
            e = ewmas[name]
            if e.n > 0 and e.var < e.min_var:
                floor_bound[name] += 1
            if name == "v2" and carried is not None:
                out["carried"].append(carried.score(rawv))
                out["carried_ref"].append(e.score(rawv))
                out[f"{name}:raw"].append(rawv)
                out[name].append(out["carried_ref"][-1])
                continue
            out[name].append(e.score(rawv))
            out[f"{name}:raw"].append(rawv)
        tick_index += 1
        if tick_index == switch_at:
            carried = _Ewma(ewmas["v1"].alpha, ewmas["v1"].min_var)
            carried.ewma, carried.var, carried.n = ewmas["v1"].ewma, ewmas["v1"].var, ewmas["v1"].n
            out["carried"], out["carried_ref"] = [], []
        prev_turns = curr_turns
    out["_floor_bound"] = floor_bound  # type: ignore[assignment]
    return out


def summarize(values: list[float], attn: float) -> str:
    if not values:
        return "n=0"
    zero = sum(1 for v in values if v == 0.0) / len(values)
    hi = sum(1 for v in values if v >= attn) / len(values)
    distinct = len({round(v, 6) for v in values})
    return (
        f"n={len(values)} min={min(values):.6g} mean={statistics.fmean(values):.6g} "
        f"p50={statistics.median(values):.6g} max={max(values):.6g} "
        f"frac_zero={zero:.3f} frac>={attn}={hi:.3f} stdev={statistics.pstdev(values):.6g} "
        f"distinct={distinct}"
    )


def node_baseline(values: list[float]) -> str:
    e = _Ewma(NODE_ALPHA, NODE_MIN_VAR)
    for v in values:
        e.fold(v)
    return f"candidateA_ewma={e.ewma:.6g} variance={e.var:.6g} n={e.n}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--route-json", type=Path)
    ap.add_argument("--chat-json", type=Path)
    ap.add_argument("--attention-level", type=float, default=0.1,
                    help="report the fraction of ticks at or above this value")
    args = ap.parse_args()
    if args.route_json:
        r = replay_route(json.loads(args.route_json.read_text()))
        print("## route_prediction_error")
        for name in ("v1", "v2"):
            print(f"{name}: {summarize(r[name], args.attention_level)}")
            print(f"{name}: {node_baseline(r[name])}")
        nz = [(i, v) for i, v in enumerate(r["v2"]) if v > 0]
        print(f"v2 nonzero ticks: {len(nz)} values={sorted({v for _, v in nz})}")
    if args.chat_json:
        c = replay_chat(json.loads(args.chat_json.read_text()))
        print("## chat_prediction_error")
        for name in ("v1", "v2", "v2+touched"):
            print(f"{name} score: {summarize(c[name], args.attention_level)}")
            print(f"{name} raw:   {summarize(c[name + ':raw'], 0.1)}")
            print(f"{name}: variance-floor bound on {c['_floor_bound'][name]} ticks; {node_baseline(c[name])}")
        pairs = list(zip(c["v1"], c["v2"]))
        diffs = [abs(a - b) for a, b in pairs]
        print(f"|v1 - v2| score per tick: mean={statistics.fmean(diffs):.4g} max={max(diffs):.4g}")
        cd = [abs(a - b) for a, b in zip(c.get("carried", []), c.get("carried_ref", []))]
        if cd:
            settle = next((i for i in range(len(cd)) if all(d < 0.01 for d in cd[i:])), len(cd))
            print(
                f"switchover (v2 carrying v1's projection baseline vs v2's own): "
                f"first-tick diff={cd[0]:.4g} max diff={max(cd):.4g} "
                f"ticks until every later diff < 0.01: {settle} of {len(cd)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
