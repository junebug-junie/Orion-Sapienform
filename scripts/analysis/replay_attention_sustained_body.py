#!/usr/bin/env python3
"""Replay stored broadcast frames through the REAL sustained-body rule. Read-only analysis.

PR report: docs/superpowers/pr-reports/2026-10-16-attention-sustained-continuous-signals-pr.md.
Design: docs/superpowers/specs/2026-10-16-attention-continuous-signal-quota-design.md (option b).

For every stored world-first frame, each continuous body candidate (biometrics, bus_synaptic) is
rebuilt from its own traced reading (value, percentile, band, age) with
``orion.attention.world_first.node_candidate`` and the node's stored readings up to the frame
(``substrate_node_prediction_error_history``), then judged by ``judge_candidate`` -- the same
calls the broadcast makes. Every other candidate keeps its traced verdict. The winner is re-picked
by the traced ranking key. With ``sustain=False`` the same path must reproduce the stored winner.

Inputs (both SELECT-only exports):

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \\
      "copy (select generated_at, projection_json->'frame'->'debug'->'world_first' \\
             from substrate_attention_broadcast_log where generated_at >= now()-interval '7 days' \\
             and projection_json->'frame'->'debug' ? 'world_first' order by generated_at) to stdout" \\
      > frames.tsv
    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \\
      "copy (select node_id, observed_at, value from substrate_node_prediction_error_history \\
             where observed_at >= now()-interval '15 days' and node_id in \\
             ('node:substrate.biometrics','node:substrate.bus_synaptic') order by observed_at) to stdout" \\
      > pe.tsv
    python3 scripts/analysis/replay_attention_sustained_body.py frames.tsv pe.tsv

Sensitivity (``--scale-node``/``--scale-factor``/``--scale-from``): multiply one node's readings
at or after ``--scale-from`` by a factor, keeping the earlier history as recorded, and re-judge
that node's current reading from scratch (``compute_prediction_error_magnitude`` + mid-rank) at
every frame. Models a step change in a producer formula (e.g. dropping a host's frozen keys
from the biometrics mean: x44/36, PR #2714) against a week that still holds the old scale.

LIMIT, stated: the history table keeps 7 days, so a frame from day 1 of the window sees fewer
than 7 days of readings when its past readings are re-judged for the dwell (the runtime had the
full week). The report splits out frames whose history covers >= 3 days.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.attention.world_first import judge_candidate, midrank_percentile, node_candidate  # noqa: E402
from orion.schemas.attention_frame import PredictionErrorMagnitudeV1  # noqa: E402
from orion.substrate.prediction_error_magnitude import (  # noqa: E402
    WINDOW_7D,
    compute_prediction_error_magnitude,
)

CONTINUOUS = ("node:substrate.biometrics", "node:substrate.bus_synaptic")


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace(" ", "T"))


def load_frames(path):
    out = []
    for line in open(path):
        ts, js = line.rstrip("\n").split("\t", 1)
        try:
            wf = json.loads(js.replace("\\\\", "\\"))
        except Exception:
            continue
        if isinstance(wf, dict) and wf.get("candidates") is not None:
            out.append((_ts(ts), wf))
    return out


def load_pe(path):
    series: dict[str, tuple[list[datetime], list[float]]] = {}
    for line in open(path):
        node, ts, v = line.rstrip("\n").split("\t")
        t, vals = series.setdefault(node, ([], []))
        t.append(_ts(ts))
        vals.append(float(v))
    return series


def key(c):
    return c["rank_score"] if c.get("rank_score") is not None else (c.get("score") or 0.0)


def pick(cands):
    if not cands:
        return None
    return max(cands, key=lambda c: (key(c), c["source_id"]))["source_id"]


def rejudge(c, t, series, *, sustain, recompute=False):
    """Real node_candidate + judge_candidate. Returns (eligible, dwell, rank key).

    Default: the traced reading. ``recompute``: the node's newest stored reading at ``t``,
    scored from scratch against the stored (possibly rescaled) week."""
    if c.get("absent") or c.get("percentile_now") is None:
        return bool(c["eligible"]), None, key(c)
    times, vals = series.get(c["source_id"], ([], []))
    hi = bisect.bisect_right(times, t)
    lo = bisect.bisect_left(times, t - WINDOW_7D)
    hist = list(zip(times[lo:hi], vals[lo:hi]))
    rank = key(c)
    if recompute and hist:
        observed_at, value = hist[-1]
        mag = compute_prediction_error_magnitude(value=value, observed_at=observed_at, history=hist, now=t)
        rank = midrank_percentile(value, [v for _, v in hist])
    else:
        observed_at = None
        mag = PredictionErrorMagnitudeV1(
            value=float(c["value"]), age_sec=float(c.get("age_sec") or 0.0),
            percentile_now=float(c["percentile_now"]), n_readings_7d=int(c.get("n_readings_7d") or 0),
            band=c.get("band") if c.get("band") not in (None, "absent") else "insufficient_history",
        )
    cand = node_candidate(
        node_id=c["source_id"], label=c["source_id"], magnitude=mag, observed_at=observed_at, now=t,
        pe_history=hist, sustain=sustain,
    )
    v = judge_candidate(cand)
    return v.eligible, v.hot_dwell_sec, (v.rank_score if v.rank_score is not None else rank)


def run(frames, series, *, recompute_nodes=frozenset(), only_from=None):
    """Winner shares: stored, rule off (re-picked), rule on. ``only_from`` restricts to frames at
    or after that time."""
    sel = [(t, wf) for t, wf in frames if only_from is None or t >= only_from]
    n = len(sel)
    sq, off, on, moved = Counter(), Counter(), Counter(), Counter()
    agree_off = 0
    dwell_at_win: list[float] = []
    for t, wf in sel:
        el_off, el_on = [], []
        for c in wf["candidates"]:
            if c["source_id"] in CONTINUOUS:
                rc = c["source_id"] in recompute_nodes
                e_off, _, k_off = rejudge(c, t, series, sustain=False, recompute=rc)
                e_on, dwell, k_on = rejudge(c, t, series, sustain=True, recompute=rc)
                if e_off:
                    el_off.append({**c, "rank_score": k_off})
                if e_on:
                    el_on.append({**c, "rank_score": k_on})
                    if dwell is not None:
                        dwell_at_win.append(dwell)
            elif c["eligible"]:
                el_off.append(c)
                el_on.append(c)
        recorded = wf.get("winner")
        w_off, w_on = pick(el_off), pick(el_on)
        sq[recorded] += 1
        off[w_off] += 1
        on[w_on] += 1
        agree_off += int(w_off == recorded)
        if recorded in CONTINUOUS and w_on != recorded:
            moved[w_on] += 1
    return n, sq, off, on, moved, agree_off, dwell_at_win


def share(c, total):
    return {str(k or "no winner"): round(v / total, 4) for k, v in c.most_common()} if total else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("pe")
    ap.add_argument("--scale-node", default="")
    ap.add_argument("--scale-factor", type=float, default=1.0)
    ap.add_argument("--scale-from", default="", help="ISO UTC; readings at/after are scaled")
    a = ap.parse_args()
    frames = load_frames(a.frames)
    series = load_pe(a.pe)
    first_pe = min(ts[0] for ts, _ in series.values())
    if a.scale_node:
        start = datetime.fromisoformat(a.scale_from)
        times, vals = series[a.scale_node]
        series[a.scale_node] = (times, [v * a.scale_factor if ts >= start else v for ts, v in zip(times, vals)])
        report = {"scale": {"node": a.scale_node, "factor": a.scale_factor, "from": start.isoformat()}}
        for label, only in (("after_change", start), ("first_24h", None), ("first_72h", None)):
            if label == "after_change":
                n, sq, off, on, _, _, dw = run(frames, series, recompute_nodes={a.scale_node}, only_from=start)
            else:
                hours = 24 if label == "first_24h" else 72
                sub = [(t, w) for t, w in frames if start <= t < start + timedelta(hours=hours)]
                n, sq, off, on, _, _, dw = run(sub, series, recompute_nodes={a.scale_node})
            report[label] = {
                "frames": n,
                "rule_off": share(off, n),
                "rule_on": share(on, n),
                "continuous_wins_with_dwell_le_300s": sum(1 for d in dw if d <= 300.0),
            }
        print(json.dumps(report, indent=1))
        return
    n, sq, off, on, moved, agree_off, dw = run(frames, series)
    mature = [(t, w) for t, w in frames if t - first_pe >= timedelta(days=3)]
    nm, sqm, _, onm, _, _, _ = run(mature, series)
    print(json.dumps({
        "frames": n,
        "span": [frames[0][0].isoformat(), frames[-1][0].isoformat()],
        "pe_history_from": first_pe.isoformat(),
        "sustain_off_reproduces_recorded_winner": round(agree_off / n, 4),
        "status_quo_recorded": share(sq, n),
        "sustained_rule": share(on, n),
        "frames_with_ge_3d_history": nm,
        "status_quo_recorded_ge_3d": share(sqm, nm),
        "sustained_rule_ge_3d": share(onm, nm),
        "frames_lost_by_biometrics_or_bus_now_go_to": share(moved, sum(moved.values()) or 1),
        "continuous_wins_with_dwell_le_300s": sum(1 for d in dw if d <= 300.0),
    }, indent=1))


if __name__ == "__main__":
    main()
