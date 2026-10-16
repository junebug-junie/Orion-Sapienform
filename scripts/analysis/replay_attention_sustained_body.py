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

LIMIT, stated: the history table keeps 7 days, so a frame from day 1 of the window sees fewer
than 7 days of readings when its past readings are re-judged for the dwell (the runtime had the
full week). The report splits out frames whose history covers >= 3 days.
"""
from __future__ import annotations

import bisect
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.attention.world_first import judge_candidate, node_candidate  # noqa: E402
from orion.schemas.attention_frame import PredictionErrorMagnitudeV1  # noqa: E402
from orion.substrate.prediction_error_magnitude import WINDOW_7D  # noqa: E402

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


def rejudge(c, t, series, *, sustain):
    """Real node_candidate + judge_candidate on the traced reading. Returns (eligible, dwell)."""
    if c.get("absent") or c.get("percentile_now") is None:
        return bool(c["eligible"]), None
    mag = PredictionErrorMagnitudeV1(
        value=float(c["value"]), age_sec=float(c.get("age_sec") or 0.0),
        percentile_now=float(c["percentile_now"]), n_readings_7d=int(c.get("n_readings_7d") or 0),
        band=c.get("band") if c.get("band") not in (None, "absent") else "insufficient_history",
    )
    times, vals = series.get(c["source_id"], ([], []))
    hi = bisect.bisect_right(times, t)
    lo = bisect.bisect_left(times, t - WINDOW_7D)
    hist = list(zip(times[lo:hi], vals[lo:hi]))
    cand = node_candidate(
        node_id=c["source_id"], label=c["source_id"], magnitude=mag, observed_at=None, now=t,
        pe_history=hist, sustain=sustain,
    )
    v = judge_candidate(cand)
    return v.eligible, v.hot_dwell_sec


def main() -> None:
    frames = load_frames(sys.argv[1])
    series = load_pe(sys.argv[2])
    first_pe = min(ts[0] for ts, _ in series.values())
    n = len(frames)
    sq, off, on = Counter(), Counter(), Counter()
    on_mature, sq_mature, n_mature = Counter(), Counter(), 0
    agree_off = 0
    moved = Counter()
    dwell_at_win: list[float] = []
    for t, wf in frames:
        cands = wf["candidates"]
        recorded = wf.get("winner")
        el_off, el_on = [], []
        for c in cands:
            if c["source_id"] in CONTINUOUS:
                e_off, _ = rejudge(c, t, series, sustain=False)
                e_on, dwell = rejudge(c, t, series, sustain=True)
                if e_off:
                    el_off.append(c)
                if e_on:
                    el_on.append(c)
                    if dwell is not None:
                        dwell_at_win.append(dwell)
            elif c["eligible"]:
                el_off.append(c)
                el_on.append(c)
        w_off, w_on = pick(el_off), pick(el_on)
        sq[recorded] += 1
        off[w_off] += 1
        on[w_on] += 1
        agree_off += int(w_off == recorded)
        if recorded in CONTINUOUS and w_on != recorded:
            moved[w_on] += 1
        if t - first_pe >= timedelta(days=3):
            n_mature += 1
            on_mature[w_on] += 1
            sq_mature[recorded] += 1

    def share(c, total):
        return {str(k or "no winner"): round(v / total, 4) for k, v in c.most_common()}

    report = {
        "frames": n,
        "span": [frames[0][0].isoformat(), frames[-1][0].isoformat()],
        "pe_history_from": first_pe.isoformat(),
        "sustain_off_reproduces_recorded_winner": round(agree_off / n, 4),
        "status_quo_recorded": share(sq, n),
        "sustained_rule": share(on, n),
        "frames_with_ge_3d_history": n_mature,
        "status_quo_recorded_ge_3d": share(sq_mature, n_mature) if n_mature else {},
        "sustained_rule_ge_3d": share(on_mature, n_mature) if n_mature else {},
        "frames_lost_by_biometrics_or_bus_now_go_to": share(moved, sum(moved.values()) or 1),
        "continuous_wins_with_dwell_le_300s": sum(1 for d in dwell_at_win if d <= 300.0),
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
