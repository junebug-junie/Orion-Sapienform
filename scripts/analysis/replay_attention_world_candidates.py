#!/usr/bin/env python3
"""Replay stored broadcast frames with proposed WORLD candidates added. Read-only analysis.

Design: docs/superpowers/specs/2026-10-16-attention-world-candidates-design.md.

Question: if the things Orion does in the world (curiosity runs, reading, paintings) could
compete for attention, how often would they win, and what would they push out?

Every stored world-first frame keeps its traced candidates and verdicts exactly (the real
broadcast's output). Each proposed candidate is built at the frame's time from the events
that had happened by then, and judged with the REAL ``orion.attention.world_first``
``judge_candidate``. The winner is re-picked with the same key ``rank_candidates`` uses
(rank score, then source id; the Borda tie-break is not in the trace -- an exact tie between
a new and an old candidate is the only case it would decide). With no proposed candidate the
script must reproduce the stored winner; it reports how often it does.

Variants (each replayed alone, so their effects do not mix):

  curiosity_nats  -- the proposal. One reading per finished curiosity run: the Bayesian
                     surprise it recorded (``curiosity_run_outcomes.realized_nats``), judged
                     against the runs of the 7 days before it. Event-written: news for
                     EVENT_ORIENTING_WINDOW_SEC after the run finished, fading after the
                     grace, like any other event-written source.
  rate:<workflow> -- the rejected comparison. The chat pattern applied to Orion's own
                     doing: finished runs of one workflow in the last 15 min against the same
                     count sampled every minute over 7 days (``chat_rate_reading``).

Inputs (SELECT-only exports, one JSON object per line):

    P='docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc'
    $P "select json_build_object('t',generated_at,'wf',projection_json->'frame'->'debug'->'world_first') \\
        from substrate_attention_broadcast_log where generated_at >= now()-interval '7 days' \\
        and projection_json->'frame'->'debug' ? 'world_first' order by generated_at" > frames.jsonl
    $P "select json_build_object('source','curiosity_nats','t',o.completed_at,'value',o.realized_nats, \\
        'ref',o.run_id,'detail',r.detail) from curiosity_run_outcomes o left join substrate_durable_run_state r \\
        on r.run_id=o.run_id and r.workflow='curiosity.investigate' and r.status='completed' \\
        where o.completed_at >= now()-interval '15 days' and o.realized_nats is not null \\
        order by o.completed_at" > events.jsonl
    $P "select json_build_object('source',workflow,'t',generated_at,'value',1,'ref',run_id,'detail',detail) \\
        from substrate_durable_run_state where status='completed' and workflow in \\
        ('curiosity.investigate','reading.turn','reverie.visual') \\
        and generated_at >= now()-interval '15 days' order by generated_at" >> events.jsonl
    python scripts/analysis/replay_attention_world_candidates.py frames.jsonl events.jsonl

Needs pydantic (the repo venv). No writes anywhere.
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.attention.world_first import (  # noqa: E402
    EVENT_ORIENTING_WINDOW_SEC,
    chat_rate_reading,
    judge_candidate,
    midrank_percentile,
)
from orion.schemas.attention_candidate import AttentionCandidateV1  # noqa: E402
from orion.substrate.prediction_error_magnitude import (  # noqa: E402
    WINDOW_7D,
    compute_prediction_error_magnitude,
)

CURIOSITY_SOURCE_ID = "world:curiosity"
RATE_WORKFLOWS = ("curiosity.investigate", "reading.turn", "reverie.visual")


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def load_frames(path: str) -> list[tuple[datetime, dict]]:
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        wf = row.get("wf")
        if isinstance(wf, dict) and wf.get("candidates") is not None:
            out.append((_ts(row["t"]), wf))
    return out


def _excerpt(source: str, detail: dict | None, cap: int = 160) -> str:
    d = detail or {}
    text = ""
    if source in ("curiosity_nats", "curiosity.investigate"):
        text = str(d.get("finding_text") or "")
    elif source == "reading.turn":
        raw = str((d.get("reading_result") or {}).get("text") or "")
        try:
            body = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                body = json.loads(m.group(0)) if m else {}
            except Exception:
                body = {}
        text = str(body.get("article_summary") or body.get("what_i_learned") or body.get("summary") or "")
    elif source == "reverie.visual":
        text = str(d.get("caption") or "(no caption)")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:cap]


def load_events(path: str) -> dict[str, list[tuple[datetime, float, str, str]]]:
    series: dict[str, list[tuple[datetime, float, str, str]]] = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        src = row["source"]
        series[src].append(
            (_ts(row["t"]), float(row["value"]), str(row.get("ref") or ""), _excerpt(src, row.get("detail")))
        )
    for v in series.values():
        v.sort(key=lambda r: r[0])
    return series


def key(c: dict) -> float:
    return c["rank_score"] if c.get("rank_score") is not None else (c.get("score") or 0.0)


def pick(cands: list[dict]) -> str | None:
    """rank_candidates' order: highest key first, then source_id ascending."""
    elig = [c for c in cands if c.get("eligible")]
    if not elig:
        return None
    return sorted(elig, key=lambda c: (-key(c), c["source_id"]))[0]["source_id"]


def curiosity_candidate(events, t: datetime, *, min_readings: int):
    """world:curiosity at ``t``: the newest finished run's realized nats against the
    runs of the 7 days before ``t``. None when no run finished in 7 days."""
    times = [e[0] for e in events]
    hi = bisect.bisect_right(times, t)
    lo = bisect.bisect_left(times, t - WINDOW_7D)
    window = events[lo:hi]
    if not window:
        return None, None
    last_t, last_v, ref, text = window[-1]
    hist = [(e[0], e[1]) for e in window]
    mag = compute_prediction_error_magnitude(
        value=last_v, observed_at=last_t, history=hist, now=t, min_readings=min_readings
    )
    cand = AttentionCandidateV1(
        candidate_id=f"attention-candidate:{CURIOSITY_SOURCE_ID}",
        source_id=CURIOSITY_SOURCE_ID,
        source_kind="external",
        label=f"a curiosity run changed my mind ({last_v:.3f} nats)",
        unusualness=mag,
        rank_percentile=midrank_percentile(last_v, [v for _, v in hist]),
        observed_at=last_t,
        value_kind="level",
        event_window_sec=EVENT_ORIENTING_WINDOW_SEC,
        evidence_refs=[CURIOSITY_SOURCE_ID, f"curiosity_run:{ref}"],
    )
    return cand, (ref, text, last_v)


def rate_candidate(events, t: datetime, source_id: str):
    times = [e[0] for e in events if e[0] <= t]
    # Fast path: an empty 15-min window reads 0 -> strict-below percentile 0 -> quiet.
    lo = bisect.bisect_right(times, t - timedelta(minutes=15))
    if lo >= len(times):
        return None, None
    mag, mid = chat_rate_reading(times, now=t)
    ref, text = events[len(times) - 1][2], events[len(times) - 1][3]
    cand = AttentionCandidateV1(
        candidate_id=f"attention-candidate:{source_id}",
        source_id=source_id,
        source_kind="external",
        label=f"{source_id}: {int(mag.value)} finished in 15 min",
        unusualness=mag,
        rank_percentile=mid,
        observed_at=times[-1],
        value_kind="count",
        evidence_refs=[source_id],
    )
    return cand, (ref, text, mag.value)


def verdict_dict(cand: AttentionCandidateV1) -> dict:
    v = judge_candidate(cand)
    return {
        "source_id": cand.source_id,
        "eligible": v.eligible,
        "score": v.score,
        "rank_score": v.rank_score,
        "reason": v.reason,
        "band": v.band,
    }


def replay(frames, series, variant: str, *, min_readings: int, examples: int):
    stored_match = 0
    base_no = new_no = won = 0
    displaced: Counter = Counter()
    eligible_ticks = 0
    won_days: Counter = Counter()
    ex: list[tuple] = []
    seen_refs: set[str] = set()
    for t, wf in frames:
        cands = list(wf["candidates"])
        base = pick(cands)
        if base == wf.get("winner"):
            stored_match += 1
        if variant == "curiosity_nats":
            cand, info = curiosity_candidate(series.get("curiosity_nats", []), t, min_readings=min_readings)
        else:
            wfname = variant.split(":", 1)[1]
            cand, info = rate_candidate(series.get(wfname, []), t, f"world:activity.{wfname}")
        new = base
        if cand is not None:
            vd = verdict_dict(cand)
            if vd["eligible"]:
                eligible_ticks += 1
            new = pick(cands + [vd])
        base_no += base is None
        new_no += new is None
        if new is not None and new != base:
            won += 1
            displaced[base or "(no winner)"] += 1
            won_days[t.date().isoformat()] += 1
            if info and info[0] not in seen_refs and len(ex) < examples:
                seen_refs.add(info[0])
                ex.append((t.isoformat(timespec="seconds"), base or "(no winner)", info))
    n = len(frames)
    return {
        "variant": variant,
        "frames": n,
        "stored_winner_reproduced": stored_match,
        "eligible_ticks": eligible_ticks,
        "new_candidate_won": won,
        "won_per_day": round(won / max(1e-9, (frames[-1][0] - frames[0][0]).total_seconds() / 86400), 1),
        "no_winner_before": round(base_no / n, 4),
        "no_winner_after": round(new_no / n, 4),
        "displaced": dict(displaced.most_common()),
        "won_by_day": dict(sorted(won_days.items())),
        "examples": ex,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("events")
    ap.add_argument("--min-readings", type=int, default=20,
                    help="curiosity_nats: fewer finished runs than this in 7 days = insufficient_history")
    ap.add_argument("--examples", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    frames = load_frames(args.frames)
    series = load_events(args.events)
    if not frames:
        print("no frames", file=sys.stderr)
        return 1
    results = [replay(frames, series, "curiosity_nats", min_readings=args.min_readings, examples=args.examples)]
    for wfname in RATE_WORKFLOWS:
        results.append(replay(frames, series, f"rate:{wfname}", min_readings=args.min_readings, examples=5))
    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return 0
    print(f"frames {frames[0][0]} .. {frames[-1][0]}  n={len(frames)}")
    for r in results:
        print(f"\n== {r['variant']}")
        for k in ("frames", "stored_winner_reproduced", "eligible_ticks", "new_candidate_won", "won_per_day",
                  "no_winner_before", "no_winner_after"):
            print(f"  {k}: {r[k]}")
        print(f"  displaced: {r['displaced']}")
        print(f"  won_by_day: {r['won_by_day']}")
        for t, base, (ref, text, v) in r["examples"]:
            print(f"  ex {t}  instead of {base}  ref={ref} value={v:.4g}  {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
