"""Arc-precision eval on real exported rows (Temporal Self patch 2, acceptance checks 1-3).

Input: ``fixtures/day_2026-10-09.jsonl.gz``, a read-only, text-free export of two local days
(10-08 for carry-in, 10-09 judged) made by ``export_fixture_day.sql``. No database, no bus.

Gates (exit 1 if any fails):

1. Evidence precision: every ``evidence_ref`` of every arc resolves to an exported row whose
   OWN subject column equals the arc's ``subject_ref`` (an oracle that reads raw columns,
   not the reducer's adapters).
2. Context purity: ``context_event_ids`` hold only subject-less kinds.
3. Process recall: every completed sleep / curiosity run / visual run / reverie chain with
   content that became available on the day is exactly one arc.
4. Replay identity: folding the two days in one pass equals folding them in 48 hourly
   chunks with ``advance_clock`` at each boundary, byte-for-byte on the closed day.
5. Rest state reachable: some of the day has no foreground arc.
6. Hand labels (``fixtures/day_2026-10-09.labels.json``) hold.

Usage: python orion/temporal_self/evals/run_arc_precision_eval.py [--timeline] [--sweep]
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from orion.temporal_self import ReducerConfig, advance_clock, drain_closed_days, fold, initial_state  # noqa: E402
from orion.temporal_self.body import summarize_body  # noqa: E402
from orion.temporal_self.broadcast import BroadcastTickView  # noqa: E402
from orion.temporal_self.day import as_utc, day_window  # noqa: E402
from orion.temporal_self.sources import ADAPTERS  # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "day_2026-10-09.jsonl.gz"
LABELS = HERE / "fixtures" / "day_2026-10-09.labels.json"
DAY = "2026-10-09"
SUBJECTLESS_PREFIXES = ("metacog_observation:", "memory_episode:", "vision_percept:", "consolidation_window_close:",
                        "chat_turn:")


def load(path: Path = FIXTURE) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r.pop("k")].append(r)
    return rows


def build_inputs(rows: dict[str, list[dict]]):
    ticks = [BroadcastTickView(r["log_id"], as_utc(r["generated_at"]), r.get("ref"), r.get("label") or "") for r in rows["broadcast"]]
    events = []
    for kind, adapter in ADAPTERS.items():
        for r in rows.get(kind, []):
            e = adapter(r)
            if e is not None:
                events.append(e)
    return ticks, events


def oracle(rows: dict[str, list[dict]]) -> dict[str, str]:
    """``<table>:<pk>`` -> subject, read straight from raw columns."""
    out: dict[str, str] = {}
    for r in rows["broadcast"]:
        if r.get("ref"):
            out[f"substrate_attention_broadcast_log:{r['log_id']}"] = r["ref"]
    for r in rows["chat_turn"]:
        if r.get("has_prompt"):  # only Juniper's turns can be conversation evidence
            out[f"chat_history_log:{r['id']}"] = r.get("session_id")
    for r in rows["field_dominance_run"]:
        out[f"field_dominance_run:{r['run_id']}"] = r["target_id"]
    for r in rows["attention_loop_raised"]:
        out[f"attention_salience_trace:{r['trace_id']}"] = r["loop_id"]
    for r in rows["attention_loop_verdict"]:
        out[f"attention_loop_outcome:{r['outcome_id']}"] = r["loop_id"]
    for r in rows["curiosity_run"]:
        out[f"curiosity_run_outcomes:{r['run_id']}"] = r["run_id"]
    for r in rows["reverie_chain"]:
        out[f"substrate_reverie_chain:{r['chain_id']}"] = r["chain_id"]
        for t in r.get("thoughts") or []:
            out[f"substrate_reverie_thought:{t['thought_id']}"] = r["chain_id"]
    for r in rows["visual_run"]:
        out[f"reverie_visual_chain:{r['chain_id']}"] = r["chain_id"]
    for r in rows["dream_cycle"]:
        out[f"dream_cycle:{r['cycle_id']}"] = r["cycle_id"]
    return out


def run(cfg: ReducerConfig, rows, ticks, events):
    start, end = day_window("2026-10-08", cfg.tz_name)[0], day_window(DAY, cfg.tz_name)[1]
    s = fold(initial_state(), ticks, events, cfg)
    s = advance_clock(s, end, cfg)
    s, days = drain_closed_days(s)
    day = next(d for d in days if d.day_id == DAY)

    # Chunked replay at hourly watermarks.
    c = initial_state()
    t = start
    while t < end:
        nxt = t + timedelta(hours=1)
        c = fold(c, [x for x in ticks if t <= x.generated_at < nxt],
                 [e for e in events if t <= (e.ended_at or e.occurred_at) < nxt], cfg)
        c = advance_clock(c, nxt, cfg)
        t = nxt
    c, cdays = drain_closed_days(c)
    cday = next(d for d in cdays if d.day_id == DAY)
    identical = day.model_dump_json() == cday.model_dump_json()
    return day, identical


def evaluate(cfg: ReducerConfig, rows, ticks, events) -> dict:
    day, identical = run(cfg, rows, ticks, events)
    subjects = oracle(rows)
    arcs = day.arcs
    total = matched = unresolved = 0
    mismatches = []
    for a in arcs:
        for ref in a.evidence_refs:
            total += 1
            if ref not in subjects:
                unresolved += 1
                mismatches.append((a.arc_id, ref, "unresolved"))
            elif subjects[ref] == a.subject_ref:
                matched += 1
            else:
                mismatches.append((a.arc_id, ref, subjects[ref]))
    juniper_turns = {f"chat_turn:{r['id']}" for r in rows["chat_turn"] if r.get("has_prompt")}
    impure = [(a.arc_id, c) for a in arcs for c in a.context_event_ids
              if not c.startswith(SUBJECTLESS_PREFIXES) or c in juniper_turns]

    d0, d1 = day_window(DAY, cfg.tz_name)

    def in_day(ts) -> bool:
        t = as_utc(ts)
        return t is not None and d0 <= t < d1

    def avail_reverie(r):
        ts = [as_utc(t["created_at"]) for t in r.get("thoughts") or []]
        return max(ts + [as_utc(r["created_at"])]) if ts else None

    expected = {
        "sleep": {r["cycle_id"] for r in rows["dream_cycle"] if r["status"] == "completed" and in_day(r["ended_at"])},
        "curiosity": {r["run_id"] for r in rows["curiosity_run"] if r.get("completed_at") and in_day(r["completed_at"])},
        "imagery": {r["chain_id"] for r in rows["visual_run"] if in_day(r["created_at"])},
        "reverie": {r["chain_id"] for r in rows["reverie_chain"] if r.get("thoughts") and avail_reverie(r) and d0 <= avail_reverie(r) < d1},
    }
    recall = {}
    for kind, want in expected.items():
        got = [a.subject_ref for a in arcs if a.kind == kind]
        recall[kind] = {"expected": len(want), "arcs": len(got), "exact": sorted(got) == sorted(want)}

    by_kind = Counter(a.kind for a in arcs)
    returns = defaultdict(list)
    dwell = defaultdict(list)
    for a in arcs:
        returns[a.kind].append(a.attention_returns)
        dwell[a.kind].append(a.cumulative_dwell_sec)

    # Rest state: share of the day covered by no foreground (attention / conversation) segment.
    spans = sorted((max(s.began_at, d0), min(s.ended_at, d1)) for a in arcs if a.kind in ("attention", "conversation") for s in a.segments)
    covered = 0.0
    cur_a = cur_b = None
    for a_, b_ in spans:
        if cur_b is None or a_ > cur_b:
            if cur_b is not None:
                covered += (cur_b - cur_a).total_seconds()
            cur_a, cur_b = a_, b_
        else:
            cur_b = max(cur_b, b_)
    if cur_b is not None:
        covered += (cur_b - cur_a).total_seconds()
    day_sec = (d1 - d0).total_seconds()
    day_ticks = [r for r in rows["broadcast"] if in_day(r["generated_at"])]

    # Body summaries per non-reverie arc (metric gate step 4).
    body_rows = {k: [dict(r, _t=as_utc(r.get("observed_at") or r.get("timestamp") or r.get("ts"))) for r in rows[k]]
                 for k in ("body_cluster", "body_cabinet", "body_spike")}
    bodies = []
    for a in arcs:
        if a.kind == "reverie":
            continue
        b0, b1 = a.began_at, a.ended_at or a.last_seen_at
        if b1 <= b0:
            b0, b1 = b0 - timedelta(seconds=30), b1 + timedelta(seconds=30)  # point arcs: the minute around them
        pick = {k: [r for r in v if b0 <= r["_t"] <= b1] for k, v in body_rows.items()}
        bodies.append((a, summarize_body(pick["body_cluster"], pick["body_cabinet"], pick["body_spike"])))

    def dist(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return {"n": len(vals), "min": round(min(vals), 3), "median": round(statistics.median(vals), 3),
                "max": round(max(vals), 3), "distinct": len({round(v, 3) for v in vals})}

    # Every hypothesis made on the day hangs off its own sleep arc (by cycle id, never time).
    hyp_by_cycle: dict[str, set[str]] = defaultdict(set)
    for r in rows["dream_hypothesis"]:
        hyp_by_cycle[r["cycle_id"]].add(f"dream_hypothesis:{r['hypothesis_id']}")
    sleep_hyp_fail = [a.subject_ref for a in arcs if a.kind == "sleep"
                      if set(a.expectation_event_ids) != hyp_by_cycle.get(a.subject_ref, set())]

    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    label_failures = check_labels(labels, arcs)

    report = {
        "day": DAY,
        "config": {"K": cfg.arc_min_ticks, "R_sec": cfg.return_window_sec, "conversation_R_sec": cfg.conversation_return_window_sec},
        "arcs_by_kind": dict(sorted(by_kind.items())),
        "evidence": {"total": total, "matched": matched, "unresolved": unresolved,
                     "precision": round(matched / total, 6) if total else None, "mismatches": mismatches[:10]},
        "context_impure": impure[:10],
        "process_recall": recall,
        "replay_identical": identical,
        "returns": {k: {"arcs": len(v), "zero": sum(1 for x in v if x == 0), "ge2": sum(1 for x in v if x >= 2),
                        "max": max(v)} for k, v in sorted(returns.items())},
        "dwell_sec": {k: dist(v) for k, v in sorted(dwell.items())},
        "rest": {"foreground_covered_share": round(covered / day_sec, 4),
                 "ticks": len(day_ticks), "no_winner_tick_share": round(sum(1 for r in day_ticks if not r.get("ref")) / max(1, len(day_ticks)), 4)},
        "warnings": day.frame.warnings,
        "body": {
            "arcs": len(bodies),
            "cabinet_temp_c_max": dist([b.cabinet_temp_c_max for _, b in bodies]),
            "chassis_watts_mean": dist([b.chassis_watts_mean for _, b in bodies]),
            "ambient_spike_count": dist([float(b.ambient_spike_count) for _, b in bodies]),
            "arcs_with_cabinet_samples": sum(1 for _, b in bodies if b.cabinet_sample_count),
        },
        "frame_at_close": {
            "expectations_resolved": Counter(x.verdict for x in day.frame.expectations_resolved_today),
            "expectations_resolved_total": day.frame.expectations_resolved_total,
            "self_change_overflow": day.frame.self_change_overflow,
            "expectations_pending": len(day.frame.expectations_pending),
            "self_change_events": len(day.frame.self_change_event_ids),
            "constraint_events": len(day.frame.constraint_event_ids),
            "open_threads": len(day.frame.open_threads),
            "entered_day_with": len(day.frame.entered_day_with),
        },
        "label_failures": label_failures,
        "sleep_hypothesis_mismatch": sleep_hyp_fail,
        "timeline": timeline(arcs, cfg),
    }
    report["passed"] = bool(
        total and matched == total and not impure and identical
        and all(r["exact"] for r in recall.values())
        and report["rest"]["foreground_covered_share"] < 1.0 and not label_failures and not sleep_hyp_fail
    )
    return report


def check_labels(labels: dict, arcs) -> list[str]:
    fails = []
    for want in labels.get("arcs", []):
        hits = [a for a in arcs if a.kind == want["kind"] and a.subject_ref == want["subject_ref"]
                and ("began_at" not in want or a.began_at.isoformat() == want["began_at"])]
        if "count" in want and len(hits) != want["count"]:
            fails.append(f"{want['kind']} {want['subject_ref']}: {len(hits)} arcs, labelled {want['count']}")
            continue
        for a in hits:
            for field in ("ended_at",):
                if field in want and (getattr(a, field) is None or getattr(a, field).isoformat() != want[field]):
                    fails.append(f"{want['kind']} {want['subject_ref']} {field}: {getattr(a, field)} != {want[field]}")
            if "min_returns" in want and a.attention_returns < want["min_returns"]:
                fails.append(f"{want['kind']} {want['subject_ref']}: returns {a.attention_returns} < {want['min_returns']}")
    return fails


def timeline(arcs, cfg) -> list[str]:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(cfg.tz_name)
    out = []
    for a in sorted(arcs, key=lambda a: a.began_at):
        if a.kind == "reverie":
            continue
        b = a.began_at.astimezone(tz).strftime("%H:%M")
        e = (a.ended_at or a.last_seen_at).astimezone(tz).strftime("%H:%M")
        out.append(f"{b}-{e} {a.kind:13s} {a.subject_ref[:40]:40s} returns={a.attention_returns} dwell={int(a.cumulative_dwell_sec)}s "
                   f"{a.closed_reason or a.status}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="report arcs/returns across K and R")
    ap.add_argument("--timeline", action="store_true")
    args = ap.parse_args()
    rows = load()
    ticks, events = build_inputs(rows)
    if args.sweep:
        for k in (2, 3, 5):
            for r_min in (10, 30, 60, 180):
                rep = evaluate(ReducerConfig(arc_min_ticks=k, return_window_sec=r_min * 60.0), rows, ticks, events)
                att, inter = rep["returns"].get("attention", {}), rep["returns"].get("interoception", {})
                print(f"K={k} R={r_min:>3}m attention arcs={att.get('arcs')} ge2={att.get('ge2')} zero={att.get('zero')} | "
                      f"interoception arcs={inter.get('arcs')} ge2={inter.get('ge2')} zero={inter.get('zero')} | "
                      f"fg_covered={rep['rest']['foreground_covered_share']} passed={rep['passed']}")
        return 0
    rep = evaluate(ReducerConfig(), rows, ticks, events)
    if not args.timeline:
        rep.pop("timeline")
    print(json.dumps(rep, indent=2, default=str))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
