"""Replay E1 and E2 over the last week of real acts (read-only eval).

    POSTGRES_URI=postgresql://... python -P -m orion.self_monitor.replay [--days 7] [--json]

Takes the organ's state as computed now (``report.collect``), then walks every ``grammar_acts``
attempt of the window in time order and asks the real gate (``gate.act_gate``) what it would have
done, spending a probe only on attempts the replay admits (a refused attempt never opens an act).
Also lists the self-questions E2 would mint from an empty memory.

Caveat, stated in the output: the state is today's, held fixed for the whole window. The real gate
sees the state as it was at each hour, and a faculty only becomes failing after 3 held evaluations,
so this is an upper bound on what the gate would have refused, not a reconstruction.

Same-job retries (option A, 2026-11-03): at a seam that has a job identity, an attempt whose own job
already opened an act inside the probe window is admitted as a retry (no probe spent), exactly as the
live gate does. ``job_key`` names it per row: orion-actions journals -> ``test_before``
(``journal:<dedupe_key>``), Prime's concern bets -> ``object_ref``; the allocator has none. Per
faculty the replay also counts jobs: ``jobs_worked`` (a real attempt of the job worked) and
``jobs_lost`` (that worked attempt is one the gate would have refused).

``reach``: which doers pass a gated seam today. ``allocator`` = an act-backed route in
config/execution_dispatch/execution_dispatch_policy.v1.yaml (for hub.world_pulse_read only the
reading PULL passes it; Hub's own scheduled read lane opens the same act without it);
``actions_journal`` = orion-actions' ``_dispatch_journal``; ``prime`` = Prime's own act opener in
orion-mind (services/orion-mind/app/prime_loop.py and its loop library orion/mindloop/runtime.py); everything else is ``not_gated``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from orion.self_monitor.adapters import act_faculty
from orion.self_monitor.gate import DEFAULT_PROBE_INTERVAL_SEC, FAILING_STATUSES, MAX_SAME_JOB_ATTEMPTS, act_gate, utc
from orion.self_monitor.questions import self_questions_for
from orion.self_monitor.view import SelfMonitorView

ALLOCATOR_ACTORS = {("thought.painting", "make"), ("hub.world_pulse_read", "look"),
                    ("hub.retro_leisure", "play"), ("aitown.session", "play")}


PRIME_GATED_VERBS = frozenset({"infer", "predict", "ask", "regulate"})


def reach(actor: str, verb: str) -> str:
    if (actor, verb) in ALLOCATOR_ACTORS:
        return "allocator"
    if actor == "actions.journal":
        return "actions_journal"
    if actor == "prime" and verb in PRIME_GATED_VERBS:
        return "prime"
    return "not_gated"


def job_key(row: Mapping[str, Any]) -> str | None:
    """The job identity the live seam passes as ``job_key`` (None: that seam has no same-job notion)."""
    seam = reach(str(row.get("actor") or ""), str(row.get("verb") or ""))
    if seam == "actions_journal":
        return row.get("test_before") or None
    if seam == "prime" and row.get("verb") == "predict":
        return row.get("object_ref") or None
    return None


def replay_acts(view: SelfMonitorView, rows: Iterable[Mapping[str, Any]], *,
                probe_interval_sec: float = DEFAULT_PROBE_INTERVAL_SEC,
                same_job: bool = True) -> dict[str, dict[str, Any]]:
    """Per failing faculty: attempts, would_refuse, would_probe, would_retry (same job), jobs,
    jobs_worked, jobs_lost, reach. Rows need actor, verb, object_ref, test_before, opened_at, and are
    walked oldest first. ``same_job=False`` replays the rule before option A (no retry admission)."""
    out: dict[str, dict[str, Any]] = {}
    last_admitted: dict[str, datetime] = {}
    job_admitted: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    jobs: dict[str, dict[str, dict[str, bool]]] = defaultdict(dict)
    for row in sorted(rows, key=lambda r: r["opened_at"]):
        actor, verb = str(row["actor"]), str(row["verb"])
        faculty = act_faculty(actor, verb, str(row.get("object_ref") or ""))
        if view.status(faculty) not in FAILING_STATUSES:
            continue
        opened = utc(row["opened_at"])
        key = job_key(row)
        in_window = [t for t in job_admitted.get((faculty, key), []) if (opened - t).total_seconds() < probe_interval_sec] if key else []
        retry = same_job and 0 < len(in_window) < MAX_SAME_JOB_ATTEMPTS
        d = act_gate(view, faculty, last_attempt_at=last_admitted.get(faculty), now=opened,
                     probe_interval_sec=probe_interval_sec, same_job_attempted=retry)
        stats = out.setdefault(faculty, {"status": view.status(faculty), "reach": reach(actor, verb),
                                         "attempts": 0, "would_refuse": 0, "would_probe": 0, "would_retry": 0,
                                         "refused_outcomes": defaultdict(int)})
        stats["attempts"] += 1
        worked = _outcome_word(row) == "worked"
        job = jobs[faculty].setdefault(key or f"row:{id(row)}", {"worked": False, "lost": False})
        job["worked"] = job["worked"] or worked
        if d.admit:
            stats["would_probe"] += int(d.probe)
            stats["would_retry"] += int(d.same_job_retry)
            last_admitted[faculty] = opened
            if key:
                job_admitted[(faculty, key)].append(opened)
        else:
            stats["would_refuse"] += 1
            stats["refused_outcomes"][_outcome_word(row)] += 1
            job["lost"] = job["lost"] or worked
    for faculty, stats in out.items():
        stats["refused_outcomes"] = dict(stats["refused_outcomes"])
        stats["jobs"] = len(jobs[faculty])
        stats["jobs_worked"] = sum(1 for j in jobs[faculty].values() if j["worked"])
        stats["jobs_lost"] = sum(1 for j in jobs[faculty].values() if j["lost"])
    return out


def _outcome_word(row: Mapping[str, Any]) -> str:
    note = str(row.get("outcome_note") or "")
    if row.get("closed_at") is None:
        return "open"
    if note.startswith(("raised:", "failed:", "abandoned", "no_read_evidence:")):
        return "failed_to_execute"
    value = row.get("outcome_value")
    return "worked" if value is not None and float(value) >= 0.5 else "did_not_work"


def main(argv: list[str] | None = None) -> int:
    from sqlalchemy import text

    from orion.self_monitor.report import _engine, collect

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument("--probe-interval-sec", type=float, default=DEFAULT_PROBE_INTERVAL_SEC)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-same-job", action="store_true",
                        help="replay the rule before option A: a same-job retry is refused like a new attempt")
    parser.add_argument("--hold", action="append", default=[], metavar="FACULTY=STATUS",
                        help="hold a faculty at STATUS (e.g. broken) for the whole window, instead of today's "
                             "status; for replaying a case the window has slid past")
    args = parser.parse_args(argv)
    url = os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL")
    if not url:
        print("POSTGRES_URI is required", file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc)
    state = collect(url, now=now, window_days=args.days)
    records = {r.faculty: r for r in state.faculties}
    for item in args.hold:
        faculty, _, status = item.partition("=")
        if faculty not in records or not status:
            print(f"--hold {item!r}: unknown faculty or empty status", file=sys.stderr)
            return 2
        records[faculty] = records[faculty].model_copy(update={"status": status})
    view = SelfMonitorView(known=True, reason="fresh", age_sec=0.0, faculties=records)
    engine = _engine(url)  # report's cached read-only engine (one per URL)
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(text(
            "SELECT actor, verb, object_ref, test_before, opened_at, closed_at, outcome_value, outcome_note FROM grammar_acts "
            "WHERE opened_at >= :since AND opened_at IS NOT NULL"), {"since": now - timedelta(days=args.days)}).mappings()]
    acts = replay_acts(view, rows, probe_interval_sec=args.probe_interval_sec, same_job=not args.no_same_job)
    mints, _ = self_questions_for({}, view, now=now)
    failing = sorted(f for f, r in view.faculties.items() if r.status in FAILING_STATUSES)
    result = {
        "computed_at": now.isoformat(), "window_days": args.days, "probe_interval_sec": args.probe_interval_sec,
        "caveat": "today's state held fixed for the whole window: an upper bound, not a reconstruction",
        "held": args.hold, "same_job_rule": not args.no_same_job,
        "failing_faculties": failing, "acts_scanned": len(rows), "gate": acts,
        "self_questions": [{"faculty": m.faculty, "status": m.status, "question_id": m.question_id,
                            "text": m.text} for m in mints],
    }
    if args.json:
        print(json.dumps(result, indent=1, default=str))
        return 0
    print(f"replay {args.days:g} days to {now:%Y-%m-%d %H:%MZ}; {len(rows)} acts; failing: {', '.join(failing) or 'none'}")
    print(f"caveat: {result['caveat']}")
    print(f"same-job retries admitted: {result['same_job_rule']}; held: {', '.join(args.hold) or 'none'}")
    for faculty, s in sorted(acts.items()):
        print(f"  {faculty:<48} {s['status']:<10} reach={s['reach']:<16} attempts={s['attempts']:<4} "
              f"refuse={s['would_refuse']:<4} probe={s['would_probe']:<3} retry={s['would_retry']:<4} "
              f"jobs={s['jobs']} worked={s['jobs_worked']} lost={s['jobs_lost']} "
              f"refused_outcomes={s['refused_outcomes']}")
    for faculty in failing:
        if faculty not in acts:
            print(f"  {faculty:<48} {view.status(faculty):<10} (not an act faculty, or no attempts: nothing to gate)")
    print("self-questions E2 would mint:")
    for m in mints:
        print(f"  [{m.question_id}] {m.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
