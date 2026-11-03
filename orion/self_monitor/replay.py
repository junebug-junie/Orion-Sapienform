"""Replay E1 and E2 over the last week of real acts (read-only eval).

    POSTGRES_URI=postgresql://... python -P -m orion.self_monitor.replay [--days 7] [--json]

Takes the organ's state as computed now (``report.collect``), then walks every ``grammar_acts``
attempt of the window in time order and asks the real gate (``gate.act_gate``) what it would have
done, spending a probe only on attempts the replay admits (a refused attempt never opens an act).
Also lists the self-questions E2 would mint from an empty memory.

Caveat, stated in the output: the state is today's, held fixed for the whole window. The real gate
sees the state as it was at each hour, and a faculty only becomes failing after 3 held evaluations,
so this is an upper bound on what the gate would have refused, not a reconstruction.

``reach``: which doers pass a gated seam today. ``allocator`` = an act-backed route in
config/execution_dispatch/execution_dispatch_policy.v1.yaml (for hub.world_pulse_read only the
reading PULL passes it; Hub's own scheduled read lane opens the same act without it);
``actions_journal`` = orion-actions' ``_dispatch_journal``; everything else is ``not_gated``.
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
from orion.self_monitor.gate import DEFAULT_PROBE_INTERVAL_SEC, FAILING_STATUSES, act_gate
from orion.self_monitor.questions import self_questions_for
from orion.self_monitor.view import SelfMonitorView

ALLOCATOR_ACTORS = {("thought.painting", "make"), ("hub.world_pulse_read", "look"),
                    ("hub.retro_leisure", "play"), ("aitown.session", "play")}


def reach(actor: str, verb: str) -> str:
    if (actor, verb) in ALLOCATOR_ACTORS:
        return "allocator"
    if actor == "actions.journal":
        return "actions_journal"
    return "not_gated"


def replay_acts(view: SelfMonitorView, rows: Iterable[Mapping[str, Any]], *,
                probe_interval_sec: float = DEFAULT_PROBE_INTERVAL_SEC) -> dict[str, dict[str, Any]]:
    """Per failing faculty: attempts, would_refuse, would_probe, reach. Rows need actor, verb,
    object_ref, opened_at, and are walked oldest first."""
    out: dict[str, dict[str, Any]] = {}
    last_admitted: dict[str, datetime] = {}
    for row in sorted(rows, key=lambda r: r["opened_at"]):
        actor, verb = str(row["actor"]), str(row["verb"])
        faculty = act_faculty(actor, verb, str(row.get("object_ref") or ""))
        if view.status(faculty) not in FAILING_STATUSES:
            continue
        opened = row["opened_at"]
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        d = act_gate(view, faculty, last_attempt_at=last_admitted.get(faculty), now=opened,
                     probe_interval_sec=probe_interval_sec)
        stats = out.setdefault(faculty, {"status": view.status(faculty), "reach": reach(actor, verb),
                                         "attempts": 0, "would_refuse": 0, "would_probe": 0,
                                         "refused_outcomes": defaultdict(int)})
        stats["attempts"] += 1
        if d.admit:
            stats["would_probe"] += int(d.probe)
            last_admitted[faculty] = opened
        else:
            stats["would_refuse"] += 1
            stats["refused_outcomes"][_outcome_word(row)] += 1
    for stats in out.values():
        stats["refused_outcomes"] = dict(stats["refused_outcomes"])
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
    from sqlalchemy import create_engine, text

    from orion.self_monitor.report import collect

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument("--probe-interval-sec", type=float, default=DEFAULT_PROBE_INTERVAL_SEC)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    url = os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL")
    if not url:
        print("POSTGRES_URI is required", file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc)
    state = collect(url, now=now, window_days=args.days)
    view = SelfMonitorView(known=True, reason="fresh", age_sec=0.0, faculties={r.faculty: r for r in state.faculties})
    engine = create_engine(url, pool_size=1, max_overflow=0, connect_args={
        "connect_timeout": 5, "options": "-c statement_timeout=60000 -c default_transaction_read_only=on"})
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(text(
            "SELECT actor, verb, object_ref, opened_at, closed_at, outcome_value, outcome_note FROM grammar_acts "
            "WHERE opened_at >= :since AND opened_at IS NOT NULL"), {"since": now - timedelta(days=args.days)}).mappings()]
    acts = replay_acts(view, rows, probe_interval_sec=args.probe_interval_sec)
    mints, _ = self_questions_for({}, view, now=now)
    failing = sorted(f for f, r in view.faculties.items() if r.status in FAILING_STATUSES)
    result = {
        "computed_at": now.isoformat(), "window_days": args.days, "probe_interval_sec": args.probe_interval_sec,
        "caveat": "today's state held fixed for the whole window: an upper bound, not a reconstruction",
        "failing_faculties": failing, "acts_scanned": len(rows), "gate": acts,
        "self_questions": [{"faculty": m.faculty, "status": m.status, "question_id": m.question_id,
                            "text": m.text} for m in mints],
    }
    if args.json:
        print(json.dumps(result, indent=1, default=str))
        return 0
    print(f"replay {args.days:g} days to {now:%Y-%m-%d %H:%MZ}; {len(rows)} acts; failing: {', '.join(failing) or 'none'}")
    print(f"caveat: {result['caveat']}")
    for faculty, s in sorted(acts.items()):
        print(f"  {faculty:<48} {s['status']:<10} reach={s['reach']:<16} attempts={s['attempts']:<4} "
              f"refuse={s['would_refuse']:<4} probe={s['would_probe']:<3} refused_outcomes={s['refused_outcomes']}")
    for faculty in failing:
        if faculty not in acts:
            print(f"  {faculty:<48} {view.status(faculty):<10} (not an act faculty, or no attempts: nothing to gate)")
    print("self-questions E2 would mint:")
    for m in mints:
        print(f"  [{m.question_id}] {m.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
