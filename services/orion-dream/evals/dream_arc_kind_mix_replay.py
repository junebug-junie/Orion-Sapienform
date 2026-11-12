#!/usr/bin/env python3
"""Eval: which kinds of Temporal Self arcs does each sleep replay, before and after the one-arc-per-kind rule?
READ-ONLY.

Live 11-12: over 5 days, 15 of 18 arcs replayed in dreams were ``interoception`` body arcs (substrate
execution / perception / chat / bus_synaptic). They run for hours and their returns count sensor flicker, so
they outweigh every other kind. ``replay.arc_candidates`` now keeps one arc per kind (the heaviest) before the
20-candidate cap; the dream arm and the control pool both draw from that list.

Two lanes:

* ``--fixture`` (default; deterministic, no database): the 11-11 04:57 window shape (four body arcs about
  different substrate nodes, one attention arc, one conversation arc) through the REAL ``arc_candidates`` and
  ``select_replay``. Checks the old rule replays 4 body arcs and the new rule replays one of each kind.
* ``--live``: every REAL completed sleep over the last ``--days``. For each: its own window (pressure.since to
  started_at), the closed arcs that ended in it (same filter as ``cycle_store.ARC_ROWS_SQL``), and the
  non-arc items it really replayed. Rebuilds the replay with the old rule (fidelity: does it match the arcs
  the sleep really replayed?) and with the new rule, and prints the kind mix per sleep and in total.
  Gate: under the new rule no sleep replays two arcs of one kind, and the arc cap (a third) still holds.

Limit: non-arc candidates that LOST their slot to an arc are not stored, so the rebuild cannot put them back.
A slot an arc gives up under the new rule is reported as ``freed`` (a real sleep would fill it with the next
existing item, or leave it empty if there was none). Arc rows can also be rewritten after a sleep (live:
10-30 arcs carry updated_at 10-31 06:07), so a few old-rule rebuilds differ from what really ran; the
fidelity line names them.

    PYTHONPATH=. .venv/bin/python services/orion-dream/evals/dream_arc_kind_mix_replay.py [--live --days 14]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SVC = Path(__file__).resolve().parents[1]
ROOT = SVC.parents[1]
sys.path[:0] = [str(ROOT), str(SVC), str(ROOT / "scripts" / "analysis")]

from app.replay import (  # noqa: E402
    ReplayItemV1,
    arc_candidates,
    arc_replay_slots,
    select_replay,
    walkway_replay_floor,
    walkway_replay_slots,
)

K = 12  # settings.DREAM_REPLAY_MAX default; the live lane reads the real setting


def _replay(existing: list[ReplayItemV1], arcs: list[ReplayItemV1], k: int) -> list[ReplayItemV1]:
    """The cycle's own call (app/cycle.py), arcs + existing + walkway already mixed in `existing`."""
    return select_replay(
        existing + arcs, k,
        source_caps={"arc": arc_replay_slots(k), "walkway": walkway_replay_slots(k)},
        source_floors={"walkway": walkway_replay_floor(k)},
    )


def _arc_kinds(replay: list[ReplayItemV1]) -> list[str]:
    return [r.tags[0] if r.tags else "?" for r in replay if r.source_kind == "arc"]


# --- fixture lane --------------------------------------------------------------------------------------

def _fixture() -> int:
    now = datetime(2026, 11, 11, 4, 57, tzinfo=timezone.utc)
    since = now - timedelta(hours=9)

    def row(aid, kind, ref, dwell, returns, minutes_ago):
        ended = now - timedelta(minutes=minutes_ago)
        return {"arc_id": aid, "kind": kind, "subject_ref": ref, "subject_label": ref,
                "began_at": ended - timedelta(seconds=dwell), "ended_at": ended, "status": "closed",
                "attention_returns": returns, "cumulative_dwell_sec": dwell}

    rows = [
        row("perc", "interoception", "node:substrate.perception", 14400.0, 9, 30),
        row("chat", "interoception", "node:substrate.chat", 10800.0, 7, 40),
        row("exec", "interoception", "node:substrate.execution", 9000.0, 6, 50),
        row("bus", "interoception", "node:substrate.bus_synaptic", 7200.0, 5, 60),
        row("att", "attention", "world:reading", 1500.0, 2, 70),
        row("conv", "conversation", "conv-1", 1200.0, 1, 80),
    ]
    existing = [ReplayItemV1(ref_id=f"metacog:{i}", source_kind="metacog", text="t", weight=1.0, reason="r")
                for i in range(3)]
    existing += [ReplayItemV1(ref_id=f"compaction_request:{i}", source_kind="compaction_request", text="t",
                              weight=0.5, reason="r") for i in range(3)]
    before = _arc_kinds(_replay(existing, arc_candidates(rows, since, now, one_per_kind=False), K))
    after = _arc_kinds(_replay(existing, arc_candidates(rows, since, now), K))
    print(f"fixture 11-11 04:57 window: before {before}")
    print(f"fixture 11-11 04:57 window: after  {after}")
    ok = before == ["interoception"] * 4 and sorted(after) == ["attention", "conversation", "interoception"]
    print("fixture: PASS" if ok else "fixture: FAIL")
    return 0 if ok else 1


# --- live lane (read-only) -----------------------------------------------------------------------------

def _ts(v: Any) -> datetime:
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


def _q(sql: str) -> Any:
    import _body_economy_data as data_mod  # read-only psql session

    return data_mod.run_query(sql)


def _live(days: int) -> int:
    from app.settings import settings

    k = int(settings.DREAM_REPLAY_MAX)
    days = max(1, int(days))
    cycles = _q(f"""
        SELECT coalesce(json_agg(json_build_object(
                 'cycle_id', cycle_id, 'started_at', started_at,
                 'since', cycle_json->'pressure'->>'since', 'replay', cycle_json->'replay')
               ORDER BY started_at), '[]'::json)
          FROM dream_cycle
         WHERE status = 'completed' AND started_at > now() - interval '{days} days';
    """)
    if not cycles:
        print(f"live: no completed sleeps in the last {days} days")
        return 1
    lo = min(_ts(c["since"]) for c in cycles if c.get("since"))
    hi = max(_ts(c["started_at"]) for c in cycles)
    arcs = _q(f"""
        SELECT coalesce(json_agg(json_build_object(
                 'arc_id', arc_id, 'kind', kind, 'subject_ref', subject_ref,
                 'subject_label', coalesce(arc_json->>'subject_label', ''),
                 'began_at', began_at, 'ended_at', ended_at, 'status', status,
                 'attention_returns', attention_returns,
                 'cumulative_dwell_sec', CASE WHEN jsonb_typeof(arc_json->'cumulative_dwell_sec') = 'number'
                     THEN (arc_json->>'cumulative_dwell_sec')::double precision ELSE 0 END)), '[]'::json)
          FROM temporal_self_arc
         WHERE status = 'closed' AND kind <> 'sleep'
           AND ended_at >= '{lo.isoformat()}' AND ended_at < '{hi.isoformat()}';
    """)
    for a in arcs:
        for f in ("began_at", "ended_at"):
            a[f] = _ts(a[f]) if a.get(f) else None

    tot_real, tot_before, tot_after = Counter(), Counter(), Counter()
    match = n = freed = 0
    mismatched: list[str] = []
    max_same_after = max_arcs_after = 0
    print(f"{'sleep':<17} {'started (UTC)':<12} {'real arcs replayed':<52} after")
    for c in cycles:
        if not c.get("since"):
            continue
        n += 1
        since, until = _ts(c["since"]), _ts(c["started_at"])
        stored = [ReplayItemV1(**r) for r in (c.get("replay") or [])]
        existing = [r for r in stored if r.source_kind != "arc"]
        real_arcs = [r for r in stored if r.source_kind == "arc"]
        rebuilt_old = _replay(existing, arc_candidates(arcs, since, until, one_per_kind=False), k)
        new = _replay(existing, arc_candidates(arcs, since, until), k)
        if sorted(r.ref_id for r in rebuilt_old if r.source_kind == "arc") == sorted(r.ref_id for r in real_arcs):
            match += 1
        else:
            mismatched.append(c["cycle_id"])
        real_k, new_k = _arc_kinds(stored), _arc_kinds(new)
        tot_real.update(real_k)
        tot_before.update(_arc_kinds(rebuilt_old))
        tot_after.update(new_k)
        freed += max(0, len(real_k) - len(new_k))
        if new_k:
            max_same_after = max(max_same_after, max(Counter(new_k).values()))
        max_arcs_after = max(max_arcs_after, len(new_k))
        print(f"{c['cycle_id']:<17} {until.strftime('%m-%d %H:%M'):<12} "
              f"{', '.join(real_k) or '-':<52} {', '.join(new_k) or '-'}")

    def share(cnt: Counter) -> str:
        total = sum(cnt.values())
        return f"{total} arcs: " + ", ".join(f"{kind} {v} ({100 * v / total:.0f}%)" for kind, v in cnt.most_common()) \
            if total else "0 arcs"

    print()
    print(f"sleeps: {n}  (last {days} days, completed)")
    print(f"real (what ran):          {share(tot_real)}")
    print(f"rebuilt, old rule:        {share(tot_before)}")
    print(f"rebuilt, one per kind:    {share(tot_after)}")
    print(f"fidelity: old-rule rebuild matches the arcs really replayed in {match}/{n} sleeps"
          + (f" (differ: {', '.join(mismatched)})" if mismatched else ""))
    print(f"arc slots freed for existing items: {freed}")
    print(f"most arcs of one kind in one sleep, new rule: {max_same_after}; most arcs in one sleep: {max_arcs_after}")
    ok = max_same_after <= 1 and max_arcs_after <= arc_replay_slots(k)
    print("live: PASS" if ok else "live: FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true", help="replay real sleeps from the live DB (read-only)")
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args(argv)
    return _live(args.days) if args.live else _fixture()


if __name__ == "__main__":
    raise SystemExit(main())
