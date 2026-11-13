"""Eval: the turn hand-back (living Prime Part 2, orion/prime/handback.py) replayed on the live cockpit record.

Reads (read-only) the ``ingress`` and ``finalize`` rows of ``cockpit_turn_sighting`` -- what orion-sql-writer
persisted from the same bus channel Prime listens to, ``orion:cockpit:hop`` -- for the last ``--days`` days,
rebuilds each row as the hop that was published, and folds them in time order through the real fold, with the
real tick rule (``observe``) run at each hop so concerns fade on the live TTL. No thought runs, so a concern a
turn opened is never dropped early: the loop-guard bounds are tested at their hardest.

Checks (exit 1 on any failure):

* deterministic  -- two replays of the same rows give the same turns, concern ids included;
* coverage       -- every finalize row of a handed-back session folds into a turn (none lost in the join);
* not_hollow     -- every folded turn has a non-empty gist (no "(no reply text)" for a completed turn);
* bounded        -- never more than TURN_CONCERNS_MAX_OPEN turn concerns held, never more than
                    TURN_CONCERNS_PER_DAY opened in any 24 h;
* replay_safe    -- folding the same rows a second time into the result changes nothing.

Reported, not checked: turns per kind, threads opened / refused (zero until the intention ledger's D0 judge
fields are deployed: before that no finalize hop carries ``promises_made``), concerns answered, sample gists.

Run: ``PYTHONPATH=. python -m orion.prime.evals.run_turn_handback_replay --days 3``.
Database: ``--psql`` (default ``docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -Atq``).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Sequence

from orion.prime import handback as hb
from orion.prime.state import PrimeSnapshotV1, PrimeStateV1, observe

DEFAULT_PSQL = "docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -Atq"

# Only the fields the fold reads leave the database (the rest of a hop's raw is not needed).
QUERY = """
select json_build_object(
  'correlation_id', correlation_id, 'seq', seq, 'ts', to_char(ts at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
  'stage', stage, 'visor_line', left(visor_line, 500), 'status', status, 'producer', producer,
  'summary', '{{}}'::jsonb,
  'raw', case when stage = 'ingress' then jsonb_strip_nulls(jsonb_build_object(
      'session_id', raw->'session_id', 'mode', raw->'mode', 'user_message', left(raw->>'user_message', 4000)))
    else jsonb_strip_nulls(jsonb_build_object(
      'final_text', left(raw->>'final_text', 2000), 'compliance_verdict', raw->'compliance_verdict',
      'reflection', jsonb_build_object('promises_made', raw->'reflection'->'promises_made',
                                       'promises_kept', raw->'reflection'->'promises_kept'))) end)
from cockpit_turn_sighting
where stage in ('ingress', 'finalize') and ts > now() - interval '{days} days'
order by ts, seq;
"""


def fetch(days: float, psql: str) -> list[dict[str, Any]]:
    out = subprocess.run([*shlex.split(psql), "-c", QUERY.format(days=float(days))], check=True,
                         capture_output=True, text=True, timeout=300).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def replay(hops: Sequence[dict[str, Any]]) -> dict[str, Any]:
    state, turns = PrimeStateV1(), hb.PrimeTurnsStateV1()
    snap = PrimeSnapshotV1(binding=None, body_ok=False)
    folded: list[hb.PrimeTurnV1] = []
    max_open = 0
    opened_times: list[datetime] = []
    for hop in hops:
        # jsonb_strip_nulls drops an absent mode, which is exactly "no mode key" on the wire.
        sig = hb.signal_from_hop(hop)
        if sig is None:
            continue
        now = sig.at
        state = observe(state, snap, now)
        state, turns, new = hb.fold_turn_signals(state, turns, [sig], now)
        folded.extend(new)
        opened_times.extend(now for t in new if t.opened_concern_id)
        max_open = max(max_open, sum(1 for c in state.concerns
                                     if c.graded is None and (c.event_ref or "").startswith(hb.TURN_EVENT_PREFIX)))
    max_day = max((sum(1 for u in opened_times if t <= u < t + hb.TURN_CONCERN_WINDOW) for t in opened_times),
                  default=0)
    return {"state": state, "turns": turns, "folded": folded, "max_open": max_open, "max_day": max_day}


def handed_back_finalize_ids(hops: Sequence[dict[str, Any]]) -> set[str]:
    started = {h["correlation_id"] for h in hops if h["stage"] == "ingress"
               and hb.classify_who((h.get("raw") or {}).get("session_id"), h.get("raw") or {})}
    return {h["correlation_id"] for h in hops if h["stage"] == "finalize" and h["correlation_id"] in started}


def checks(hops: Sequence[dict[str, Any]]) -> tuple[dict[str, bool], dict[str, Any]]:
    a, b = replay(hops), replay(hops)
    strip = lambda r: [t.model_dump() for t in r["folded"]]  # noqa: E731 -- concern ids are derived, so exact
    expected = handed_back_finalize_ids(hops)
    got = {t.correlation_id for t in a["folded"]}
    last = max((hb.signal_from_hop(h).at for h in hops if hb.signal_from_hop(h) is not None), default=None)
    again_state, again_turns, again = (a["state"], a["turns"], []) if last is None else hb.fold_turn_signals(
        a["state"], a["turns"], [s for s in (hb.signal_from_hop(h) for h in hops) if s is not None], last)
    result = {
        "deterministic": strip(a) == strip(b),
        "coverage": expected <= got,
        "not_hollow": all(t.gist and t.gist != "(no reply text)" for t in a["folded"]
                          if t.verdict in hb.COUNTED_VERDICTS),
        "bounded": a["max_open"] <= hb.TURN_CONCERNS_MAX_OPEN and a["max_day"] <= hb.TURN_CONCERNS_PER_DAY,
        "replay_safe": again == [] and again_state == a["state"] and again_turns.recent == a["turns"].recent,
    }
    report = {
        "hops": len(hops),
        "turns_by_who": dict(Counter(t.who for t in a["folded"])),
        "missing": sorted(expected - got)[:10],
        "threads_opened": sum(1 for t in a["folded"] if t.opened_concern_id),
        "threads_refused": sum(1 for t in a["folded"] if t.thread_refused),
        "turns_with_promises": sum(1 for t in a["folded"] if t.promise_refs),
        "concerns_answered": sum(len(t.settled_concern_ids) for t in a["folded"]),
        "max_turn_concerns_open": a["max_open"],
        "max_turn_concerns_24h": a["max_day"],
        "sample_gists": [f"{t.who}: {t.gist}" for t in a["folded"][-6:]],
    }
    return result, report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--psql", default=DEFAULT_PSQL)
    args = ap.parse_args(argv)
    hops = fetch(args.days, args.psql)
    result, report = checks(hops)
    print(json.dumps({"checks": result, **report}, indent=1, ensure_ascii=False))
    return 0 if all(result.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
