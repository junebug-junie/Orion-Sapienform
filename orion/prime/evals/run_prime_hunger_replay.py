"""Eval: how often Prime would have thought, and why, with hunger needs instead of the 6-hour quiet floor, replayed on
live data (integration design Part B, patch 5; orion/prime/hunger.py).

The fixture is a live pull (``--pull``, read-only SQL) of one window:

* every Prime thought (grammar_acts actor prime verb infer): its start and its close note. A thought whose note does
  not start with ``woke: quiet`` woke on a real change; those changes still wake Prime, so they are replayed as
  changes. A floor-only thought is NOT replayed: the floor is deleted.
* how many concerns Prime held after each thought (the note's ``kept N`` + added - dropped). Live, Prime keeps every
  concern at every thought (the notes read ``kept 3``), so the replay keeps them all at every replayed thought: the
  horizon need then measures the time since Prime last thought while it holds a concern.
* every experience with ``surprise_nats`` in the window, with its own source kind's 90th percentile and count over
  the 30 days strictly before it (the endpoint's rule);
* Juniper's messages to Orion in Hub (chat_history_log, source hub_orion) from 30 days before the window on.

The replay runs the real rules every 60 s (orion/prime/hunger.py ``compute_needs``, ``track_troughs``, ``pick_wake``, ``note_thought``,
``after_answer``; orion/prime/state.py ``wants_to_think``) with the live refractory interval (300 s), threshold 0.5
and budget 24. Every eligible tick is a thought (an upper bound: the bid can still defer). Two answer policies bound
the truth: ``idle`` (no thought ever moves or edits a concern: guard 1 halves every shown need) and ``busy`` (every
thought acts: no halving). Planted cases check the guards directly.

Checks:

* budget_holds        -- no replayed 24 h window has more than 24 need-woken thoughts, either policy;
* no_floor_wakes      -- no replayed thought woke on time alone (the floor is gone);
* concerns_get_thought-- while a concern is held, no stretch without a thought exceeds the concern's 24 h life
                         (the floor's one job: a held concern is thought about before it fades);
* never_falling_bounded -- a planted need pinned at drive 1.0 with the same content, Prime never acting, wakes Prime
                         at most once in 7 days (guards 1 and 5);
* ever_new_bounded    -- a planted need with NEW content every 10 minutes at drive 0.99, Prime never acting, stays
                         within the budget (24 per 24 h) every day;
* rest_point          -- with no change, no concern, no surprise and Juniper having written 30 minutes ago, no need wakes Prime.

Reported, not gated: thoughts per day per need and per change, against the live timer's thoughts per day and
floor-only thoughts.

Run: ``PYTHONPATH=. python orion/prime/evals/run_prime_hunger_replay.py`` (exit 1 on any failed check).
Refresh the fixture: ``PYTHONPATH=. python orion/prime/evals/run_prime_hunger_replay.py --pull`` (needs the live DB
through ``docker exec orion-athena-sql-db``; read-only).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from orion.prime import hunger as hg
from orion.prime.state import PrimeConcernV1, PrimeStateV1, wants_to_think

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "prime_hunger_inputs_live_2026-11-16.json"
TICK = timedelta(seconds=60)
REFRACTORY = timedelta(seconds=300)
WINDOW_DAYS = 10

_KEPT = re.compile(r"concerns \+(\d+) -(\d+) kept (\d+)")


# --- the pull (read-only) --------------------------------------------------------------------------------------

PULL_SQL = {
    "thoughts": """
SELECT coalesce(json_agg(json_build_object('at', opened_at, 'note', left(outcome_note, 300)) ORDER BY opened_at), '[]')
FROM grammar_acts WHERE actor = 'prime' AND verb = 'infer' AND opened_at > %(lo)s AND opened_at <= %(hi)s""",
    "surprises": """
SELECT coalesce(json_agg(x ORDER BY x.at), '[]') FROM (
  SELECT e.event_id, e.source_kind, e.occurred_at AS at, e.surprise_nats AS nats,
    (SELECT percentile_cont(0.9) WITHIN GROUP (ORDER BY h.surprise_nats) FROM temporal_self_event h
      WHERE h.source_kind = e.source_kind AND h.surprise_nats IS NOT NULL
        AND h.occurred_at >= e.occurred_at - interval '30 days' AND h.occurred_at < e.occurred_at) AS p90,
    (SELECT count(*) FROM temporal_self_event h
      WHERE h.source_kind = e.source_kind AND h.surprise_nats IS NOT NULL
        AND h.occurred_at >= e.occurred_at - interval '30 days' AND h.occurred_at < e.occurred_at) AS n
  FROM temporal_self_event e
  WHERE e.surprise_nats IS NOT NULL AND e.occurred_at > %(lo)s AND e.occurred_at <= %(hi)s
    AND e.source_kind <> 'prime_thought') x""",
    "juniper": """
SELECT coalesce(json_agg((created_at AT TIME ZONE 'UTC') ORDER BY created_at), '[]') FROM chat_history_log
WHERE source = 'hub_orion' AND btrim(coalesce(prompt, '')) <> ''
  AND created_at >= ((%(lo)s::timestamptz - interval '30 days') AT TIME ZONE 'UTC')
  AND created_at <= (%(hi)s::timestamptz AT TIME ZONE 'UTC')""",
}


def _psql(sql: str) -> Any:
    out = subprocess.run(["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney",
                          "-Atc", f"SET default_transaction_read_only = on; {sql}"],
                         capture_output=True, text=True, check=True, timeout=120)
    text = out.stdout.strip()
    return json.loads(text[len("SET"):].strip() if text.startswith("SET") else text)


def pull(hi: datetime) -> dict[str, Any]:
    lo = hi - timedelta(days=WINDOW_DAYS)
    data: dict[str, Any] = {"window": {"start": lo.isoformat(), "end": hi.isoformat()}}
    for key, sql in PULL_SQL.items():
        data[key] = _psql(sql.replace("%(lo)s", f"'{lo.isoformat()}'").replace("%(hi)s", f"'{hi.isoformat()}'"))
    return data


# --- the replay ------------------------------------------------------------------------------------------------


def _t(s: str) -> datetime:
    t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _held_after(note: str, before: int) -> int:
    m = _KEPT.search(note or "")
    if not m:
        return before
    return max(0, min(3, int(m.group(3)) + int(m.group(1))))


def _concerns(n: int, at: datetime) -> list[PrimeConcernV1]:
    return [PrimeConcernV1(concern_id=f"{i:016x}", text=f"held concern {i}", opened_at=at, last_evidence_at=at)
            for i in range(n)]


def replay(data: dict[str, Any], *, policy: str, planted: Optional[str] = None,
           days: Optional[float] = None) -> dict[str, Any]:
    """Thoughts as (time, why). ``planted``: never_falling | ever_new | rest (synthetic needs, no live data)."""
    start = _t(data["window"]["start"])
    end = start + timedelta(days=days) if days is not None else _t(data["window"]["end"])
    live = [(_t(r["at"]), r["note"] or "") for r in data.get("thoughts") or []]
    changes = sorted(t for t, note in live if not note.startswith("woke: quiet"))
    held_steps = []  # (time, held after it)
    held = 0
    for t, note in live:
        held = _held_after(note, held)
        held_steps.append((t, held))
    surprises = [s for s in data.get("surprises") or [] if s.get("p90") is not None]
    juniper = sorted(_t(x) for x in data.get("juniper") or [])

    state = PrimeStateV1(last_thought_at=start)
    thoughts: list[tuple[datetime, str]] = []
    ci = hi_ = 0
    held = 0
    last_keep = start
    now = start
    while now <= end:
        while ci < len(changes) and changes[ci] <= now:
            state = state.model_copy(update={"pending_changes": [*state.pending_changes, "a real change"]})
            ci += 1
        while hi_ < len(held_steps) and held_steps[hi_][0] <= now:
            held = held_steps[hi_][1]
            hi_ += 1
        inputs: dict[str, Any] = {"surprises": [], "juniper": {"message_times": []}}
        if planted is None:
            state = state.model_copy(update={"concerns": _concerns(held, last_keep)})
            inputs["surprises"] = [
                {"event_id": s["event_id"], "organ": "x", "source_kind": s["source_kind"], "label": s["source_kind"],
                 "surprise_nats": s["nats"], "source_p90": s["p90"], "source_n": s["n"]}
                for s in surprises if now - timedelta(hours=24) < _t(s["at"]) <= now]
            inputs["juniper"] = {"message_times": [t for t in juniper if now - timedelta(days=30) <= t <= now],
                                 "last_event_id": "chat_turn:last", "last_prompt": "..."}
        elif planted == "rest":
            # 60 past conversations 12 h apart, and she wrote 30 minutes ago: a pause, not a gap.
            inputs["juniper"] = {"message_times": [start - timedelta(hours=12 * k) for k in range(60)]
                                 + [now - timedelta(minutes=30)], "last_event_id": "e", "last_prompt": "hi"}
        needs = hg.compute_needs(state, now, inputs)
        if planted == "never_falling":
            needs = [hg.NeedV1(name="surprise", drive=1.0, damped=hg.damp_factor(state.needs, "surprise", now),
                               content="a deadline that cannot be met", ref="experience:stuck", recover_sec=86400)]
        elif planted == "ever_new":
            slot = int((now - start) / timedelta(minutes=10))
            needs = [hg.NeedV1(name="surprise", drive=0.99, damped=0.99 * hg.damp_factor(state.needs, "surprise", now),
                               content=f"new surprise {slot}", ref=f"experience:s{slot}", recover_sec=86400)]
        state = hg.track_troughs(state, needs)
        wake = hg.pick_wake(needs, state, now)
        if wants_to_think(state, now, refractory=REFRACTORY, need_wake=wake is not None):
            why = f"need:{wake.name}" if wake is not None and not state.pending_changes else (
                "change" if state.pending_changes else "?")
            if wake is not None and state.pending_changes:
                why = f"change+need:{wake.name}"
            thoughts.append((now, why))
            state = hg.note_thought(state, now, wake if wake is not None else None)
            acted = policy == "busy"
            # Live, every thought keeps every held concern: that answers each horizon need on a held concern.
            kept = [c.concern_id for c in state.concerns] if planted is None else []
            state = hg.after_answer(state, hg.shown_needs(needs), now, acted=acted,
                                    touched=hg.answered_refs(kept=kept))
            state = state.model_copy(update={"pending_changes": [], "last_thought_at": now})
            last_keep = now  # live: every thought keeps every held concern
        state = hg.prune(state, now)
        now += TICK
    return {"thoughts": thoughts, "start": start, "end": end, "live": live, "held_steps": held_steps}


def _per_day(thoughts: list[tuple[datetime, str]], start: datetime, end: datetime) -> float:
    return len(thoughts) / max((end - start).total_seconds() / 86400.0, 1e-9)


def _max_need_wakes_24h(thoughts: list[tuple[datetime, str]]) -> int:
    times = [t for t, why in thoughts if why.startswith("need:")]
    best = 0
    for i, t in enumerate(times):
        best = max(best, sum(1 for u in times[i:] if u - t < timedelta(hours=24)))
    return best


def _longest_gap_while_held(r: dict[str, Any]) -> timedelta:
    """The longest stretch with no thought during which Prime held a concern the whole time. A concern dropped or
    settled ends the stretch (it no longer needs a thought); a replayed thought ends it too (it keeps them all)."""
    points = sorted([(t, "thought", 0) for t, _ in r["thoughts"]] + [(t, "held", n) for t, n in r["held_steps"]]
                    + [(r["end"], "end", 0)])
    held, since, worst = 0, None, timedelta(0)
    for t, kind, n in points:
        if since is not None:
            worst = max(worst, t - since)
        if kind == "held":
            held = n
        if kind == "thought" or held == 0:
            since = t if held > 0 else None
        elif since is None:
            since = t
    return worst


def run(data: Optional[dict[str, Any]] = None) -> list[tuple[str, bool, str]]:
    data = data or json.loads(FIXTURE.read_text())
    results: list[tuple[str, bool, str]] = []
    idle, busy = replay(data, policy="idle"), replay(data, policy="busy")
    start, end = idle["start"], idle["end"]
    live = idle["live"]
    floor_only = sum(1 for _, n in live if n.startswith("woke: quiet"))
    report = []
    for name, r in (("idle", idle), ("busy", busy)):
        why = Counter(w for _, w in r["thoughts"])
        report.append(f"{name}: {len(r['thoughts'])} thoughts = {_per_day(r['thoughts'], start, end):.1f}/day "
                      f"({', '.join(f'{k} {v}' for k, v in sorted(why.items()))})")
    results.append(("report", True, f"live timer: {len(live)} thoughts = {_per_day(live, start, end):.1f}/day, "
                                     f"{floor_only} woke on the floor alone | " + " | ".join(report)))
    worst = max(_max_need_wakes_24h(idle["thoughts"]), _max_need_wakes_24h(busy["thoughts"]))
    results.append(("budget_holds", worst <= hg.DAILY_BUDGET, f"most need wakes in any 24 h: {worst}"))
    floorish = [w for _, w in idle["thoughts"] + busy["thoughts"] if w == "?"]
    results.append(("no_floor_wakes", not floorish, f"{len(floorish)} thoughts woke on time alone"))
    gap = max(_longest_gap_while_held(idle), _longest_gap_while_held(busy))
    results.append(("concerns_get_thought", gap <= timedelta(hours=24),
                    f"longest stretch with a held concern and no thought: {gap.total_seconds() / 3600:.1f} h"))
    base = {"window": {"start": "2026-11-01T00:00:00+00:00", "end": "2026-11-08T00:00:00+00:00"}}
    nf = replay(base, policy="idle", planted="never_falling")
    results.append(("never_falling_bounded", len(nf["thoughts"]) <= 1,
                    f"planted never-falling need, same content, never acted on: {len(nf['thoughts'])} thoughts in 7 days"))
    en = replay(base, policy="idle", planted="ever_new")
    by_day = Counter(t.date() for t, _ in en["thoughts"])
    worst_en = _max_need_wakes_24h(en["thoughts"])
    results.append(("ever_new_bounded", worst_en <= hg.DAILY_BUDGET,
                    f"planted ever-new need: {len(en['thoughts'])} thoughts in 7 days, most in any 24 h {worst_en}, "
                    f"per day {dict(sorted((d.isoformat()[5:], n) for d, n in by_day.items()))}"))
    rest = replay(base, policy="idle", planted="rest", days=2)
    results.append(("rest_point", not rest["thoughts"], f"{len(rest['thoughts'])} thoughts at rest over 2 days"))
    return results


def main(argv: list[str]) -> int:
    if "--pull" in argv:
        hi = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        FIXTURE.write_text(json.dumps(pull(hi), indent=1) + "\n")
        print(f"wrote {FIXTURE}")
    results = run()
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
