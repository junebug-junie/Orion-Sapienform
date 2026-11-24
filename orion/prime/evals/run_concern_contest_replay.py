"""Eval: Prime's concern slots 11-17 20:01Z..11-24 02:00Z replayed without and with the concern contest (#3574, #3656).

Design: docs/superpowers/specs/2026-11-17-prime-concerns-decided-by-attention-design.md, patch 2 (acceptance checks 3-5
on recorded data). The window starts when patch 1 (the evidence clock, #3649) went live, so both arms run with the
clock on, as live.

The fixture (``fixtures/concern_contest_live_2026-11-17_24.json``) is a read-only pull of everything that moved Prime's
concern slots in the window:

* the 3 concerns held at the start (two 11-13 letter threads, one 11-16 thread);
* every promise the hand-back folded that we have a record of: the 3 it opened (``concern_record``), the 11-17 21:55Z
  refusal (#3656) and the 6 refusals of 11-23 (orion-mind log ``prime_turn_handback``). orion-mind's log before
  11-22 22:47Z is gone, so a promise refused 11-18..11-22 is missing (hub.promise acts show 4 chat promises on
  11-19/20 that opened no concern; whether the hand-back saw them is unknown, so they are left out);
* every day letter (``orion:prime:day_letter`` done; all had 5 threads), dream concern and concern Prime named
  (``concern_record``; the two event-want concerns carry their want's ``salience`` from ``want_ledger``; for
  a20521a9 the event_ref is the one its thought woke on, inferred);
* every settle (``concern_record``), every answered Prime thought (``grammar_acts`` infer, outcome 1, its close time)
  and every drop those thoughts made (``dropped <id>``);
* every question Prime asked about a concern (``ask`` acts: open renews the concern, and it is in flight until the
  act closes) and every take-on (``intention:`` make acts).

The replay applies the real pure functions every minute (orion/prime/state.py ``observe`` and ``touch_concern``,
orion/prime/handback.py ``fold_turn_signals``, orion/prime/day_letter.py ``thread_concerns``, orion/prime/slots.py
``seat_or_wait`` and ``run_contest``, orion/prime/acts.py ``settle_concern``) twice: contest OFF (no contest pass at all,
as live) and ON.

Limit: the model's live answers are held fixed. A drop, ask or take-on about a concern the ON arm no longer holds
changes nothing, and Prime is never shown the new concerns it would have held, so it never acts on them here. Thoughts
are the live ones only: the horizon need for a concern the ON arm holds (and live did not) would wake Prime more
often, so the ON arm displaces later here than it would live (a quiet concern yields only after Prime thought once
since it went quiet).

Checks (exit 1 on any failure):

* fidelity      -- OFF reproduces the live record: threads held per letter 0/2/0/1/1/0, the 3 promises live opened
                   open and the 7 live refused do not, the same 5 concerns fade;
* threads_held  -- ON: every letter gets at least 1 thread held (live: 3 of 6 letters got 0);
* promise_3656  -- ON: the 11-17 21:55Z promise (d391d241) opens at once, displacing a non-promise concern;
* no_promise_lost -- ON: every promise opens or is still waiting at the end; none is dropped. A promise that waited
                   until its 72 h ran out would be counted as expired and fail this;
* no_churn      -- ON: no contest displacement of a concern quiet for less than half its life, with a question open,
                   or that Prime had not thought about since it went quiet and got its slot (acceptance check 4);
* never_a_promise -- ON: no promise is ever displaced (Q3);
* notes         -- ON: every displacement's close note starts "displaced: by <id>; score <held> held vs <new> new;".

Run: ``PYTHONPATH=. python orion/prime/evals/run_concern_contest_replay.py``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from orion.prime import handback as hb
from orion.prime import slots
from orion.prime.acts import settle_concern
from orion.prime.day_letter import thread_concern_id, thread_concerns
from orion.prime.state import (MAX_CONCERNS, PrimeConcernV1, PrimeNominationV1, PrimeSnapshotV1,
                               PrimeStateV1, close_concern, concern_ttl, is_promise_concern, observe, touch_concern)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "concern_contest_live_2026-11-17_24.json"
TICK = timedelta(minutes=1)


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s)


@dataclass
class Run:
    held_per_letter: dict[str, int] = field(default_factory=dict)
    promise_open: dict[str, str] = field(default_factory=dict)  # corr -> "at once" | "later" | "waiting" | "refused"
    faded: list[str] = field(default_factory=list)
    displaced: list[dict[str, Any]] = field(default_factory=list)
    expired: list[str] = field(default_factory=list)
    max_waiting: int = 0
    log: list[str] = field(default_factory=list)
    final: PrimeStateV1 = field(default_factory=PrimeStateV1)


def _signals(p: dict[str, Any]) -> list[hb.TurnSignalV1]:
    at = _t(p["at"])
    return [hb.TurnSignalV1(correlation_id=p["correlation_id"], stage="ingress", at=at - timedelta(seconds=30),
                            who=p["who"], prompt_excerpt="(replay)"),
            hb.TurnSignalV1(correlation_id=p["correlation_id"], stage="finalize", at=at, final_excerpt="(replay)",
                            verdict="completed", promise_indices=[0], promise_quotes=["(replay)"])]


def replay(fx: dict[str, Any], *, contest: bool) -> Run:
    events: list[tuple[datetime, int, str, dict]] = []
    for s in fx["settles"]:
        events.append((_t(s["at"]), 0, "settle", s))
    for d in fx["drops"]:
        events.append((_t(d["at"]), 1, "drop", d))
    for a in fx["asks"]:
        events.append((_t(a["opened_at"]), 2, "ask_open", a))
        if a["closed_at"]:
            events.append((_t(a["closed_at"]), 2, "ask_close", a))
    for k in fx["take_ons"]:
        events.append((_t(k["opened_at"]), 2, "take_on", k))
        events.append((_t(k["closed_at"]), 2, "take_off", k))
    for p in fx["promises"]:
        events.append((_t(p["at"]), 3, "promise", p))
    for th in fx["thoughts"]:
        events.append((_t(th), 1, "thought", {}))
    for o in fx["opens"]:
        events.append((_t(o["at"]), 4, "open", o))
    for lt in fx["letters"]:
        events.append((_t(lt["at"]), 5, "letter", lt))
    events.sort(key=lambda e: (e[0], e[1]))

    run = Run()
    now, end = _t(fx["start"]), _t(fx["end"])
    state = PrimeStateV1(concerns=[PrimeConcernV1(concern_id=c["concern_id"], text=f"held {c['concern_id']}",
                                                  event_ref=c["event_ref"], opened_at=_t(c["opened_at"]),
                                                  last_evidence_at=_t(c["last_evidence_at"]),
                                                  act_opened_at=_t(c["opened_at"]))
                                   for c in fx["initial"]])
    turns = hb.PrimeTurnsStateV1()
    taken_on: set[str] = set()
    promise_ids = {hb.turn_concern_id(p["correlation_id"]): p["correlation_id"] for p in fx["promises"]}

    def contest_pass(at: datetime) -> None:
        nonlocal state, turns
        if not contest:
            return
        before = {c.concern_id for c in state.concerns}

        def bound(st: PrimeStateV1, _p: PrimeConcernV1):
            return hb.thread_refusal(st, turns, at, exempt=frozenset(taken_on), room=False)

        state, rep = slots.run_contest(state, at, taken_on=frozenset(taken_on), promise_bound=bound)
        for cid in rep.seated:
            if cid in promise_ids and cid not in before:
                turns = turns.model_copy(update={"turn_concerns_opened_at": [*turns.turn_concerns_opened_at, at]})
                run.promise_open[promise_ids[cid]] = "later"
                turns = hb.note_opened_later(turns, cid)
                run.log.append(f"{at:%m-%d %H:%MZ} waiting promise {cid} opened")
        for cid in rep.expired:
            run.expired.append(cid)
            run.log.append(f"{at:%m-%d %H:%MZ} waiting concern {cid} expired")

    def track(before: PrimeStateV1, after: PrimeStateV1, at: datetime) -> None:
        """Each new displaced record, with how long the pushed-out concern had been quiet and whether a promise's
        intake (no quiet rule) or the contest pushed it out."""
        held = {c.concern_id: c for c in before.concerns}
        for r in after.concern_record[len(before.concern_record):]:
            if r.end != "displaced" or r.concern_id not in held:
                continue
            c = held[r.concern_id]
            run.displaced.append({"concern_id": r.concern_id, "at": at, "note": r.note,
                                  "quiet": at - c.last_evidence_at, "half_life": concern_ttl(c) / 2,
                                  "intake": "a new promise always gets a slot" in r.note,
                                  "promise": is_promise_concern(c),
                                  "question_open": any(n.concern_id == c.concern_id for n in before.open_nominations),
                                  "seen": (before.last_thought_at is not None and before.last_thought_at
                                           >= max(c.last_evidence_at + concern_ttl(c) / 2, c.opened_at)),
                                  "held_for": at - c.opened_at})
            run.log.append(f"{at:%m-%d %H:%MZ} displaced {r.concern_id} (quiet "
                           f"{(at - c.last_evidence_at).total_seconds() / 3600:.1f} h): {r.note}")

    i = 0
    while now <= end:
        while i < len(events) and events[i][0] <= now:
            at, _, kind, ev = events[i]
            i += 1
            before_event = state
            cid = ev.get("concern_id")
            if kind == "settle":
                state, _ = settle_concern(state, cid, arm="treated", produced=True, now=at, note="as live")
            elif kind == "drop":
                gone = next((c for c in state.concerns if c.concern_id == cid), None)
                if gone is not None:
                    state = close_concern(state, gone, end="dropped", now=at, note="dropped unresolved: as live")
                    state = state.model_copy(update={"concerns": [c for c in state.concerns if c.concern_id != cid]})
            elif kind == "thought":
                # Prime looked at its concerns (when it last thought decides whether a quiet one can lose its slot).
                state = state.model_copy(update={"last_thought_at": at})
            elif kind == "ask_open":
                state = touch_concern(state, cid, at)
                state = state.model_copy(update={"open_nominations": [*state.open_nominations, PrimeNominationV1(
                    act_id=f"act_{ev['opened_at']}", thought_id="t", focus_id="f", question="q", why="w",
                    arm="treated", opened_at=at, concern_id=cid)]})
            elif kind == "ask_close":
                state = state.model_copy(update={"open_nominations": [
                    n for n in state.open_nominations if n.act_id != f"act_{ev['opened_at']}"]})
            elif kind == "take_on":
                if any(c.concern_id == cid for c in state.concerns):
                    taken_on.add(cid)
            elif kind == "take_off":
                taken_on.discard(cid)
            elif kind == "promise":
                state, turns, [turn] = hb.fold_turn_signals(state, turns, _signals(ev), at,
                                                            taken_on=frozenset(taken_on), contest=contest)
                corr = ev["correlation_id"]
                if turn.opened_concern_id:
                    run.promise_open[corr] = "at once"
                else:
                    run.promise_open[corr] = "waiting" if contest else "refused"
                run.log.append(f"{at:%m-%d %H:%MZ} promise {corr[:8]} ({ev['who']}): "
                               f"{'opened ' + turn.opened_concern_id if turn.opened_concern_id else turn.thread_refused}")
            elif kind == "open":
                concern = PrimeConcernV1(concern_id=cid, text=f"{ev['kind']} {cid}", event_ref=ev["event_ref"],
                                         opened_at=at, last_evidence_at=at,
                                         evidence_salience=ev.get("salience"))
                if contest:
                    state, seated = slots.seat_or_wait(state, concern, at)
                elif len(state.concerns) < MAX_CONCERNS:
                    state = state.model_copy(update={"concerns": [*state.concerns, concern]})
                    seated = True
                else:
                    seated = False
                run.log.append(f"{at:%m-%d %H:%MZ} {ev['kind']} concern {cid}: {'held' if seated else 'no room'}")
            elif kind == "letter":  # noqa: SIM114 -- one branch per event kind
                day = date.fromisoformat(ev["letter_date"])
                threads = [f"thread {k}" for k in range(ev["threads"])]
                state, _opened, _ = thread_concerns(state, day, threads, at)
                contest_pass(at)
                mine = {thread_concern_id(day, k) for k in range(ev["threads"])}
                held = sum(1 for c in state.concerns if c.concern_id in mine)
                run.held_per_letter[ev["letter_date"]] = held
                run.log.append(f"{at:%m-%d %H:%MZ} letter {day}: {held} held, "
                               f"{sum(1 for w in state.waiting if w.concern_id in mine)} waiting")
            track(before_event, state, at)
            if not contest:
                # As live: a concern with no room was forgotten (no contest pass ever reads the list).
                state = state.model_copy(update={"waiting": []})
        for cid in taken_on:
            state = touch_concern(state, cid, now)
        before = {c.concern_id for c in state.concerns}
        state = observe(state, PrimeSnapshotV1(), now)
        for cid in before - {c.concern_id for c in state.concerns}:
            run.faded.append(cid)
            run.log.append(f"{now:%m-%d %H:%MZ} concern {cid} faded")
        before_tick = state
        contest_pass(now)
        track(before_tick, state, now)
        run.max_waiting = max(run.max_waiting, len(state.waiting))
        now += TICK
    run.log.sort()
    run.final = state
    return run


def main() -> int:
    fx = json.loads(FIXTURE.read_text())
    off = replay(fx, contest=False)
    on = replay(fx, contest=True)
    live_held = {lt["letter_date"]: lt["live_held"] for lt in fx["letters"]}
    live_opened = {p["correlation_id"] for p in fx["promises"] if p["live"] == "opened"}
    first = fx["promises"][0]["correlation_id"]
    promise_cids = {hb.turn_concern_id(p["correlation_id"]) for p in fx["promises"]}
    still_waiting = {w.concern_id for w in on.final.waiting}

    checks = {
        "fidelity": (off.held_per_letter == live_held
                     and {c for c, how in off.promise_open.items() if how == "at once"} == live_opened
                     and sorted(off.faded) == sorted(fx["live"]["faded"])),
        "threads_held": all(n >= 1 for n in on.held_per_letter.values()) and len(on.held_per_letter) == 6,
        "promise_3656": on.promise_open.get(first) == "at once" and any(
            f"by {hb.turn_concern_id(first)}" in d["note"] for d in on.displaced),
        "no_promise_lost": (not (set(on.expired) & promise_cids)
                            and all(how in ("at once", "later") or hb.turn_concern_id(c) in still_waiting
                                    for c, how in on.promise_open.items())),
        "no_churn": all(d["intake"] or (d["quiet"] >= d["half_life"] and not d["question_open"] and d["seen"])
                        for d in on.displaced),
        "never_a_promise": not any(d["promise"] for d in on.displaced),
        "notes": all(d["note"].startswith("displaced: by ") and " held vs " in d["note"] for d in on.displaced),
    }
    settled_live = {x["concern_id"]: _t(x["at"]) for x in fx["settles"]}
    cost = sorted(d["concern_id"] for d in on.displaced
                  if d["concern_id"] in settled_live and settled_live[d["concern_id"]] > d["at"])
    for name, run in (("contest OFF (as live, evidence clock on)", off), ("contest ON", on)):
        print(f"{name}:")
        for line in run.log:
            print("  " + line)
        opened = sum(1 for how in run.promise_open.values() if how in ("at once", "later"))
        print(f"  threads held per letter {run.held_per_letter}")
        print(f"  promises opened {opened} of {len(run.promise_open)} "
              f"({sum(1 for h in run.promise_open.values() if h == 'at once')} at once); "
              f"displaced {len(run.displaced)}; faded {len(run.faded)}; waiting expired {len(run.expired)}; "
              f"waiting list max {run.max_waiting}, at end {len(run.final.waiting)}")
    quick = [d["concern_id"] for d in on.displaced if d["held_for"] < timedelta(hours=1)]
    print(f"churn watch: {len(quick)} displacement(s) within 1 h of the concern getting its slot: "
          f"{', '.join(quick) or 'none'} (promise intake has no quiet rule)")
    print(f"cost: {len(cost)} concern(s) the ON arm displaced went on to settle live, after the displacement: "
          f"{', '.join(cost) or 'none'} (Prime asked about each later; the ON arm cannot show what it would have "
          "asked instead)")
    print("checks:")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
