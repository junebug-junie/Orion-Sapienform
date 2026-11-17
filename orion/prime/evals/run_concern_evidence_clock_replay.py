"""Eval: Prime's concern slots 11-14..11-17 replayed with and without the evidence clock (#3574).

Design: docs/superpowers/specs/2026-11-17-prime-concerns-decided-by-attention-design.md, acceptance check 1 (patch 1).

The fixture (``fixtures/concern_clock_live_2026-11-14_17.json``) is a read-only pull of what moved Prime's concern
slots in the window, from the live ledger:

* every Prime thought (``grammar_acts`` actor prime verb infer, its open time) and the concern it dropped (its close
  note's ``dropped <id>``). Every thought in the window kept every concern it held (the notes read ``kept N`` with N =
  all held), so the replay keeps all held concerns at every thought;
* the two promise concerns the turn hand-back opened (Prime's state ``concern_record``: opened_at, ``turn:<corr>``),
  and the day-letter turn that settled one (11-17 09:12Z);
* each day letter's threads (``orion_day_letter.carry_forward_md``, first words only: a thread's concern id is a
  hash of its day and index, never its text) and when Prime's tick opened them (the ``make`` act's close);
* sent questions (verb ask) and take-ons (``intention:`` make acts) about a concern: there were none (0 rows since
  11-14), so with the clock on nothing renews any concern in this window.

The replay applies the real pure functions (orion/prime/state.py ``observe`` every minute, orion/prime/thought.py
``apply_answer``, orion/prime/day_letter.py ``thread_concerns``, orion/prime/handback.py ``thread_refusal``,
orion/prime/acts.py ``settle_concern``) twice: clock off (a keep renews, as live) and clock on.

Limit: the model's live answers are held fixed. A model that reacts to the new prompt (keeping no longer renews) by
sending questions is not simulated, so ``no_evidence`` is true by construction here; it records that the live window
had no evidence source at all. ``promise_lives`` passes because the settle came ~51.5 h after the promise opened.

Checks (exit 1 on any failure):

* fidelity      -- clock off reproduces the live record: threads held per letter 2/0/0/1, the same 3 concern ids held
                   at the end, 0 faded (the replay is only worth reading if it gives back what happened);
* threads_fade  -- clock on: the two 11-13 threads fade 24 h after they opened (11-15 10:06Z), with no evidence;
* letter_1114   -- clock on: the 11-14 letter (11-15 07:33Z) still finds no room (it came before the fade; with
                   patch 1 alone there is no waiting list, so its threads are lost);
* letter_1115   -- clock on: the 11-15 letter (11-16 07:48Z) gets 2 threads held;
* letter_1116   -- clock on: the 11-16 letter gets 2 threads held (the 11-15 threads faded at 11-17 07:48Z and the
                   promise was settled and dropped at 09:12Z);
* promise_lives -- clock on: the 11-15 promise is not faded before its letter-turn settle (72 h from 05:42Z);
* no_evidence   -- clock on: 0 renewals in the window. Plainly: with today's evidence sources patch 1 is a 24 h timer
                   (Orion's review of #3610, point 1). That timer is what ends the jam.

Run: ``PYTHONPATH=. python orion/prime/evals/run_concern_evidence_clock_replay.py``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from orion.prime import handback as hb
from orion.prime.acts import settle_concern
from orion.prime.day_letter import thread_concern_id, thread_concerns
from orion.prime.state import PrimeConcernV1, PrimeSnapshotV1, PrimeStateV1, observe, touch_concern
from orion.prime.thought import PrimeAnswerV1, apply_answer

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "concern_clock_live_2026-11-14_17.json"
TICK = timedelta(minutes=1)


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s)


@dataclass
class Run:
    held_per_letter: dict[str, int] = field(default_factory=dict)
    faded: list[tuple[str, datetime]] = field(default_factory=list)
    renewals: int = 0
    held_at_end: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


def replay(fx: dict[str, Any], *, evidence_clock: bool) -> Run:
    events: list[tuple[datetime, int, str, dict]] = []
    # Order inside one instant: settle, thought, promise, letter (a settle at 09:12:13 precedes the thought at :14).
    for s in fx["settles"]:
        events.append((_t(s["at"]), 0, "settle", s))
    for th in fx["thoughts"]:
        events.append((_t(th["at"]), 1, "thought", th))
    for p in fx["promises"]:
        events.append((_t(p["at"]), 2, "promise", p))
    for m in [*fx["sent_moves"], *fx["take_ons"]]:
        events.append((_t(m["at"]), 3, "touch", m))
    for lt in fx["letters"]:
        events.append((_t(lt["opened_at"]), 4, "letter", lt))
    events.sort(key=lambda e: (e[0], e[1]))

    run = Run()
    now = _t(fx["start"])
    end = _t(fx["end"])
    state = PrimeStateV1(last_thought_at=now)
    turns = hb.PrimeTurnsStateV1()
    i = 0
    while now <= end:
        while i < len(events) and events[i][0] <= now:
            at, _, kind, ev = events[i]
            i += 1
            if kind == "settle":
                state, _line = settle_concern(state, ev["concern_id"], arm="treated", produced=True, now=at,
                                              note=ev["note"])
            elif kind == "thought":
                held = [c.concern_id for c in state.concerns]
                answer = PrimeAnswerV1.model_validate({
                    "reading": "replay",
                    "concerns": {"keep": [c for c in held if c not in ev["drop"]],
                                 "drop": [{"concern_id": d, "why": "as live"} for d in ev["drop"] if d in held]}})
                state = apply_answer(state, answer, now=at, arc_ids=[], focus_pending=False,
                                     evidence_clock=evidence_clock).state
            elif kind == "promise":
                cid = hb.turn_concern_id(ev["correlation_id"])
                assert cid == ev["concern_id"], f"turn_concern_id drifted: {cid} != {ev['concern_id']}"
                refused = hb.thread_refusal(state, turns, at)
                if refused:
                    run.log.append(f"{at:%m-%d %H:%MZ} promise {cid} refused: {refused}")
                    continue
                concern = PrimeConcernV1(concern_id=cid, text="a promise (replay)",
                                         event_ref=f"{hb.TURN_EVENT_PREFIX}{ev['correlation_id']}",
                                         opened_at=at, last_evidence_at=at)
                state = state.model_copy(update={"concerns": [*state.concerns, concern]})
                turns = turns.model_copy(update={"turn_concerns_opened_at": [*turns.turn_concerns_opened_at, at]})
            elif kind == "touch":
                before = state
                state = touch_concern(state, ev["concern_id"], at)
                run.renewals += state is not before
            elif kind == "letter":
                day = date.fromisoformat(ev["letter_date"])
                state, opened, no_room = thread_concerns(state, day, ev["threads"], at)
                run.held_per_letter[ev["letter_date"]] = len(opened)
                run.log.append(f"{at:%m-%d %H:%MZ} letter {day}: {len(opened)} held, {no_room} without room")
        before = {c.concern_id for c in state.concerns}
        state = observe(state, PrimeSnapshotV1(), now)
        for cid in before - {c.concern_id for c in state.concerns}:
            run.faded.append((cid, now))
            run.log.append(f"{now:%m-%d %H:%MZ} concern {cid} faded")
        now += TICK
    run.held_at_end = [c.concern_id for c in state.concerns]
    return run


def main() -> int:
    fx = json.loads(FIXTURE.read_text())
    off = replay(fx, evidence_clock=False)
    on = replay(fx, evidence_clock=True)
    live = fx["live"]
    ids = {lt["letter_date"]: lt for lt in fx["letters"]}
    t13 = {thread_concern_id(date(2026, 11, 13), 0), thread_concern_id(date(2026, 11, 13), 1)}
    fade_13 = {cid: at for cid, at in on.faded if cid in t13}
    promise = "438e69ed6bb3d798"
    checks = {
        "fidelity": (off.held_per_letter == live["held_per_letter"] and off.held_at_end == live["held_at_end"]
                     and len(off.faded) == live["faded"]),
        "threads_fade": (set(fade_13) == t13 and all(
            timedelta(hours=24) < at - _t(ids["2026-11-13"]["opened_at"]) <= timedelta(hours=24) + 2 * TICK
            for at in fade_13.values())),
        "letter_1114": on.held_per_letter.get("2026-11-14") == 0,
        "letter_1115": on.held_per_letter.get("2026-11-15") == 2,
        "letter_1116": on.held_per_letter.get("2026-11-16") == 2,
        "promise_lives": all(cid != promise for cid, _ in on.faded),
        "no_evidence": on.renewals == 0,
    }
    print("clock OFF (a keep renews; as live):")
    for line in off.log:
        print("  " + line)
    print(f"  held per letter {off.held_per_letter}; faded {len(off.faded)}; held at end {off.held_at_end}")
    print("clock ON (only evidence renews):")
    for line in on.log:
        print("  " + line)
    print(f"  held per letter {on.held_per_letter}; faded {len(on.faded)}; renewals {on.renewals}; "
          f"held at end {on.held_at_end}")
    print("checks:")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
