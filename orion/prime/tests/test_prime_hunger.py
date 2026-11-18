"""Prime hunger (integration design Part B, orion/prime/hunger.py): one test per rule."""

from __future__ import annotations

import math
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.prime import hunger as hg
from orion.prime.state import (CONCERN_TTL, PROMISE_CONCERN_TTL, PrimeConcernV1, PrimeNeedsStateV1, PrimeStateV1,
                               observe, PrimeSnapshotV1, wants_to_think)

T = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
REFRACTORY = timedelta(seconds=300)
REPO = Path(__file__).resolve().parents[3]


def concern(cid="a" * 16, *, kept=T, opened=None, event_ref=None, graded=None):
    return PrimeConcernV1(concern_id=cid, text="cabinet heat", opened_at=opened or kept, last_evidence_at=kept,
                          event_ref=event_ref, graded=graded)


def juniper(gaps_h, *, last_ago, now=T):
    """Message times: conversations separated by ``gaps_h``, the last one ``last_ago`` before ``now``."""
    t = now - last_ago
    times = [t]
    for g in reversed(gaps_h):
        t = t - timedelta(hours=g)
        times.insert(0, t)
    return {"message_times": times, "last_chat_id": "c1", "last_prompt": "are you there?", "last_event_id": "chat_turn:c1"}


def surprise(nats, p90=0.0123, n=138, eid="prediction_surprise:x"):
    return {"event_id": eid, "organ": "prediction", "source_kind": "prediction_surprise", "label": "forecast crossed",
            "surprise_nats": nats, "source_p90": p90, "source_n": n}


# --- rest point 0 ---------------------------------------------------------------------------------------------


def test_every_drive_is_zero_at_rest():
    # horizon: a concern kept just now
    assert hg.horizon_needs(PrimeStateV1(concerns=[concern()]), T) == []
    assert hg.horizon_drive(timedelta(0), CONCERN_TTL) == 0.0
    # surprise: the live calm shape -- every prediction_surprise row at the same 0.0123 nats (its own p90)
    assert hg.surprise_drive(0.0123, 0.0123, 138) == 0.0
    # connection: she wrote 30 minutes ago (a pause inside a conversation, not a gap between them)
    assert hg.connection_drive(juniper([12.0] * 30, last_ago=timedelta(minutes=30))["message_times"], T) == 0.0
    # and nothing at all wakes Prime
    state = PrimeStateV1(concerns=[concern()], last_thought_at=T - timedelta(hours=1))
    needs = hg.compute_needs(state, T, {"surprises": [surprise(0.0123)],
                                        "juniper": juniper([12.0] * 30, last_ago=timedelta(minutes=30))})
    assert needs == [] and hg.pick_wake(needs, state, T) is None


def test_the_quiet_floor_is_deleted_not_left_unused():
    """Kill means kill: no QUIET_FLOOR symbol anywhere in code (docs may tell its history)."""
    out = subprocess.run(["git", "grep", "-n", "QUIET_FLOOR", "--", "*.py", "*.yaml", "*.yml", "*_example"],
                         cwd=REPO, capture_output=True, text=True, check=False)
    hits = [h for h in out.stdout.splitlines() if not h.startswith("orion/prime/tests/test_prime_hunger.py")]
    assert hits == []
    import orion.prime.state as st

    assert not hasattr(st, "QUIET_FLOOR")
    quiet = observe(PrimeStateV1(seen_binding=[], last_thought_at=T), PrimeSnapshotV1(binding=[], body_ok=True),
                    T + timedelta(hours=30))
    assert quiet.pending_changes == []


# --- each drive -----------------------------------------------------------------------------------------------


def test_horizon_counts_from_half_and_reaches_the_threshold_at_three_quarters():
    assert hg.horizon_drive(CONCERN_TTL * 0.5, CONCERN_TTL) == 0.0
    assert hg.horizon_drive(CONCERN_TTL * 0.75, CONCERN_TTL) == pytest.approx(0.5)
    assert hg.horizon_drive(CONCERN_TTL, CONCERN_TTL) == 1.0
    assert hg.horizon_drive(CONCERN_TTL * 2, CONCERN_TTL) == 1.0
    state = PrimeStateV1(concerns=[concern(kept=T - timedelta(hours=18))])
    [n] = hg.compute_needs(state, T, None)
    assert n.name == "horizon" and n.drive == pytest.approx(0.5) and n.ref == f"prime_concern:{'a' * 16}"
    assert "fades unless I keep it in 6 h" in n.content and n.recover_sec == CONCERN_TTL.total_seconds()


def test_horizon_uses_the_promise_ttl_and_the_adapter_deadline_and_skips_settled_and_taken_on():
    promise = concern("b" * 16, kept=T - timedelta(hours=18), event_ref="turn:corr")
    assert hg.horizon_needs(PrimeStateV1(concerns=[promise]), T) == []  # 18 h of 72 h
    deadline = T + timedelta(hours=1)
    [n] = hg.horizon_needs(PrimeStateV1(concerns=[promise.model_copy(update={"opened_at": T - timedelta(hours=11)})]),
                           T, horizon_of=lambda c: deadline)
    assert n[0] == "horizon" and n[1] == pytest.approx((11 / 12 - 0.5) / 0.5) and "ends in 1 h" in n[2]
    assert n[2].startswith("promise ") and n[4] == PROMISE_CONCERN_TTL.total_seconds()
    old = concern(kept=T - timedelta(hours=20))
    assert hg.horizon_needs(PrimeStateV1(concerns=[old]), T, taken_on={old.concern_id}) == []
    assert hg.horizon_needs(PrimeStateV1(concerns=[old.model_copy(update={"graded": "settled"})]), T) == []


def test_surprise_needs_above_its_own_p90_and_at_least_ln2_and_enough_history():
    assert hg.surprise_drive(math.log(2), 0.0123, 138) == pytest.approx(0.5)
    assert hg.surprise_drive(0.22, 0.0022, 61) == 0.0  # live max action_outcome 7 days: above p90, under ln 2
    assert hg.surprise_drive(4.2, 0.0123, 19) == 0.0  # under 20 past events: no reference
    assert hg.surprise_drive(4.2, 5.0, 138) == 0.0  # not above its own source's usual
    assert hg.surprise_drive(4.2, 0.0123, 138) == pytest.approx(1 - math.exp(-4.2))
    [n] = hg.compute_needs(PrimeStateV1(), T, {"surprises": [surprise(4.2)]})
    assert n.name == "surprise" and n.experience_id == "prediction_surprise:x"
    assert "[experience:prediction_surprise:x]" in n.content


def test_connection_is_her_gap_against_her_own_usual_gap():
    gaps = [float(h) for h in range(3, 43)]  # 40 gaps between conversations, median ~23 h
    assert hg.connection_drive(juniper(gaps, last_ago=timedelta(hours=23))["message_times"], T) == pytest.approx(0.5)
    assert hg.connection_drive(juniper(gaps, last_ago=timedelta(hours=100))["message_times"], T) == 1.0
    # Under 20 gaps between conversations: unknown (None), never calm and never a need.
    assert hg.connection_drive(juniper([5.0] * 19, last_ago=timedelta(hours=99))["message_times"], T) is None
    # Pauses under 2 h are inside one conversation: they are not her gaps.
    assert hg.conversation_gaps([T, T + timedelta(minutes=30), T + timedelta(hours=3)]) == [timedelta(hours=2, minutes=30)]
    [n] = hg.compute_needs(PrimeStateV1(), T, {"juniper": juniper(gaps, last_ago=timedelta(hours=30))})
    # The ref is the message (stable for the whole silence); the citable experience rides alongside.
    assert n.name == "connection" and n.ref == "chat_history_log:c1" and n.experience_id == "chat_turn:c1"
    assert "'are you there?'" in n.content and "[experience:chat_turn:c1]" in n.content
    j = {**juniper(gaps, last_ago=timedelta(hours=30)), "last_event_id": None}
    [m] = hg.compute_needs(PrimeStateV1(), T, {"juniper": j})
    assert m.ref == n.ref and m.key == n.key and m.experience_id is None and "not yet an experience" in m.content


def test_total_hunger_is_the_strongest_need_not_a_sum():
    weak = [concern(f"{i:016x}", kept=T - timedelta(hours=17)) for i in range(3)]  # each ~0.42
    needs = hg.compute_needs(PrimeStateV1(concerns=weak), T, None)
    assert len(needs) == 3 and max(n.damped for n in needs) < hg.THRESHOLD
    assert hg.pick_wake(needs, PrimeStateV1(concerns=weak), T) is None


# --- the wake ----------------------------------------------------------------------------------------------------


def test_a_need_crossing_the_threshold_wakes_prime_with_no_pending_change():
    state = PrimeStateV1(concerns=[concern(kept=T - timedelta(hours=17, minutes=50))], last_thought_at=T - timedelta(hours=1))
    assert hg.pick_wake(hg.compute_needs(state, T, None), state, T) is None
    later = T + timedelta(minutes=10)
    wake = hg.pick_wake(hg.compute_needs(state, later, None), state, later)
    assert wake is not None and wake.name == "horizon"
    assert not wants_to_think(state, later, refractory=REFRACTORY)
    assert wants_to_think(state, later, refractory=REFRACTORY, need_wake=True)
    # The refractory interval and a back-off still hold a need wake.
    assert not wants_to_think(state.model_copy(update={"last_thought_at": later - timedelta(seconds=60)}), later,
                              refractory=REFRACTORY, need_wake=True)
    assert not wants_to_think(state.model_copy(update={"retry_not_before": later + timedelta(minutes=1)}), later,
                              refractory=REFRACTORY, need_wake=True)


def test_guard1_a_thought_that_changes_nothing_halves_the_need_and_it_recovers_over_its_horizon():
    state = PrimeStateV1(concerns=[concern(kept=T - timedelta(hours=24) + timedelta(minutes=1))])
    [n] = hg.compute_needs(state, T, None)
    assert n.damped == pytest.approx(n.drive)
    assert hg.after_answer(state, [n], T, acted=True) is state  # it acted: nothing halved
    damped = hg.after_answer(state, [n], T, acted=False)
    assert hg.damp_factor(damped.needs, "horizon", T) == pytest.approx(0.5)
    assert hg.damp_factor(damped.needs, "horizon", T + CONCERN_TTL / 2) == pytest.approx(0.75)
    assert hg.damp_factor(damped.needs, "horizon", T + CONCERN_TTL) == 1.0
    # Halved again before it recovered: compounding on the current factor.
    twice = hg.after_answer(damped, [n], T, acted=False)
    assert hg.damp_factor(twice.needs, "horizon", T) == pytest.approx(0.25)
    # Damping is per need name: a surprise is not damped by a horizon no-op.
    assert hg.damp_factor(damped.needs, "surprise", T) == 1.0
    # Survives a restart (persisted in Prime's state).
    back = PrimeStateV1.model_validate_json(damped.model_dump_json())
    assert hg.damp_factor(back.needs, "horizon", T) == pytest.approx(0.5)


def test_guard4_the_daily_budget_holds_need_wakes_but_never_a_change():
    state = PrimeStateV1(concerns=[concern(kept=T - timedelta(hours=23))], last_thought_at=T - timedelta(hours=1))
    for i in range(hg.DAILY_BUDGET):
        state = hg.note_thought(state, T - timedelta(hours=23) + timedelta(minutes=50 * i), None)
    needs = hg.compute_needs(state, T, None)
    assert needs and needs[0].damped >= hg.THRESHOLD
    assert hg.budget_spent(state.needs, T) and hg.pick_wake(needs, state, T) is None
    changed = state.model_copy(update={"pending_changes": ["an act of mine settled"]})
    assert wants_to_think(changed, T, refractory=REFRACTORY)
    # The window slides: an hour later the oldest thoughts are out of it.
    later = T + timedelta(hours=1)
    assert not hg.budget_spent(state.needs, later)


def test_guard5_the_same_content_cannot_wake_twice_unless_its_drive_rose():
    state = PrimeStateV1(concerns=[concern(kept=T - timedelta(hours=19))])
    [n] = hg.compute_needs(state, T, None)
    fired = hg.note_thought(state, T, n)
    assert hg.pick_wake([n], fired, T + timedelta(minutes=10)) is None
    rose_a_little = n.model_copy(update={"drive": n.drive + hg.MIN_RISE / 2, "damped": n.drive + hg.MIN_RISE / 2})
    assert hg.pick_wake([rose_a_little], fired, T) is None
    rose = n.model_copy(update={"drive": n.drive + hg.MIN_RISE, "damped": n.drive + hg.MIN_RISE})
    assert hg.pick_wake([rose], fired, T) is not None
    # A different content line of the same need is not blocked.
    other = n.model_copy(update={"ref": "prime_concern:" + "c" * 16})
    assert hg.pick_wake([other], fired, T) is not None


def test_guard5_rearms_from_the_trough_but_an_unread_source_never_rearms():
    state = PrimeStateV1(concerns=[concern(kept=T - timedelta(hours=19))])
    [n] = hg.compute_needs(state, T, None)
    fired = hg.note_thought(state, T, n)
    # Kept: the need is gone (drive 0) -> the line's record falls to 0, so a later rise to 0.5 may wake Prime again.
    rearmed = hg.track_troughs(fired, [])
    assert rearmed.needs.fired[n.key].drive == 0.0 and hg.pick_wake([n], rearmed, T) is not None
    # A connection line whose source was not read is unknown, not calm: its record stays.
    c = hg.NeedV1(name="connection", drive=0.8, damped=0.8, content="x", ref="chat_history_log:c1", recover_sec=3600)
    cf = hg.note_thought(PrimeStateV1(), T, c)
    assert hg.track_troughs(cf, [], hunger_read=False).needs.fired[c.key].drive == 0.8
    assert hg.track_troughs(cf, []).needs.fired[c.key].drive == 0.0
    # A surprise's surprisal is fixed: leaving the read (cited, aged out, cut by the cap) never re-arms it.
    sp = hg.NeedV1(name="surprise", drive=0.9, damped=0.9, content="y", ref="experience:p:1", recover_sec=86400)
    sf = hg.note_thought(PrimeStateV1(), T, sp)
    assert hg.track_troughs(sf, []).needs.fired[sp.key].drive == 0.9
    # An event want's line falls only when the event wants were read.
    ev = hg.NeedV1(name="horizon", drive=0.5, damped=0.5, content="z", ref="curiosity_run:r1", recover_sec=3600)
    ef = hg.note_thought(PrimeStateV1(), T, ev)
    assert hg.track_troughs(ef, [], events_read=False).needs.fired[ev.key].drive == 0.5
    assert hg.track_troughs(ef, []).needs.fired[ev.key].drive == 0.0


def test_a_planted_never_falling_need_is_dampened_and_bounded():
    """The spec's danger: a need that never falls (a deadline that cannot be met). Same content, never acted on."""
    state, t, thoughts = PrimeStateV1(last_thought_at=T), T, 0
    while t < T + timedelta(days=7):
        need = hg.NeedV1(name="horizon", drive=1.0, damped=hg.damp_factor(state.needs, "horizon", t),
                         content="a promise that cannot be kept", ref="prime_concern:" + "f" * 16, recover_sec=86400)
        state = hg.track_troughs(state, [need])
        wake = hg.pick_wake([need], state, t)
        if wake is not None and wants_to_think(state, t, refractory=REFRACTORY, need_wake=True):
            thoughts += 1
            state = hg.after_answer(hg.note_thought(state, t, wake), [need], t, acted=False)
            state = state.model_copy(update={"last_thought_at": t})
        t += timedelta(minutes=1)
    assert thoughts == 1
    assert state.needs.damp["horizon"].factor == pytest.approx(0.5)


# --- the trace and the frame --------------------------------------------------------------------------------------


def test_the_trace_is_stable_and_parseable():
    n = hg.NeedV1(name="surprise", drive=0.9, damped=0.9, content="x", ref="experience:prediction_surprise:a",
                  recover_sec=86400)
    note = hg.woke_prefix(n) + " | concerns +0 -0 kept 1 | no move"
    assert note.startswith("woke: need:surprise experience:prediction_surprise:a")
    assert hg.parse_woke_need(note) == ("surprise", "experience:prediction_surprise:a")
    assert hg.parse_woke_need("woke: first wake: I have not thought yet | no move") is None
    # A need and a change together: the ref stays clean.
    mixed = hg.woke_prefix(n) + " (also: an act of mine settled) | no move"
    assert hg.parse_woke_need(mixed) == ("surprise", "experience:prediction_surprise:a")
    assert hg.parse_woke_need("woke: need:horizon prime_concern:abc; x | y") == ("horizon", "prime_concern:abc")
    assert hg.parse_woke_need(None) is None


def test_frame_lines_show_content_not_numbers_and_mark_the_wake():
    state = PrimeStateV1(concerns=[concern(kept=T - timedelta(hours=20))])
    needs = hg.compute_needs(state, T, None)
    lines = hg.frame_lines(needs, needs[0])
    assert lines[0].startswith("- [horizon] concern 'cabinet heat'") and lines[0].endswith("(this is what woke you)")
    assert "drive" not in " ".join(lines) and f"{needs[0].drive:.2f}" not in " ".join(lines)
    assert hg.frame_lines([], None) == ["(nothing is pulling at you right now)"]
    assert "could not be read: boom" in hg.frame_lines([], None, unread="boom")[-1]


def test_needs_state_is_omitted_until_used_and_round_trips():
    assert "needs" not in PrimeStateV1().model_dump_json()
    s = hg.note_thought(PrimeStateV1(), T, None)
    assert '"needs"' in s.model_dump_json()
    assert PrimeStateV1.model_validate_json(s.model_dump_json()).needs == s.needs
    assert isinstance(s.needs, PrimeNeedsStateV1)
    # prune drops what has recovered and what left the budget window
    pruned = hg.prune(s, T + timedelta(days=2))
    assert pruned.needs.thoughts_at == []


def test_an_unheld_funded_event_want_is_offered_once_before_it_expires():
    """Event mode: funded event wants are Prime's only concern source. With the floor gone, a short-window event
    could expire between thoughts; the horizon need offers it once, at three quarters of its window."""
    want = {"want_id": "w", "family": "hub.curiosity_finding", "event_ref": "curiosity_run:r1",
            "decided_at": (T - timedelta(hours=9)).isoformat(), "expires_at": (T + timedelta(hours=3)).isoformat(),
            "label": "a philosophical finding about time", "salience": 0.95}
    events = {"wants": [want]}
    [n] = hg.compute_needs(PrimeStateV1(), T, None, events=events)
    assert n.name == "horizon" and n.drive == hg.EVENT_DRIVE_CAP and n.ref == "curiosity_run:r1"
    assert "[event_id curiosity_run:r1]" in n.content and "in 3 h" in n.content
    # Capped: as it rises it is never re-offered (guard 5 sees no rise).
    fired = hg.note_thought(PrimeStateV1(), T, n)
    [later] = hg.compute_needs(fired, T + timedelta(hours=2), None, events=events)
    assert later.drive == hg.EVENT_DRIVE_CAP and hg.pick_wake([later], fired, T + timedelta(hours=2)) is None
    # Half its window in: nothing yet. Already held as a concern: nothing.
    early = {"wants": [{**want, "decided_at": (T - timedelta(hours=6)).isoformat(),
                        "expires_at": (T + timedelta(hours=6)).isoformat()}]}
    assert hg.compute_needs(PrimeStateV1(), T, None, events=early) == []
    held = PrimeStateV1(concerns=[concern(event_ref="curiosity_run:r1")])
    assert [x for x in hg.compute_needs(held, T, None, events=events) if x.ref == "curiosity_run:r1"] == []


def test_a_top_tenth_funded_event_wakes_prime_at_exactly_the_band():
    want = {"want_id": "w", "family": "hub.world_pulse_read", "event_ref": "world_pulse_read_seed:reading:x",
            "decided_at": (T - timedelta(hours=9)).isoformat(), "expires_at": (T + timedelta(hours=3)).isoformat(),
            "label": "a read finished", "salience": hg.EVENT_WAKE_SALIENCE}
    [n] = hg.compute_needs(PrimeStateV1(), T, None, events={"wants": [want]})
    assert n.name == "horizon" and n.ref == "world_pulse_read_seed:reading:x"


@pytest.mark.parametrize("salience", [0.0, 0.09, 0.89, None])
def test_an_ordinary_funded_event_does_not_wake_prime(salience):
    """Live 11-16..17: 11 of 12 thoughts were woken by funded reading wants of salience ~0 and all ended 'no move'.
    Below its source's top tenth an event waits in the frame; it does not wake Prime."""
    want = {"want_id": "w", "family": "hub.world_pulse_read", "event_ref": "world_pulse_read_seed:reading:x",
            "decided_at": (T - timedelta(hours=9)).isoformat(), "expires_at": (T + timedelta(hours=3)).isoformat(),
            "label": "a read finished", "salience": salience}
    assert hg.compute_needs(PrimeStateV1(), T, None, events={"wants": [want]}) == []


def test_guard1_a_need_the_thought_answered_is_not_halved():
    """Keeping a concern is the answer to its horizon need; citing a surprise answers it."""
    h = hg.NeedV1(name="horizon", drive=0.6, damped=0.6, content="x", ref=f"prime_concern:{'a' * 16}", recover_sec=86400)
    s = hg.NeedV1(name="surprise", drive=0.9, damped=0.9, content="y", ref="experience:p:1", recover_sec=86400)
    touched = hg.answered_refs(kept=["a" * 16], cited=["p:1"])
    assert touched == {f"prime_concern:{'a' * 16}", "experience:p:1"}
    assert hg.after_answer(PrimeStateV1(), [h, s], T, acted=False, touched=touched).needs.damp == {}
    only_h = hg.after_answer(PrimeStateV1(), [h, s], T, acted=False, touched=hg.answered_refs(kept=["a" * 16]))
    assert set(only_h.needs.damp) == {"surprise"}
    assert hg.answered_refs(dropped=[f"{'b' * 16} (no longer matters)"]) == {f"prime_concern:{'b' * 16}"}


def _reading(i: int, decided: datetime, window=timedelta(hours=3)) -> dict:
    # salience 0.95: a top-tenth reading, so the case still holds if a salience gate on event wakes lands (#3559).
    return {"want_id": f"w{i}", "family": "hub.world_pulse_read", "event_ref": f"world_pulse_read_seed:reading:{i}",
            "salience": 0.95, "decided_at": decided.isoformat(), "expires_at": (decided + window).isoformat(),
            "label": f"a read finished: {i}"}


def test_guard1_kept_concerns_do_not_shield_an_unanswered_event_need_issue_3552():
    """#3552, live 11-16..17: Prime held 3 concerns and kept all 3 on every thought. A new reading want (event
    horizon need, capped at 0.5 = the threshold) woke Prime 11 times in 9 h, each "kept 3 | no move". Keeping a
    concern answered its own horizon need, and the whole name ``horizon`` counted as answered, so the reading want
    that woke the thought was never halved. Guard 5 cannot hold it: every reading has a new ref."""
    cids = [ch * 16 for ch in "abc"]
    # The live shape: each concern was kept just now, but its adapter deadline counts from when it opened, so its
    # horizon need is shown (drive 0.2) on every thought.
    held = [concern(c, kept=T, opened=T - timedelta(days=3)) for c in cids]
    deadline = {c.concern_id: c.opened_at + timedelta(days=5) for c in held}
    horizon_of = lambda c: deadline[c.concern_id]  # noqa: E731
    state = PrimeStateV1(concerns=held, last_thought_at=T - timedelta(hours=1))
    first = _reading(1, T - timedelta(hours=2, minutes=15))  # three quarters of its 3 h window: drive at the cap
    needs = hg.compute_needs(state, T, None, events={"wants": [first]}, horizon_of=horizon_of)
    wake = hg.pick_wake(needs, state, T)
    assert wake is not None and wake.ref == first["event_ref"] and wake.drive == hg.EVENT_DRIVE_CAP
    shown = hg.shown_needs(needs, wake)
    assert {n.ref for n in shown} == {first["event_ref"], *(f"prime_concern:{c}" for c in cids)}
    state = hg.after_answer(hg.note_thought(state, T, wake), shown, T, acted=False,
                            touched=hg.answered_refs(kept=cids))
    # The unanswered reading is halved (under its own event family), the kept concerns' horizon is not.
    assert set(state.needs.damp) == {"horizon:event:hub.world_pulse_read"}
    assert hg.damp_factor(state.needs, "horizon", T) == 1.0
    # The next reading (a new ref) reaches the cap at the next tick past the refractory interval: no wake.
    later = T + timedelta(minutes=10)
    second = _reading(2, later - timedelta(hours=2, minutes=15))
    needs2 = hg.compute_needs(state, later, None, events={"wants": [first, second]}, horizon_of=horizon_of)
    nxt = [n for n in needs2 if n.ref == second["event_ref"]]
    assert nxt and nxt[0].drive == hg.EVENT_DRIVE_CAP and nxt[0].damped < hg.THRESHOLD
    assert hg.pick_wake(needs2, state, later) is None
    # A kept concern's own horizon need is not damped: it reads at its full drive.
    for n in needs2:
        if n.ref.startswith("prime_concern:"):
            assert n.damped == pytest.approx(n.drive) and n.drive > 0
    # Once per recovery window: after the reading's own 3 h window, a new reading may wake Prime again.
    back = T + timedelta(hours=3)
    third = _reading(3, back - timedelta(hours=2, minutes=15))
    needs3 = hg.compute_needs(state, back, None, events={"wants": [third]}, horizon_of=horizon_of)
    assert (w := hg.pick_wake(needs3, state, back)) is not None and w.ref == third["event_ref"]


def test_guard1_an_event_family_damp_does_not_touch_other_families_and_old_state_still_loads():
    """The halving is kept per event family: an ignored reading says nothing about an office visit. An older saved
    state (damp keyed by need name only) still loads and still damps concern horizon needs."""
    reading = hg.NeedV1(name="horizon", drive=0.5, damped=0.5, content="r", ref="world_pulse_read_seed:reading:1",
                        recover_sec=3 * 3600, damp_as=hg.event_damp_key("hub.world_pulse_read"))
    state = hg.after_answer(PrimeStateV1(), [reading], T, acted=False)
    assert set(state.needs.damp) == {"horizon:event:hub.world_pulse_read"}
    visit = {"want_id": "v", "family": "vision.office_visit", "event_ref": "office_visit:1",
             "decided_at": (T - timedelta(minutes=90)).isoformat(), "expires_at": (T + timedelta(minutes=30)).isoformat(),
             "label": "someone in the office", "salience": 0.95}
    [n] = hg.compute_needs(state, T, None, events={"wants": [visit]})
    assert n.damped == n.drive == hg.EVENT_DRIVE_CAP and n.damp_key == "horizon:event:vision.office_visit"
    old = PrimeStateV1.model_validate({"needs": {"damp": {"horizon": {"factor": 0.5, "at": T.isoformat(),
                                                                       "recover_sec": 86400.0}}}})
    [c] = hg.compute_needs(old.model_copy(update={"concerns": [concern(kept=T - timedelta(hours=20))]}), T, None)
    assert c.damp_key == "horizon" and c.damped == pytest.approx(c.drive * 0.5)
    back = PrimeStateV1.model_validate_json(state.model_dump_json())
    assert back.needs.damp == state.needs.damp


def test_prime_own_thought_bid_is_never_an_event_need():
    """Review finding: Prime's own bid listed as a funded event want would wake Prime about itself, forever."""
    own = {"want_id": "prime-think-1", "family": "prime", "event_ref": "prime_need:horizon:prime_concern:x",
           "decided_at": (T - timedelta(hours=1)).isoformat(), "expires_at": (T + timedelta(minutes=1)).isoformat()}
    assert hg.compute_needs(PrimeStateV1(), T, None, events={"wants": [own]}) == []
    assert hg.compute_needs(PrimeStateV1(), T, None, events={"wants": [{**own, "family": "other"}]}) == []


def test_the_need_that_woke_is_always_shown_first():
    needs = [hg.NeedV1(name="horizon", drive=0.9, damped=0.9, content=f"c{i}", ref=f"prime_concern:{i:016x}",
                       recover_sec=86400) for i in range(6)]
    woke = hg.NeedV1(name="surprise", drive=0.6, damped=0.6, content="s", ref="experience:p:9", recover_sec=86400)
    shown = hg.shown_needs([*needs, woke], woke)
    assert shown[0] == woke and len(shown) == hg.MAX_SHOWN
    assert hg.frame_lines([*needs, woke], woke)[0].endswith("(this is what woke you)")


# --- learning (choice-as-EFE Step 1; orion-prediction GET /learning-progress) ------------------------------------


def learning_read(direction, p_value, sid="office.temp_c", recent=0.34, prior=0.11):
    return {"series": [{"series_id": sid, "status": "ok" if direction else "insufficient", "direction": direction,
                        "p_value": p_value, "learning_progress": recent - prior,
                        "recent": {"start": "2026-11-09", "end": "2026-11-15", "mean_skill": recent},
                        "prior": {"start": "2026-11-02", "end": "2026-11-08", "mean_skill": prior}}]}


def test_learning_flat_or_insufficient_is_rest_drive_zero():
    st = PrimeStateV1()
    assert hg.compute_needs(st, T, None, learning=learning_read("flat", 0.07)) == []
    assert hg.compute_needs(st, T, None, learning=learning_read(None, None)) == []
    assert hg.compute_needs(st, T, None, learning=None) == []
    assert hg.learning_drive("flat", 0.001) == 0.0
    assert hg.learning_drive("better", None) == 0.0


def test_learning_real_move_wakes_once_with_content():
    st = PrimeStateV1()
    needs = hg.compute_needs(st, T, None, learning=learning_read("better", 0.02))
    assert len(needs) == 1
    n = needs[0]
    assert n.name == "learning" and n.ref == "learning:office.temp_c:better"
    assert n.damp_key == "learning:office.temp_c"
    assert n.drive == pytest.approx(0.98)
    assert "office temperature forecasts got better" in n.content and "0.11" in n.content and "0.34" in n.content
    assert hg.pick_wake(needs, st, T) == n
    assert hg.parse_woke_need(hg.woke_prefix(n) + "; no move") == ("learning", "learning:office.temp_c:better")
    # Guard 5: the same move cannot wake Prime again while it stays up.
    st2 = hg.note_thought(st, T, n)
    assert hg.pick_wake(hg.compute_needs(st2, T, None, learning=learning_read("better", 0.001)), st2, T) is None
    # A flip to the opposite direction is a new line and may wake Prime.
    flip = hg.compute_needs(st2, T, None, learning=learning_read("worse", 0.01, recent=0.1, prior=0.34))
    assert hg.pick_wake(flip, st2, T) is not None


def test_learning_worse_is_a_need_too():
    n = hg.compute_needs(PrimeStateV1(), T, None, learning=learning_read("worse", 0.01, recent=0.1, prior=0.3))[0]
    assert "got worse" in n.content and n.drive > 0.95


def test_learning_rearms_only_after_a_read_flat_not_after_an_unread():
    st = hg.note_thought(PrimeStateV1(), T, hg.compute_needs(PrimeStateV1(), T, None,
                                                               learning=learning_read("better", 0.02))[0])
    key = "learning:learning:office.temp_c:better"
    unread = hg.track_troughs(st, [], learning_read=False)
    assert unread.needs.fired[key].drive > 0.9
    flat = hg.track_troughs(st, hg.compute_needs(st, T, None, learning=learning_read("flat", 0.7)), learning_read=True)
    assert flat.needs.fired[key].drive == 0.0
    later = T + timedelta(days=10)
    again = hg.compute_needs(flat, later, None, learning=learning_read("better", 0.02))
    assert hg.pick_wake(again, flat, later) is not None


def test_learning_unread_line_says_unknown_not_calm():
    lines = hg.frame_lines([], None, learning_unread="URLError: refused")
    assert any("could not be read" in x and "forecasts" in x for x in lines)


def test_learning_series_are_damped_apart():
    # A thought shown cabinet's move that changes nothing must not hold office's wake.
    st = PrimeStateV1()
    cab = hg.compute_needs(st, T, None, learning=learning_read("better", 0.02, sid="athena.cabinet_temp_c"))
    st = hg.after_answer(hg.note_thought(st, T, cab[0]), cab, T, acted=False)
    both = {"series": learning_read("better", 0.02, sid="athena.cabinet_temp_c")["series"]
            + learning_read("worse", 0.01)["series"]}
    needs = {n.ref: n for n in hg.compute_needs(st, T, None, learning=both)}
    assert needs["learning:athena.cabinet_temp_c:better"].damped < 0.5
    assert needs["learning:office.temp_c:worse"].damped == needs["learning:office.temp_c:worse"].drive
