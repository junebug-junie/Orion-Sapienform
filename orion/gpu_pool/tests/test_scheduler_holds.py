"""Stage 4.3 scheduling rules (H1-H4, S1-S2 in orion/gpu_pool/scheduler.py): durable-run holds,
their child calls, shared gaps, hold recall, and the preconditions on loading a swap seat.
Spec: docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md."""
from __future__ import annotations

from datetime import timedelta

from orion.gpu_pool.scheduler import (
    Abort, CardLive, Grant, Recall, SwapBlocked, SwapLoad, SwapUnload, Unavailable, schedule,
)
from orion.gpu_pool.tests.test_scheduler import CFG, COLD, T0, cards, grants, lease, live, of, run

CLEAR = {"thermal": None}
AFTER = CFG.swap_after_wait_sec("agent-gpu2")


def hold(status="queued", role=None, priority="background", **kw):
    return lease("agent", status, role, priority=priority, kind="hold", retryable=True, **kw)


def child(hold_id, status="queued", role=None, priority="background", **kw):
    return lease("agent", status, role, priority=priority, hold_lease_id=hold_id, **kw)


# --- H1: holds are placed like leases, one per role --------------------------------------
def test_hold_is_granted_home_first():
    h = hold(lease_id="h")
    assert grants(run([h])) == {"h": "agent"}


def test_at_most_one_hold_per_role_second_goes_elsewhere_or_waits():
    h1 = hold("granted", "agent", lease_id="h1")
    h2 = hold(lease_id="h2")
    # agent-gpu2 is not loaded and gpu0 is not lent: the second hold waits, never shares agent.
    assert grants(run([h1, h2])) == {}
    swapped = cards(gpu2=CardLive("gpu2", swapped_in={"agent-gpu2"}))
    assert grants(run([h1, h2], crds=swapped)) == {"h2": "agent-gpu2"}


def test_hold_waits_for_a_busy_slot_rather_than_preempting_it():
    busy = lease("agent", "granted", "agent")
    h = hold(lease_id="h")
    assert grants(run([busy, h])) == {}


# --- H2: a run's own calls never queue behind the run ------------------------------------
def test_child_uses_its_holds_slot_without_a_second_one():
    h = hold("granted", "agent", lease_id="h")
    c = child("h", lease_id="c")
    assert grants(run([h, c])) == {"c": "agent"}


def test_child_jumps_the_roles_queue_ahead_of_higher_priority_work():
    h = hold("granted", "agent", lease_id="h")
    c = child("h", lease_id="c", age=0)
    older_interactive = lease("agent", priority="interactive", lease_id="i", age=100)
    got = grants(run([h, c, older_interactive]))
    assert got == {"c": "agent"}   # the interactive call waits one inference, the run never does


def test_hold_plus_active_child_is_one_slot_not_two():
    h = hold("granted", "agent", lease_id="h")
    c = child("h", "granted", "agent", lease_id="c")
    sys_call = lease("agent", priority="interactive", lease_id="s")
    assert "s" not in grants(run([h, c, sys_call]))
    c2 = child("h", lease_id="c2")
    assert grants(run([h, c, c2])) == {}   # its own second call waits for the first, on the same slot


def test_child_of_a_lost_hold_is_unavailable_not_queued_forever():
    c = child("gone", lease_id="c")
    assert [(u.lease_id, u.reason) for u in of(Unavailable, run([c]))] == [("c", "hold_not_granted")]
    released_hold = hold("released", "agent", lease_id="h")
    c2 = child("h", lease_id="c2")
    assert ("c2", "hold_not_granted") in [(u.lease_id, u.reason) for u in of(Unavailable, run([released_hold, c2]))]


def test_child_still_runs_while_its_hold_is_recalled():
    h = hold("recalling", "agent", lease_id="h", recall_by=T0 + timedelta(seconds=300))
    c = child("h", lease_id="c")
    assert grants(run([h, c])) == {"c": "agent"}


# --- H3: gaps are shared with STRICTLY higher priority -----------------------------------
def test_higher_priority_call_uses_the_gap_between_the_runs_calls():
    h = hold("granted", "agent", lease_id="h")          # background, no child in flight
    s = lease("agent", priority="system", lease_id="s")
    assert grants(run([h, s])) == {"s": "agent"}


def test_equal_or_lower_priority_never_interleaves():
    h = hold("granted", "agent", lease_id="h", priority="system")
    same = lease("agent", priority="system", lease_id="same")
    low = lease("agent", priority="background", lease_id="low")
    assert grants(run([h, same, low])) == {}


def test_a_hold_never_interleaves_another_hold():
    h = hold("granted", "agent", lease_id="h", priority="background")
    other = hold(lease_id="o", priority="interactive")
    assert grants(run([h, other])) == {}


def test_after_an_interleaver_the_runs_next_call_waits_one_inference_then_goes_first():
    h = hold("granted", "agent", lease_id="h")
    s = lease("agent", "granted", "agent", priority="system", lease_id="s")   # interleaved
    c = child("h", lease_id="c")
    s2 = lease("agent", priority="interactive", lease_id="s2", age=50)
    assert grants(run([h, s, c, s2])) == {}             # slot busy: both wait
    assert grants(run([h, c, s2])) == {"c": "agent"}     # s released: the run's call goes first


def test_interleaver_does_not_take_the_gap_of_a_multi_slot_role_twice():
    roles = live(agent=live()["agent"].__class__("agent", True, 2, 131072, False))
    h = hold("granted", "agent", lease_id="h")
    s1 = lease("agent", priority="system", lease_id="s1", age=2)
    s2 = lease("agent", priority="system", lease_id="s2", age=1)
    # 2 slots, one reserved by the idle hold: the free one plus the gap -> both system calls fit.
    assert grants(run([h, s1, s2], roles=roles)) == {"s1": "agent", "s2": "agent"}
    b = lease("agent", priority="background", lease_id="b")
    assert grants(run([h, b], roles=roles)) == {"b": "agent"}   # the truly free slot, not the gap
    b2 = lease("agent", priority="background", lease_id="b2", age=1)
    assert grants(run([h, b, b2], roles=roles)) == {"b2": "agent"}   # oldest first, one slot


# --- H4: hold recall --------------------------------------------------------------------
def test_hold_recall_uses_the_hold_grace_then_aborts():
    crds = cards(gpu0=CardLive("gpu0", lent=True))
    h = hold("granted", "chat", lease_id="h", granted_at=T0 - timedelta(seconds=10))
    owner = lease("chat", priority="interactive", lease_id="o")
    # gpu0 still lent; the chat owner waits -> the borrowing hold is recalled with the HOLD grace
    decisions = run([h, owner], crds=crds)
    [r] = of(Recall, decisions)
    assert r.lease_id == "h" and r.reason == "owner_waiting"
    assert r.recall_by == T0 + timedelta(seconds=CFG.defaults.hold_clawback_grace_sec)
    assert CFG.defaults.hold_clawback_grace_sec == 600
    # the owner uses the gap meanwhile (interactive > background)
    assert grants(decisions) == {"o": "chat"}
    recalling = hold("recalling", "chat", lease_id="h", recall_by=T0 - timedelta(seconds=1))
    assert [a.lease_id for a in of(Abort, run([recalling, owner], crds=crds))] == ["h"]


def test_children_are_never_recalled_on_their_own():
    crds = cards(gpu0=CardLive("gpu0", lent=False))    # unlent: borrowers on chat are recalled
    h = hold("granted", "chat", lease_id="h")
    c = child("h", "granted", "chat", lease_id="c")
    assert {r.lease_id for r in of(Recall, run([h, c], crds=crds))} == {"h"}


def test_a_long_system_hold_on_the_home_seat_is_never_capped():
    # Only background holds have a clock (H6); the home seat itself has no max_hold_sec.
    h = hold("granted", "agent", lease_id="h", priority="system", granted_at=T0 - timedelta(hours=8))
    assert not of(Recall, run([h]))


# --- H6: a background hold is taken back at defaults.background_max_hold_sec (#3391) -------
CAP = CFG.defaults.background_max_hold_sec


def test_the_background_hold_cap_ships_on_above_orions_own_longest_hold():
    # 11-05..11-15, 1,650 durable-run holds: max 8,545 s. The trial that prompted this: 25,126 s.
    assert CAP == 9000 and 8545 < CAP < 25126


def test_a_background_hold_is_recalled_at_the_cap_and_not_before():
    under = hold("granted", "agent", lease_id="h", granted_at=T0 - timedelta(seconds=CAP - 1))
    assert not of(Recall, run([under]))
    at = hold("granted", "agent", lease_id="h", granted_at=T0 - timedelta(seconds=CAP))
    [r] = of(Recall, run([at]))
    assert (r.lease_id, r.reason) == ("h", "background_max_hold")
    assert r.recall_by == T0 + timedelta(seconds=CFG.defaults.hold_clawback_grace_sec)


def test_the_cap_is_for_any_work_class_and_a_hold_nobody_retries():
    # The 11-13 trial: a plain background hold on the agent lane, not retryable, 7 h long.
    trial = lease("operator_agent", "granted", "agent", priority="background", kind="hold",
                  lease_id="trial", granted_at=T0 - timedelta(seconds=25126))
    assert [(r.lease_id, r.reason) for r in of(Recall, run([trial]))] == [("trial", "background_max_hold")]


def test_the_capped_hold_is_aborted_after_its_grace_and_the_waiting_run_gets_the_seat():
    # A hold nobody retries (the 11-13 trial): the abort ends it, so it is not in the next tick.
    waiting = hold(lease_id="q")
    recalling = lease("operator_agent", "recalling", "agent", priority="background", kind="hold", lease_id="h",
                      reason="background_max_hold", granted_at=T0 - timedelta(seconds=CAP + 600),
                      recall_by=T0 - timedelta(seconds=1))
    decisions = run([recalling, waiting])
    assert [a.lease_id for a in of(Abort, decisions)] == ["h"]
    assert grants(run([waiting])) == {"q": "agent"}


def test_a_capped_retryable_hold_keeps_its_place_in_the_queue():
    # A durable run cut mid-node is re-queued in place (lease_graph: same created_at), and the queue
    # is priority then age. So it wins the seat back from a YOUNGER background hold; anything of
    # higher priority goes first. The cap frees the seat for others, it does not send Orion's own
    # run to the back.
    capped = hold(lease_id="capped", age=CAP + 1000)
    younger = hold(lease_id="younger", age=3000)
    assert grants(run([capped, younger])) == {"capped": "agent"}
    chat_turn = lease("agent", priority="interactive", lease_id="i", age=10)
    assert grants(run([capped, chat_turn])) == {"i": "agent"}


def test_operator_and_non_background_holds_are_not_capped_by_it():
    long_ago = T0 - timedelta(seconds=CAP * 3)
    operator = lease("agent", "granted", "agent", priority="background", kind="hold", operator=True,
                     lease_id="op", granted_at=long_ago)
    interactive = hold("granted", "agent", lease_id="i", priority="interactive", granted_at=long_ago)
    assert not of(Recall, run([operator])) and not of(Recall, run([interactive]))


def test_no_cap_when_the_default_is_none():
    from orion.gpu_pool.config import Defaults

    assert Defaults.model_validate({"background_max_hold_sec": None}).background_max_hold_sec is None
    cfg = CFG.model_copy(update={"defaults": CFG.defaults.model_copy(update={"background_max_hold_sec": None})})
    h = hold("granted", "agent", lease_id="h", granted_at=T0 - timedelta(hours=8))
    assert not of(Recall, run([h], cfg=cfg))


def test_seat_loaded_past_max_hold_drains_recalls_then_unloads():
    assert CFG.roles["agent-gpu2"].max_hold_sec == 9000   # ~2.5 h, Juniper 2026-09-29 (stage 5)
    loaded = cards(gpu2=CardLive("gpu2", swapped_in={"agent-gpu2"}, loaded_at=T0 - timedelta(seconds=9000)))
    h = hold("granted", "agent-gpu2", lease_id="h")
    q = hold(lease_id="q")
    decisions = run([hold("granted", "agent", lease_id="home"), h, q], crds=loaded, cfg=COLD)
    assert [(r.lease_id, r.reason) for r in of(Recall, decisions)] == [("h", "max_hold")]
    assert "q" not in grants(decisions)                 # nothing new lands on a draining seat
    assert not of(SwapUnload, decisions)                # busy: unload waits for the hold to leave
    assert [(u.role, u.reason) for u in of(SwapUnload, run([], crds=loaded, cfg=COLD))] == [("agent-gpu2", "max_hold")]


def test_observed_seat_without_loaded_at_is_never_max_held():
    observed = cards(gpu2=CardLive("gpu2", swapped_in={"agent-gpu2"}))
    # Granted 2 h ago: under the background hold cap (H6), so only the seat clock is in play.
    h = hold("granted", "agent-gpu2", lease_id="h", granted_at=T0 - timedelta(hours=2))
    assert not of(Recall, run([h], crds=observed))


# --- S1: seat-load preconditions ---------------------------------------------------------
def _demand():
    return [lease("agent", "granted", "agent"),
            hold(lease_id="w", queued_since=T0 - timedelta(seconds=AFTER + 1))]


def test_after_wait_sec_per_seat_keeps_todays_1200s_trigger():
    assert AFTER == 1200
    early = [lease("agent", "granted", "agent"), hold(lease_id="w", queued_since=T0 - timedelta(seconds=AFTER - 1))]
    assert not of(SwapLoad, schedule(CFG, live(), cards(), early, T0, guards=CLEAR))
    assert [s.role for s in of(SwapLoad, schedule(CFG, live(), cards(), _demand(), T0, guards=CLEAR))] == ["agent-gpu2"]


def test_min_residency_blocks_reload_and_says_so():
    crds = cards(gpu2=CardLive("gpu2", residency_until=T0 + timedelta(seconds=10)))
    decisions = schedule(CFG, live(), crds, _demand(), T0, guards=CLEAR)
    assert not of(SwapLoad, decisions)
    assert [(b.role, b.reason) for b in of(SwapBlocked, decisions)] == [("agent-gpu2", "min_residency")]


def test_cooldown_after_failed_load_is_reported():
    crds = cards(gpu2=CardLive("gpu2", cooldown_until=T0 + timedelta(seconds=10)))
    assert [b.reason for b in of(SwapBlocked, schedule(CFG, live(), crds, _demand(), T0, guards=CLEAR))] == ["cooldown"]


def test_guards_block_loading_by_name_and_fail_closed():
    hot = {"thermal": "hot"}
    [b] = of(SwapBlocked, schedule(CFG, live(), cards(), _demand(), T0, guards=hot))
    assert (b.reason, b.detail) == ("guard:thermal", "hot")
    unread: dict = {}   # a guard the caller never read fails closed
    assert [b.reason for b in of(SwapBlocked, schedule(CFG, live(), cards(), _demand(), T0, guards=unread))] \
        == ["guard:thermal"]
    # stage 5.4: an overdue image baseline no longer blocks the 27B load (visual_baseline deleted)
    assert CFG.roles["agent-gpu2"].swap.guards == ["thermal"]


def test_no_block_reported_without_demand():
    crds = cards(gpu2=CardLive("gpu2", residency_until=T0 + timedelta(seconds=10)))
    assert not of(SwapBlocked, schedule(CFG, live(), crds, [], T0, guards={"thermal": "hot"}))


# --- S2: swap state gates grants ----------------------------------------------------------
def test_fault_card_grants_nothing_on_any_of_its_roles():
    crds = cards(gpu2=CardLive("gpu2", swap_state="fault", swap_role="agent-gpu2"))
    w = lease("world", lease_id="w")
    d = lease("diffusion", lease_id="d")
    assert grants(run([w, d], crds=crds)) == {}
    assert grants(run([lease("fast", lease_id="f")], crds=crds)) == {"f": "fast"}   # other cards unaffected


def test_mid_load_blocks_the_seat_and_its_evictions_but_not_other_residents():
    crds = cards(gpu2=CardLive("gpu2", swap_state="loading", swap_role="agent-gpu2"))
    assert grants(run([lease("world", lease_id="w"), lease("diffusion", lease_id="d")], crds=crds)) == {"w": "world"}
    assert not of(SwapLoad, schedule(CFG, live(), crds, _demand(), T0, guards=CLEAR))


# --- review findings (2026-09-25) --------------------------------------------------------------
def test_retrying_child_of_a_gone_hold_ends_instead_of_crashing_the_tick():
    c = child("gone", "retry_wait", lease_id="c", not_before=T0 - timedelta(seconds=1))
    assert [(u.lease_id, u.reason) for u in of(Unavailable, run([c]))] == [("c", "hold_not_granted")]


def test_a_hold_never_takes_a_second_slot_for_its_calls():
    roles = live(metacog=live()["metacog"].__class__("metacog", True, 4, 4096, False))
    h = lease("metacog", "granted", "metacog", kind="hold", lease_id="h", priority="background")
    c1 = lease("metacog", "granted", "metacog", hold_lease_id="h", lease_id="c1", priority="background")
    c2 = lease("metacog", hold_lease_id="h", lease_id="c2", priority="background")
    owner = lease("metacog", priority="interactive", lease_id="o", age=100)
    got = grants(run([h, c1, c2, owner], roles=roles))
    assert "c2" not in got and got["o"] == "metacog"


def test_a_hold_does_not_borrow_a_role_its_owner_is_using():
    roles = live(chat=live()["chat"].__class__("chat", True, 2, 65536, True))
    crds = cards(gpu0=CardLive("gpu0", lent=True))
    busy_agent = lease("agent", "granted", "agent")
    owner = lease("chat", "granted", "chat", priority="interactive")
    h = hold(lease_id="h")
    decisions = run([busy_agent, owner, h], roles=roles, crds=crds)
    assert "h" not in grants(decisions) and not of(Recall, decisions)


# --- 4.5 finding: an unloaded seat has no live context; min_ctx holds must still load it -------
def _big_demand():
    return [lease("agent", "granted", "agent"),
            hold(lease_id="w", min_ctx_tokens=32768, queued_since=T0 - timedelta(seconds=AFTER + 1))]


def _unloaded_seat():
    return live(**{"agent-gpu2": live()["agent-gpu2"].__class__("agent-gpu2", False, 0, None, None)})


def test_min_ctx_hold_loads_the_seat_from_its_last_seen_context():
    decisions = schedule(CFG, _unloaded_seat(), cards(), _big_demand(), T0,
                         seen_ctx={"agent-gpu2": 131072}, guards=CLEAR)
    assert [(s.role, s.reason) for s in of(SwapLoad, decisions)] == [("agent-gpu2", "demand")]
    assert "w" not in grants(decisions)                   # placement still waits for live context


def test_min_ctx_hold_with_no_known_seat_context_says_so():
    decisions = schedule(CFG, _unloaded_seat(), cards(), _big_demand(), T0, seen_ctx={}, guards=CLEAR)
    assert not of(SwapLoad, decisions)
    assert [(b.role, b.reason, b.detail) for b in of(SwapBlocked, decisions)] == \
        [("agent-gpu2", "ctx_unknown", "min_ctx_tokens=32768")]


def test_a_seat_known_to_be_too_small_is_not_loaded_for_it():
    decisions = schedule(CFG, _unloaded_seat(), cards(), _big_demand(), T0,
                         seen_ctx={"agent-gpu2": 16384}, guards=CLEAR)
    assert not of(SwapLoad, decisions) and not of(SwapBlocked, decisions)
