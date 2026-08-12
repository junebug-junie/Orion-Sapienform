"""Dispatch-arena starvation tests (2026-08-12).

The defect these cover was measured, not theorised. Over three live hours the
same five read-only templates held all five dispatch slots on essentially
every tick:

    inspect_bus_channel_catalog          DISPATCHED  267
    inspect_attended_target              DISPATCHED  267
    summarize_transport_contract_drift   DISPATCHED  267
    inspect_field_topology_catalog       DISPATCHED  243
    watch_transport_backpressure         DISPATCHED  209
    ...
    prune_build_cache   BLOCKED:max_dispatch_candidates_exceeded   4
    prune_build_cache   DISPATCHED                                 0

Three separate faults, three sections below:

  1. The loss was INVISIBLE -- every blocked record stored dispatch_kind
     "noop" regardless of what was actually blocked, so no query could tell a
     starved mutating action from a starved inspect. Finding the above
     required reconstructing the kind from a proposal_id string.
  2. Capacity was first-come-first-served within the frame, so a whole class
     of action could be crowded out permanently by a noisier one.
  3. Nothing was ever aged, so "lost this tick" and "lost for six hours"
     ranked identically forever.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

from orion.execution_dispatch.builder import (
    build_execution_dispatch_frame,
    effective_priority,
    starvation_key,
)
from orion.execution_dispatch.policy import (
    MAINTENANCE_SCOPE,
    DispatchLimitsV1,
    load_execution_dispatch_policy,
)
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"
NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def _load_store_module():
    """Same by-path load idiom as tests/test_execution_dispatch_runtime_store.py
    -- the service's app/ is not an importable package from repo root."""
    spec = importlib.util.spec_from_file_location(
        "execution_dispatch_runtime_store_for_starvation_tests",
        REPO / "services" / "orion-execution-dispatch-runtime" / "app" / "store.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _candidate(
    name: str,
    *,
    kind: str = "inspect",
    priority: float = 0.9,
    target: str | None = None,
) -> ProposalCandidateV1:
    return ProposalCandidateV1(
        proposal_id=f"proposal:{name}:tick:none",
        proposal_kind=kind,
        title=name,
        description=name,
        target_id=target or f"target:{name}",
        target_kind="system",
        priority_score=priority,
        urgency_score=priority,
        confidence_score=0.9,
        risk_score=0.2,
        reversibility_score=1.0,
        proposed_effect="increase_observability" if kind != "maintain" else "preserve_stability",
        required_policy_gate="operator_review",
    )


def _decision(candidate: ProposalCandidateV1, idx: int) -> PolicyDecisionV1:
    return PolicyDecisionV1(
        decision_id=f"decision:{idx}",
        proposal_id=candidate.proposal_id,
        decision="approved_maintenance" if candidate.proposal_kind == "maintain" else "approved_read_only",
        policy_gate="operator_review",
        risk_score=candidate.risk_score,
        reversibility_score=candidate.reversibility_score,
        confidence_score=candidate.confidence_score,
    )


def _build(candidates: list[ProposalCandidateV1], *, prev_counts=None, policy=None):
    policy = policy or load_execution_dispatch_policy(POLICY_PATH)
    decisions = [_decision(c, i) for i, c in enumerate(candidates)]
    policy_frame = PolicyDecisionFrameV1(
        frame_id="policy.frame:1",
        generated_at=NOW,
        source_proposal_frame_id="proposal.frame:1",
        overall_risk=0.2,
        decisions=decisions,
    )
    proposal_frame = ProposalFrameV1(
        frame_id="proposal.frame:1",
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        source_attention_frame_id="none",
        overall_action_pressure=0.9,
        overall_risk=0.2,
        candidates=candidates,
    )
    return build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal_frame,
        field_tick_id="tick",
        policy=policy,
        now=NOW,
        override_dispatch_mode="dispatch_read_only",
        prev_starvation_counts=prev_counts,
    )


# --- 1. the loss is visible -------------------------------------------------


def test_a_starved_candidate_records_what_it_actually_was():
    """The regression test for the fault that hid all the others.

    Before this patch every blocked record stored dispatch_kind="noop", so a
    starved MUTATING action and a starved inspect were byte-identical in the
    one column that would have distinguished them.
    """
    crowd = [_candidate(f"inspect_{i}", priority=0.95) for i in range(6)]
    prune = _candidate("prune_build_cache", kind="maintain", priority=0.30)
    # No reservation here: this test is about the RECORD, not about winning.
    policy = load_execution_dispatch_policy(POLICY_PATH)
    policy.limits.reserved_slots_by_scope = {}
    frame = _build(crowd + [prune], policy=policy)

    starved = [b for b in frame.blocked_candidates if b.source_proposal_id == prune.proposal_id]
    assert starved, "the low-priority maintain candidate should have lost on capacity"
    assert starved[0].dispatch_kind == "maintain", (
        "a starved maintenance action must be identifiable as one in the stored "
        "record -- recording it as 'noop' is what made this defect invisible"
    )
    assert starved[0].starvation_ticks == 1
    assert any(r.startswith("starved_consecutive_ticks:") for r in starved[0].reasons)


def test_reasons_zero_stays_the_old_string_so_existing_queries_survive():
    crowd = [_candidate(f"inspect_{i}", priority=0.95) for i in range(8)]
    frame = _build(crowd)
    starved = [b for b in frame.blocked_candidates if "max_dispatch" in b.reasons[0]]
    assert starved
    assert starved[0].reasons[0] == "max_dispatch_candidates_exceeded"


def test_the_frame_says_out_loud_that_it_truncated():
    crowd = [_candidate(f"inspect_{i}", priority=0.95) for i in range(8)]
    frame = _build(crowd)
    assert any(w.startswith("dispatch_starved kind=inspect") for w in frame.warnings), (
        f"a cap that truncates silently reads as 'covered everything': {frame.warnings}"
    )


def test_persistent_starvation_escalates_to_its_own_warning():
    crowd = [_candidate(f"inspect_{i}", priority=0.99) for i in range(6)]
    loser = _candidate("loser", priority=0.10)
    prev = {starvation_key(loser): 40}
    frame = _build(crowd + [loser], prev_counts=prev)
    assert any(w.startswith("dispatch_starvation_persistent") for w in frame.warnings)


# --- 2. the reserved lane ---------------------------------------------------


def test_a_maintenance_action_wins_a_slot_against_a_full_read_only_arena():
    """The live shape: five higher-priority read-only templates, one low
    maintenance action. Before the reservation the maintenance action lost
    every tick forever, which is the measured behaviour above."""
    crowd = [_candidate(f"inspect_{i}", priority=0.90) for i in range(5)]
    prune = _candidate("prune_build_cache", kind="maintain", priority=0.30)
    frame = _build(crowd + [prune])

    admitted = {c.source_proposal_id for c in frame.candidates}
    assert prune.proposal_id in admitted, (
        "a reserved maintenance slot must survive a fully-occupied read-only arena"
    )
    assert len(frame.candidates) == 5, "the reservation must not raise total capacity"


def test_an_unclaimed_reservation_is_reclaimed_not_wasted():
    """A reservation that costs throughput on every idle tick is not a fix.
    With no maintenance candidate present, all five slots still go to work."""
    crowd = [_candidate(f"inspect_{i}", priority=0.90) for i in range(8)]
    frame = _build(crowd)
    assert len(frame.candidates) == 5
    assert all(c.dispatch_kind != "maintain" for c in frame.candidates)


def test_the_reservation_does_not_widen_what_may_fire():
    """The reserved lane removes a capacity race, not a safety gate. With
    mutating dispatch off, a reserved maintenance candidate is still blocked."""
    policy = load_execution_dispatch_policy(POLICY_PATH)
    policy.mode.allow_mutating_dispatch = False
    prune = _candidate("prune_build_cache", kind="maintain", priority=0.99)
    frame = _build([prune], policy=policy)

    assert not frame.candidates
    blocked = frame.blocked_candidates[0]
    assert blocked.reasons == ["route_scope_not_read_only"]
    assert blocked.blocked_by == [MAINTENANCE_SCOPE]


def test_the_shipped_config_actually_reserves_a_maintenance_slot():
    """Config truth is not runtime truth, but a config that never reserved
    anything would make every test above a test of nothing."""
    policy = load_execution_dispatch_policy(POLICY_PATH)
    assert policy.limits.reserved_slots_by_scope.get(MAINTENANCE_SCOPE) == 1
    reserved_total = sum(policy.limits.reserved_slots_by_scope.values())
    assert reserved_total < policy.limits.max_dispatch_candidates, (
        "reserving every slot would starve the read-only lane instead -- the "
        "same defect pointed the other way"
    )


# --- 3. aging ---------------------------------------------------------------


def test_aging_is_bounded_and_cannot_invert_a_large_priority_gap():
    limits = DispatchLimitsV1(
        starvation_aging_bonus_per_tick=0.002,
        starvation_aging_bonus_cap=0.25,
    )
    assert effective_priority(priority_score=0.5, starved_ticks=0, limits=limits) == 0.5
    assert effective_priority(priority_score=0.5, starved_ticks=50, limits=limits) == 0.6
    # Cap holds no matter how long it loses.
    assert effective_priority(priority_score=0.5, starved_ticks=10_000, limits=limits) == 0.75
    assert effective_priority(priority_score=0.5, starved_ticks=10_000, limits=limits) < 0.99


def test_aging_eventually_wins_a_slot_for_a_persistent_loser():
    crowd = [_candidate(f"inspect_{i}", priority=0.80) for i in range(6)]
    loser = _candidate("loser", priority=0.70)

    fresh = _build(crowd + [loser])
    assert loser.proposal_id not in {c.source_proposal_id for c in fresh.candidates}

    aged = _build(crowd + [loser], prev_counts={starvation_key(loser): 100})
    assert loser.proposal_id in {c.source_proposal_id for c in aged.candidates}, (
        "starvation must be finite: a candidate that keeps losing has to get in eventually"
    )


def test_counters_advance_for_losers_and_reset_for_winners():
    crowd = [_candidate(f"inspect_{i}", priority=0.95) for i in range(5)]
    loser = _candidate("loser", priority=0.10)
    frame = _build(crowd + [loser], prev_counts={starvation_key(loser): 7})

    assert frame.starvation_counts[starvation_key(loser)] == 8
    for winner in crowd:
        assert starvation_key(winner) not in frame.starvation_counts, (
            "a candidate that got dispatched is not starving; keeping its key would "
            "let a winner accrue an aging bonus it never earned"
        )


def test_the_counter_map_stays_bounded_to_what_is_currently_starving():
    """This repo has shipped the unbounded version of exactly this shape
    twice (the evidence_event_ids caps). A key for something no longer being
    proposed must drop out rather than accumulate forever."""
    crowd = [_candidate(f"inspect_{i}", priority=0.95) for i in range(8)]
    stale = {"maintain:some:target:that:is:gone": 900, "inspect:vanished": 12}
    frame = _build(crowd, prev_counts=stale)
    assert "maintain:some:target:that:is:gone" not in frame.starvation_counts
    assert "inspect:vanished" not in frame.starvation_counts


def test_two_templates_sharing_a_kind_and_target_do_not_share_a_counter():
    """Regression test for a live no-op bug, found in this patch's own frames
    minutes after deploy. These two are real, from production:

        inspect_execution_pressure  inspect  capability:orchestration
        inspect_attended_target     inspect  capability:orchestration

    The first version keyed on `{proposal_kind}:{target_id}` alone, so the
    one that wins nearly every tick (avg rank 3.0) popped the key belonging
    to the one that starves (avg rank 7.5) -- and the candidate that most
    needed aging could never accrue any.
    """
    winner = _candidate("inspect_attended_target", priority=0.95, target="capability:orchestration")
    loser = _candidate("inspect_execution_pressure", priority=0.10, target="capability:orchestration")
    assert starvation_key(winner) != starvation_key(loser)

    crowd = [_candidate(f"inspect_other_{i}", priority=0.90) for i in range(4)]
    frame = _build(crowd + [winner, loser], prev_counts={starvation_key(loser): 5})

    assert winner.proposal_id in {c.source_proposal_id for c in frame.candidates}
    assert frame.starvation_counts[starvation_key(loser)] == 6, (
        "the admitted sibling must not reset a genuinely-starving candidate's counter"
    )


def test_a_malformed_proposal_id_degrades_aging_instead_of_crashing_dispatch():
    from orion.execution_dispatch.builder import proposal_template_key  # noqa: PLC0415

    assert proposal_template_key("proposal:prune_build_cache:tick_x:attention.frame:tick_x") == (
        "prune_build_cache"
    )
    # attention_frame_id contains colons -- splitting from the right would
    # pick up the wrong segment, which is why this splits from the left.
    assert proposal_template_key("proposal:t:tick:a:b:c:d") == "t"
    for bad in ("", "nonsense", "proposal:", "proposal::tick", "other:t:tick"):
        assert proposal_template_key(bad) is None

    odd = _candidate("x").model_copy(update={"proposal_id": "nonsense"})
    assert starvation_key(odd).startswith("kind:")
    frame = _build([odd])  # must not raise
    assert frame.frame_id


def test_starvation_key_is_stable_across_ticks():
    """The whole mechanism depends on this. proposal_id embeds the
    field_tick_id, so keying on it would make every tick a first offence and
    no counter would ever exceed 1."""
    tick_a = _candidate("prune_build_cache", kind="maintain", target="host:docker_build_cache")
    tick_b = tick_a.model_copy(
        update={"proposal_id": "proposal:prune_build_cache:tick_DIFFERENT:attention"}
    )
    assert tick_a.proposal_id != tick_b.proposal_id
    assert starvation_key(tick_a) == starvation_key(tick_b)


def test_admitted_candidates_carry_how_long_they_waited():
    crowd = [_candidate(f"inspect_{i}", priority=0.80) for i in range(4)]
    loser = _candidate("loser", priority=0.70)
    frame = _build(crowd + [loser], prev_counts={starvation_key(loser): 33})
    got = [c for c in frame.candidates if c.source_proposal_id == loser.proposal_id]
    assert got and got[0].starvation_ticks == 33, (
        "'finally got in after 33 ticks' is the evidence that says whether aging "
        "works, and it is unrecoverable after the fact if only blocked rows carry it"
    )


def test_persisted_counters_survive_the_shapes_real_rows_actually_have():
    """Rows written before this field existed have no key at all, and this
    value only feeds an ordering bonus -- a malformed one must cost the bonus
    for a tick, not stall the dispatch runtime."""
    coerce_starvation_counts = _load_store_module()._coerce_starvation_counts

    assert coerce_starvation_counts(None) == {}
    assert coerce_starvation_counts({}) == {}
    assert coerce_starvation_counts("not a map") == {}
    assert coerce_starvation_counts({"maintain:x": 3}) == {"maintain:x": 3}
    # Postgres hands JSON numbers back as int already, but a hand-edited row
    # or an older writer can produce a string or a null.
    assert coerce_starvation_counts({"a": "4", "b": None, "c": 2}) == {"a": 4, "c": 2}


def test_admission_does_not_depend_on_the_order_candidates_arrive_in():
    """Determinism that is worth asserting. Calling the builder twice with the
    same list in the same process can only fail if the sort key is actively
    random -- the real risk is input-order sensitivity, so permute the input.
    """
    cands = [_candidate(f"tied_{i}", priority=0.5) for i in range(9)]
    baseline = [c.source_proposal_id for c in _build(cands).candidates]
    assert baseline
    for shift in range(1, len(cands)):
        rotated = cands[shift:] + cands[:shift]
        assert [c.source_proposal_id for c in _build(rotated).candidates] == baseline


def test_the_send_budget_cannot_silently_drop_an_admitted_candidate():
    """Guards the coupling that makes clearing counters on ADMISSION safe.

    _admit_candidates clears a candidate's starvation counter the moment it
    wins a slot, but worker._send_prepared_candidates takes candidates in
    order and can stop early. If the per-tick send budget were smaller than
    the admission budget, the LOWEST-ranked admitted candidate -- after
    aging, exactly the starved one this whole mechanism exists for -- would
    be dropped with its counter already zeroed, and would restart its climb
    forever while the frame recorded it as admitted.

    Equal today (5/5). This fails rather than letting them drift apart.
    """
    limits = load_execution_dispatch_policy(POLICY_PATH).limits
    assert limits.max_dispatches_per_tick >= limits.max_dispatch_candidates, (
        "lowering max_dispatches_per_tick below max_dispatch_candidates makes "
        "starvation accounting silently wrong -- move the counter reset to a "
        "confirmed send first (see the NOTE at builder.py's admitted .pop())"
    )


def test_a_full_multi_tick_loop_lets_a_persistent_loser_break_through():
    """Closed loop: every tick is fed the PREVIOUS tick's own
    starvation_counts, exactly as the worker does.

    This is the test that was missing when the collision bug shipped. Every
    other aging test hand-injects prev_counts, so none of them can catch a
    break in the counter LIFECYCLE -- only in the bonus arithmetic. Here the
    frame's real output is the next tick's real input, so a counter that
    fails to accumulate (collision, reset-by-sibling, dropped carry) shows up
    as the loser simply never getting in.
    """
    # No reservation: this must be AGING doing the work, not the lane. The
    # gap has to be inside the bonus cap for aging to be able to close it at
    # all -- 0.85 vs 0.70 is 0.15 against a 0.25 cap, so ~75 ticks.
    policy = load_execution_dispatch_policy(POLICY_PATH)
    policy.limits.reserved_slots_by_scope = {}
    gap = 0.15
    assert gap < policy.limits.starvation_aging_bonus_cap

    crowd = [_candidate(f"inspect_leader_{i}", priority=0.85) for i in range(5)]
    loser = _candidate(
        "prune_build_cache", kind="maintain", priority=0.85 - gap, target="host:cache"
    )

    counts: dict[str, int] = {}
    breakthrough = None
    streaks = []
    for tick in range(400):
        frame = _build(crowd + [loser], prev_counts=counts, policy=policy)
        counts = dict(frame.starvation_counts)
        streaks.append(counts.get(starvation_key(loser), 0))
        if loser.proposal_id in {c.source_proposal_id for c in frame.candidates}:
            breakthrough = tick
            break

    assert breakthrough is not None, (
        f"400 closed-loop ticks and the loser never got in; streak reached "
        f"{max(streaks)} -- a counter that stops accumulating is the failure mode "
        f"this test exists for"
    )
    assert streaks[:5] == [1, 2, 3, 4, 5], f"counters must accumulate, got {streaks[:8]}"
    expected = gap / policy.limits.starvation_aging_bonus_per_tick
    assert abs(breakthrough - expected) <= 2, (
        f"broke through at tick {breakthrough}, expected ~{expected:.0f} from "
        f"gap {gap} / {policy.limits.starvation_aging_bonus_per_tick} per tick"
    )


def test_aging_alone_cannot_close_a_gap_larger_than_its_cap():
    """The other half of the loop test: a bounded bonus must stay bounded
    under real iteration, not just in the arithmetic helper. A candidate more
    than `cap` below the cut line must starve forever -- if it ever breaks
    through, the bonus is being applied more than once or is not capped."""
    policy = load_execution_dispatch_policy(POLICY_PATH)
    policy.limits.reserved_slots_by_scope = {}
    cap = policy.limits.starvation_aging_bonus_cap

    crowd = [_candidate(f"inspect_leader_{i}", priority=0.95) for i in range(5)]
    hopeless = _candidate("hopeless", priority=0.95 - cap - 0.05)

    counts: dict[str, int] = {}
    for _ in range(500):
        frame = _build(crowd + [hopeless], prev_counts=counts, policy=policy)
        counts = dict(frame.starvation_counts)
        assert hopeless.proposal_id not in {c.source_proposal_id for c in frame.candidates}, (
            "aging must not be able to invert a gap wider than its own cap"
        )
    assert counts[starvation_key(hopeless)] == 500
