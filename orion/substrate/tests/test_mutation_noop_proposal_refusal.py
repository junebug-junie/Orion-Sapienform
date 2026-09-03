"""The routing proposal generator must read the live surface, not a constant
-- AND, as of 2026-09-03, must not generate routing proposals at all.

Regression cover for a confirmed-live defect (2026-09-03): the surface had
been at 0.58 since 2026-09-02T04:11:17, and the generator kept proposing 0.58
on top of it 5-6 times an hour for at least 36 hours -- ~190 identical
proposals, each burning a proposal -> decision -> adoption cycle and taking a
15-minute rollback lock for a change that could not occur.

PR #2058 (merged 2026-09-03) stops such a patch being ADOPTED. This module
covers the other half: stopping it being GENERATED, so the pipeline does not
spend a proposal, a trial and a decision to arrive at a block. That no-op
guard lives in `_routing_threshold_payloads()` and is still correct -- it is
exercised directly below, not through `ProposalFactory.plan_for_pressure()`
any more.

The proposal-level rollback of 0.50 was NOT reaching adoptions: commit
255f8252e (2026-09-02 23:55Z, already on main) overwrites it with the live
value inside PatchApplier.apply. Deriving it here still matters because
SubstrateTrialRunner._routing_baseline_threshold reads the PROPOSAL's rollback,
so trials were replaying a 0.58 candidate against a 0.50 baseline while live
was 0.58 -- untouched by #2058.

**2026-09-03, same session:** `plan_for_pressure()` now refuses every
"routing" pressure outright, unconditionally, before it ever reaches
`_routing_threshold_payloads()`. Confirmed live: (a) the evidence this
surface receives is graph-review telemetry that has nothing to do with what
`chat_reflective_lane_threshold` gates, and (b) with
`AUTO_ROUTER_LLM_ENABLED=false` in production, the lowest confidence any
heuristic routing decision can carry at `execution_depth >= 2` is 0.61 --
above the hardcoded patch target of 0.58 -- so `decision_confidence <
routing_threshold` can never fire at this target regardless of evidence.
See `orion/substrate/mutation_proposals.py`'s `_ROUTING_TARGET_PARKED_REASON`
and `project_routing_threshold_mutation_parked_2026-09-03.md`.

The no-op-guard tests below now call `_routing_threshold_payloads()`
directly to keep exercising that logic -- it is still correct and will be
needed again once the target is unparked. The `plan_for_pressure()`-level
tests assert the new park behavior instead.
"""

from __future__ import annotations

from orion.core.schemas.substrate_mutation import MutationPressureV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1
from orion.substrate.mutation_apply import PatchApplier
from orion.substrate.mutation_decision import DecisionEngine
from orion.substrate.mutation_detectors import MutationDetectors
from orion.substrate.mutation_monitor import PostAdoptionMonitor
from orion.substrate.mutation_pressure import PressureAccumulator, PressurePolicy
from orion.substrate.mutation_proposals import (
    _ROUTING_TARGET_PARKED_REASON,
    _routing_threshold_payloads,
    ProposalFactory,
)
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner
from orion.substrate.mutation_worker import SubstrateAdaptationWorker


def _routing_surface(value: float = 0.50, *, degraded: bool = False):
    """Stub for ProposalFactory's live-surface reader. 0.50 is what the routing
    patch was always implicitly written against, so pre-existing assertions
    (patch 0.58, rollback 0.50) still hold -- now derived, not hardcoded."""
    return lambda: {"value": value, "raw": {"value": value}, "degraded": degraded}


def _routing_pressure() -> MutationPressureV1:
    return MutationPressureV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="routing",
        pressure_kind="runtime_failure",
        pressure_score=8.0,
        evidence_refs=["telemetry:1"],
        source_signal_ids=["signal-1"],
    )


def _reader(value=0.50, *, degraded=False):
    """Mirrors inspect_chat_reflective_lane_threshold(): the stored payload
    under "raw", plus a top-level "value" that silently becomes a default when
    nothing is stored."""
    return lambda: {
        "value": value,
        "raw": {"value": value},
        "degraded": degraded,
    }


# --- plan_for_pressure(): the routing surface is parked, full stop ---------


def test_plan_for_pressure_refuses_every_routing_pressure() -> None:
    """The park guard fires before the surface reader is ever touched."""
    plan = ProposalFactory(routing_surface_reader=_reader(0.58)).plan_for_pressure(
        _routing_pressure()
    )
    assert plan.proposal is None
    assert plan.refusal_reason == _ROUTING_TARGET_PARKED_REASON


def test_plan_for_pressure_parks_regardless_of_a_real_gap() -> None:
    """Before parking, a genuine gap (0.50 live vs 0.58 target) would have
    produced a real proposal. It must not any more."""
    plan = ProposalFactory(routing_surface_reader=_reader(0.50)).plan_for_pressure(
        _routing_pressure()
    )
    assert plan.proposal is None
    assert plan.refusal_reason == _ROUTING_TARGET_PARKED_REASON


def test_plan_for_pressure_parks_with_no_reader_at_all() -> None:
    """The park guard does not depend on a surface reader existing."""
    plan = ProposalFactory().plan_for_pressure(_routing_pressure())
    assert plan.proposal is None
    assert plan.refusal_reason == _ROUTING_TARGET_PARKED_REASON


def test_plan_for_pressure_parks_even_when_the_reader_would_raise() -> None:
    def _boom():
        raise RuntimeError("postgres down")

    plan = ProposalFactory(routing_surface_reader=_boom).plan_for_pressure(
        _routing_pressure()
    )
    assert plan.proposal is None
    assert plan.refusal_reason == _ROUTING_TARGET_PARKED_REASON


def test_non_routing_surfaces_are_unaffected_by_the_park() -> None:
    """Only "routing" is parked; every other class behaves as before."""
    pressure = MutationPressureV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="graph_consolidation",
        pressure_kind="runtime_failure",
        pressure_score=8.0,
        evidence_refs=["telemetry:1"],
        source_signal_ids=["signal-1"],
    )
    plan = ProposalFactory(routing_surface_reader=_routing_surface()).plan_for_pressure(pressure)
    assert plan.refusal_reason is None
    assert plan.proposal is not None
    assert plan.proposal.mutation_class == "graph_consolidation_param_patch"


# --- _routing_threshold_payloads(): the no-op guard is still correct -------
# Exercised directly now that plan_for_pressure() no longer reaches it.


def test_refuses_to_propose_the_value_the_surface_already_holds() -> None:
    """The live defect: target 0.58 proposed onto a surface already at 0.58."""
    patch, rollback, refusal = _routing_threshold_payloads(target_value=0.58, reader=_reader(0.58))
    assert patch == {}
    assert rollback == {}
    assert refusal == "routing_threshold_already_at_target"


def test_a_surface_below_target_still_proposes() -> None:
    """The refusal must be about equality, not about refusing everything.

    Uses a real gap, not 0.5799: the tolerance here is a float-equality epsilon
    (values round-trip through JSON and Postgres), NOT a minimum-meaningful-
    movement policy. Whether a 1e-4 threshold move is worth a proposal, a trial
    and a 15-minute surface lock is a separate question this patch does not
    answer -- see the PR report's concerns.
    """
    patch, rollback, refusal = _routing_threshold_payloads(target_value=0.58, reader=_reader(0.50))
    assert refusal is None
    assert patch == {"chat_reflective_lane_threshold": 0.58}
    assert rollback == {"chat_reflective_lane_threshold": 0.50}


def test_rollback_payload_is_the_value_being_replaced_not_a_constant() -> None:
    """A rollback to a stale constant reverts someone else's change, not this one."""
    _, rollback, refusal = _routing_threshold_payloads(target_value=0.58, reader=_reader(0.62))
    assert refusal is None
    assert rollback == {"chat_reflective_lane_threshold": 0.62}


def test_absent_reader_refuses_rather_than_guessing_a_baseline() -> None:
    patch, rollback, refusal = _routing_threshold_payloads(target_value=0.58, reader=None)
    assert patch == {}
    assert rollback == {}
    assert refusal == "routing_surface_reader_absent"


def test_a_sticky_degraded_flag_does_not_block_a_real_reading() -> None:
    """`degraded()` is sticky and fires on the failed Postgres probe even when
    the sqlite fallback is healthy. Gating on it refused every routing proposal
    on any non-Postgres deployment."""
    _, _, refusal = _routing_threshold_payloads(
        target_value=0.58, reader=_reader(0.50, degraded=True)
    )
    assert refusal is None


def test_an_unstored_surface_refuses_rather_than_trusting_the_default() -> None:
    """No stored payload -> the top-level value is a hardcoded 0.75 default.
    Rolling back to a default is the bug this whole patch removes."""
    reader = lambda: {"value": 0.75, "raw": {}, "degraded": False}
    patch, rollback, refusal = _routing_threshold_payloads(target_value=0.58, reader=reader)
    assert patch == {}
    assert rollback == {}
    assert refusal == "routing_surface_value_missing"


def test_a_raising_reader_refuses_and_does_not_propagate() -> None:
    def _boom():
        raise RuntimeError("postgres down")

    patch, rollback, refusal = _routing_threshold_payloads(target_value=0.58, reader=_boom)
    assert patch == {}
    assert rollback == {}
    assert refusal == "routing_surface_read_failed"


def test_an_empty_snapshot_refuses() -> None:
    patch, rollback, refusal = _routing_threshold_payloads(target_value=0.58, reader=lambda: {})
    assert patch == {}
    assert rollback == {}
    assert refusal == "routing_surface_value_missing"


def test_a_store_outage_is_named_as_an_outage_not_an_unwritten_surface() -> None:
    """RuntimeControlSurfaceStore.get() swallows exceptions and returns None,
    so a Postgres outage arrives as an empty payload. Reporting that as
    "never written" tells the operator a different and far calmer story."""
    reader = lambda: {
        "value": 0.75,
        "raw": {},
        "degraded": True,
        "error": "connection refused",
    }
    patch, rollback, refusal = _routing_threshold_payloads(target_value=0.58, reader=reader)
    assert patch == {}
    assert rollback == {}
    assert refusal == "routing_surface_read_failed"


# --- worker-level: refusal is traced and cools the pressure ----------------


def test_worker_traces_the_refusal_instead_of_skipping_silently(monkeypatch) -> None:
    """A refusal is an outcome of the cycle, not an absence of one.

    Before PR #2058/#2071-era work, the worker did a bare `continue` on a None
    proposal, so a generator refusing 100% of the time was indistinguishable
    from one nothing ever asked. As of 2026-09-03 the routing surface refuses
    100% of the time by design (parked) -- this asserts that refusal is still
    traced, now with the park reason.
    """
    traces: list[dict] = []
    store = SubstrateMutationStore()
    worker = SubstrateAdaptationWorker(
        store=store,
        detectors=MutationDetectors(),
        pressure=PressureAccumulator(policy=PressurePolicy(activation_threshold=0.2)),
        proposals=ProposalFactory(routing_surface_reader=_reader(0.58)),
        trial_runner=SubstrateTrialRunner(
            scorer=ClassSpecificScorer(),
            corpus_registry=ReplayCorpusRegistry(
                corpus_by_class={}, baseline_metric_ref_by_class={}
            ),
        ),
        decision_engine=DecisionEngine(),
        applier=PatchApplier(surfaces={}),
        monitor=PostAdoptionMonitor(),
        trace_logger=traces.append,
    )
    monkeypatch.setenv("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true")
    telemetry = [
        GraphReviewTelemetryRecordV1(
            invocation_surface="operator_review",
            execution_outcome="failed",
            selection_reason="x",
            # autonomy_graph is the zone that maps to the routing surface, and
            # >=1200ms raises routing_runtime_degradation -- see
            # mutation_detectors._target_surface_for_zone / _build_rich_routing_signals.
            target_zone="autonomy_graph",
            runtime_duration_ms=1500,
            degraded=True,
            anchor_scope="orion",
            subject_ref="entity:orion",
        )
        for _ in range(12)
    ]
    # Routing pressure accumulates across cycles before it activates.
    for _ in range(12):
        worker.run_cycle(telemetry=telemetry, measured_metrics_by_proposal={})
        if any(t.get("event") == "mutation_proposal_refused" for t in traces):
            break

    refusals = [t for t in traces if t.get("event") == "mutation_proposal_refused"]
    assert refusals, f"no refusal traced; events={sorted({t.get('event') for t in traces})}"
    assert any(
        f"reason={_ROUTING_TARGET_PARKED_REASON}" in (t.get("notes") or [])
        for t in refusals
    )
    assert not [t for t in traces if t.get("event") == "mutation_proposal_enqueued"]


def test_a_refusal_cools_the_pressure_instead_of_re_evaluating_every_tick(
    monkeypatch,
) -> None:
    """A refusal that cannot change without an external write must back off.

    mark_proposal_emitted is only reached on the success path, so before this
    patch cooldown_until stayed None and PressureAccumulator only decays when a
    NEW signal arrives -- the cycle re-read the control surface (two Postgres
    round-trips) and re-emitted the same refusal on every tick, forever. As of
    2026-09-03 the routing surface is parked (no reader read happens at all),
    but the cooldown behavior itself must still hold for any refusal reason.
    """
    traces: list[dict] = []
    store = SubstrateMutationStore()
    worker = SubstrateAdaptationWorker(
        store=store,
        detectors=MutationDetectors(),
        pressure=PressureAccumulator(policy=PressurePolicy(activation_threshold=0.2)),
        proposals=ProposalFactory(routing_surface_reader=_reader(0.58)),
        trial_runner=SubstrateTrialRunner(
            scorer=ClassSpecificScorer(),
            corpus_registry=ReplayCorpusRegistry(
                corpus_by_class={}, baseline_metric_ref_by_class={}
            ),
        ),
        decision_engine=DecisionEngine(),
        applier=PatchApplier(surfaces={}),
        monitor=PostAdoptionMonitor(),
        trace_logger=traces.append,
    )
    monkeypatch.setenv("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true")
    telemetry = [
        GraphReviewTelemetryRecordV1(
            invocation_surface="operator_review",
            execution_outcome="failed",
            selection_reason="x",
            target_zone="autonomy_graph",
            runtime_duration_ms=1500,
            degraded=True,
            anchor_scope="orion",
            subject_ref="entity:orion",
        )
        for _ in range(12)
    ]
    cycles = 24
    for _ in range(cycles):
        worker.run_cycle(telemetry=telemetry, measured_metrics_by_proposal={})

    refusals = [t for t in traces if t.get("event") == "mutation_proposal_refused"]
    assert refusals, "expected at least one refusal"
    # The point: bounded, not one per cycle.
    assert len(refusals) < cycles, (
        f"refused {len(refusals)} times in {cycles} cycles -- pressure is not cooling"
    )
