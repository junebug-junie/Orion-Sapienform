"""The routing proposal generator must read the live surface, not a constant.

Regression cover for a confirmed-live defect (2026-09-03): the surface had
been at 0.58 since 2026-09-02T04:11:17, and the generator kept proposing 0.58
on top of it 5-6 times an hour for at least 36 hours -- ~190 identical
proposals, each burning a proposal -> decision -> adoption cycle and taking a
15-minute rollback lock, each carrying a rollback payload of 0.50 that had not
been the live value since the previous day.

Both halves are covered here: the no-op refusal, and the rollback payload being
derived from the value actually being replaced.
"""

from __future__ import annotations

import os

from orion.core.schemas.substrate_mutation import MutationPressureV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1
from orion.substrate.mutation_apply import PatchApplier
from orion.substrate.mutation_decision import DecisionEngine
from orion.substrate.mutation_detectors import MutationDetectors
from orion.substrate.mutation_monitor import PostAdoptionMonitor
from orion.substrate.mutation_pressure import PressureAccumulator, PressurePolicy
from orion.substrate.mutation_proposals import ProposalFactory
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner
from orion.substrate.mutation_worker import SubstrateAdaptationWorker


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


def test_refuses_to_propose_the_value_the_surface_already_holds() -> None:
    """The live defect: target 0.58 proposed onto a surface already at 0.58."""
    plan = ProposalFactory(routing_surface_reader=_reader(0.58)).plan_for_pressure(
        _routing_pressure()
    )
    assert plan.proposal is None
    assert plan.refusal_reason == "routing_threshold_already_at_target"


def test_a_surface_just_below_target_still_proposes() -> None:
    """The refusal must be about equality, not about refusing everything."""
    plan = ProposalFactory(routing_surface_reader=_reader(0.5799)).plan_for_pressure(
        _routing_pressure()
    )
    assert plan.refusal_reason is None
    assert plan.proposal is not None
    assert plan.proposal.patch.patch["chat_reflective_lane_threshold"] == 0.58


def test_rollback_payload_is_the_value_being_replaced_not_a_constant() -> None:
    """A rollback to a stale constant reverts someone else's change, not this one."""
    plan = ProposalFactory(routing_surface_reader=_reader(0.62)).plan_for_pressure(
        _routing_pressure()
    )
    assert plan.proposal is not None
    assert plan.proposal.patch.rollback_payload == {"chat_reflective_lane_threshold": 0.62}


def test_absent_reader_refuses_rather_than_guessing_a_baseline() -> None:
    plan = ProposalFactory().plan_for_pressure(_routing_pressure())
    assert plan.proposal is None
    assert plan.refusal_reason == "routing_surface_reader_absent"


def test_a_sticky_degraded_flag_does_not_block_a_real_reading() -> None:
    """`degraded()` is sticky and fires on the failed Postgres probe even when
    the sqlite fallback is healthy. Gating on it refused every routing proposal
    on any non-Postgres deployment."""
    plan = ProposalFactory(
        routing_surface_reader=_reader(0.50, degraded=True)
    ).plan_for_pressure(_routing_pressure())
    assert plan.refusal_reason is None
    assert plan.proposal is not None


def test_an_unstored_surface_refuses_rather_than_trusting_the_default() -> None:
    """No stored payload -> the top-level value is a hardcoded 0.75 default.
    Rolling back to a default is the bug this whole patch removes."""
    plan = ProposalFactory(
        routing_surface_reader=lambda: {"value": 0.75, "raw": {}, "degraded": False}
    ).plan_for_pressure(_routing_pressure())
    assert plan.proposal is None
    assert plan.refusal_reason == "routing_surface_value_missing"


def test_a_raising_reader_refuses_and_does_not_propagate() -> None:
    def _boom():
        raise RuntimeError("postgres down")

    plan = ProposalFactory(routing_surface_reader=_boom).plan_for_pressure(
        _routing_pressure()
    )
    assert plan.proposal is None
    assert plan.refusal_reason == "routing_surface_read_failed"


def test_an_empty_snapshot_refuses() -> None:
    plan = ProposalFactory(routing_surface_reader=lambda: {}).plan_for_pressure(
        _routing_pressure()
    )
    assert plan.proposal is None
    assert plan.refusal_reason == "routing_surface_value_missing"


def test_non_routing_surfaces_are_unaffected_by_the_reader() -> None:
    """Only routing has a live surface; every other class must behave as before."""
    pressure = MutationPressureV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="graph_consolidation",
        pressure_kind="runtime_failure",
        pressure_score=8.0,
        evidence_refs=["telemetry:1"],
        source_signal_ids=["signal-1"],
    )
    plan = ProposalFactory().plan_for_pressure(pressure)
    assert plan.refusal_reason is None
    assert plan.proposal is not None
    assert plan.proposal.mutation_class == "graph_consolidation_param_patch"


def test_worker_traces_the_refusal_instead_of_skipping_silently() -> None:
    """A refusal is an outcome of the cycle, not an absence of one.

    Before this patch the worker did a bare `continue` on a None proposal, so a
    generator refusing 100% of the time was indistinguishable from one nothing
    ever asked. With the no-op refusal now firing on every live routing cycle,
    an untraced skip would have made the whole loop look idle.
    """
    traces: list[dict] = []
    store = SubstrateMutationStore()
    worker = SubstrateAdaptationWorker(
        store=store,
        detectors=MutationDetectors(),
        pressure=PressureAccumulator(policy=PressurePolicy(activation_threshold=0.2)),
        # Surface already at target -> every routing proposal is refused.
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
    previous = os.environ.get("SUBSTRATE_MUTATION_AUTONOMY_ENABLED")
    os.environ["SUBSTRATE_MUTATION_AUTONOMY_ENABLED"] = "true"
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
    try:
        # Routing pressure accumulates across cycles before it activates.
        for _ in range(12):
            worker.run_cycle(
                telemetry=telemetry,
                measured_metrics_by_proposal={},
            )
            if any(t.get("event") == "mutation_proposal_refused" for t in traces):
                break
    finally:
        if previous is None:
            os.environ.pop("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", None)
        else:
            os.environ["SUBSTRATE_MUTATION_AUTONOMY_ENABLED"] = previous

    refusals = [t for t in traces if t.get("event") == "mutation_proposal_refused"]
    assert refusals, f"no refusal traced; events={sorted({t.get('event') for t in traces})}"
    assert any(
        "reason=routing_threshold_already_at_target" in (t.get("notes") or [])
        for t in refusals
    )
    assert not [t for t in traces if t.get("event") == "mutation_proposal_enqueued"]
