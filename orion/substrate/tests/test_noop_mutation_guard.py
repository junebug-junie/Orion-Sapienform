"""A change that changes nothing must not consume an adoption or a surface lock.

Confirmed in production 2026-09-03. The moment the surface lock was released,
the pipeline adopted 0.58 over a live value of 0.58 and wrote a history row
reading `0.58 -> 0.58`. The routing patch value is a hardcoded constant
(`_default_patch_for_class`), so once the surface reaches it every later
proposal re-applies the number already live -- an adoption, a lock and a history
row every rollback window, none of which change Orion's behaviour, and each of
which blocks a real proposal for the length of the window.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.core.schemas.substrate_mutation import (
    MutationDecisionV1,
    MutationPatchV1,
    MutationProposalV1,
)
from orion.substrate import mutation_control_surface
from orion.substrate.mutation_apply import PatchApplier
from orion.substrate.mutation_decision import DecisionEngine
from orion.substrate.mutation_detectors import MutationDetectors
from orion.substrate.mutation_monitor import PostAdoptionMonitor
from orion.substrate.mutation_pressure import PressureAccumulator
from orion.substrate.mutation_proposals import ProposalFactory
from orion.substrate.mutation_queue import SubstrateMutationStore


@pytest.fixture
def isolated_surface(monkeypatch, tmp_path):
    store = mutation_control_surface.RuntimeControlSurfaceStore(
        sql_db_path=str(tmp_path / "control.sqlite3")
    )
    monkeypatch.setattr(mutation_control_surface, "_CONTROL_SURFACE_STORE", store)
    return store


def _proposal(value: float) -> MutationProposalV1:
    return MutationProposalV1(
        mutation_class="routing_threshold_patch",
        target_surface="routing",
        lane="operational",
        risk_tier="low",
        rationale="test",
        anchor_scope="orion",
        subject_ref="orion",
        expected_effect="reduce_runtime_executed",
        evidence_refs=["e-1"],
        source_signal_ids=["s-1"],
        source_pressure_id="pr-1",
        patch=MutationPatchV1(
            mutation_class="routing_threshold_patch",
            target_surface="routing",
            target_ref="routing",
            patch={"chat_reflective_lane_threshold": value},
            rollback_payload={"chat_reflective_lane_threshold": 0.5},
        ),
    )


def test_a_patch_matching_the_live_value_is_a_noop(isolated_surface) -> None:
    """The exact production shape: 0.58 proposed over a live 0.58."""
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.58, actor="seed")

    reason = PatchApplier(surfaces={}).noop_reason(proposal=_proposal(0.58))

    assert reason is not None
    assert "patch_is_noop" in reason


def test_a_patch_that_moves_the_value_is_not_a_noop(isolated_surface) -> None:
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.5, actor="seed")

    assert PatchApplier(surfaces={}).noop_reason(proposal=_proposal(0.58)) is None


def test_an_uncomparable_surface_is_never_called_a_noop(isolated_surface) -> None:
    """"Cannot tell" must not read as "no change" -- apply proceeds instead."""
    other = _proposal(0.58).model_copy(update={"mutation_class": "recall_weighting_patch"})

    assert PatchApplier(surfaces={}).noop_reason(proposal=other) is None


def test_the_worker_records_the_skip_rather_than_swallowing_it(
    isolated_surface, monkeypatch
) -> None:
    """Driven through the real run_cycle, not by calling the helper directly.

    The guard's whole value is that the cycle *reaches* it and leaves a record.
    A test that calls ``noop_reason`` and then records the block by hand proves
    only that the helper returns a string.
    """
    from orion.core.schemas.substrate_mutation import MutationTrialV1
    from orion.substrate.mutation_worker import SubstrateAdaptationWorker

    class _PassingTrials:
        """Minimal stand-in: the trial lane is not what is under test here."""

        class corpus_registry:  # noqa: N801 - matches the attribute the worker reads
            @staticmethod
            def ready_for_class(_mutation_class: str) -> bool:
                return True

        def run_trial(self, *, proposal, **_kwargs) -> MutationTrialV1:
            return MutationTrialV1(
                proposal_id=proposal.proposal_id,
                mutation_class=proposal.mutation_class,
                replay_corpus_id="corpus-test",
                baseline_metric_ref="baseline-test",
                status="passed",
            )

    monkeypatch.setenv("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true")
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.58, actor="seed")
    store = SubstrateMutationStore()
    proposal = _proposal(0.58)
    store.add_proposal(proposal, priority=60)
    worker = SubstrateAdaptationWorker(
        store=store,
        detectors=MutationDetectors(),
        pressure=PressureAccumulator(),
        proposals=ProposalFactory(),
        trial_runner=_PassingTrials(),
        decision_engine=DecisionEngine(),
        applier=PatchApplier(surfaces={}),
        monitor=PostAdoptionMonitor(),
    )

    result = worker.run_cycle(
        telemetry=[],
        measured_metrics_by_proposal={},
        now=datetime.now(timezone.utc),
    )

    assert any("patch_is_noop" in n for n in result["notes"]), result["notes"]
    assert result["adoptions"] == 0
    assert store.active_surface("routing") is None  # no lock taken
    assert any(
        "patch_is_noop" in str(row.get("reason"))
        for row in store.recent_blocked_applies(limit=5)
    )


def test_the_noop_guard_leaves_the_surface_untouched(isolated_surface) -> None:
    """No adoption, so no lock, so a real proposal is not blocked behind it."""
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.58, actor="seed")
    store = SubstrateMutationStore()
    applier = PatchApplier(surfaces={})
    proposal = _proposal(0.58)

    assert applier.noop_reason(proposal=proposal) is not None
    assert store.active_surface("routing") is None
    assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.58


def test_the_smoke_script_never_writes_to_the_ambient_control_surface() -> None:
    """The smoke must not be able to move Orion's real routing threshold.

    A pytest fixture writing to the ambient control surface is how
    `value=0.5, actor="scheduler_seed"` reached Orion's live row with 4,925
    updates on it. `run_smoke` isolated its mutation store but read the global
    control surface, so it both inherited ambient state and could have written
    to it. It now runs against its own in-memory surface.
    """
    from orion.substrate.scripts.smoke_mutation_v21 import run_smoke

    before_store = mutation_control_surface._CONTROL_SURFACE_STORE
    sentinel = mutation_control_surface.RuntimeControlSurfaceStore(
        sql_db_path=None, postgres_url=None
    )
    mutation_control_surface._CONTROL_SURFACE_STORE = sentinel
    try:
        mutation_control_surface.set_chat_reflective_lane_threshold(value=0.71, actor="operator")

        lines = run_smoke(emit=False)

        # The ambient surface is untouched, and restored afterwards.
        assert mutation_control_surface._CONTROL_SURFACE_STORE is sentinel
        assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.71
        assert len(sentinel.history("routing.chat_reflective_lane_threshold")) == 1
        # ... and the smoke still exercises the apply path it exists to prove.
        assert any("decision=auto_promote" in line and "applied=True" in line for line in lines)
    finally:
        mutation_control_surface._CONTROL_SURFACE_STORE = before_store
