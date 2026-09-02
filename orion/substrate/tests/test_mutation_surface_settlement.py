"""A change that succeeds must hand its surface back.

``record_adoption`` takes the one-live-mutation-per-surface lock. Until
``record_settlement`` existed, ``record_rollback`` was the only thing that
returned it -- so a mutation that *worked* held its surface forever. Live on
2026-09-02 a single adoption at 04:11 UTC blocked every later proposal for
thirteen hours: 77 of them, each decided ``hold /
active_surface_mutation_exists``, with zero rollbacks ever recorded.

The second property here matters as much as the first. The monitor is fed a
delta score, and for the whole life of the system it has been fed ``None`` --
nothing in production ever supplied one. A missing reading has to mean
*unknown*, never *bad*: if silence rolled changes back, an unmeasured surface
would revert on every cycle, which is the same trap facing the other way.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.substrate_mutation import MutationAdoptionV1
from orion.substrate.mutation_apply import PatchApplier
from orion.substrate.mutation_decision import DecisionEngine
from orion.substrate.mutation_detectors import MutationDetectors
from orion.substrate.mutation_monitor import PostAdoptionMonitor
from orion.substrate.mutation_pressure import PressureAccumulator
from orion.substrate.mutation_proposals import ProposalFactory
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.mutation_worker import SubstrateAdaptationWorker


def _adoption(*, applied_at: datetime, window_sec: int = 900) -> MutationAdoptionV1:
    return MutationAdoptionV1(
        proposal_id="p-1",
        decision_id="d-1",
        target_surface="routing",
        applied_patch={"chat_reflective_lane_threshold": 0.58},
        rollback_payload={"chat_reflective_lane_threshold": 0.5},
        applied_at=applied_at,
        rollback_window_sec=window_sec,
    )


def _worker(store: SubstrateMutationStore) -> SubstrateAdaptationWorker:
    """Real worker. Only `store` and the trace hook are exercised here."""
    return SubstrateAdaptationWorker(
        store=store,
        detectors=MutationDetectors(),
        pressure=PressureAccumulator(),
        proposals=ProposalFactory(),
        trial_runner=None,
        decision_engine=DecisionEngine(),
        applier=PatchApplier(surfaces={}),
        monitor=PostAdoptionMonitor(),
    )


def _adopted_store() -> tuple[SubstrateMutationStore, MutationAdoptionV1]:
    store = SubstrateMutationStore()
    adoption = _adoption(applied_at=datetime.now(timezone.utc) - timedelta(hours=2))
    assert store.record_adoption(adoption) == []
    assert store.active_surface("routing") == adoption.adoption_id
    return store, adoption


def test_settlement_releases_the_surface_and_keeps_the_change() -> None:
    store, adoption = _adopted_store()

    assert store.record_settlement(adoption.adoption_id) is True

    assert store.active_surface("routing") is None
    settled = store._adoptions[adoption.adoption_id]
    assert settled.status == "settled"
    assert settled.applied_patch == {"chat_reflective_lane_threshold": 0.58}


def test_a_released_surface_accepts_a_new_adoption() -> None:
    """The 77-proposal backlog: proving the lock actually reopens."""
    store, adoption = _adopted_store()
    store.record_settlement(adoption.adoption_id)

    nxt = _adoption(applied_at=datetime.now(timezone.utc))
    nxt = nxt.model_copy(update={"proposal_id": "p-2", "adoption_id": "a-2"})

    assert store.record_adoption(nxt) == []
    assert store.active_surface("routing") == "a-2"


def test_settlement_is_idempotent_and_never_reopens_a_rollback() -> None:
    store, adoption = _adopted_store()
    assert store.record_settlement(adoption.adoption_id) is True

    assert store.record_settlement(adoption.adoption_id) is False
    assert store.record_settlement("no-such-adoption") is False


def test_worker_settles_only_after_the_rollback_window_has_elapsed() -> None:
    """``rollback_window_sec`` was read by nothing at all; this is what uses it."""
    store = SubstrateMutationStore()
    now = datetime.now(timezone.utc)
    adoption = _adoption(applied_at=now, window_sec=900)
    store.record_adoption(adoption)
    worker = _worker(store)

    early = worker._settle_if_window_elapsed(
        adoption=adoption, cycle_id="c", lineage_id="l",
        now=now + timedelta(seconds=899), notes=[], reason="test",
    )
    assert early is False
    assert store.active_surface("routing") == adoption.adoption_id

    late = worker._settle_if_window_elapsed(
        adoption=adoption, cycle_id="c", lineage_id="l",
        now=now + timedelta(seconds=901), notes=[], reason="test",
    )
    assert late is True
    assert store.active_surface("routing") is None


def test_a_missing_delta_settles_and_never_rolls_back(monkeypatch) -> None:
    """Silence is 'unknown'. The monitor has been fed None for its entire life.

    Driven through the real ``run_cycle`` rather than the helper, because the
    live failure was in how the cycle *reached* the helper: the old code hit
    ``if delta is None: continue`` and skipped every adoption forever.
    """
    monkeypatch.setenv("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true")
    store, adoption = _adopted_store()
    worker = _worker(store)

    result = worker.run_cycle(
        telemetry=[],
        measured_metrics_by_proposal={},
        post_adoption_delta_by_proposal=None,
        post_adoption_delta_by_target_surface=None,
        now=datetime.now(timezone.utc),
    )

    assert store._rollbacks == {}
    assert store._adoptions[adoption.adoption_id].status == "settled"
    assert store.active_surface("routing") is None
    assert f"settled:{adoption.proposal_id}" in result["notes"]


def test_a_regressing_delta_still_rolls_back_rather_than_settling(monkeypatch) -> None:
    """Settlement must not swallow the failure path it sits next to."""
    monkeypatch.setenv("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true")
    store, adoption = _adopted_store()
    worker = _worker(store)

    result = worker.run_cycle(
        telemetry=[],
        measured_metrics_by_proposal={},
        post_adoption_delta_by_target_surface={"routing": -0.9},
        now=datetime.now(timezone.utc),
    )

    assert store._adoptions[adoption.adoption_id].status == "rolled_back"
    assert store.active_surface("routing") is None
    assert f"rolled_back:{adoption.proposal_id}" in result["notes"]
    assert f"settled:{adoption.proposal_id}" not in result["notes"]


def test_adoption_records_the_real_prior_value_not_the_class_constant(
    monkeypatch, tmp_path
) -> None:
    """The undo button must point at a reading, not at a hardcoded number.

    ``mutation_apply`` read the live threshold and then dropped it, because the
    proposal already carries ``{"chat_reflective_lane_threshold": 0.50}`` from
    ``_default_rollback_for_class`` and the merge used ``setdefault``. Every
    recorded rollback value was therefore a constant. It matched reality once,
    on 2026-09-02, by coincidence -- the live value happened to be 0.5 because a
    pytest fixture had been writing it there.
    """
    from orion.core.schemas.substrate_mutation import MutationDecisionV1, MutationPatchV1, MutationProposalV1
    from orion.substrate import mutation_control_surface

    monkeypatch.setenv("SUBSTRATE_MUTATION_CONTROL_SQL_DB_PATH", str(tmp_path / "cs.sqlite3"))
    monkeypatch.setattr(
        mutation_control_surface,
        "_CONTROL_SURFACE_STORE",
        mutation_control_surface.RuntimeControlSurfaceStore(sql_db_path=str(tmp_path / "cs.sqlite3")),
    )
    # A live value that is deliberately NOT the class constant.
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.71, actor="operator")

    proposal = MutationProposalV1(
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
            patch={"chat_reflective_lane_threshold": 0.58},
            rollback_payload={"chat_reflective_lane_threshold": 0.50},  # the constant
        ),
    )
    decision = MutationDecisionV1(proposal_id=proposal.proposal_id, action="auto_promote", reason="test")

    adoption = PatchApplier(surfaces={}).apply(proposal=proposal, decision=decision)

    assert adoption is not None
    assert adoption.rollback_payload["chat_reflective_lane_threshold"] == 0.71
    assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.58
