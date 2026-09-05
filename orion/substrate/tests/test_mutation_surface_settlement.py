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


def _routing_surface(value: float = 0.50, *, degraded: bool = False):
    """Stub for ProposalFactory's live-surface reader. 0.50 is what the routing
    patch was always implicitly written against, so pre-existing assertions
    (patch 0.58, rollback 0.50) still hold -- now derived, not hardcoded."""
    return lambda: {"value": value, "raw": {"value": value}, "degraded": degraded}


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
        proposals=ProposalFactory(routing_surface_reader=_routing_surface()),
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


def test_settlement_is_idempotent_and_rejects_an_unknown_id() -> None:
    store, adoption = _adopted_store()
    assert store.record_settlement(adoption.adoption_id) is True

    assert store.record_settlement(adoption.adoption_id) is False
    assert store.record_settlement("no-such-adoption") is False


def test_settlement_never_reopens_a_rolled_back_adoption() -> None:
    """Terminal means terminal: a reverted change must not become 'kept'."""
    from orion.core.schemas.substrate_mutation import MutationRollbackV1

    store, adoption = _adopted_store()
    store.record_rollback(
        MutationRollbackV1(
            adoption_id=adoption.adoption_id,
            proposal_id=adoption.proposal_id,
            reason="regression",
            payload=dict(adoption.rollback_payload),
        )
    )
    assert store._adoptions[adoption.adoption_id].status == "rolled_back"

    assert store.record_settlement(adoption.adoption_id) is False
    assert store._adoptions[adoption.adoption_id].status == "rolled_back"


def test_settling_a_non_holder_leaves_the_real_holder_locked() -> None:
    """Two applied adoptions can share a surface after a reload.

    ``_recover_active_surfaces`` rebuilds the lock from every applied adoption,
    so the holder is whichever one it saw last. Settling the other must not hand
    away a lock it never held.
    """
    store = SubstrateMutationStore()
    holder = _adoption(applied_at=datetime.now(timezone.utc) - timedelta(hours=2))
    store.record_adoption(holder)
    non_holder = holder.model_copy(
        update={"adoption_id": "a-other", "proposal_id": "p-other", "status": "applied"}
    )
    store._adoptions["a-other"] = non_holder  # the post-reload split-brain shape

    assert store.record_settlement("a-other") is True

    assert store.active_surface("routing") == holder.adoption_id


def test_worker_skips_an_already_terminal_adoption_on_a_later_cycle(monkeypatch) -> None:
    """Without this guard a rolled-back adoption is re-rolled-back every cycle."""
    monkeypatch.setenv("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true")
    store, adoption = _adopted_store()
    worker = _worker(store)

    first = worker.run_cycle(
        telemetry=[], measured_metrics_by_proposal={},
        post_adoption_delta_by_target_surface={"routing": -0.9},
        now=datetime.now(timezone.utc),
    )
    assert f"rolled_back:{adoption.proposal_id}" in first["notes"]

    second = worker.run_cycle(
        telemetry=[], measured_metrics_by_proposal={},
        post_adoption_delta_by_target_surface={"routing": -0.9},
        now=datetime.now(timezone.utc),
    )
    assert second["notes"] == []
    assert len(store._rollbacks) == 1


def test_a_healthy_delta_settles_rather_than_holding_the_surface(monkeypatch) -> None:
    """Forward-looking: nothing in production supplies a delta yet.

    The no-delta path at mutation_worker.py is what actually fires live today.
    This pins the measured path so it is not silently wrong when a producer is
    wired up for it.
    """
    monkeypatch.setenv("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true")
    store, adoption = _adopted_store()
    worker = _worker(store)

    result = worker.run_cycle(
        telemetry=[], measured_metrics_by_proposal={},
        post_adoption_delta_by_target_surface={"routing": 0.4},  # improvement
        now=datetime.now(timezone.utc),
    )

    assert store._rollbacks == {}
    assert store._adoptions[adoption.adoption_id].status == "settled"
    assert f"settled:{adoption.proposal_id}" in result["notes"]


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

    # Exactly at the boundary the window has elapsed, so it settles.
    at_boundary = worker._settle_if_window_elapsed(
        adoption=adoption, cycle_id="c", lineage_id="l",
        now=now + timedelta(seconds=900), notes=[], reason="test",
    )
    assert at_boundary is True
    assert store.active_surface("routing") is None
    store._adoptions[adoption.adoption_id] = adoption  # reset for the late case
    store._active_surface_by_target["routing"] = adoption.adoption_id

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


# test_adoption_records_the_real_prior_value_not_the_class_constant() removed
# 2026-09-05: it tested mutation_apply.py's routing_threshold_patch-specific
# apply branch (the real-live-value-vs-hardcoded-rollback-constant bug fix),
# which is now unreachable -- "routing_threshold_patch" is retired
# (mutation_contracts.py's RETIRED_MUTATION_CLASSES), and PatchApplier.apply()
# refuses it unconditionally before that branch is ever reached. See this
# change's PR description.


def test_adoption_retention_never_evicts_the_surface_lock_holder(monkeypatch) -> None:
    """Releasing the lock on success removed what was bounding this table.

    A stranded lock would be unreleasable: record_settlement and record_rollback
    both look the adoption up by id, so evicting the holder recreates the exact
    permanent-hold bug this branch exists to fix.
    """
    monkeypatch.setenv("SUBSTRATE_MUTATION_RETENTION_MAX_ADOPTIONS", "50")
    store = SubstrateMutationStore()
    holder = _adoption(applied_at=datetime.now(timezone.utc) - timedelta(days=30))
    store.record_adoption(holder)  # oldest, so first in line for eviction

    for i in range(60):
        extra = holder.model_copy(
            update={
                "adoption_id": f"a-{i}",
                "proposal_id": f"p-{i}",
                "target_surface": f"surface-{i}",
                "created_at": datetime.now(timezone.utc),
            }
        )
        store._adoptions[extra.adoption_id] = extra
    store._compact_artifacts()

    assert len(store._adoptions) <= 50
    assert holder.adoption_id in store._adoptions
    assert store.active_surface("routing") == holder.adoption_id
    assert store.record_settlement(holder.adoption_id) is True
