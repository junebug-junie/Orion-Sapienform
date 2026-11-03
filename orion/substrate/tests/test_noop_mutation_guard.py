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

import pytest

from orion.core.schemas.substrate_mutation import (
    MutationPatchV1,
    MutationProposalV1,
)
from orion.substrate import mutation_control_surface
from orion.substrate.mutation_apply import PatchApplier
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
    """"Cannot tell" must not read as "no change" -- apply proceeds instead.

    The surface is seeded to the patch value on purpose. Without it the live
    default (0.75) already differs from the patch (0.58), so the comparison
    returns None regardless of the class gate and the test passes for the wrong
    reason -- it was green even with the gate deleted entirely.
    """
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.58, actor="seed")
    other = _proposal(0.58).model_copy(update={"mutation_class": "recall_weighting_patch"})

    assert PatchApplier(surfaces={}).noop_reason(proposal=other) is None


# test_the_worker_records_the_skip_rather_than_swallowing_it() removed
# 2026-09-05: drove this guard through a real run_cycle() on a
# "routing_threshold_patch" proposal, which can no longer reach the noop
# check at all -- DecisionEngine.decide() now rejects the retired class
# (mutation_contracts.py's RETIRED_MUTATION_CLASSES) before the cycle ever
# gets to PatchApplier.apply()/noop_reason(). The noop-detection logic
# itself is still covered directly by the tests above and below (calling
# noop_reason() in isolation, not through a full worker cycle). See this
# change's PR description.


def test_the_noop_guard_leaves_the_surface_untouched(isolated_surface) -> None:
    """No adoption, so no lock, so a real proposal is not blocked behind it."""
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.58, actor="seed")
    store = SubstrateMutationStore()
    applier = PatchApplier(surfaces={})
    proposal = _proposal(0.58)

    assert applier.noop_reason(proposal=proposal) is not None
    assert store.active_surface("routing") is None
    assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.58


def test_the_smoke_script_never_writes_to_the_ambient_control_surface(
    monkeypatch, tmp_path
) -> None:
    """The smoke must not be able to move Orion's real routing threshold.

    Driven with an ambient control-surface env var deliberately in scope,
    because that is the condition under which the leak happens and the
    condition orion-hub actually runs in. An earlier version of this test built
    its sentinel with `RuntimeControlSurfaceStore(sql_db_path=None,
    postgres_url=None)` -- which is not an isolation request at all, since
    `__post_init__` fills either slot from the environment. It therefore could
    not see the leak it existed to catch, and reproduced it instead.
    """
    from orion.substrate.scripts.smoke_mutation_v21 import run_smoke

    ambient_db = tmp_path / "ambient-control-surface.sqlite3"
    monkeypatch.setenv("SUBSTRATE_MUTATION_CONTROL_SQL_DB_PATH", str(ambient_db))
    for key in ("SUBSTRATE_CONTROL_PLANE_POSTGRES_URL", "SUBSTRATE_POLICY_POSTGRES_URL", "DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)

    ambient = mutation_control_surface.RuntimeControlSurfaceStore(sql_db_path=str(ambient_db))
    monkeypatch.setattr(mutation_control_surface, "_CONTROL_SURFACE_STORE", ambient)
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.71, actor="operator")

    lines = run_smoke(emit=False)

    # The ambient surface still holds the operator's value and only their row.
    assert mutation_control_surface._CONTROL_SURFACE_STORE is ambient
    assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.71
    history = ambient.history("routing.chat_reflective_lane_threshold")
    assert [row["actor"] for row in history] == ["operator"]
    # ... and the smoke still exercises the apply path it exists to prove.
    assert any("decision=auto_promote" in line and "applied=True" in line for line in lines)


def test_a_multi_key_patch_is_never_called_a_noop(isolated_surface) -> None:
    """Judging a multi-key patch on one key would silently drop the other change.

    `mutation_contracts.py` allows `autonomy_route_threshold` alongside the lane
    threshold. Not reachable from today's single-key proposal factory, but the
    failure it would cause -- a real change discarded as "no change" -- is worse
    than the no-op adoption this guard exists to prevent.
    """
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.58, actor="seed")
    proposal = _proposal(0.58)
    multi = proposal.model_copy(
        update={
            "patch": proposal.patch.model_copy(
                update={
                    "patch": {
                        "chat_reflective_lane_threshold": 0.58,
                        "autonomy_route_threshold": 0.9,
                    }
                }
            )
        }
    )

    assert PatchApplier(surfaces={}).noop_reason(proposal=multi) is None


def test_an_out_of_range_patch_over_a_saturated_surface_is_a_noop(
    isolated_surface,
) -> None:
    """The setter clamps to [0, 1], so 1.5 over a live 1.0 writes nothing.

    Comparing the raw patch value would let that through, mint an adoption and
    record a change that did not happen -- the same falsehood the guard exists
    to stop, arriving by a different door.
    """
    mutation_control_surface.set_chat_reflective_lane_threshold(value=1.0, actor="seed")

    reason = PatchApplier(surfaces={}).noop_reason(proposal=_proposal(1.5))

    assert reason is not None
    assert "patch_is_noop" in reason
