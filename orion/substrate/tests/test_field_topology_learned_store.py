from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.core.schemas.substrate_mutation import MutationPatchV1, MutationProposalV1
from orion.substrate.field_topology_learned_store import (
    EFFECTIVE_DELTA_EPSILON,
    FieldTopologyLearnedWeightsStore,
)
from orion.substrate.field_topology_plasticity import MUTATION_CLASS, edge_ref_for


def _proposal(*, edge_ref: str = "cap:a->cap:b", delta: float = 0.08) -> MutationProposalV1:
    patch = MutationPatchV1(
        mutation_class=MUTATION_CLASS,
        target_surface="field_topology",
        target_ref=edge_ref,
        patch={"edge_weight_delta": delta},
        rollback_payload={"edge_weight_delta": 0.0},
    )
    return MutationProposalV1(
        lane="operational",
        mutation_class=MUTATION_CLASS,
        risk_tier="low",
        target_surface="field_topology",
        anchor_scope="orion",
        subject_ref=f"field_edge:{edge_ref}",
        rationale="test proposal",
        expected_effect="test",
        evidence_refs=["causal_geometry:snapshot:test"],
        source_signal_ids=["causal_geometry:snapshot:test"],
        source_pressure_id="causal-geometry-derived-test",
        patch=patch,
    )


def test_propose_then_list_pending() -> None:
    store = FieldTopologyLearnedWeightsStore()
    proposal = _proposal()

    store.propose(proposal)

    pending = store.list_pending()
    assert len(pending) == 1
    assert pending[0].proposal_id == proposal.proposal_id
    assert store.status_for(proposal.proposal_id) == "pending_review"


def test_adopt_moves_proposal_into_overlay() -> None:
    store = FieldTopologyLearnedWeightsStore()
    proposal = _proposal(edge_ref=edge_ref_for("cap:reasoning", "cap:memory"), delta=0.1)
    store.propose(proposal)
    now = datetime.now(timezone.utc)

    result = store.adopt(proposal.proposal_id, operator_id="june", now=now)

    assert result["ok"] is True
    assert store.status_for(proposal.proposal_id) == "adopted"
    assert store.list_pending() == []
    overlay = store.current_overlay(now=now)
    assert edge_ref_for("cap:reasoning", "cap:memory") in overlay
    assert overlay[edge_ref_for("cap:reasoning", "cap:memory")] == 0.1


def test_reject_does_not_appear_in_overlay() -> None:
    store = FieldTopologyLearnedWeightsStore()
    proposal = _proposal(edge_ref="cap:x->cap:y", delta=0.05)
    store.propose(proposal)

    store.reject(proposal.proposal_id, operator_id="june", reason="not_convincing")

    assert store.status_for(proposal.proposal_id) == "rejected"
    assert store.list_pending() == []
    overlay = store.current_overlay()
    assert "cap:x->cap:y" not in overlay


def test_reject_after_adopt_is_a_no_op_and_leaves_overlay_live() -> None:
    """Regression: reject() previously had no pending-status guard (unlike adopt()),
    so rejecting an already-adopted proposal flipped its visible status to
    "rejected" while leaving the live overlay entry untouched -- an operator using
    the hub's Reject button on an adopted proposal would see it marked rejected
    while the weight override kept silently applying in diffusion."""
    store = FieldTopologyLearnedWeightsStore()
    proposal = _proposal(edge_ref="cap:m->cap:n", delta=0.09)
    store.propose(proposal)
    store.adopt(proposal.proposal_id, operator_id="june")

    store.reject(proposal.proposal_id, operator_id="june", reason="changed my mind")

    assert store.status_for(proposal.proposal_id) == "adopted"
    overlay = store.current_overlay()
    assert overlay.get("cap:m->cap:n") == pytest.approx(0.09)


def test_adopt_is_the_only_way_into_overlay_and_rejects_non_pending() -> None:
    store = FieldTopologyLearnedWeightsStore()
    proposal = _proposal()
    store.propose(proposal)
    store.reject(proposal.proposal_id, operator_id="june")

    result = store.adopt(proposal.proposal_id, operator_id="june")

    assert result["ok"] is False
    assert "not_pending" in result["reason"]


def test_decay_reduces_effective_delta_over_simulated_time() -> None:
    store = FieldTopologyLearnedWeightsStore(half_life_hours=24.0)
    proposal = _proposal(edge_ref="cap:decay->cap:target", delta=0.12)
    store.propose(proposal)
    t0 = datetime(2026, 7, 16, tzinfo=timezone.utc)

    store.adopt(proposal.proposal_id, operator_id="june", now=t0)

    immediate = store.current_overlay(now=t0)["cap:decay->cap:target"]
    assert immediate == 0.12

    half_life_later = store.current_overlay(now=t0 + timedelta(hours=24.0))["cap:decay->cap:target"]
    assert abs(half_life_later - 0.06) < 1e-6

    much_later = store.current_overlay(now=t0 + timedelta(hours=24.0 * 20))
    assert "cap:decay->cap:target" not in much_later  # decayed below EFFECTIVE_DELTA_EPSILON


def test_effective_delta_epsilon_is_a_named_threshold_not_magic_number() -> None:
    # Cheap regression guard: confirms the constant exists and is small/positive,
    # so a future edit can't silently make current_overlay() return everything
    # (epsilon=0) or nothing (epsilon too large) without a visible diff here.
    assert 0 < EFFECTIVE_DELTA_EPSILON < 1e-2


def test_sqlite_persistence_round_trip(tmp_path: Path) -> None:
    db_path = str(tmp_path / "field_topology_learned.sqlite3")
    store = FieldTopologyLearnedWeightsStore(sql_db_path=db_path)
    proposal = _proposal(edge_ref="cap:p->cap:q", delta=0.07)
    store.propose(proposal)
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    store.adopt(proposal.proposal_id, operator_id="june", now=now)
    assert store.source_kind() == "sqlite"
    assert not store.degraded()

    reloaded = FieldTopologyLearnedWeightsStore(sql_db_path=db_path)

    assert reloaded.source_kind() == "sqlite"
    assert reloaded.status_for(proposal.proposal_id) == "adopted"
    overlay = reloaded.current_overlay(now=now)
    assert overlay["cap:p->cap:q"] == 0.07


def test_new_store_module_never_imports_patch_applier() -> None:
    # Same AST-based import check as test_field_topology_plasticity.py -- checks for
    # an actual import statement, not a prose mention (this module's docstring
    # explains *why* PatchApplier is intentionally not used).
    import orion.substrate.field_topology_learned_store as store_module

    tree = ast.parse(Path(store_module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any("mutation_apply" in alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            assert "mutation_apply" not in module_name
            assert not any(alias.name == "PatchApplier" for alias in node.names)
