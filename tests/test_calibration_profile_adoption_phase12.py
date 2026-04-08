from __future__ import annotations

import pytest
from pathlib import Path

from orion.core.schemas.calibration_adoption import (
    CalibrationAdoptionRequestV1,
    CalibrationProfileV1,
    CalibrationRollbackRequestV1,
    CalibrationRolloutScopeV1,
)
from orion.reasoning.calibration_profiles import CalibrationRuntimeContext, InMemoryCalibrationProfileStore, SqlCalibrationProfileStore


def _profile(**kwargs) -> CalibrationProfileV1:
    base = CalibrationProfileV1(
        created_by="operator:alice",
        rationale="phase12_canary",
        source_recommendation_ids=["endogenous-rec-1"],
        source_evaluation_request_id="endogenous-eval-1",
        overrides={"trigger.reflective_threshold": "0.5", "cooldown.workflow.reflective_journal": "240"},
        scope=CalibrationRolloutScopeV1(invocation_surfaces=["operator_review"], sample_percent=100, operator_only=True),
    )
    return base.model_copy(update=kwargs)


def test_profile_validation_rejects_invalid_override_key() -> None:
    store = InMemoryCalibrationProfileStore()
    bad = _profile(overrides={"unknown.threshold": "1"})
    with pytest.raises(ValueError, match="invalid_override_keys"):
        store.apply(CalibrationAdoptionRequestV1(action="stage", operator_id="op", rationale="bad", profile=bad))


def test_manual_stage_then_activate_is_required_and_deterministic() -> None:
    store = InMemoryCalibrationProfileStore()
    profile = _profile()
    staged = store.apply(CalibrationAdoptionRequestV1(action="stage", operator_id="op", rationale="stage", profile=profile))
    assert staged.accepted is True
    assert staged.active_profile_id is None

    before = store.resolve(context=CalibrationRuntimeContext(invocation_surface="operator_review", operator_surface=True, seed="a"))
    assert before.mode == "baseline"

    activated = store.apply(
        CalibrationAdoptionRequestV1(action="activate", operator_id="op", rationale="go", profile_id=profile.profile_id)
    )
    assert activated.accepted is True
    assert activated.active_profile_id == profile.profile_id

    first = store.resolve(context=CalibrationRuntimeContext(invocation_surface="operator_review", operator_surface=True, seed="x"))
    second = store.resolve(context=CalibrationRuntimeContext(invocation_surface="operator_review", operator_surface=True, seed="x"))
    assert first.mode == "adopted"
    assert first.profile_id == second.profile_id
    assert first.overrides == second.overrides


def test_scope_and_canary_keep_baseline_outside_target() -> None:
    store = InMemoryCalibrationProfileStore()
    profile = _profile(
        scope=CalibrationRolloutScopeV1(invocation_surfaces=["chat_reflective_lane"], sample_percent=1, operator_only=False)
    )
    store.apply(CalibrationAdoptionRequestV1(action="stage", operator_id="op", rationale="stage", profile=profile))
    store.apply(CalibrationAdoptionRequestV1(action="activate", operator_id="op", rationale="activate", profile_id=profile.profile_id))

    out_of_scope = store.resolve(context=CalibrationRuntimeContext(invocation_surface="operator_review", operator_surface=True, seed="x"))
    assert out_of_scope.mode == "baseline"

    sampled_out = store.resolve(context=CalibrationRuntimeContext(invocation_surface="chat_reflective_lane", seed="seed-1"))
    assert sampled_out.mode in {"baseline", "adopted"}


def test_rollback_to_previous_and_baseline_preserves_audit() -> None:
    store = InMemoryCalibrationProfileStore()
    p1 = _profile(profile_id="p1")
    p2 = _profile(profile_id="p2", overrides={"trigger.reflective_threshold": "0.55"})

    store.apply(CalibrationAdoptionRequestV1(action="stage", operator_id="op", rationale="s1", profile=p1))
    store.apply(CalibrationAdoptionRequestV1(action="activate", operator_id="op", rationale="a1", profile_id="p1"))
    store.apply(CalibrationAdoptionRequestV1(action="stage", operator_id="op", rationale="s2", profile=p2))
    store.apply(CalibrationAdoptionRequestV1(action="activate", operator_id="op", rationale="a2", profile_id="p2"))

    rolled = store.rollback(CalibrationRollbackRequestV1(operator_id="op", rationale="revert", rollback_mode="to_previous"))
    assert rolled.active_profile_id == "p1"
    assert rolled.rolled_back_profile_id == "p2"

    baseline = store.rollback(CalibrationRollbackRequestV1(operator_id="op", rationale="baseline", rollback_mode="to_baseline"))
    assert baseline.active_profile_id is None

    audit = store.list_audit(limit=10)
    assert any(item.event_type == "activated" for item in audit)
    assert any(item.event_type in {"rolled_back", "reverted_to_baseline"} for item in audit)


def test_sql_store_stage_activate_rollback_is_restart_safe(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'calibration_state.db'}"
    store = SqlCalibrationProfileStore(database_url=db_url)
    profile = _profile(profile_id="sql-p1")

    staged = store.apply(CalibrationAdoptionRequestV1(action="stage", operator_id="op", rationale="stage", profile=profile))
    assert staged.accepted is True
    assert store.active_profile() is None

    activated = store.apply(CalibrationAdoptionRequestV1(action="activate", operator_id="op", rationale="activate", profile_id="sql-p1"))
    assert activated.accepted is True
    assert store.active_profile() is not None
    assert store.active_profile().profile_id == "sql-p1"

    restarted = SqlCalibrationProfileStore(database_url=db_url)
    assert restarted.active_profile() is not None
    assert restarted.active_profile().profile_id == "sql-p1"

    rolled = restarted.rollback(CalibrationRollbackRequestV1(operator_id="op", rationale="baseline", rollback_mode="to_baseline"))
    assert rolled.accepted is True
    assert restarted.active_profile() is None


def test_sql_store_staged_profile_does_not_auto_activate(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'calibration_state_stage_only.db'}"
    store = SqlCalibrationProfileStore(database_url=db_url)
    store.apply(CalibrationAdoptionRequestV1(action="stage", operator_id="op", rationale="stage", profile=_profile(profile_id="sql-stage")))
    resolved = store.resolve(context=CalibrationRuntimeContext(invocation_surface="operator_review", operator_surface=True, seed="x"))
    assert resolved.mode == "baseline"
