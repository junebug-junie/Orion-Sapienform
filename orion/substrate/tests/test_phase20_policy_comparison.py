from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.core.schemas.substrate_policy_adoption import (
    SubstratePolicyAdoptionRequestV1,
    SubstratePolicyOverridesV1,
    SubstratePolicyRolloutScopeV1,
)
from orion.core.schemas.substrate_policy_comparison import SubstratePolicyComparisonRequestV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1
from orion.substrate.policy_comparison import SubstratePolicyComparisonService
from orion.substrate.policy_profiles import SubstratePolicyProfileStore
from orion.substrate.review_telemetry import GraphReviewTelemetryRecorder


def _scope() -> SubstratePolicyRolloutScopeV1:
    return SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"])


def _seed_profiles(store: SubstratePolicyProfileStore) -> tuple[str, str]:
    first = store.adopt(
        SubstratePolicyAdoptionRequestV1(
            rollout_scope=_scope(),
            policy_overrides=SubstratePolicyOverridesV1(normal_revisit_seconds=1200),
            activate_now=True,
            operator_id="op",
            rationale="first",
        )
    )
    second = store.adopt(
        SubstratePolicyAdoptionRequestV1(
            rollout_scope=_scope(),
            policy_overrides=SubstratePolicyOverridesV1(normal_revisit_seconds=900),
            activate_now=True,
            operator_id="op",
            rationale="second",
        )
    )
    assert first.profile_id and second.profile_id
    return first.profile_id, second.profile_id


def _record(
    recorder: GraphReviewTelemetryRecorder,
    *,
    profile_id: str | None,
    outcome: str = "executed",
    cycles: int = 1,
    followup: bool = False,
    zone: str = "concept_graph",
    surface: str = "operator_review",
) -> None:
    recorder.record(
        GraphReviewTelemetryRecordV1(
            policy_profile_id=profile_id,
            invocation_surface=surface,
            target_zone=zone,
            selection_reason="test",
            execution_outcome=outcome,
            runtime_duration_ms=120,
            cycle_count_before=cycles,
            frontier_followup_invoked=followup,
            selected_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
    )


def test_pair_resolution_and_selected_pair_with_sql_state(tmp_path) -> None:
    db_path = tmp_path / "policy.sqlite3"
    store = SubstratePolicyProfileStore(sql_db_path=str(db_path))
    previous_id, current_id = _seed_profiles(store)
    telemetry = GraphReviewTelemetryRecorder(sql_db_path=str(tmp_path / "telemetry.sqlite3"))
    svc = SubstratePolicyComparisonService(policy_store=store, telemetry_recorder=telemetry)

    a = svc.resolve_pair(request=SubstratePolicyComparisonRequestV1(pair_mode="baseline_vs_active"))
    assert a.baseline_profile_id == "baseline"
    assert a.candidate_profile_id == current_id

    b = svc.resolve_pair(request=SubstratePolicyComparisonRequestV1(pair_mode="previous_vs_current"))
    assert b.baseline_profile_id == previous_id
    assert b.candidate_profile_id == current_id

    c = svc.resolve_pair(
        request=SubstratePolicyComparisonRequestV1(
            pair_mode="selected_pair",
            baseline_profile_id=previous_id,
            candidate_profile_id=current_id,
        )
    )
    assert c.baseline_profile_id == previous_id
    assert c.candidate_profile_id == current_id


def test_comparison_metrics_and_verdicts_are_deterministic(tmp_path) -> None:
    store = SubstratePolicyProfileStore(sql_db_path=str(tmp_path / "policy.sqlite3"))
    previous_id, current_id = _seed_profiles(store)
    telemetry = GraphReviewTelemetryRecorder(sql_db_path=str(tmp_path / "telemetry.sqlite3"))
    svc = SubstratePolicyComparisonService(policy_store=store, telemetry_recorder=telemetry)

    for _ in range(12):
        _record(telemetry, profile_id=previous_id, outcome="failed", cycles=3, followup=False)
    for _ in range(12):
        _record(telemetry, profile_id=current_id, outcome="executed", cycles=1, followup=True)

    result = svc.compare(request=SubstratePolicyComparisonRequestV1(pair_mode="previous_vs_current", sample_limit=200))
    report = result["report"]
    assert report["verdict"] == "improved"
    assert report["confidence"] >= 0.75
    assert result["samples"]["baseline"] == 12
    assert result["samples"]["candidate"] == 12
    metric_names = {item["metric"] for item in report["metric_deltas"]}
    assert "failed_rate" in metric_names
    assert "frontier_followup_rate" in metric_names
    assert "strict_zone_surface_rate" in metric_names


def test_insufficient_data_and_missing_profiles_fail_safely(tmp_path) -> None:
    store = SubstratePolicyProfileStore(sql_db_path=str(tmp_path / "policy.sqlite3"))
    _, current_id = _seed_profiles(store)
    telemetry = GraphReviewTelemetryRecorder(sql_db_path=str(tmp_path / "telemetry.sqlite3"))
    svc = SubstratePolicyComparisonService(policy_store=store, telemetry_recorder=telemetry)

    _record(telemetry, profile_id=None, outcome="noop")
    _record(telemetry, profile_id=current_id, outcome="executed")

    insufficient = svc.compare(request=SubstratePolicyComparisonRequestV1(pair_mode="baseline_vs_active", sample_limit=50))
    assert insufficient["report"]["verdict"] == "insufficient_data"
    assert insufficient["advisory"]["mutating"] is False

    with pytest.raises(ValueError):
        svc.resolve_pair(
            request=SubstratePolicyComparisonRequestV1(
                pair_mode="selected_pair",
                baseline_profile_id="missing-profile",
                candidate_profile_id=current_id,
            )
        )


def test_postgres_preferred_source_kind_for_control_plane_reads(monkeypatch) -> None:
    monkeypatch.setattr(
        SubstratePolicyProfileStore,
        "_ensure_postgres_schema",
        lambda self: None,
    )
    monkeypatch.setattr(
        SubstratePolicyProfileStore,
        "_load_from_postgres",
        lambda self: None,
    )
    monkeypatch.setattr(
        GraphReviewTelemetryRecorder,
        "_ensure_postgres_schema",
        lambda self: None,
    )
    monkeypatch.setattr(
        GraphReviewTelemetryRecorder,
        "_load_from_postgres",
        lambda self: [],
    )

    store = SubstratePolicyProfileStore(postgres_url="postgresql://unit:test@localhost/db")
    recorder = GraphReviewTelemetryRecorder(postgres_url="postgresql://unit:test@localhost/db")
    assert store.source_kind() == "postgres"
    assert recorder.source_kind() == "postgres"


def test_postgres_failure_is_explicit_fallback_not_silent(tmp_path) -> None:
    store = SubstratePolicyProfileStore(
        postgres_url="postgresql://invalid:invalid@localhost:1/nope",
        sql_db_path=str(tmp_path / "policy.sqlite3"),
    )
    recorder = GraphReviewTelemetryRecorder(
        postgres_url="postgresql://invalid:invalid@localhost:1/nope",
        sql_db_path=str(tmp_path / "telemetry.sqlite3"),
    )
    assert store.degraded() is True
    assert recorder.degraded() is True
    assert store.last_error()
    assert recorder.last_error()
