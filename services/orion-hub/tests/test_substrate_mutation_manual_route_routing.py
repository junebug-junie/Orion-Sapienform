from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate import mutation_control_surface
from scripts import api_routes


@pytest.fixture
def routing_manual_fixture(monkeypatch):
    control_db = str((Path("/tmp") / f"orion-mutation-control-test-{uuid4()}.sqlite3"))
    monkeypatch.setenv("SUBSTRATE_MUTATION_CONTROL_SQL_DB_PATH", control_db)
    mutation_control_surface._CONTROL_SURFACE_STORE = mutation_control_surface.RuntimeControlSurfaceStore(sql_db_path=control_db)
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.5, actor="test_seed")
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret-token")
    monkeypatch.setenv("SUBSTRATE_MUTATION_MANUAL_APPLY_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true")
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", SubstrateMutationStore())
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_SURFACES", {"routing": {"chat_reflective_lane_threshold": 0.5}})
    return {
        "token": "secret-token",
        "telemetry_routing": [
            {
                "invocation_surface": "operator_review",
                "execution_outcome": "failed",
                "selection_reason": "manual-routing-proof",
                "selected_priority": 95,
                "runtime_duration_ms": 12,
                "anchor_scope": "orion",
                "subject_ref": "entity:routing-proof",
                "target_zone": "autonomy_graph",
            },
            {
                "invocation_surface": "operator_review",
                "execution_outcome": "executed",
                "selection_reason": "manual-routing-borderline",
                "selected_priority": 55,
                "runtime_duration_ms": 9,
                "anchor_scope": "orion",
                "subject_ref": "entity:routing-proof",
                "target_zone": "autonomy_graph",
            },
        ],
        "routing_pass_metrics": {
            "routing_threshold_patch": {"success_rate_delta": 0.2, "latency_ms_delta": 0.0}
        },
        "routing_fail_metrics": {
            "routing_threshold_patch": {"success_rate_delta": -0.2, "latency_ms_delta": -1.0}
        },
    }


def test_routing_dry_run_produces_trial_and_decision_without_side_effects(routing_manual_fixture) -> None:
    payload = api_routes.api_substrate_mutation_runtime_execute_once(
        request=api_routes.SubstrateMutationExecuteRequest(
            dry_run=True,
            apply_enabled=False,
            telemetry=routing_manual_fixture["telemetry_routing"],
            class_metrics=routing_manual_fixture["routing_pass_metrics"],
        ),
        x_orion_operator_token=routing_manual_fixture["token"],
    )
    assert payload["summary"]["signals_produced"] >= 1
    assert payload["summary"]["proposals_created"] >= 1
    assert payload["summary"]["trials_run"] >= 1
    assert payload["summary"]["decisions_made"] >= 1
    assert payload["summary"]["applies_completed"] == 0
    assert api_routes.SUBSTRATE_MUTATION_SURFACES["routing"]["chat_reflective_lane_threshold"] == 0.5


def test_routing_apply_succeeds_for_auto_promote_and_can_rollback(routing_manual_fixture) -> None:
    payload = api_routes.api_substrate_mutation_runtime_execute_once(
        request=api_routes.SubstrateMutationExecuteRequest(
            dry_run=False,
            apply_enabled=True,
            telemetry=routing_manual_fixture["telemetry_routing"],
            class_metrics=routing_manual_fixture["routing_pass_metrics"],
            post_adoption_delta_by_target_surface={"routing": -0.2},
        ),
        x_orion_operator_token=routing_manual_fixture["token"],
    )
    assert payload["summary"]["applies_attempted"] >= 1
    assert payload["summary"]["applies_completed"] >= 1
    assert payload["summary"]["rollbacks_recorded"] >= 1
    assert api_routes.SUBSTRATE_MUTATION_SURFACES["routing"]["chat_reflective_lane_threshold"] == 0.5
    assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.5


def test_routing_manual_apply_changes_real_live_routing_surface(routing_manual_fixture) -> None:
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.5, actor="test_before_apply")
    payload = api_routes.api_substrate_mutation_runtime_execute_once(
        request=api_routes.SubstrateMutationExecuteRequest(
            dry_run=False,
            apply_enabled=True,
            telemetry=routing_manual_fixture["telemetry_routing"],
            class_metrics=routing_manual_fixture["routing_pass_metrics"],
            post_adoption_delta_by_target_surface={},
        ),
        x_orion_operator_token=routing_manual_fixture["token"],
    )
    assert payload["summary"]["applies_completed"] >= 1
    assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.58


def test_routing_manual_apply_can_use_replay_metrics_without_class_metrics(routing_manual_fixture) -> None:
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.5, actor="replay_only_before_apply")
    payload = api_routes.api_substrate_mutation_runtime_execute_once(
        request=api_routes.SubstrateMutationExecuteRequest(
            dry_run=False,
            apply_enabled=True,
            telemetry=routing_manual_fixture["telemetry_routing"],
            class_metrics={},
        ),
        x_orion_operator_token=routing_manual_fixture["token"],
    )
    assert payload["summary"]["trials_run"] >= 1
    assert payload["summary"]["decisions_made"] >= 1
    assert payload["summary"]["applies_completed"] >= 1
    assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.58


def test_require_review_never_applies_via_manual_route(routing_manual_fixture) -> None:
    payload = api_routes.api_substrate_mutation_runtime_execute_once(
        request=api_routes.SubstrateMutationExecuteRequest(
            dry_run=False,
            apply_enabled=True,
            telemetry=[
                {
                    "invocation_surface": "operator_review",
                        "execution_outcome": "failed",
                    "selection_reason": "manual-prompt-proof",
                    "runtime_duration_ms": 5,
                    "anchor_scope": "orion",
                    "subject_ref": "entity:prompt-proof",
                    "target_zone": "self_relationship_graph",
                }
            ],
            class_metrics={"approved_prompt_profile_variant_promotion": {"quality_score_delta": 0.2, "safety_incident_delta": 0.0}},
        ),
        x_orion_operator_token=routing_manual_fixture["token"],
    )
    assert payload["summary"]["applies_completed"] == 0
    assert any("decision=require_review" in str(event) or event.get("decision") == "require_review" for event in payload["trace"]["events"])


def test_one_live_surface_invariant_blocks_before_side_effects(routing_manual_fixture) -> None:
    api_routes.SUBSTRATE_MUTATION_STORE._active_surface_by_target["routing"] = "existing-adoption"
    payload = api_routes.api_substrate_mutation_runtime_execute_once(
        request=api_routes.SubstrateMutationExecuteRequest(
            dry_run=False,
            apply_enabled=True,
            telemetry=routing_manual_fixture["telemetry_routing"],
            class_metrics=routing_manual_fixture["routing_pass_metrics"],
        ),
        x_orion_operator_token=routing_manual_fixture["token"],
    )
    assert payload["summary"]["applies_completed"] == 0
    assert any(event.get("decision") == "hold" for event in payload["trace"]["events"])
    assert api_routes.SUBSTRATE_MUTATION_SURFACES["routing"]["chat_reflective_lane_threshold"] == 0.5


def test_failed_trial_blocks_apply(routing_manual_fixture) -> None:
    payload = api_routes.api_substrate_mutation_runtime_execute_once(
        request=api_routes.SubstrateMutationExecuteRequest(
            dry_run=False,
            apply_enabled=True,
            telemetry=routing_manual_fixture["telemetry_routing"],
            class_metrics=routing_manual_fixture["routing_fail_metrics"],
        ),
        x_orion_operator_token=routing_manual_fixture["token"],
    )
    assert payload["summary"]["trials_run"] >= 1
    assert payload["summary"]["applies_completed"] == 0

