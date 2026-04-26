from __future__ import annotations

import os
import sys
import threading
import time
import importlib.util
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
hub_scripts_pkg = HUB_ROOT / "scripts" / "__init__.py"
if (
    "scripts" not in sys.modules
    or not str(getattr(sys.modules.get("scripts"), "__file__", "")).startswith(str(HUB_ROOT))
):
    spec = importlib.util.spec_from_file_location(
        "scripts",
        str(hub_scripts_pkg),
        submodule_search_locations=[str(HUB_ROOT / "scripts")],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["scripts"] = module
        spec.loader.exec_module(module)

from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate import mutation_control_surface
from scripts import api_routes


@pytest.fixture
def scheduler_fixture(monkeypatch):
    control_db = str((Path("/tmp") / f"orion-mutation-control-scheduler-{uuid4()}.sqlite3"))
    monkeypatch.setenv("SUBSTRATE_MUTATION_CONTROL_SQL_DB_PATH", control_db)
    mutation_control_surface._CONTROL_SURFACE_STORE = mutation_control_surface.RuntimeControlSurfaceStore(sql_db_path=control_db)
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.5, actor="scheduler_seed")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_PROPOSALS_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_APPLY_ENABLED", "false")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_MONITOR_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED", "false")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED", "false")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_ROLLBACK_DELTA_THRESHOLD", "-0.05")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_INTERVAL_SEC", "5")
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", SubstrateMutationStore())
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_SURFACES", {"routing": {"chat_reflective_lane_threshold": 0.5}})
    monkeypatch.setattr(api_routes, "substrate_autonomy_runtime_supported", lambda: (True, "supported"))
    return {
        "routing_telemetry": [
            {
                "invocation_surface": "operator_review",
                "execution_outcome": "failed",
                "selection_reason": "scheduler-proof",
                "selected_priority": 95,
                "runtime_duration_ms": 12,
                "anchor_scope": "orion",
                "subject_ref": "entity:scheduler-proof",
                "target_zone": "autonomy_graph",
            },
            {
                "invocation_surface": "operator_review",
                "execution_outcome": "executed",
                "selection_reason": "scheduler-borderline",
                "selected_priority": 55,
                "runtime_duration_ms": 9,
                "anchor_scope": "orion",
                "subject_ref": "entity:scheduler-proof",
                "target_zone": "autonomy_graph",
            },
        ],
        "routing_metrics": {
            "routing_threshold_patch": {"success_rate_delta": 0.2, "latency_ms_delta": 0.0},
        },
    }


def test_scheduler_disabled_mode_noop(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ENABLED", "false")
    payload = api_routes.execute_substrate_mutation_scheduled_cycle()
    assert payload["status"] == "disabled"
    assert payload["event"] == "mutation_scheduler_tick"


def test_scheduler_enabled_unsupported_store_noops_loudly(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ENABLED", "true")
    monkeypatch.setattr(api_routes, "substrate_autonomy_runtime_supported", lambda: (False, "unsupported_store_kind:memory"))
    payload = api_routes.execute_substrate_mutation_scheduled_cycle()
    assert payload["status"] == "unsafe_mode_noop"
    assert payload["reason"] == "unsupported_store_kind:memory"


def test_scheduler_enabled_mode_runs_single_cycle(scheduler_fixture) -> None:
    payload = api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=scheduler_fixture["routing_telemetry"],
        class_metrics_override=scheduler_fixture["routing_metrics"],
    )
    assert payload["status"] == "completed"
    assert payload["summary"]["signals_processed"] >= 1


def test_scheduler_lock_contention_blocks_second_runner(scheduler_fixture) -> None:
    acquired = api_routes.SUBSTRATE_AUTONOMY_LOCAL_CYCLE_LOCK.acquire(blocking=False)
    assert acquired is True
    try:
        payload = api_routes.execute_substrate_mutation_scheduled_cycle(
            telemetry_override=scheduler_fixture["routing_telemetry"],
            class_metrics_override=scheduler_fixture["routing_metrics"],
        )
    finally:
        api_routes.SUBSTRATE_AUTONOMY_LOCAL_CYCLE_LOCK.release()
    assert payload["status"] == "blocked"
    assert payload["reason"] == "local_cycle_lock_not_acquired"


def test_scheduler_concurrent_attempts_do_not_duplicate_cycle(monkeypatch, scheduler_fixture) -> None:
    calls = {"count": 0}
    gate = threading.Barrier(2)
    lock = threading.Lock()
    original_run_cycle = api_routes.SubstrateAdaptationWorker.run_cycle

    def slow_run_cycle(self, **kwargs):  # type: ignore[no-untyped-def]
        with lock:
            calls["count"] += 1
        time.sleep(0.15)
        return original_run_cycle(self, **kwargs)

    monkeypatch.setattr(api_routes.SubstrateAdaptationWorker, "run_cycle", slow_run_cycle)
    results: list[dict] = []

    def runner() -> None:
        gate.wait()
        payload = api_routes.execute_substrate_mutation_scheduled_cycle(
            telemetry_override=scheduler_fixture["routing_telemetry"],
            class_metrics_override=scheduler_fixture["routing_metrics"],
        )
        results.append(payload)

    first = threading.Thread(target=runner)
    second = threading.Thread(target=runner)
    first.start()
    second.start()
    first.join()
    second.join()

    statuses = sorted(result["status"] for result in results)
    assert statuses == ["blocked", "completed"]
    assert calls["count"] == 1


def test_scheduler_apply_gating_respected_when_apply_disabled(monkeypatch, scheduler_fixture) -> None:
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_APPLY_ENABLED", "false")
    # First tick emits proposals; second tick deterministically processes due queue items.
    api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=scheduler_fixture["routing_telemetry"],
        class_metrics_override=scheduler_fixture["routing_metrics"],
    )
    payload = api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=[],
        class_metrics_override=scheduler_fixture["routing_metrics"],
    )
    assert payload["summary"]["applies_attempted"] >= 1
    assert payload["summary"]["applies_executed"] == 0
    assert payload["summary"]["applies_blocked"] >= 1
    assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.5


def test_scheduler_apply_enabled_updates_live_routing_surface(monkeypatch, scheduler_fixture) -> None:
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_APPLY_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED", "true")
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.5, actor="before_scheduler_apply")
    api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=scheduler_fixture["routing_telemetry"],
        class_metrics_override=scheduler_fixture["routing_metrics"],
    )
    payload = api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=[],
        class_metrics_override=scheduler_fixture["routing_metrics"],
    )
    assert payload["summary"]["applies_executed"] >= 1
    assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.58


def test_scheduler_apply_enabled_can_use_replay_metrics_without_class_override(monkeypatch, scheduler_fixture) -> None:
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_APPLY_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED", "true")
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.5, actor="before_scheduler_replay_only")
    api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=scheduler_fixture["routing_telemetry"],
        class_metrics_override={},
    )
    payload = api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=scheduler_fixture["routing_telemetry"],
        class_metrics_override={},
    )
    assert payload["summary"]["trials_executed"] >= 1
    assert payload["summary"]["decisions_made"] >= 1
    assert payload["summary"]["applies_executed"] >= 1
    assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.58


def test_scheduler_operator_gated_class_never_auto_applies(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_PROPOSALS_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_APPLY_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED", "true")
    monkeypatch.setattr(api_routes, "substrate_autonomy_runtime_supported", lambda: (True, "supported"))
    control_db = str((Path("/tmp") / f"orion-mutation-control-scheduler-prompt-{uuid4()}.sqlite3"))
    monkeypatch.setenv("SUBSTRATE_MUTATION_CONTROL_SQL_DB_PATH", control_db)
    mutation_control_surface._CONTROL_SURFACE_STORE = mutation_control_surface.RuntimeControlSurfaceStore(sql_db_path=control_db)
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.44, actor="prompt_class_seed")
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", SubstrateMutationStore())
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_SURFACES", {"prompts": {"approved_profile_variant": "v1"}})
    telemetry = [
        {
            "invocation_surface": "operator_review",
            "execution_outcome": "failed",
            "selection_reason": "scheduler-prompt-proof",
            "runtime_duration_ms": 9,
            "anchor_scope": "orion",
            "subject_ref": "entity:prompt-proof",
            "target_zone": "self_relationship_graph",
        }
    ]
    class_metrics = {
        "cognitive_social_continuity_repair": {
            "operator_acceptance_rate": 0.8,
        }
    }
    api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=telemetry,
        class_metrics_override=class_metrics,
    )
    payload = api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=[],
        class_metrics_override=class_metrics,
    )
    assert payload["summary"]["applies_executed"] == 0
    assert payload["summary"]["decisions_made"] == 0
    assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.44


def test_scheduler_routing_ramp_decision_only_mode(monkeypatch, scheduler_fixture) -> None:
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_APPLY_ENABLED", "false")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED", "false")
    api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=scheduler_fixture["routing_telemetry"],
        class_metrics_override={},
    )
    payload = api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=scheduler_fixture["routing_telemetry"],
        class_metrics_override={},
    )
    assert payload["summary"]["proposals_created"] >= 0
    assert payload["summary"]["trials_executed"] >= 1
    assert payload["summary"]["decisions_made"] >= 1
    assert payload["summary"]["applies_executed"] == 0


def test_scheduler_global_apply_true_but_routing_gate_disabled_blocks_apply(monkeypatch, scheduler_fixture) -> None:
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_APPLY_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED", "false")
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.5, actor="routing_gate_disabled")
    api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=scheduler_fixture["routing_telemetry"],
        class_metrics_override={},
    )
    payload = api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=scheduler_fixture["routing_telemetry"],
        class_metrics_override={},
    )
    assert payload["summary"]["applies_attempted"] >= 1
    assert payload["summary"]["applies_executed"] == 0
    assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.5


def test_scheduler_routing_monitoring_regression_triggers_rollback(monkeypatch, scheduler_fixture) -> None:
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_APPLY_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_ROLLBACK_DELTA_THRESHOLD", "-0.05")
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.5, actor="before_monitoring_regression")
    api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=scheduler_fixture["routing_telemetry"],
        class_metrics_override={},
    )
    payload = api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=scheduler_fixture["routing_telemetry"],
        class_metrics_override={},
        post_adoption_delta_by_target_surface_override={"routing": -0.2},
    )
    assert payload["summary"]["applies_executed"] >= 1
    assert any(event.get("event") == "mutation_rollback_recorded" for event in payload["trace"])
    assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.5


def test_scheduler_cognitive_proposals_enabled_never_auto_apply(monkeypatch, scheduler_fixture) -> None:
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_APPLY_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED", "false")
    telemetry = [
        {
            "invocation_surface": "chat_reflective_lane",
            "execution_outcome": "executed",
            "selection_reason": "stance_drift",
            "runtime_duration_ms": 0,
            "anchor_scope": "orion",
            "subject_ref": "entity:cognitive-proof",
            "target_zone": "self_relationship_graph",
            "pressure_events": [
                {
                    "source_service": "orion-hub",
                    "source_event_id": "feedback-cog-1",
                    "correlation_id": "corr-cog-1",
                    "pressure_category": "social_addressedness_gap",
                    "confidence": 0.72,
                    "evidence_refs": ["feedback:fb-cog-1"],
                }
            ],
        }
    ]
    class_metrics = {
        "approved_prompt_profile_variant_promotion": {
            "quality_score_delta": 0.2,
            "safety_incident_delta": 0.0,
        }
    }
    api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=telemetry,
        class_metrics_override=class_metrics,
    )
    payload = api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=[],
        class_metrics_override=class_metrics,
    )
    cognitive_proposals = [
        proposal for proposal in api_routes.SUBSTRATE_MUTATION_STORE._proposals.values() if proposal.lane == "cognitive"
    ]
    assert cognitive_proposals
    assert all(
        proposal.mutation_class in {
            "cognitive_contradiction_reconciliation",
            "cognitive_identity_continuity_adjustment",
            "cognitive_stance_continuity_adjustment",
            "cognitive_social_continuity_repair",
        }
        for proposal in cognitive_proposals
    )
    assert payload["summary"]["applies_executed"] == 0
    assert all(
        proposal.rollout_state in {"pending_review", "trialed", "proposed", "queued", "rejected"}
        for proposal in cognitive_proposals
    )


def test_scheduler_recall_strategy_proposals_are_operator_gated_no_apply(monkeypatch, scheduler_fixture) -> None:
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_APPLY_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED", "false")
    telemetry = [
        {
            "invocation_surface": "chat_reflective_lane",
            "execution_outcome": "executed",
            "selection_reason": "producer_pressure_events:recall-evt",
            "runtime_duration_ms": 0,
            "anchor_scope": "orion",
            "subject_ref": "entity:recall-proof",
            "target_zone": "autonomy_graph",
            "pressure_events": [
                {
                    "source_service": "orion-recall",
                    "source_event_id": "recall-evt-1",
                    "correlation_id": "corr-recall-1",
                    "pressure_category": "missing_exact_anchor",
                    "confidence": 0.82,
                    "evidence_refs": ["recall_decision:abc"],
                    "metadata": {
                        "v1_v2_compare": {"v1_latency_ms": 120, "v2_latency_ms": 95, "selected_count_delta": 2},
                        "anchor_plan": {"time_window_days": 1},
                        "selected_evidence_cards": [{"id": "page-1"}],
                    },
                }
            ],
        }
    ]
    api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=telemetry,
        class_metrics_override={},
    )
    payload = api_routes.execute_substrate_mutation_scheduled_cycle(
        telemetry_override=[],
        class_metrics_override={},
    )
    recall_proposals = [
        proposal
        for proposal in api_routes.SUBSTRATE_MUTATION_STORE._proposals.values()
        if proposal.mutation_class
        in {
            "recall_strategy_profile_candidate",
            "recall_anchor_policy_candidate",
            "recall_page_index_profile_candidate",
            "recall_graph_expansion_policy_candidate",
        }
    ]
    assert recall_proposals
    assert payload["summary"]["applies_executed"] == 0
    assert all(proposal.rollout_state in {"pending_review", "trialed", "proposed", "queued", "rejected"} for proposal in recall_proposals)
