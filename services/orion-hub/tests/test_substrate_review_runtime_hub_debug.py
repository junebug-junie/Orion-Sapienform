from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path

import pytest
from fastapi import HTTPException

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

from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeResultV1
from scripts import api_routes
from orion.core.schemas.substrate_mutation import MutationPressureEvidenceV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1


TEMPLATE_PATH = HUB_ROOT / "templates" / "index.html"
APP_JS_PATH = HUB_ROOT / "static" / "js" / "app.js"


def test_template_has_substrate_review_debug_row_modal_and_standalone_link() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert 'id="substrateReviewDebugPanel"' in template
    assert 'id="substrateReviewDebugToggle"' in template
    assert 'id="substrateReviewDebugOpenModal"' in template
    assert 'id="substrateReviewModalRoot"' in template
    assert 'id="substrateReviewModalBackdrop"' in template
    assert 'id="substrateReviewModalDialog"' in template
    assert 'id="substrateReviewActionExecuteOnce"' in template
    assert 'id="substrateReviewActionExecuteFollowup"' in template
    assert 'id="substrateReviewActionSmokeCheck"' in template
    assert 'id="substrateReviewActionRefresh"' in template
    assert 'id="substrateReviewModalSubstrateLink" href="/substrate"' in template


def test_app_js_wires_substrate_review_debug_panel_modal_and_actions() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "function clearSubstrateReviewDebugPanel()" in app_js
    assert "function updateSubstrateReviewDebugPanel(statusPayload)" in app_js
    assert "function toggleSubstrateReviewDebugPanel()" in app_js
    assert "function openSubstrateReviewModal()" in app_js
    assert "function closeSubstrateReviewModal()" in app_js
    assert "function runSubstrateReviewExecuteOnce()" in app_js
    assert "function runSubstrateReviewExecuteOnceWithFollowup()" in app_js
    assert "function refreshSubstrateReviewStatus()" in app_js
    assert "function runSubstrateReviewSmokeCheck()" in app_js
    assert "substrateReviewDebugOpenModal.addEventListener('click'" in app_js
    assert "substrateReviewActionExecuteOnce.addEventListener('click'" in app_js
    assert "substrateReviewActionExecuteFollowup.addEventListener('click'" in app_js
    assert "substrateReviewActionRefresh.addEventListener('click'" in app_js
    assert "substrateReviewActionSmokeCheck.addEventListener('click'" in app_js


def test_app_js_includes_substrate_in_shell_tab_switching_without_full_navigation() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert 'const substrateTabButton = document.getElementById("substratePageLink");' in app_js
    assert 'const substratePanel = document.getElementById("substrate");' in app_js
    assert 'const isSubstrate = tabKey === "substrate";' in app_js
    assert 'substratePanel.classList.toggle("hidden", !isSubstrate);' in app_js
    assert 'styleTabButton(substrateTabButton, isSubstrate);' in app_js
    assert 'substrateTabButton.addEventListener("click", (event) => {' in app_js
    assert "event.preventDefault();" in app_js
    assert 'setActiveTab("substrate");' in app_js
    assert 'history.replaceState(null, "", "#substrate");' in app_js
    assert 'window.location.hash === "#substrate"' in app_js


def test_app_js_supports_substrate_embed_refresh_without_merging_substrate_bundle() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert 'const substratePanelFrame = document.getElementById("substratePanelFrame");' in app_js
    assert 'const substratePanelRefresh = document.getElementById("substratePanelRefresh");' in app_js
    assert 'substratePanelFrame.contentWindow?.location.reload();' in app_js
    assert "substrate.js" not in app_js


def test_substrate_review_status_endpoint_returns_typed_summary_and_source_posture() -> None:
    payload = api_routes.api_substrate_review_runtime_status(queue_limit=10, telemetry_limit=10)

    assert "generated_at" in payload
    assert "summary" in payload
    assert "source" in payload
    assert "queue_count" in payload["summary"]
    assert "due_count" in payload["summary"]
    assert "control_plane" in payload["source"]
    assert "semantic" in payload["source"]


def test_execute_once_endpoint_is_single_cycle_operator_only_and_followup_default_off(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_execute_once(self, *, request, now=None):
        observed["calls"] = int(observed.get("calls", 0)) + 1
        observed["surface"] = request.invocation_surface
        observed["followup"] = request.execute_frontier_followup_allowed
        observed["strict_override"] = request.operator_override_strict_zone
        return GraphReviewRuntimeResultV1(
            request_id=request.request_id,
            correlation_id=request.correlation_id,
            outcome="executed",
            selected_queue_item_id="queue-1",
            notes=["single_cycle_execution"],
        )

    monkeypatch.setattr(type(api_routes.SUBSTRATE_REVIEW_RUNTIME_EXECUTOR), "execute_once", fake_execute_once)
    payload = api_routes.api_substrate_review_runtime_execute_once()

    assert observed["calls"] == 1
    assert observed["surface"] == "operator_review"
    assert observed["followup"] is False
    assert observed["strict_override"] is False
    assert payload["request"]["single_cycle"] is True
    assert payload["result"]["outcome"] == "executed"


def test_execute_once_followup_endpoint_keeps_followup_explicit(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_execute_once(self, *, request, now=None):
        observed["followup"] = request.execute_frontier_followup_allowed
        observed["surface"] = request.invocation_surface
        return GraphReviewRuntimeResultV1(
            request_id=request.request_id,
            correlation_id=request.correlation_id,
            outcome="operator_only",
            notes=["strict_zone_guardrail"],
        )

    monkeypatch.setattr(type(api_routes.SUBSTRATE_REVIEW_RUNTIME_EXECUTOR), "execute_once", fake_execute_once)
    payload = api_routes.api_substrate_review_runtime_execute_once_followup()

    assert observed["surface"] == "operator_review"
    assert observed["followup"] is True
    assert payload["request"]["allow_frontier_followup"] is True
    assert payload["result"]["outcome"] in {"operator_only", "executed", "noop", "failed", "suppressed", "terminated"}


def test_smoke_check_is_bounded_and_returns_source_and_checks() -> None:
    payload = api_routes.api_substrate_review_runtime_smoke_check()

    assert "checks" in payload
    assert "source" in payload
    assert "queue_available" in payload["checks"]
    assert "runtime_eligible" in payload["checks"]
    assert "semantic_available" in payload["checks"]
    assert "control_plane_available" in payload["checks"]
    assert "control_plane" in payload["source"]


def test_debug_run_route_returns_structured_payload(monkeypatch) -> None:
    monkeypatch.setattr(api_routes, "_substrate_runtime_status_payload", lambda **kwargs: {"summary": {"queue_count": 1, "due_count": 1}})
    monkeypatch.setattr(api_routes, "_sql_review_queue_payload", lambda **kwargs: {"data": {"queue_items": [{"queue_item_id": "q-1"}]}})
    monkeypatch.setattr(api_routes, "_graphdb_overview_payload", lambda **kwargs: {"data": {"nodes": 3}})
    monkeypatch.setattr(api_routes, "_graphdb_hotspots_payload", lambda **kwargs: {"data": {"hotspots": ["h1"]}})
    monkeypatch.setattr(api_routes, "_sql_review_executions_payload", lambda **kwargs: {"data": {"executions": []}})
    monkeypatch.setattr(api_routes, "_bootstrap_substrate_review_frontier", lambda **kwargs: {"items_enqueued": 2, "notes": ["bootstrap_seeded"]})
    monkeypatch.setattr(
        api_routes,
        "_execute_substrate_review_cycle",
        lambda **kwargs: {"result": {"outcome": "executed", "audit_summary": {"selection_reason": "eligible_item_selected"}}},
    )
    monkeypatch.setattr(api_routes, "_substrate_source_posture", lambda: {"control_plane": {"degraded": False}, "semantic": {"degraded": False}})

    payload = api_routes.api_substrate_review_runtime_debug_run()
    assert "generated_at" in payload
    assert "baseline" in payload
    assert "bootstrap" in payload
    assert "post_bootstrap" in payload
    assert "execute_once" in payload
    assert "final" in payload
    assert "diagnosis" in payload
    assert "source_posture" in payload


def test_debug_diagnosis_classification_cases() -> None:
    base_source = {"control_plane": {"degraded": False}, "semantic": {"degraded": False}}

    unavailable = api_routes._classify_substrate_debug_diagnosis(
        source_posture=base_source,
        bootstrap_payload={"error": "route down"},
        baseline_queue_payload={"data": {"queue_items": []}},
        post_bootstrap_queue_payload={"data": {"queue_items": []}},
        execute_payload={"result": {"outcome": "noop", "audit_summary": {"selection_reason": "no eligible queue items"}}},
        final_queue_payload={"data": {"queue_items": []}},
    )
    assert "bootstrap route/path unavailable" in unavailable["categories"]

    zero_items = api_routes._classify_substrate_debug_diagnosis(
        source_posture=base_source,
        bootstrap_payload={"items_enqueued": 0, "notes": ["seed_skipped:hotspot_region"]},
        baseline_queue_payload={"data": {"queue_items": []}},
        post_bootstrap_queue_payload={"data": {"queue_items": []}},
        execute_payload={"result": {"outcome": "noop", "audit_summary": {"selection_reason": "no eligible queue items"}}},
        final_queue_payload={"data": {"queue_items": []}},
    )
    assert "bootstrap produced zero items" in zero_items["categories"]
    assert "likely weak seed heuristics for current graph shape" in zero_items["categories"]

    success = api_routes._classify_substrate_debug_diagnosis(
        source_posture=base_source,
        bootstrap_payload={"items_enqueued": 2, "notes": ["bootstrap_seeded"]},
        baseline_queue_payload={"data": {"queue_items": []}},
        post_bootstrap_queue_payload={"data": {"queue_items": [{"id": 1}]}},
        execute_payload={"result": {"outcome": "executed", "audit_summary": {"selection_reason": "eligible_item_selected"}}},
        final_queue_payload={"data": {"queue_items": []}},
    )
    assert "bootstrap produced items and execute-once succeeded" in success["categories"]

    enqueue_but_empty = api_routes._classify_substrate_debug_diagnosis(
        source_posture=base_source,
        bootstrap_payload={"items_enqueued": 1, "notes": ["bootstrap_seeded"]},
        baseline_queue_payload={"data": {"queue_items": []}},
        post_bootstrap_queue_payload={"data": {"queue_items": []}},
        execute_payload={"result": {"outcome": "noop", "audit_summary": {"selection_reason": "no eligible queue items"}}},
        final_queue_payload={"data": {"queue_items": []}},
    )
    assert "bootstrap claimed to enqueue but queue remained empty" in enqueue_but_empty["categories"]

    queue_nonempty_noop = api_routes._classify_substrate_debug_diagnosis(
        source_posture=base_source,
        bootstrap_payload={"items_enqueued": 1, "notes": ["bootstrap_seeded"]},
        baseline_queue_payload={"data": {"queue_items": []}},
        post_bootstrap_queue_payload={"data": {"queue_items": [{"id": 1}]}},
        execute_payload={"result": {"outcome": "noop", "audit_summary": {"selection_reason": "no eligible queue items"}}},
        final_queue_payload={"data": {"queue_items": [{"id": 1}]}},
    )
    assert "queue nonempty but execute-once still noop due to no eligible items" in queue_nonempty_noop["categories"]

    control_degraded = api_routes._classify_substrate_debug_diagnosis(
        source_posture={"control_plane": {"degraded": True}, "semantic": {"degraded": False}},
        bootstrap_payload={"items_enqueued": 0, "notes": []},
        baseline_queue_payload={"data": {"queue_items": []}},
        post_bootstrap_queue_payload={"data": {"queue_items": []}},
        execute_payload={"result": {"outcome": "noop", "audit_summary": {"selection_reason": "no eligible queue items"}}},
        final_queue_payload={"data": {"queue_items": []}},
    )
    assert "control plane degraded" in control_degraded["categories"]

    semantic_degraded = api_routes._classify_substrate_debug_diagnosis(
        source_posture={"control_plane": {"degraded": False}, "semantic": {"degraded": True}},
        bootstrap_payload={"items_enqueued": 0, "notes": ["seed_skipped:concept_region"]},
        baseline_queue_payload={"data": {"queue_items": []}},
        post_bootstrap_queue_payload={"data": {"queue_items": []}},
        execute_payload={"result": {"outcome": "noop", "audit_summary": {"selection_reason": "no eligible queue items"}}},
        final_queue_payload={"data": {"queue_items": []}},
    )
    assert "semantic substrate degraded" in semantic_degraded["categories"]


def test_mutation_execute_once_route_requires_operator_guard(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret-token")
    with pytest.raises(HTTPException) as exc:
        api_routes.api_substrate_mutation_runtime_execute_once(
            request=api_routes.SubstrateMutationExecuteRequest(),
            x_orion_operator_token=None,
        )
    assert exc.value.status_code == 403
    assert "operator_guard_rejected" in str(exc.value.detail)


def test_mutation_execute_once_route_allows_operator_and_returns_summary(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret-token")
    observed: dict[str, object] = {}

    def fake_execute(*, request):
        observed["request"] = request
        return {
            "request": {"single_cycle": True, "dry_run": request.dry_run},
            "summary": {
                "signals_produced": 1,
                "pressures_updated": 1,
                "proposals_created": 1,
                "queue_items_touched": ["q-1"],
                "trials_run": 1,
                "decisions_made": 1,
                "applies_attempted": 1,
                "applies_blocked": 1,
                "applies_completed": 0,
                "monitoring_windows_opened": 0,
                "kill_switch_or_policy_blockers": ["dry_run_forces_apply_disabled"],
            },
            "source": {"mutation_store_kind": "memory"},
            "trace": {"events": [], "notes": []},
        }

    monkeypatch.setattr(api_routes, "_execute_substrate_mutation_cycle", fake_execute)
    payload = api_routes.api_substrate_mutation_runtime_execute_once(
        request=api_routes.SubstrateMutationExecuteRequest(dry_run=True, max_signals=3, max_proposals=2, max_trials=2),
        x_orion_operator_token="secret-token",
    )
    assert observed["request"] is not None
    assert payload["request"]["single_cycle"] is True
    assert payload["summary"]["signals_produced"] == 1
    assert payload["summary"]["applies_blocked"] == 1
    assert payload["source"]["mutation_store_kind"] == "memory"


def test_mutation_execute_once_emits_structured_lifecycle_logs(monkeypatch, caplog) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret-token")
    monkeypatch.setattr(
        api_routes,
        "_execute_substrate_mutation_cycle",
        lambda **kwargs: {
            "request": {"single_cycle": True},
            "summary": {"signals_produced": 0},
            "source": {"mutation_store_kind": "memory"},
            "trace": {"events": [], "notes": []},
        },
    )
    # direct helper coverage for structured log shape
    caplog.set_level("INFO")
    api_routes._emit_mutation_lifecycle_logs(
        route_invocation_id="manual-1",
        events=[
            {
                "event": "mutation_proposal_enqueued",
                "cycle_id": "cycle-1",
                "proposal_id": "proposal-1",
                "queue_item_id": "queue-1",
                "lineage_id": "signal-1",
            }
        ],
    )
    assert any("substrate_mutation_lifecycle" in rec.message for rec in caplog.records)


def test_mutation_lineage_readonly_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(
        api_routes,
        "_mutation_lineage_payload",
        lambda **kwargs: {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "source": {"kind": "memory", "degraded": False},
            "data": {"lifecycles": [], "active_surfaces": [], "recent_blocked_applies": [], "recent_rollbacks": []},
        },
    )
    lineage = api_routes.api_substrate_mutation_runtime_lineage(limit=5)
    assert "source" in lineage
    assert "data" in lineage
    monkeypatch.setattr(api_routes.SUBSTRATE_MUTATION_STORE, "active_surfaces_snapshot", lambda: [{"target_surface": "routing", "adoption_id": "a-1"}])
    monkeypatch.setattr(api_routes.SUBSTRATE_MUTATION_STORE, "recent_blocked_applies", lambda limit=20: [{"proposal_id": "p-1"}])
    monkeypatch.setattr(api_routes.SUBSTRATE_MUTATION_STORE, "recent_rollbacks", lambda limit=20: [{"rollback_id": "r-1"}])
    monkeypatch.setattr(
        api_routes.SUBSTRATE_MUTATION_STORE,
        "recent_signals",
        lambda limit=20, target_surface=None: [
            {
                "signal_id": "sig-1",
                "detected_at": "2026-01-01T00:00:00+00:00",
                "source_ref": "review-telemetry:tel-1",
                "event_kind": "routing_runtime_degradation",
                "target_surface": "routing",
                "strength": 0.68,
                "evidence_refs": [
                    "pressure_event:pressure-evt-1",
                    "telemetry:tel-1",
                    "source_kind:runtime_degradation_signal",
                ],
                "metadata": {"source_kind": "runtime_degradation_signal"},
            },
            {
                "signal_id": "sig-recall-1",
                "detected_at": "2026-01-01T00:00:01+00:00",
                "source_ref": "review-telemetry:tel-1",
                "event_kind": "pressure_event:recall_miss_or_dissatisfaction",
                "target_surface": "recall_strategy_profile",
                "strength": 0.72,
                "evidence_refs": [
                    "pressure_event:pressure-evt-1",
                    "recall_compare:selected_count_delta=2",
                ],
                "metadata": {"source_kind": "producer_pressure_event"},
            },
        ],
    )
    monkeypatch.setattr(
        api_routes.SUBSTRATE_MUTATION_STORE,
        "lifecycle_for_proposal",
        lambda proposal_id: {
            "proposal": {"proposal_id": proposal_id, "lane": "cognitive", "target_surface": "cognitive_stance_continuity_adjustment"},
            "signals": [{"signal_id": "sig-cog-1"}],
            "pressure": {"pressure_kind": "stance_drift_pressure"},
            "queue_item": None,
            "trials": [],
            "decisions": [{"action": "require_review"}],
            "adoption": None,
            "rollback": None,
        },
    )
    monkeypatch.setattr(
        api_routes.SUBSTRATE_MUTATION_STORE,
        "_proposals",
        {
            "proposal-cog-1": type(
                "_P",
                (),
                {
                    "lane": "cognitive",
                    "target_surface": "cognitive_stance_continuity_adjustment",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "model_dump": lambda self, mode="json": {
                        "proposal_id": "proposal-cog-1",
                        "lane": "cognitive",
                        "target_surface": "cognitive_stance_continuity_adjustment",
                    },
                },
            )()
        },
    )
    monkeypatch.setattr(
        api_routes.SUBSTRATE_MUTATION_STORE,
        "record_cognitive_review",
        lambda review: type(
            "_D",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "draft_id": "draft-1",
                    "proposal_id": review.proposal_id,
                    "status": "draft_only_not_applied",
                }
            },
        )(),
    )
    monkeypatch.setattr(api_routes.SUBSTRATE_MUTATION_STORE, "recent_cognitive_drafts", lambda limit=20: [{"draft_id": "draft-1"}])
    monkeypatch.setattr(api_routes.SUBSTRATE_MUTATION_STORE, "recent_cognitive_reviews", lambda limit=20: [{"review_id": "review-1"}])
    monkeypatch.setattr(
        api_routes.SUBSTRATE_REVIEW_TELEMETRY_STORE,
        "query",
        lambda query: [
            GraphReviewTelemetryRecordV1(
                invocation_surface="chat_reflective_lane",
                execution_outcome="executed",
                selection_reason="producer_pressure_events:evt-1",
                runtime_duration_ms=0,
                anchor_scope="orion",
                subject_ref="entity:orion",
                target_zone="autonomy_graph",
                pressure_events=[
                    MutationPressureEvidenceV1(
                        pressure_event_id="pressure-evt-1",
                        source_service="orion-hub",
                        source_event_id="feedback-1",
                        correlation_id="corr-1",
                        pressure_category="recall_miss_or_dissatisfaction",
                        confidence=0.84,
                        evidence_refs=["feedback:feedback-1"],
                        metadata={
                            "v1_v2_compare": {"v1_latency_ms": 120, "v2_latency_ms": 90, "selected_count_delta": 2},
                            "anchor_plan": {"time_window_days": 1},
                            "selected_evidence_cards": [{"id": "page-1"}],
                        },
                    )
                ],
            )
        ],
    )
    active = api_routes.api_substrate_mutation_runtime_active_surfaces()
    blocked = api_routes.api_substrate_mutation_runtime_blocked_applies(limit=5)
    rollbacks = api_routes.api_substrate_mutation_runtime_rollbacks(limit=5)
    monkeypatch.setattr(
        api_routes,
        "inspect_chat_reflective_lane_threshold",
        lambda: {"surface": "routing.chat_reflective_lane_threshold", "value": 0.5, "source_kind": "sqlite", "degraded": False},
    )
    monkeypatch.setattr(
        api_routes,
        "_routing_replay_inspection_payload",
        lambda **kwargs: {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "data": {
                "case_count": 2,
                "corpus_composition": {"rich_signal_case_count": 1, "rich_signal_coverage": 0.5},
                "derived_metrics": {"evaluator_confidence": 0.65},
            },
        },
    )
    monkeypatch.setattr(
        api_routes,
        "_routing_live_ramp_posture_payload",
        lambda: {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "data": {
                "proposals_enabled": True,
                "apply_enabled": False,
                "last_decision": {"decision_id": "d-1"},
                "last_adoption": None,
                "last_rollback": None,
                "live_surface": {"surface": "routing.chat_reflective_lane_threshold", "value": 0.5},
            },
        },
    )
    monkeypatch.setattr(
        api_routes,
        "build_mutation_cognition_context",
        lambda **kwargs: {"mutation_scope": "routing_threshold_patch_only", "live_ramp_active": True},
    )
    live = api_routes.api_substrate_mutation_runtime_live_routing_surface()
    replay = api_routes.api_substrate_mutation_runtime_routing_replay_inspect(limit=12)
    posture = api_routes.api_substrate_mutation_runtime_routing_live_ramp_posture()
    cognition = api_routes.api_substrate_mutation_runtime_cognition_context()
    pressure_sources = api_routes.api_substrate_mutation_runtime_routing_pressure_sources(limit=5)
    producer_pressure = api_routes.api_substrate_mutation_runtime_producer_pressure_events(limit=5)
    recall_pressure = api_routes.api_substrate_mutation_runtime_recall_pressure_events(limit=5)
    recall_proposals = api_routes.api_substrate_mutation_runtime_recall_strategy_proposals(limit=5)
    recall_eval = api_routes.api_substrate_mutation_runtime_recall_v1_v2_latest_eval()
    cognitive_pressure = api_routes.api_substrate_mutation_runtime_cognitive_pressure(limit=5)
    cognitive_proposals = api_routes.api_substrate_mutation_runtime_cognitive_proposals(limit=5)
    cognitive_detail = api_routes.api_substrate_mutation_runtime_cognitive_proposal_detail("proposal-cog-1")
    cognitive_evidence = api_routes.api_substrate_mutation_runtime_cognitive_proposal_evidence("proposal-cog-1")
    cognitive_lineage = api_routes.api_substrate_mutation_runtime_cognitive_proposal_lineage("proposal-cog-1")
    cognitive_review = api_routes.api_substrate_mutation_runtime_cognitive_proposal_review(
        "proposal-cog-1",
        api_routes.CognitiveProposalReviewRequest(state="accepted_as_draft"),
    )
    cognitive_drafts = api_routes.api_substrate_mutation_runtime_cognitive_drafts(limit=5)
    assert active["data"]["active_surfaces"][0]["target_surface"] == "routing"
    assert blocked["data"]["recent_blocked_applies"][0]["proposal_id"] == "p-1"
    assert rollbacks["data"]["recent_rollbacks"][0]["rollback_id"] == "r-1"
    assert live["data"]["surface"] == "routing.chat_reflective_lane_threshold"
    assert live["data"]["value"] == 0.5
    assert replay["data"]["case_count"] == 2
    assert replay["data"]["corpus_composition"]["rich_signal_case_count"] == 1
    assert replay["data"]["derived_metrics"]["evaluator_confidence"] == 0.65
    assert posture["data"]["proposals_enabled"] is True
    assert posture["data"]["apply_enabled"] is False
    assert posture["data"]["last_decision"]["decision_id"] == "d-1"
    assert cognition["data"]["mutation_scope"] == "routing_threshold_patch_only"
    assert cognition["data"]["live_ramp_active"] is True
    assert pressure_sources["data"]["routing_pressure_sources"][0]["source_kind"] == "runtime_degradation_signal"
    assert pressure_sources["data"]["routing_pressure_sources"][0]["derived_signal_kind"] == "routing_runtime_degradation"
    assert producer_pressure["data"]["recent_events"][0]["pressure_event_id"] == "pressure-evt-1"
    assert producer_pressure["data"]["recent_events"][0]["linked_signal_ids"] == ["sig-1", "sig-recall-1"]
    assert recall_pressure["data"]["recent_recall_pressure_events"][0]["pressure_event_id"] == "pressure-evt-1"
    assert recall_pressure["data"]["recent_recall_pressure_events"][0]["linked_signal_ids"] == ["sig-recall-1"]
    assert recall_proposals["data"]["recent_recall_strategy_proposals"] == []
    assert recall_eval["data"]["latest_recall_v1_v2_eval_summary"]["v1_v2_compare"]["selected_count_delta"] == 2
    assert "recent_recall_eval_suite_rows" in recall_eval["data"]
    assert cognitive_pressure["data"]["recent_cognitive_pressure"] == []
    assert cognitive_proposals["data"]["recent_cognitive_proposals"][0]["lane"] == "cognitive"
    assert cognitive_detail["data"]["proposal"]["proposal_id"] == "proposal-cog-1"
    assert cognitive_evidence["data"]["proposal_id"] == "proposal-cog-1"
    assert cognitive_lineage["data"]["lineage"]["proposal"]["proposal_id"] == "proposal-cog-1"
    assert cognitive_review["data"]["review"]["state"] == "accepted_as_draft"
    assert cognitive_review["data"]["draft_recommendation"]["status"] == "draft_only_not_applied"
    assert cognitive_drafts["data"]["draft_recommendations"][0]["draft_id"] == "draft-1"


def test_hub_substrate_debug_surface_does_not_embed_standalone_page() -> None:
    index_html = TEMPLATE_PATH.read_text(encoding="utf-8")
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert 'id="substratePanelFrame"' in index_html
    assert 'src="/substrate"' in index_html
    assert "substrate.js" not in index_html
    assert "window.OrionHub" not in app_js


def test_autonomy_readiness_endpoint_returns_unified_read_only_snapshot() -> None:
    payload = api_routes.api_substrate_autonomy_readiness()

    assert payload["schema_version"] == "autonomy_readiness.v1"
    assert "generated_at" in payload
    assert "overall" in payload
    assert "scheduler" in payload
    assert "policy_matrix" in payload
    assert "surfaces" in payload
    assert "routing" in payload
    assert "recall" in payload
    assert "cognitive" in payload
    assert "pressure" in payload
    assert "recent_activity" in payload
    assert "safe_next_actions" in payload
    assert "warnings" in payload
    assert payload["recall"]["production_mode"] == "v1"
    assert payload["recall"]["live_apply_enabled"] is False
    assert payload["cognitive"]["live_apply_enabled"] is False
    assert "manual_canary" in payload["recall"]
    assert "recommended_canary_action" in payload["recall"]["manual_canary"]


def test_autonomy_readiness_endpoint_tolerates_partial_failures(monkeypatch) -> None:
    monkeypatch.setattr(api_routes, "_routing_live_ramp_posture_payload", lambda: (_ for _ in ()).throw(RuntimeError("routing unavailable")))
    monkeypatch.setattr(api_routes, "readiness_from_telemetry_records", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("readiness unavailable")))

    payload = api_routes.api_substrate_autonomy_readiness()

    assert payload["schema_version"] == "autonomy_readiness.v1"
    assert isinstance(payload["warnings"], list)
    assert any("routing_posture_unavailable" in item for item in payload["warnings"])
    assert any("recall_posture_unavailable" in item for item in payload["warnings"])


def test_autonomy_readiness_policy_matrix_blocks_recall_and_cognitive_live_apply() -> None:
    payload = api_routes.api_substrate_autonomy_readiness()
    surfaces = payload["policy_matrix"]["surfaces"]
    by_surface = {str(item.get("surface")): item for item in surfaces}

    assert by_surface["routing_threshold_patch"]["apply"] == "gated_auto"
    assert by_surface["recall_strategy_profile"]["status"] == "shadow_staged_only"
    assert "recall_weighting_patch_live_apply" in by_surface["recall_strategy_profile"]["forbidden"]
    assert by_surface["cognitive_self_model"]["apply"] == "forbidden"
    assert payload["surfaces"]["live"] == [] or all(
        row.get("surface") == "routing_threshold_patch" for row in payload["surfaces"]["live"]
    )


def test_recall_canary_status_rollups_shape() -> None:
    payload = api_routes.api_substrate_recall_canary_status(limit=20)
    data = payload["data"]
    assert data["schema_version"] == "recall_canary_status.v1"
    assert "judgment_counts" in data
    assert "failure_mode_counts" in data
    assert "recent_judgments" in data
    assert "review_artifact_count" in data
