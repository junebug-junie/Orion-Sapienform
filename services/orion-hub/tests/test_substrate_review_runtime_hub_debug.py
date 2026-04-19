from __future__ import annotations

import os
import sys
from pathlib import Path

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

from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeResultV1
from scripts import api_routes


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


def test_hub_substrate_debug_surface_does_not_embed_standalone_page() -> None:
    index_html = TEMPLATE_PATH.read_text(encoding="utf-8")
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert 'id="substratePanelFrame"' in index_html
    assert 'src="/substrate"' in index_html
    assert "substrate.js" not in index_html
    assert "window.OrionHub" not in app_js
