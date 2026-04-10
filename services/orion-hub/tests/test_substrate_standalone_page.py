from __future__ import annotations

import sys
from pathlib import Path

import os

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

from scripts import api_routes
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1
from orion.core.schemas.substrate_policy_adoption import (
    SubstratePolicyAdoptionRequestV1,
    SubstratePolicyOverridesV1,
    SubstratePolicyRolloutScopeV1,
)


def test_substrate_route_and_template_and_bundle_are_standalone() -> None:
    template_path = HUB_ROOT / "templates" / "substrate.html"
    js_path = HUB_ROOT / "static" / "js" / "substrate.js"

    template = template_path.read_text(encoding="utf-8")
    script = js_path.read_text(encoding="utf-8")

    assert "Substrate Inspector" in template
    assert "substratePolicyComparison" in template
    assert 'id="substrateBootstrapButton"' in template
    assert 'id="substrateExecuteOnceButton"' in template
    assert 'id="substrateDebugRunButton"' in template
    assert 'id="substrateRuntimeStatus"' in template
    assert 'id="substrateDebugRunResult"' in template
    assert 'id="substrateDiagnosisSummary"' in template
    assert "/static/js/substrate.js?v={{HUB_UI_ASSET_VERSION}}" in template
    assert "window.OrionHub" not in script
    assert "\'/api/substrate/overview?limit=10\'" in script
    assert "\'/api/substrate/review-runtime/status\'" in script
    assert "\'/api/substrate/review-runtime/bootstrap\'" in script
    assert "\'/api/substrate/review-runtime/execute-once\'" in script
    assert "\'/api/substrate/review-runtime/debug-run\'" in script
    assert "policy-comparison?pair_mode=baseline_vs_active" in script

    route_paths = {route.path for route in api_routes.router.routes}
    assert "/substrate" in route_paths


def test_backend_substrate_endpoints_have_source_metadata_and_expected_split() -> None:
    first = api_routes.SUBSTRATE_POLICY_STORE.adopt(
        SubstratePolicyAdoptionRequestV1(
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            policy_overrides=SubstratePolicyOverridesV1(normal_revisit_seconds=1100),
            activate_now=True,
            operator_id="hub-test",
            rationale="operator-inspection-first",
        )
    )
    second = api_routes.SUBSTRATE_POLICY_STORE.adopt(
        SubstratePolicyAdoptionRequestV1(
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            policy_overrides=SubstratePolicyOverridesV1(normal_revisit_seconds=999),
            activate_now=True,
            operator_id="hub-test",
            rationale="operator-inspection",
        )
    )
    for _ in range(6):
        api_routes.SUBSTRATE_REVIEW_TELEMETRY_STORE.record(
            GraphReviewTelemetryRecordV1(
                policy_profile_id=first.profile_id,
                invocation_surface="operator_review",
                target_zone="concept_graph",
                selection_reason="test",
                execution_outcome="failed",
                runtime_duration_ms=100,
                cycle_count_before=3,
            )
        )
        api_routes.SUBSTRATE_REVIEW_TELEMETRY_STORE.record(
            GraphReviewTelemetryRecordV1(
                policy_profile_id=second.profile_id,
                invocation_surface="operator_review",
                target_zone="concept_graph",
                selection_reason="test",
                execution_outcome="executed",
                runtime_duration_ms=100,
                cycle_count_before=1,
                frontier_followup_invoked=True,
            )
        )
    overview = api_routes.api_substrate_overview(limit=5)
    hotspots = api_routes.api_substrate_hotspots(limit=5)
    queue = api_routes.api_substrate_review_queue(limit=5)
    executions = api_routes.api_substrate_review_executions(limit=5)
    telemetry = api_routes.api_substrate_telemetry_summary(limit=5)
    calibration = api_routes.api_substrate_calibration(limit=5)
    comparison = api_routes.api_substrate_policy_comparison(pair_mode="previous_vs_current", sample_limit=100)

    assert overview["source"]["kind"] in {"graphdb", "fallback", "cache"}
    assert hotspots["source"]["kind"] in {"graphdb", "fallback", "cache"}
    assert overview["source"]["query_kind"] == "overview"
    assert hotspots["source"]["query_kind"] == "hotspots"

    assert queue["source"]["kind"] in {"sql", "sqlite", "postgres", "fallback", "memory"}
    assert executions["source"]["kind"] in {"sql", "sqlite", "postgres", "fallback", "memory"}
    assert telemetry["source"]["kind"] in {"sql", "sqlite", "postgres", "fallback", "memory"}
    assert calibration["source"]["kind"] in {"sql", "sqlite", "postgres", "fallback", "memory"}
    assert comparison["source"]["kind"] in {"sql", "sqlite", "postgres", "fallback", "memory"}
    assert "staged_profiles" in calibration["data"]
    assert "recent_audit_events" in calibration["data"]
    assert comparison["data"]["report"]["verdict"] in {"improved", "neutral", "degraded", "insufficient_data"}
    assert comparison["data"]["advisory"]["mutating"] is False
    assert comparison["data"]["pair"]["candidate_profile_id"] == second.profile_id

    assert "degraded" in overview["source"]
    assert "truncated" in hotspots["source"]
    assert "query" in queue["source"]


def test_main_hub_has_substrate_navigation_link_in_shell_tabs() -> None:
    index_html = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")

    assert 'id="substratePageLink"' in index_html
    assert 'href="#substrate"' in index_html
    assert 'data-hash-target="#substrate"' in index_html
    assert ">Substrate<" in index_html
    substrate_link_block = index_html.split('id="substratePageLink"', 1)[1].split("</a>", 1)[0]
    assert 'role="button"' in substrate_link_block


def test_main_hub_has_isolated_substrate_shell_panel_embedding_standalone_page() -> None:
    index_html = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")

    assert '<section id="substrate" data-panel="substrate"' in index_html
    assert 'id="substratePanelFrame"' in index_html
    assert 'src="/substrate"' in index_html
    assert 'id="substratePanelStandaloneLink" href="/substrate"' in index_html


def test_substrate_page_keeps_main_shell_untouched() -> None:
    index_html = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    app_js = (HUB_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")

    assert "substrate.js" not in index_html
    assert "id=\"substratePanelFrame\"" in index_html
    assert "/api/substrate/overview" not in app_js
    assert "substrateRefreshButton" not in app_js


def test_policy_comparison_source_honesty_prefers_postgres_metadata(monkeypatch) -> None:
    monkeypatch.setattr(api_routes.SUBSTRATE_POLICY_STORE, "source_kind", lambda: "postgres")
    monkeypatch.setattr(api_routes.SUBSTRATE_REVIEW_TELEMETRY_STORE, "source_kind", lambda: "postgres")
    monkeypatch.setattr(api_routes.SUBSTRATE_POLICY_STORE, "degraded", lambda: False)
    monkeypatch.setattr(api_routes.SUBSTRATE_REVIEW_TELEMETRY_STORE, "degraded", lambda: False)
    monkeypatch.setattr(api_routes.SUBSTRATE_POLICY_STORE, "last_error", lambda: None)
    monkeypatch.setattr(api_routes.SUBSTRATE_REVIEW_TELEMETRY_STORE, "last_error", lambda: None)
    payload = api_routes.api_substrate_policy_comparison(pair_mode="baseline_vs_active", sample_limit=50)
    assert payload["source"]["kind"] == "postgres"
