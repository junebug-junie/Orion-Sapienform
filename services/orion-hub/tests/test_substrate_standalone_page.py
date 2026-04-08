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
    assert "/static/js/substrate.js?v={{HUB_UI_ASSET_VERSION}}" in template
    assert "window.OrionHub" not in script
    assert "\'/api/substrate/overview?limit=10\'" in script

    route_paths = {route.path for route in api_routes.router.routes}
    assert "/substrate" in route_paths


def test_backend_substrate_endpoints_have_source_metadata_and_expected_split() -> None:
    api_routes.SUBSTRATE_POLICY_STORE.adopt(
        SubstratePolicyAdoptionRequestV1(
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            policy_overrides=SubstratePolicyOverridesV1(normal_revisit_seconds=999),
            activate_now=False,
            operator_id="hub-test",
            rationale="operator-inspection",
        )
    )
    overview = api_routes.api_substrate_overview(limit=5)
    hotspots = api_routes.api_substrate_hotspots(limit=5)
    queue = api_routes.api_substrate_review_queue(limit=5)
    executions = api_routes.api_substrate_review_executions(limit=5)
    telemetry = api_routes.api_substrate_telemetry_summary(limit=5)
    calibration = api_routes.api_substrate_calibration(limit=5)

    assert overview["source"]["kind"] in {"graphdb", "fallback", "cache"}
    assert hotspots["source"]["kind"] in {"graphdb", "fallback", "cache"}
    assert overview["source"]["query_kind"] == "overview"
    assert hotspots["source"]["query_kind"] == "hotspots"

    assert queue["source"]["kind"] == "sql"
    assert executions["source"]["kind"] == "sql"
    assert telemetry["source"]["kind"] == "sql"
    assert calibration["source"]["kind"] == "sql"
    assert "staged_profiles" in calibration["data"]
    assert "recent_audit_events" in calibration["data"]

    assert "degraded" in overview["source"]
    assert "truncated" in hotspots["source"]
    assert "query" in queue["source"]


def test_substrate_page_keeps_main_shell_untouched() -> None:
    index_html = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    app_js = (HUB_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")

    assert "substrate.js" not in index_html
    assert "Substrate Inspector" not in index_html
    assert "/api/substrate/overview" not in app_js
    assert "substrateRefreshButton" not in app_js
