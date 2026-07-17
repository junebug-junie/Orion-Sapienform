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

TEMPLATE_PATH = HUB_ROOT / "templates" / "drives-analytics.html"
JS_PATH = HUB_ROOT / "static" / "js" / "drives-analytics.js"

CONTROL_IDS = (
    "drivesSubjectSelect",
    "drivesWindowSelect",
    "drivesColorMode",
    "drivesRefreshButton",
    "drivesAutoRefresh",
    "drivesLastLoaded",
    "drivesCoverageBanner",
)

CARD_IDS = (
    "drivesGaugesCard",
    "drivesContributorsCard",
    "drivesKpiStrip",
    "drivesSeriesCard",
    "drivesDivergenceCard",
    "drivesGoalsCard",
    "drivesCrossLinks",
)

TOOLTIP_KEYS = (
    "gauges",
    "contributors",
    "kpi",
    "series",
    "divergence",
    "goals",
    "crosslinks",
)


def test_drives_analytics_route_is_registered() -> None:
    route_paths = {route.path for route in api_routes.router.routes}
    assert "/drives-analytics" in route_paths


def test_template_has_control_and_card_ids() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    for control_id in CONTROL_IDS:
        assert f'id="{control_id}"' in template
    for card_id in CARD_IDS:
        assert f'id="{card_id}"' in template


def test_template_references_standalone_asset_version_bundle() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert "/static/js/drives-analytics.js?v={{HUB_UI_ASSET_VERSION}}" in template


def test_script_does_not_reference_hub_shell_global() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "window.OrionHub" not in script


def test_script_has_real_dom_renderers_not_raw_json_dump() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "function renderGauges(" in script
    assert "function renderKpiStrip(" in script
    assert "function renderSeries(" in script
    assert "function renderContributors(" in script
    assert "function renderDivergence(" in script
    assert "function renderGoals(" in script
    assert "function renderCrossLinks(" in script
    assert "target.textContent = JSON.stringify(payload, null, 2);" not in script


def test_tooltip_copy_covers_all_cards_with_three_required_angles() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "TOOLTIP_COPY" in script
    for key in TOOLTIP_KEYS:
        assert f"{key}:" in script or f'"{key}"' in script
    assert "definition:" in script
    assert "design:" in script
    assert "reading:" in script


def test_polling_pauses_on_hidden_and_resumes_on_visible() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "visibilitychange" in script
    assert "setInterval(" in script
    assert "clearInterval(" in script
    assert 'document.visibilityState === \'hidden\'' in script or 'document.visibilityState === "hidden"' in script


def test_contributors_live_window_toggle_and_null_attribution_language() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "Live" in script
    assert "Window" in script
    assert "null_attribution_row_count" in script


def test_color_mode_and_window_picker_values_present() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "combined" in script
    assert "align" in script
    assert "funnel" in script
    for hours in ("1", "6", "24", "168"):
        assert hours in script or hours in TEMPLATE_PATH.read_text(encoding="utf-8")


def test_no_mutation_fetch_calls_in_page_js() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    for verb in ("POST", "PUT", "DELETE", "PATCH"):
        assert f"method: '{verb}'" not in script
        assert f'method: "{verb}"' not in script


def test_goals_card_has_no_action_buttons() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    # causal-geometry.js has adopt/reject buttons wired to POST endpoints; this page must
    # not, since the Goals card is strictly read-only orientation, never a mutation console.
    assert "adoptButton" not in script
    assert "rejectButton" not in script
    assert "postAction(" not in script
    assert "proposals/${" not in script


def test_gate_verdict_never_labeled_bare_go() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "GO_DRIVE_ONLY" in script
    assert "drive economy OK (partial)" in script


def test_endpoints_referenced_in_script() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    for path in (
        "/api/drives-analytics/subjects",
        "/api/drives-analytics/snapshot",
        "/api/drives-analytics/window",
        "/api/drives-analytics/series",
        "/api/drives-analytics/goal-alignment",
        "/api/drives-analytics/divergence",
    ):
        assert path in script


def test_cross_links_card_content() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "/spark/ui" in script
    assert "/#pressure" in script
    assert "/causal-geometry" in script
    assert "Pressure" in script and "not the same thing as the DriveEngine economy" in script
    assert "<details" in script


def test_divergence_card_surfaces_fallback_banner_and_autonomy_note() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "store_path_is_fallback_default" in script
    assert "autonomy_state_v2_note" in script
