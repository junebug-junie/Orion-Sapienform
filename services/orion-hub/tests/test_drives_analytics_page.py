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


# ---------------------------------------------------------------------------
# Hub shell tab wiring (Task 8) -- mirrors test_causal_geometry_page.py's
# shell-level tests for the #drives tab.
# ---------------------------------------------------------------------------

APP_JS_PATH = HUB_ROOT / "static" / "js" / "app.js"
INDEX_HTML_PATH = HUB_ROOT / "templates" / "index.html"


def test_main_hub_has_drives_navigation_link_in_shell_tabs() -> None:
    index_html = INDEX_HTML_PATH.read_text(encoding="utf-8")

    assert 'id="drivesAnalyticsTabButton"' in index_html
    assert 'href="#drives"' in index_html
    assert 'data-hash-target="#drives"' in index_html
    assert ">Drives<" in index_html
    drives_link_block = index_html.split('id="drivesAnalyticsTabButton"', 1)[1].split("</a>", 1)[0]
    assert 'role="button"' in drives_link_block


def test_main_hub_has_isolated_drives_shell_panel_embedding_standalone_page() -> None:
    index_html = INDEX_HTML_PATH.read_text(encoding="utf-8")

    assert '<section id="drives" data-panel="drives"' in index_html
    assert 'id="drivesAnalyticsPanelFrame"' in index_html
    assert 'src="/drives-analytics"' in index_html
    assert 'id="drivesAnalyticsPanelStandaloneLink" href="/drives-analytics"' in index_html


def test_drives_page_keeps_main_shell_untouched() -> None:
    index_html = INDEX_HTML_PATH.read_text(encoding="utf-8")
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "drives-analytics.js" not in index_html
    assert 'id="drivesAnalyticsPanelFrame"' in index_html
    assert "/api/drives-analytics" not in app_js
    assert "drivesRefreshButton" not in app_js


def test_app_js_includes_drives_in_shell_tab_switching_without_full_navigation() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert 'const drivesAnalyticsTabButton = document.getElementById("drivesAnalyticsTabButton");' in app_js
    assert 'const drivesAnalyticsPanel = document.getElementById("drives");' in app_js
    assert 'const isDrivesAnalytics = effectiveTab === "drives";' in app_js
    assert 'drivesAnalyticsPanel.classList.toggle("hidden", !isDrivesAnalytics);' in app_js
    assert 'styleTabButton(drivesAnalyticsTabButton, isDrivesAnalytics);' in app_js
    assert 'drivesAnalyticsTabButton.addEventListener("click", (event) => {' in app_js
    assert 'setActiveTab("drives");' in app_js
    assert 'history.replaceState(null, "", "#drives");' in app_js
    assert 'h === "#drives"' in app_js


def test_app_js_supports_drives_embed_refresh_without_merging_bundle() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert 'const drivesAnalyticsPanelFrame = document.getElementById("drivesAnalyticsPanelFrame");' in app_js
    assert 'const drivesAnalyticsPanelRefresh = document.getElementById("drivesAnalyticsPanelRefresh");' in app_js
    assert 'drivesAnalyticsPanelFrame.contentWindow?.location.reload();' in app_js
    assert "drives-analytics.js" not in app_js


def test_app_js_never_navigates_the_parent_window_for_drives_tab() -> None:
    """The crux of 'doesn't kill the session': no full-page navigation for this tab.

    The only reload allowed for this feature is on the iframe's own contentWindow, triggered
    by the explicit refresh button -- never window.location/href on the parent Hub page.
    """
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    lines = app_js.splitlines()
    drives_lines = [line for line in lines if "drivesAnalytics" in line]
    assert drives_lines, "expected drivesAnalytics lines to exist in app.js"

    for line in drives_lines:
        assert "window.location.href" not in line
        assert "window.location =" not in line

    reload_lines = [line for line in drives_lines if ".reload()" in line]
    assert len(reload_lines) == 1
    assert "drivesAnalyticsPanelFrame.contentWindow?.location.reload();" in reload_lines[0]
