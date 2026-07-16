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

APP_JS_PATH = HUB_ROOT / "static" / "js" / "app.js"


def test_causal_geometry_route_and_template_and_bundle_are_standalone() -> None:
    template_path = HUB_ROOT / "templates" / "causal_geometry.html"
    js_path = HUB_ROOT / "static" / "js" / "causal-geometry.js"

    template = template_path.read_text(encoding="utf-8")
    script = js_path.read_text(encoding="utf-8")

    assert "Causal Geometry Inspector" in template
    assert 'id="causalGeometryRefreshButton"' in template
    assert 'id="causalGeometrySnapshot"' in template
    assert 'id="causalGeometryHistory"' in template
    assert 'id="causalGeometryProposals"' in template
    assert "/static/js/causal-geometry.js?v={{HUB_UI_ASSET_VERSION}}" in template
    assert "window.OrionHub" not in script
    assert "'/api/causal-geometry/snapshot'" in script
    assert "'/api/causal-geometry/history?limit=20'" in script
    assert "'/api/causal-geometry/proposals?limit=50'" in script
    assert "/adopt" in script
    assert "/reject" in script

    route_paths = {route.path for route in api_routes.router.routes}
    assert "/causal-geometry" in route_paths
    assert "/api/causal-geometry/snapshot" in route_paths
    assert "/api/causal-geometry/history" in route_paths
    assert "/api/causal-geometry/proposals" in route_paths
    assert "/api/causal-geometry/proposals/{proposal_id}/adopt" in route_paths
    assert "/api/causal-geometry/proposals/{proposal_id}/reject" in route_paths


def test_main_hub_has_causal_geometry_navigation_link_in_shell_tabs() -> None:
    index_html = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")

    assert 'id="causalGeometryTabButton"' in index_html
    assert 'href="#causal-geometry"' in index_html
    assert 'data-hash-target="#causal-geometry"' in index_html
    assert ">Causal Geometry<" in index_html
    causal_geometry_link_block = index_html.split('id="causalGeometryTabButton"', 1)[1].split("</a>", 1)[0]
    assert 'role="button"' in causal_geometry_link_block


def test_main_hub_has_isolated_causal_geometry_shell_panel_embedding_standalone_page() -> None:
    index_html = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")

    assert '<section id="causal-geometry" data-panel="causal-geometry"' in index_html
    assert 'id="causalGeometryPanelFrame"' in index_html
    assert 'src="/causal-geometry"' in index_html
    assert 'id="causalGeometryPanelStandaloneLink" href="/causal-geometry"' in index_html


def test_causal_geometry_page_keeps_main_shell_untouched() -> None:
    index_html = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "causal-geometry.js" not in index_html
    assert "id=\"causalGeometryPanelFrame\"" in index_html
    assert "/api/causal-geometry" not in app_js
    assert "causalGeometryRefreshButton" not in app_js


def test_app_js_includes_causal_geometry_in_shell_tab_switching_without_full_navigation() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert 'const causalGeometryTabButton = document.getElementById("causalGeometryTabButton");' in app_js
    assert 'const causalGeometryPanel = document.getElementById("causal-geometry");' in app_js
    assert 'const isCausalGeometry = effectiveTab === "causal-geometry";' in app_js
    assert 'causalGeometryPanel.classList.toggle("hidden", !isCausalGeometry);' in app_js
    assert 'styleTabButton(causalGeometryTabButton, isCausalGeometry);' in app_js
    assert 'causalGeometryTabButton.addEventListener("click", (event) => {' in app_js
    assert 'setActiveTab("causal-geometry");' in app_js
    assert 'history.replaceState(null, "", "#causal-geometry");' in app_js
    assert 'h === "#causal-geometry"' in app_js


def test_app_js_supports_causal_geometry_embed_refresh_without_merging_bundle() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert 'const causalGeometryPanelFrame = document.getElementById("causalGeometryPanelFrame");' in app_js
    assert 'const causalGeometryPanelRefresh = document.getElementById("causalGeometryPanelRefresh");' in app_js
    assert 'causalGeometryPanelFrame.contentWindow?.location.reload();' in app_js
    assert "causal-geometry.js" not in app_js


def test_app_js_never_navigates_the_parent_window_for_causal_geometry_tab() -> None:
    """The crux of 'doesn't kill the session': no full-page navigation for this tab.

    The only reload allowed for this feature is on the iframe's own contentWindow, triggered
    by the explicit refresh button -- never window.location/href on the parent Hub page.
    """
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    lines = app_js.splitlines()
    causal_geometry_lines = [line for line in lines if "causalGeometry" in line or "causal-geometry" in line]
    assert causal_geometry_lines, "expected causal-geometry lines to exist in app.js"

    for line in causal_geometry_lines:
        assert "window.location.href" not in line
        assert "window.location =" not in line

    reload_lines = [line for line in causal_geometry_lines if ".reload()" in line]
    assert len(reload_lines) == 1
    assert "causalGeometryPanelFrame.contentWindow?.location.reload();" in reload_lines[0]


def test_api_causal_geometry_snapshot_route_registered_and_degrades_gracefully() -> None:
    payload = api_routes.api_causal_geometry_snapshot()
    assert "source" in payload
    assert "data" in payload
