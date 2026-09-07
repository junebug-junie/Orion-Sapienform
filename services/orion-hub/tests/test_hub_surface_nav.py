"""Static-content tests for the Hub-tab nav entry pointing at /hub-surface.

Regression test: hub_surface_routes.py (PR #2137) shipped the standalone
/hub-surface page with no way to reach it from the main Hub nav -- the only
access path was typing the URL directly. This covers the nav tab + embedded
panel + iframe wiring added to close that gap.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "services" / "orion-hub" / "templates" / "index.html"
PANEL_JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "hub_surface_tab.js"


def test_template_declares_hub_surface_tab_and_iframe() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert 'id="hubSurfaceTabButton"' in template
    assert 'data-hash-target="#hub-surface"' in template
    assert '<section id="hub-surface" data-panel="hub-surface"' in template
    assert 'id="hubSurfacePanelFrame"' in template
    assert 'src="/hub-surface"' in template
    assert 'id="hubSurfacePanelStandaloneLink" href="/hub-surface"' in template


def test_template_includes_cache_busted_panel_script() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert (
        '<script src="/static/js/hub_surface_tab.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>'
        in template
    )


def test_panel_js_is_tab_controller_only() -> None:
    panel_js = PANEL_JS_PATH.read_text(encoding="utf-8")

    assert "function activatePanel()" in panel_js
    assert "function deactivatePanel()" in panel_js
    assert "#hub-surface" in panel_js
    assert "hubSurfacePanelRefresh" in panel_js
    assert "hubSurfacePanelFrame" in panel_js
