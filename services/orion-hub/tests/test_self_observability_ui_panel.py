"""Static-content tests for the Hub Self-Observability panel (self-observability v2)."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "services" / "orion-hub" / "templates" / "index.html"
PANEL_JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "self_observability.js"

PANEL_ELEMENT_IDS = (
    "selfObsStatus",
    "selfObsRefresh",
    "selfObsAttentionType",
    "selfObsDwellTicks",
    "selfObsNodeCount",
    "selfObsCoalitionDesc",
    "selfObsStability",
    "selfObsGapCount",
    "selfObsGapList",
    "selfObsPresenceHealth",
    "selfObsLastTurnAge",
    "selfObsTurnsPerMin",
)


def test_template_declares_self_observability_tab_and_panel() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert 'id="selfObservabilityTabButton"' in template
    assert 'data-hash-target="#self-observability"' in template
    assert '<section id="self-observability" data-panel="self-observability"' in template
    for element_id in PANEL_ELEMENT_IDS:
        assert f'id="{element_id}"' in template, element_id


def test_template_includes_cache_busted_panel_script() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert (
        '<script src="/static/js/self_observability.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>'
        in template
    )


def test_panel_js_fetches_summary_and_renders_all_cards() -> None:
    panel_js = PANEL_JS_PATH.read_text(encoding="utf-8")

    assert '"/api/substrate/observability/summary"' in panel_js
    for element_id in PANEL_ELEMENT_IDS:
        assert f'"{element_id}"' in panel_js, element_id
    # Null-degrading sections and tab lifecycle.
    assert "no data yet" in panel_js
    assert "function activatePanel()" in panel_js
    assert "function deactivatePanel()" in panel_js
    assert "startAutoRefresh" in panel_js
    assert "stopAutoRefresh" in panel_js
