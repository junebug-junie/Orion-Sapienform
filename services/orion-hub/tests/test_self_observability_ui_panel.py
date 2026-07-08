"""Static-content tests for the Hub Self tab (self-brain iframe)."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "services" / "orion-hub" / "templates" / "index.html"
PANEL_JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "self_observability.js"

# DOM hooks + endpoint from the legacy 4-card summary body. These were migrated
# to the standalone brain page; if the panel script or template references any
# of them again it will crash at runtime (the elements no longer exist).
REMOVED_SUMMARY_IDS = (
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
LEGACY_SUMMARY_ENDPOINT = "/api/substrate/observability/summary"


def test_template_declares_self_tab_and_brain_iframe() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert 'id="selfObservabilityTabButton"' in template
    assert 'data-hash-target="#self-observability"' in template
    assert '<section id="self-observability" data-panel="self-observability"' in template
    assert 'id="selfBrainFrame"' in template
    assert '/static/self-brain.html?v={{HUB_UI_ASSET_VERSION}}' in template


def test_template_dropped_legacy_summary_cards() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert LEGACY_SUMMARY_ENDPOINT not in template
    for element_id in REMOVED_SUMMARY_IDS:
        assert f'id="{element_id}"' not in template, element_id


def test_template_includes_cache_busted_panel_script() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert (
        '<script src="/static/js/self_observability.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>'
        in template
    )


def test_panel_js_is_tab_controller_only() -> None:
    panel_js = PANEL_JS_PATH.read_text(encoding="utf-8")

    # Tab lifecycle retained (app.js does not route #self-observability).
    assert "function activatePanel()" in panel_js
    assert "function deactivatePanel()" in panel_js
    assert "#self-observability" in panel_js

    # No dead summary fetch / card rendering that would crash against the
    # removed DOM elements.
    assert LEGACY_SUMMARY_ENDPOINT not in panel_js
    for element_id in REMOVED_SUMMARY_IDS:
        assert element_id not in panel_js, element_id
