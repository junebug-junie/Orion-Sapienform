from __future__ import annotations

from pathlib import Path


HUB_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = HUB_ROOT / "templates" / "index.html"
APP_JS_PATH = HUB_ROOT / "static" / "js" / "app.js"


def test_template_includes_recall_canary_profile_dropdown_and_safety_copy() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert 'value="recall_v2_shadow_default"' in template
    assert 'id="recallCanaryProfileSelect"' in template
    assert 'id="recallCanaryProfileEmptyState"' in template
    assert 'id="recallCanarySafetyBadges"' in template
    assert "No recall profiles available for canary testing." in template
    assert "Production recall remains V1" in template
    assert "No production promotion" in template
    assert 'id="recallCanaryOperatorTokenInput"' not in template
    assert 'id="recallCanaryRememberTokenSession"' not in template
    assert 'id="recallCanaryToggle"' in template
    assert 'id="recallCanaryBody"' in template
    assert 'id="recallCanaryJudgmentSelect"' in template
    assert "Run a manual canary query below, then judge the result or create a review artifact." in template
    assert "Use panel controls below for judgments and review artifacts." not in template


def test_app_js_wires_recall_canary_profile_loading_selection_and_payload() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")
    assert "const RECALL_CANARY_PROFILE_STORAGE_KEY = 'orion_recall_canary_profile_v1';" in app_js
    assert "function hydrateRecallCanaryProfileSelect(data = {})" in app_js
    assert "localStorage.getItem(RECALL_CANARY_PROFILE_STORAGE_KEY)" in app_js
    assert "localStorage.setItem(RECALL_CANARY_PROFILE_STORAGE_KEY, selectedValue);" in app_js
    assert "body: JSON.stringify({ query_text: queryText, profile_id: profileId })" in app_js
    assert "'X-Orion-Operator-Token': operatorToken" not in app_js
    assert "function toggleRecallCanaryPanel()" in app_js
    assert "selected profile:" in app_js
    assert "recallCanaryRunButton.disabled = disabled;" in app_js


def test_recall_canary_profile_ui_does_not_expose_unsafe_actions() -> None:
    merged = TEMPLATE_PATH.read_text(encoding="utf-8") + "\n" + APP_JS_PATH.read_text(encoding="utf-8")
    forbidden = [
        "Make Default",
        "Promote to Production",
        "Apply Recall Patch",
        "Apply Recall Profile",
        "Enable Recall V2",
        "Switch Production Recall",
        "Production Default",
        "Live Apply",
        "Auto Promote",
    ]
    for token in forbidden:
        assert token not in merged
