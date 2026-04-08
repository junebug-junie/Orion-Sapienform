from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
APP_JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "app.js"
TEMPLATE_PATH = REPO_ROOT / "services" / "orion-hub" / "templates" / "index.html"


def test_app_js_adds_feedback_controls_only_for_orion_messages() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")
    assert "if (sender === 'Orion')" in app_js
    assert "const thumbsUp = document.createElement('button');" in app_js
    assert "const thumbsDown = document.createElement('button');" in app_js
    assert "openResponseFeedbackModal('up', meta, text)" in app_js
    assert "openResponseFeedbackModal('down', meta, text)" in app_js
    assert "if (sender === 'Orion') {" in app_js


def test_app_js_submits_feedback_payload_with_response_linkage_fields() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")
    assert "target_turn_id: responseFeedbackDraft.targetTurnId" in app_js
    assert "target_message_id: responseFeedbackDraft.targetMessageId" in app_js
    assert "target_correlation_id: responseFeedbackDraft.targetCorrelationId" in app_js
    assert "feedback_value: responseFeedbackDraft.feedbackValue" in app_js
    assert "categories: Array.from(responseFeedbackDraft.categories)" in app_js
    assert "fetch(`${API_BASE_URL}/api/chat/response-feedback`" in app_js
    assert "linkage_strategy: linkage.linkageStrategy" in app_js
    assert "loadResponseFeedbackOptions()" in app_js


def test_template_includes_response_feedback_modal() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert 'id="responseFeedbackModal"' in template
    assert 'id="responseFeedbackCategoryList"' in template
    assert 'id="responseFeedbackNotes"' in template
    assert 'id="responseFeedbackSubmit"' in template
