from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "services" / "orion-hub" / "templates" / "index.html"
APP_JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "app.js"


def test_template_includes_chat_stance_panel_and_outer_modal_button() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert 'id="chatStanceDebugPanel"' in template
    assert 'id="chatStanceDebugToggle"' in template
    assert 'id="chatStanceDebugOpenModal"' in template
    assert template.index('id="chatStanceDebugOpenModal"') < template.index('id="chatStanceDebugBody"')
    assert "Chat Stance" in template


def test_template_includes_chat_stance_modal_shell() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert 'id="chatStanceDebugModalRoot" class="hidden fixed inset-0 z-[120]' in template
    assert 'id="chatStanceDebugModalBackdrop" class="fixed inset-0 z-[120]' in template
    assert 'id="chatStanceDebugModalDialog" class="fixed inset-x-4 top-8 bottom-8 z-[121]' in template
    assert 'id="chatStanceDebugModalBody"' in template


def test_app_js_chat_stance_empty_state_and_grouped_modal_sections() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")
    assert "function clearChatStanceDebugPanel()" in app_js
    assert "No chat stance debug payload on this turn." in app_js
    assert "function updateChatStanceDebugPanel(payload)" in app_js
    assert "buildChatStanceSection('Overview'" in app_js
    assert "buildChatStanceSection('Source Inputs by Category'" in app_js
    assert "buildChatStanceSection('Synthesized Brief'" in app_js
    assert "buildChatStanceSection('Enforcement / Fallback'" in app_js
    assert "buildChatStanceSection('Final Prompt Contract'" in app_js
    assert "buildChatStanceSection('Raw compact JSON'" in app_js


def test_app_js_wires_chat_stance_modal_and_payload_plumbing() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")
    assert "chatStanceDebug: d.chat_stance_debug," in app_js
    assert "updateChatStanceDebugPanel(meta.chatStanceDebug || meta.chat_stance_debug);" in app_js
    assert "chatStanceDebugToggle.addEventListener('click', toggleChatStanceDebugPanel);" in app_js
    assert "chatStanceDebugOpenModal.addEventListener('click', (event) => {" in app_js
    assert "openChatStanceDebugModal();" in app_js
