from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
APP_JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "app.js"


def test_chat_stance_modal_includes_attention_curiosity_section() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")
    assert "buildChatStanceSection('Attention / Curiosity'" in app_js
    assert "model.source_inputs && model.source_inputs.attention_frame" in app_js
