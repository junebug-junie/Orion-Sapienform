from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


def _render_hub_index(monkeypatch: pytest.MonkeyPatch, *, proposal_review_enabled: bool) -> str:
    monkeypatch.setenv("HUB_PROPOSAL_REVIEW_ENABLED", "true" if proposal_review_enabled else "false")
    for mod in ("scripts.main", "scripts.settings", "app.settings"):
        sys.modules.pop(mod, None)
    import app.settings as app_settings

    app_settings.get_settings.cache_clear()
    import scripts.main as hub_main

    return hub_main.render_hub_index_html(memory_pool_ok=False)


def test_hub_main_layout_is_fifty_fifty_and_scrollable_chat() -> None:
    template = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert template.count("md:w-1/2 bg-gray-900 rounded-2xl shadow-lg p-5 flex flex-col space-y-4 h-[56rem] min-h-0") >= 2
    assert 'id="conversation" class="flex-1 min-h-0' in template


def test_hub_nav_is_horizontal_scroll_and_settings_toggle_removed() -> None:
    template = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'id="hubPrimaryNav"' in template
    assert "hub-tab-nav" in template
    assert 'id="settingsToggle"' not in template
    assert 'id="settingsOpenButton"' in template


def test_orion_vision_panel_is_full_width() -> None:
    template = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert "Orion's Vision" in template
    assert "lg:w-1/2" not in template.split("Orion's Vision")[1].split("</section>")[0]


def test_orphan_messages_panel_removed_and_proposal_review_gated() -> None:
    template = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    main_py = (HUB_ROOT / "scripts" / "main.py").read_text(encoding="utf-8")
    assert "messagesPanel" not in template
    assert "messagesToggle" not in template
    assert "{{HUB_PROPOSAL_REVIEW_PANEL}}" in template
    assert "{{HUB_PROPOSAL_REVIEW_SCRIPT}}" in template
    assert "proposalReviewPanel" not in template
    assert "HUB_PROPOSAL_REVIEW_ENABLED" in main_py
    assert "Pending Decisions" in main_py

def test_service_logs_use_row_cards_only() -> None:
    template = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    ui = (HUB_ROOT / "static" / "js" / "service-logs-ui.js").read_text(encoding="utf-8")
    assert "serviceLogsMasterTerminal" not in template
    assert "serviceLogsMasterTerminal" not in ui
    assert 'id="serviceLogsTerminals" class="flex flex-col gap-4"' in template


def test_app_js_formats_pending_attention_in_mdt() -> None:
    app_js = (HUB_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "formatHubLocalTime" in app_js
    assert "America/Denver" in app_js
    assert "loadChatMessages" not in app_js
    assert "syncMessagesPanelVisibility" not in app_js

def test_substrate_lattice_coerces_non_array_join_fields() -> None:
    js = (HUB_ROOT / "static" / "js" / "substrate-lattice.js").read_text(encoding="utf-8")
    routes = (HUB_ROOT / "scripts" / "substrate_lattice_routes.py").read_text(encoding="utf-8")
    assert "function _asList" in js
    assert "def _coerce_str_list" in routes


def test_hub_cfg_exposes_proposal_review_flag() -> None:
    main_py = (HUB_ROOT / "scripts" / "main.py").read_text(encoding="utf-8")
    assert '"proposalReviewEnabled"' in main_py
    assert '"notifyEnabled"' not in main_py


def test_render_hub_index_html_injects_proposal_review_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered = _render_hub_index(monkeypatch, proposal_review_enabled=True)
    assert 'id="proposalReviewPanel"' in rendered
    assert "proposal-review-ui.js" in rendered
    assert '"proposalReviewEnabled":true' in rendered.replace(" ", "")


def test_render_hub_index_html_omits_proposal_review_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered = _render_hub_index(monkeypatch, proposal_review_enabled=False)
    assert 'id="proposalReviewPanel"' not in rendered
    assert "proposal-review-ui.js" not in rendered
    assert '"proposalReviewEnabled":false' in rendered.replace(" ", "")
