from __future__ import annotations

import os
import sys
from pathlib import Path

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

INDEX_HTML = HUB_ROOT / "templates" / "index.html"
APP_JS = HUB_ROOT / "static" / "js" / "app.js"


def test_index_has_mind_tab_and_panel() -> None:
    index_html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="mindTabButton"' in index_html
    assert 'href="#mind"' in index_html
    assert 'data-hash-target="#mind"' in index_html
    assert ">Mind<" in index_html
    assert '<section id="mind" data-panel="mind"' in index_html
    assert 'id="mindHoursInput"' in index_html
    assert 'id="mindRefreshButton"' in index_html
    assert 'id="mindFilterOk"' in index_html
    assert 'id="mindDefaultOnSendToggle"' in index_html
    assert 'id="mindRunsTableBody"' in index_html
    assert 'id="mindRunsModal"' in index_html
    assert 'id="mindRunsModalList"' in index_html
    assert 'id="mindRunsModalDetails"' in index_html


def test_app_js_wires_mind_hash_and_recent_api() -> None:
    app_js = APP_JS.read_text(encoding="utf-8")
    assert 'getElementById("mindTabButton")' in app_js
    assert 'getElementById("mind")' in app_js
    assert 'setActiveTab("mind")' in app_js
    assert 'history.replaceState(null, "", "#mind")' in app_js
    assert 'window.location.hash === "#mind"' in app_js or 'h === "#mind"' in app_js
    assert "/api/mind/runs/recent" in app_js
    assert "/api/mind/runs?" in app_js
    assert "/api/mind/runs/" in app_js
    assert "refreshMindRuns" in app_js
    assert "openMindRunsModal" in app_js
    assert "payload.context.metadata.mind_enabled = true" in app_js
