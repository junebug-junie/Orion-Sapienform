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


def test_index_has_forge_tab_and_panel() -> None:
    index_html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="forgeTabButton"' in index_html
    assert 'href="#forge"' in index_html
    assert 'data-hash-target="#forge"' in index_html
    assert ">Forge<" in index_html
    assert '<section id="forge" data-panel="forge"' in index_html
    assert 'id="forgeStatusStrip"' in index_html
    assert 'id="forgeCompileTask"' in index_html
    assert 'id="forgeCompileButton"' in index_html
    assert 'id="forgeDebugDetails"' in index_html
    assert "Source Delta Review" in index_html
    assert 'id="forgeSourcePath"' in index_html
    assert 'id="forgeSourceId"' in index_html
    assert 'id="forgeSourceKind"' in index_html
    assert 'id="forgeSourceDryRun"' in index_html
    assert 'id="forgeSourceWriteReview"' in index_html
    assert 'id="forgeSourceIngestButton"' in index_html
    assert 'id="forgeSourceIngestResult"' in index_html
    assert 'id="forgeSourceIngestError"' in index_html
    assert "does not mutate accepted claims" in index_html
    assert 'value="docs/PR-orion-knowledge-forge-source-delta-v1.2.md"' not in index_html


def test_app_js_wires_forge_hash_and_knowledge_api() -> None:
    app_js = APP_JS.read_text(encoding="utf-8")
    assert 'getElementById("forgeTabButton")' in app_js
    assert 'getElementById("forge")' in app_js
    assert 'setActiveTab("forge")' in app_js
    assert 'history.replaceState(null, "", "#forge")' in app_js
    assert 'window.location.hash === "#forge"' in app_js or 'h === "#forge"' in app_js
    assert "KNOWLEDGE_PROXY_BASE" in app_js
    assert "/api/knowledge" in app_js
    assert "refreshForgeTab" in app_js
    assert "/api/knowledge/status" in app_js or 'knowledgeForgeFetch("/status")' in app_js
    assert "context-packs/compile" in app_js
    assert "runForgeSearch" in app_js
    assert "runForgeCompile" in app_js
    assert "runForgeSourceIngest" in app_js
    assert "/sources/ingest" in app_js
    assert "forgeRenderSourceIngestResult" in app_js
    assert "forgeSourceIngestContentPreview" in app_js
    assert "forgeDebugSourceIngest" in app_js
