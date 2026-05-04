from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
APP_JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "app.js"
MEMORY_JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "memory.js"
TEMPLATE_PATH = REPO_ROOT / "services" / "orion-hub" / "templates" / "index.html"


def test_template_includes_memory_graph_bridge_modal_and_a11y() -> None:
    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert 'id="memoryGraphBridgeModal"' in html
    assert 'id="memoryGraphBridgeSuggest"' in html
    assert 'id="memoryGraphBridgeToMemory"' in html
    assert 'id="memoryGraphBridgeChainHints"' in html
    assert 'id="memoryGraphBridgeModalTitle"' in html
    assert 'role="dialog"' in html
    assert "aria-modal=\"true\"" in html
    assert "aria-labelledby=\"memoryGraphBridgeModalTitle\"" in html


def test_app_js_wires_memory_graph_bridge_handlers() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")
    assert "function collectConversationTurnsUpTo" in app_js
    assert "skippedWithoutId" in app_js
    assert "memoryGraphBridgeTurnsCache" in app_js
    assert "verbs: ['memory_graph_suggest']" in app_js
    assert "setupMemoryGraphBridgeModal();" in app_js
    assert "closeMemoryGraphBridgeModal();" in app_js
    assert "CustomEvent('orion-hub-memory-graph-draft-import'" in app_js
    assert "backfillLatestUserTurnIdForGraph" in app_js
    assert "extractCortexStepErrorHint" in app_js


def test_memory_js_listens_for_bridge_import_event() -> None:
    memory_js = MEMORY_JS_PATH.read_text(encoding="utf-8")
    assert 'orion-hub-memory-graph-draft-import' in memory_js
    assert "orion_memory_graph_draft_import" in memory_js
