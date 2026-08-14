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
ORG_SIGNALS_JS = HUB_ROOT / "static" / "js" / "organ-signals-graph-ui.js"
MAIN_PY = HUB_ROOT / "scripts" / "main.py"


def test_index_has_organ_signals_tab_and_graph_hosts() -> None:
    index_html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="signalsTabButton"' in index_html
    assert 'href="#signals"' in index_html
    assert ">Organ signals<" in index_html
    assert '<section id="signals" data-panel="signals"' in index_html
    assert 'id="organSignalsCyHost"' in index_html
    assert 'id="organ-signals-layer-filter"' in index_html
    assert 'organ-signals-graph-ui.js?v={{HUB_UI_ASSET_VERSION}}' in index_html


def test_app_js_wires_signals_hash_and_graph() -> None:
    app_js = APP_JS.read_text(encoding="utf-8")
    assert 'getElementById("signalsTabButton")' in app_js
    assert 'getElementById("signals")' in app_js
    assert 'setActiveTab("signals")' in app_js
    assert 'h === "#signals"' in app_js
    assert "ensureOrganSignalsGraph" in app_js


def test_main_mtime_token_includes_organ_signals_js() -> None:
    """The cache-bust token must move when this module changes.

    Was a grep of main.py for the literal filename, back when
    _ui_asset_mtime_token() hardcoded a four-file list. That list was the bug --
    it silently excluded every other JS module the template loads, so editing
    one produced no new ?v= and browsers kept serving the cached copy
    (found 2026-08-14). main.py now globs static/js, so assert the behavior
    the grep was standing in for: touching this file changes the token.
    """
    import time

    from scripts.main import _ui_asset_mtime_token

    assert ORG_SIGNALS_JS.exists()
    before = _ui_asset_mtime_token()
    original = ORG_SIGNALS_JS.stat().st_mtime
    # The token is max(mtimes), so simulate what an edit actually does: set the
    # mtime to *now-ish*, which necessarily exceeds every other file's. Bumping
    # this file's own old mtime by a fixed delta does not, since some other
    # asset may still be newer -- that made an earlier version of this test pass
    # for the wrong reason and then fail once more files entered the candidate set.
    stamp = time.time() + 10_000
    try:
        os.utime(ORG_SIGNALS_JS, (stamp, stamp))
        assert _ui_asset_mtime_token() != before
    finally:
        os.utime(ORG_SIGNALS_JS, (original, original))
    assert _ui_asset_mtime_token() == before


def test_organ_signals_graph_ui_exports_attach() -> None:
    src = ORG_SIGNALS_JS.read_text(encoding="utf-8")
    assert "window.OrionOrganSignalsGraphUI" in src
    assert "/api/signals/active" in src
    assert "filterSignalsByLayer" in src
    assert "organ-signals-layer-filter" in src or "layerFilterEl" in src
