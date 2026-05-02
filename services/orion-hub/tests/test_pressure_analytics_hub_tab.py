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
PRESSURE_STATIC = HUB_ROOT / "static" / "pressure-analytics.html"


def test_index_has_pressure_tab_section_and_frame() -> None:
    index_html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="pressureAnalyticsTabButton"' in index_html
    assert 'href="#pressure"' in index_html
    assert 'data-hash-target="#pressure"' in index_html
    assert ">Pressure<" in index_html
    assert '<section id="pressure" data-panel="pressure"' in index_html
    assert 'id="pressureAnalyticsFrame"' in index_html
    assert 'src="/static/pressure-analytics.html?v={{HUB_UI_ASSET_VERSION}}"' in index_html
    assert 'id="pressureAnalyticsRefresh"' in index_html


def test_app_js_wires_pressure_hash_and_refresh() -> None:
    app_js = APP_JS.read_text(encoding="utf-8")
    assert 'getElementById("pressureAnalyticsTabButton")' in app_js
    assert 'getElementById("pressure")' in app_js
    assert 'getElementById("pressureAnalyticsFrame")' in app_js
    assert 'getElementById("pressureAnalyticsRefresh")' in app_js
    assert 'setActiveTab("pressure")' in app_js
    assert 'history.replaceState(null, "", "#pressure")' in app_js
    assert 'window.location.hash === "#pressure"' in app_js or 'h === "#pressure"' in app_js
    assert "pressurePanel.classList.toggle" in app_js
    assert "isPressure" in app_js
    assert "pressureAnalyticsRefresh" in app_js


def test_static_pressure_page_has_root_section_ids() -> None:
    html = PRESSURE_STATIC.read_text(encoding="utf-8")
    for needle in [
        'id="pressureAnalyticsRoot"',
        'id="pressureKpiStrip"',
        'id="pressureTimeline"',
        'id="pressureHeatmap"',
        'id="pressureLifecycleFunnel"',
        'id="pressureEvidenceTable"',
        'id="pressureInspector"',
    ]:
        assert needle in html
