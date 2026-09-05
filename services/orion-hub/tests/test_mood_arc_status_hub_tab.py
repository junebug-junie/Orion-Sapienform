from __future__ import annotations

import sys
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

INDEX_HTML = HUB_ROOT / "templates" / "index.html"
APP_JS = HUB_ROOT / "static" / "js" / "app.js"
MOOD_ARC_STATUS_STATIC = HUB_ROOT / "static" / "mood-arc-status.html"


def test_index_has_mood_arc_status_tab_nav_button() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="moodArcStatusTabButton"' in html
    assert 'href="#mood-arc-status"' in html
    assert 'data-hash-target="#mood-arc-status"' in html
    assert ">Mood Arc Status<" in html


def test_index_has_mood_arc_status_section_and_frame() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert '<section id="mood-arc-status" data-panel="mood-arc-status"' in html
    assert 'id="moodArcStatusFrame"' in html
    assert 'src="/static/mood-arc-status.html?v={{HUB_UI_ASSET_VERSION}}"' in html


def test_app_js_wires_mood_arc_status_hash_and_tab() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    assert 'getElementById("moodArcStatusTabButton")' in js
    assert 'getElementById("mood-arc-status")' in js
    assert 'getElementById("moodArcStatusFrame")' in js
    assert 'setActiveTab("mood-arc-status")' in js
    assert "#mood-arc-status" in js


def test_mood_arc_status_static_page_has_root_ids() -> None:
    html = MOOD_ARC_STATUS_STATIC.read_text(encoding="utf-8")
    for needle in [
        'id="masRoot"',
        'id="masLiveCard"',
        'id="masPhiCard"',
        'id="masAutoRefresh"',
        'id="masReconCanvas"',
        'id="masChannelCanvas"',
        'id="masTriggerCanvas"',
        'id="masTooltip"',
        'id="masInterpretation"',
    ]:
        assert needle in html, f"Missing: {needle}"


def test_mood_arc_status_static_page_fetches_all_four_endpoints() -> None:
    html = MOOD_ARC_STATUS_STATIC.read_text(encoding="utf-8")
    assert "/api/mood-arc-status/live" in html
    assert "/api/mood-arc-status/phi-v2-inventory" in html
    assert "/api/mood-arc-status/inference-trace" in html
    assert "/api/mood-arc-status/downstream-triggers" in html
