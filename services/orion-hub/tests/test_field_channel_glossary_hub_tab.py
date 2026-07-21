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
GLOSSARY_STATIC = HUB_ROOT / "static" / "field-channel-glossary.html"


def test_index_has_field_channel_glossary_tab_nav_button() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="fieldChannelGlossaryTabButton"' in html
    assert 'href="#field-channel-glossary"' in html
    assert 'data-hash-target="#field-channel-glossary"' in html
    assert ">Field Channels<" in html


def test_index_has_field_channel_glossary_section_and_frame() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert '<section id="field-channel-glossary" data-panel="field-channel-glossary"' in html
    assert 'id="fieldChannelGlossaryFrame"' in html
    assert 'src="/static/field-channel-glossary.html?v={{HUB_UI_ASSET_VERSION}}"' in html


def test_app_js_wires_field_channel_glossary_hash_and_tab() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    assert 'getElementById("fieldChannelGlossaryTabButton")' in js
    assert 'getElementById("field-channel-glossary")' in js
    assert 'getElementById("fieldChannelGlossaryFrame")' in js
    assert 'setActiveTab("field-channel-glossary")' in js
    assert "#field-channel-glossary" in js


def test_glossary_static_page_has_root_ids() -> None:
    html = GLOSSARY_STATIC.read_text(encoding="utf-8")
    for needle in [
        'id="fieldChannelGlossaryRoot"',
        'id="fcgCategoryStrip"',
        'id="fcgTable"',
        'id="fcgTableBody"',
        'id="fcgWindowHours"',
        'id="fcgShowAll"',
    ]:
        assert needle in html, f"Missing: {needle}"


def test_glossary_static_page_fetches_both_endpoints() -> None:
    html = GLOSSARY_STATIC.read_text(encoding="utf-8")
    assert "/api/field-channel-glossary/channels" in html
    assert "/api/field-channel-glossary/health" in html
