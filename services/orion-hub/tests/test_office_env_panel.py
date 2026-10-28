"""Static wiring: the Office strip's template ids, script tag and API URLs agree."""
from __future__ import annotations

import re
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
OFFICE_JS = (HUB_ROOT / "static" / "js" / "office-env.js").read_text(encoding="utf-8")


def test_script_is_loaded_and_exists() -> None:
    assert "/static/js/office-env.js?v={{HUB_UI_ASSET_VERSION}}" in INDEX_HTML


def test_every_element_id_the_js_binds_exists_in_the_template() -> None:
    ids = re.findall(r'getElementById\("([^"]+)"\)', OFFICE_JS)
    assert "officeEnvStrip" in ids
    for element_id in ids:
        if element_id == "cabinet":
            continue  # asserted below: the strip must live inside it
        assert f'id="{element_id}"' in INDEX_HTML, element_id


def test_strip_lives_inside_the_cabinet_subview() -> None:
    cabinet = INDEX_HTML.index('<section id="cabinet"')
    strip = INDEX_HTML.index('id="officeEnvStrip"')
    energy = INDEX_HTML.index('id="energyStrip"')
    assert cabinet < strip < energy


def test_window_buttons_match_the_supported_windows() -> None:
    assert set(re.findall(r'data-office-env-window="([^"]+)"', INDEX_HTML)) == {"24h", "3d", "7d"}


def test_js_urls_match_the_router_prefix() -> None:
    assert '"/api/office/env/latest"' in OFFICE_JS
    assert '"/api/office/env/history?window="' in OFFICE_JS


def test_subtab_is_renamed_but_keeps_its_id() -> None:
    assert re.search(r'id="biometricsSubtabCabinet"[^>]*>Cabinet / Office</button>', INDEX_HTML)
