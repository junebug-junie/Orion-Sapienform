"""Static wiring for the Hub Orion Energy tab (button, section, iframe, app.js, page ids)."""

from __future__ import annotations

import re
from pathlib import Path

HUB = Path(__file__).resolve().parents[1]
INDEX = (HUB / "templates/index.html").read_text(encoding="utf-8")
APP_JS = (HUB / "static/js/app.js").read_text(encoding="utf-8")
PAGE = (HUB / "static/orion-energy.html").read_text(encoding="utf-8")
PAGE_JS = (HUB / "static/js/orion-energy.js").read_text(encoding="utf-8")


def test_index_has_button_section_and_versioned_frame() -> None:
    assert 'id="orionEnergyTabButton"' in INDEX
    assert 'data-hash-target="#orion-energy"' in INDEX
    assert '<section id="orion-energy" data-panel="orion-energy"' in INDEX
    assert 'src="/static/orion-energy.html?v={{HUB_UI_ASSET_VERSION}}"' in INDEX


def test_app_js_routes_hash_click_and_visibility() -> None:
    for needle in ('getElementById("orionEnergyTabButton")', 'getElementById("orion-energy")',
                   'getElementById("orionEnergyFrame")', 'setActiveTab("orion-energy")',
                   'h === "#orion-energy"', "styleTabButton(orionEnergyTabButton, isOrionEnergy)",
                   'orionEnergyPanelEl.classList.toggle("hidden", !isOrionEnergy)'):
        assert needle in APP_JS, needle


def test_every_id_the_page_js_writes_exists_in_the_page() -> None:
    written = set(re.findall(r'getElementById\("(oe[A-Z][A-Za-z]+)"\)', PAGE_JS))
    assert {"oeKwhChart", "oeDeltaChart", "oeDeltaLegend", "oeTooltip", "oeTable"} <= written
    for element_id in written:
        assert f'id="{element_id}"' in PAGE, element_id


def test_page_offers_every_grain_and_loads_versioned_script() -> None:
    for grain in ("hour", "day", "week", "month", "quarter", "year"):
        assert f'data-grain="{grain}"' in PAGE
    assert '"/static/js/orion-energy.js" + location.search' in PAGE
    assert "/api/energy/orion?grain=" in PAGE_JS


def test_tiles_start_unknown_not_zero() -> None:
    for element_id in ("oeOrionKwh", "oeHouseKwh", "oeShare", "oeOrionUsd", "oeHouseUsd"):
        match = re.search(rf'id="{element_id}"[^>]*>([^<]*)<', PAGE)
        assert match and match.group(1).strip() == "unknown", element_id
