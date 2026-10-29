"""Static wiring: the climate overlay strip's ids, chips, script order and API URL agree."""
from __future__ import annotations

import re
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
OVERLAY_JS = (HUB_ROOT / "static" / "js" / "cabinet-climate-overlay.js").read_text(encoding="utf-8")


def test_every_element_id_the_js_binds_exists_in_the_template() -> None:
    ids = re.findall(r'getElementById\("([^"]+)"\)', OVERLAY_JS)
    assert "climateOverlayStrip" in ids
    for element_id in ids:
        assert f'id="{element_id}"' in INDEX_HTML, element_id


def test_strip_lives_inside_the_cabinet_subview() -> None:
    cabinet = INDEX_HTML.index('<section id="cabinet"')
    office = INDEX_HTML.index('id="officeEnvStrip"')
    strip = INDEX_HTML.index('id="climateOverlayStrip"')
    energy = INDEX_HTML.index('id="energyStrip"')
    assert cabinet < office < strip < energy


def test_filter_chips_cover_exactly_the_js_series() -> None:
    chips = re.findall(r'data-climate-overlay-series="([^"]+)"', INDEX_HTML)
    js_keys = re.findall(r'\{ key: "([a-z_]+)", label:', OVERLAY_JS)
    assert chips == js_keys == ["office_temp_c", "cabinet_temp_c", "outdoor_temp_c", "orion_watts"]


def test_window_buttons_match_the_supported_windows() -> None:
    assert set(re.findall(r'data-climate-overlay-window="([^"]+)"', INDEX_HTML)) == {"24h", "3d", "7d"}


def test_js_url_matches_the_router() -> None:
    assert '"/api/cabinet/climate/overlay?window="' in OVERLAY_JS


def test_shared_hover_loads_before_every_chart_that_uses_it() -> None:
    hover = INDEX_HTML.index("/static/js/chart-hover.js?v={{HUB_UI_ASSET_VERSION}}")
    for script in ("cabinet-sensors.js", "office-env.js", "cabinet-climate-overlay.js"):
        assert hover < INDEX_HTML.index(f"/static/js/{script}?v={{{{HUB_UI_ASSET_VERSION}}}}"), script


def test_office_charts_wire_the_shared_hover() -> None:
    office_js = (HUB_ROOT / "static" / "js" / "office-env.js").read_text(encoding="utf-8")
    assert "chartHover.attachSeriesHover(" in office_js
