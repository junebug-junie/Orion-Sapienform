"""Static wiring checks for the Hub Energy strip (template, asset, activation)."""

from __future__ import annotations

import re
from pathlib import Path

HUB = Path(__file__).resolve().parents[1]
INDEX = (HUB / "templates/index.html").read_text()
JS = (HUB / "static/js/energy-strip.js").read_text()
BIOMETRICS_VIEW_JS = (HUB / "static/js/biometrics-view.js").read_text()

STRIP_IDS = (
    "energyStrip", "energyImporterState", "energyCycleToDate", "energyProjected",
    "energyForecast", "energyMarginal", "energyPressure", "energyReconcileActual",
    "energyReconcileForecast", "energyDailyBars", "energyCoveredThrough", "energyStaleNote",
)


def test_strip_markup_present() -> None:
    for element_id in STRIP_IDS:
        assert f'id="{element_id}"' in INDEX


def test_every_id_the_js_writes_exists_in_the_template() -> None:
    written = set(re.findall(r'"(energy[A-Z][A-Za-z]+)"', JS))
    assert written, "energy-strip.js should address strip elements by id"
    for element_id in written:
        assert f'id="{element_id}"' in INDEX, element_id


def test_strip_defaults_to_unknown_not_zero() -> None:
    for element_id in ("energyCycleToDate", "energyProjected", "energyForecast", "energyMarginal", "energyPressure"):
        match = re.search(rf'id="{element_id}"[^>]*>([^<]*)<', INDEX)
        assert match and match.group(1).strip() == "unknown", element_id


def test_strip_sits_after_the_cooling_strip() -> None:
    assert INDEX.index('id="cabinetCoolingWattsChart"') < INDEX.index('id="energyStrip"')


def test_script_loaded_with_cache_bust() -> None:
    tag = '<script src="/static/js/energy-strip.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>'
    assert tag in INDEX
    # Must load before biometrics-view.js, which activates it.
    assert INDEX.index(tag) < INDEX.index('<script src="/static/js/biometrics-view.js')
    assert (HUB / "static/js/energy-strip.js").is_file()


def test_js_polls_both_endpoints() -> None:
    assert "/api/energy/latest" in JS and "/api/energy/usage/daily" in JS
    assert "window.OrionEnergyStrip" in JS


def test_cabinet_subview_activates_and_deactivates_the_strip() -> None:
    assert "window.OrionEnergyStrip.activate()" in BIOMETRICS_VIEW_JS
    assert BIOMETRICS_VIEW_JS.count("window.OrionEnergyStrip.deactivate()") == 2


def test_strip_lives_inside_the_cabinet_panel() -> None:
    # The cabinet panel's first "</section>" is its end; the strip must not nest a <section>.
    start = INDEX.index('id="cabinet" data-panel="cabinet"')
    end = INDEX.index("</section>", start)
    assert start < INDEX.index('id="energyStrip"') < end
