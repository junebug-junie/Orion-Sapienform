"""Static wiring: the Office strip's template ids, script tag and API URLs agree."""
from __future__ import annotations

import re
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
OFFICE_JS = (HUB_ROOT / "static" / "js" / "office-env.js").read_text(encoding="utf-8")


def test_script_is_loaded_and_exists() -> None:
    assert "/static/js/office-env.js?v={{HUB_UI_ASSET_VERSION}}" in INDEX_HTML


def _js_bound_ids() -> list[str]:
    """Literal getElementById calls plus the strip configs' ``id:`` / ``...Id:`` keys."""
    ids = re.findall(r'getElementById\("([^"]+)"\)', OFFICE_JS)
    ids += re.findall(r'\b(?:id|stripId|statusId|ageId|liveId):\s*"([^"]+)"', OFFICE_JS)
    return ids


def test_every_element_id_the_js_binds_exists_in_the_template() -> None:
    ids = _js_bound_ids()
    for expected in (
        "officeEnvStrip", "officeEnvPm25", "officeEnvPm10", "officeEnvPm1", "officeEnvLux",
        "officeEnvLuxChart", "closetEnvStrip", "closetEnvCo2", "closetEnvCo2Chart",
    ):
        assert expected in ids, expected
    for element_id in ids:
        if element_id == "cabinet":
            continue  # asserted below: the strip must live inside it
        assert INDEX_HTML.count(f'id="{element_id}"') == 1, element_id


def test_strip_lives_inside_the_cabinet_subview() -> None:
    cabinet = INDEX_HTML.index('<section id="cabinet"')
    strip = INDEX_HTML.index('id="officeEnvStrip"')
    energy = INDEX_HTML.index('id="energyStrip"')
    assert cabinet < strip < energy


def test_closet_strip_lives_inside_the_cabinet_subview() -> None:
    cabinet = INDEX_HTML.index('<section id="cabinet"')
    office = INDEX_HTML.index('id="officeEnvStrip"')
    closet = INDEX_HTML.index('id="closetEnvStrip"')
    energy = INDEX_HTML.index('id="energyStrip"')
    assert cabinet < office < closet < energy


def test_window_buttons_match_the_supported_windows() -> None:
    assert set(re.findall(r'data-office-env-window="([^"]+)"', INDEX_HTML)) == {"24h", "3d", "7d"}
    assert set(re.findall(r'data-closet-env-window="([^"]+)"', INDEX_HTML)) == {"24h", "3d", "7d"}
    assert '"data-closet-env-window"' in OFFICE_JS


def test_js_urls_match_the_router_prefix() -> None:
    assert '"/api/office/env/latest"' in OFFICE_JS
    assert '"/api/office/env/history?window="' in OFFICE_JS
    assert '"/api/closet/env/latest"' in OFFICE_JS
    assert '"/api/closet/env/history?window="' in OFFICE_JS


def test_closet_router_is_registered() -> None:
    api_routes = (HUB_ROOT / "scripts" / "api_routes.py").read_text(encoding="utf-8")
    assert "from .closet_env_routes import router as closet_env_router" in api_routes
    assert "router.include_router(closet_env_router)" in api_routes


def test_subtab_is_renamed_but_keeps_its_id() -> None:
    assert re.search(r'id="biometricsSubtabCabinet"[^>]*>Cabinet / Office</button>', INDEX_HTML)
