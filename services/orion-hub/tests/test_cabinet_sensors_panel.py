"""Cabinet sensors operator tab: wiring contract + static-content assertions.

This tab visualizes Athena Nano host snapshots from
GET /api/cabinet/sensors/latest (cabinet_sensors_routes.py, Task 1). Modeled
directly on test_field_attention_operator_panel.py: the panel needs the same
app.js registration points (element binding, missing-panel fallback,
visibility toggle, button styling, hash routing) plus a script tag plus a nav
anchor -- missing any one produces a tab that silently falls back to Hub on a
specific interaction rather than failing loudly.

The backend route is covered by test_cabinet_sensors_api.py; this file does
not duplicate API behavior.
"""

from __future__ import annotations

from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]

INDEX_HTML = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (HUB_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
CABINET_SENSORS_JS = (HUB_ROOT / "static" / "js" / "cabinet-sensors.js").read_text(
    encoding="utf-8"
)
FIELD_ATTENTION_JS = (HUB_ROOT / "static" / "js" / "field-attention.js").read_text(
    encoding="utf-8"
)


# --------------------------------------------------------------------------
# Template wiring
# --------------------------------------------------------------------------


def test_template_declares_nav_button_panel_and_script_tag() -> None:
    assert 'data-hash-target="#cabinet"' in INDEX_HTML
    assert 'id="cabinetTabButton"' in INDEX_HTML
    assert 'id="cabinet" data-panel="cabinet"' in INDEX_HTML
    assert "/static/js/cabinet-sensors.js?v={{HUB_UI_ASSET_VERSION}}" in INDEX_HTML
    for mount_id in (
        "cabinetStatus",
        "cabinetSensorGrid",
        "cabinetPressureStrip",
        "cabinetRefreshBtn",
    ):
        assert f'id="{mount_id}"' in INDEX_HTML, mount_id


def test_template_panel_does_not_reuse_field_attention_naming() -> None:
    """Distinct subsystem -- a collision would bind DOM lookups to the wrong tab."""
    section_start = INDEX_HTML.index('id="cabinet" data-panel="cabinet"')
    section_end = INDEX_HTML.index("</section>", section_start)
    section_html = INDEX_HTML[section_start:section_end]
    assert "fieldAttention" not in section_html
    assert 'id="field-attention"' not in section_html


def test_template_names_no_snapshot_service_and_hub_activity_label() -> None:
    assert "orion-cabinet-sensors.service" in INDEX_HTML
    assert "activity (Hub)" in INDEX_HTML


# --------------------------------------------------------------------------
# app.js registration points
# --------------------------------------------------------------------------


def test_app_js_registers_the_panel_in_every_place_setactivetab_needs() -> None:
    """All Field Attention-style registration points -- see module docstring."""
    assert 'document.getElementById("cabinetTabButton")' in APP_JS
    assert 'document.getElementById("cabinet")' in APP_JS
    assert 'tabKey === "cabinet" && !cabinetPanel' in APP_JS
    assert 'const isCabinet = effectiveTab === "cabinet";' in APP_JS
    assert 'cabinetPanel.classList.toggle("hidden", !isCabinet);' in APP_JS
    assert "styleTabButton(cabinetTabButton, isCabinet);" in APP_JS
    assert 'h === "#cabinet" && cabinetPanel' in APP_JS
    assert 'history.replaceState(null, "", "#cabinet");' in APP_JS


def test_app_js_resets_an_unresolvable_cabinet_hash_to_hub() -> None:
    assert '|| h === "#cabinet"' in APP_JS


def test_app_js_drives_the_panels_poll_lifecycle_on_tab_switch() -> None:
    assert "window.OrionCabinetSensors.activate()" in APP_JS
    assert "window.OrionCabinetSensors.deactivate()" in APP_JS


def test_app_js_does_not_confuse_cabinet_with_field_attention() -> None:
    assert "window.OrionCabinetSensors" not in FIELD_ATTENTION_JS
    assert "window.OrionFieldAttention" not in CABINET_SENSORS_JS


# --------------------------------------------------------------------------
# cabinet-sensors.js content
# --------------------------------------------------------------------------


def test_cabinet_sensors_js_is_standalone_and_reads_only_its_own_api() -> None:
    assert "window.OrionCabinetSensors" in CABINET_SENSORS_JS
    assert "activate:" in CABINET_SENSORS_JS and "deactivate:" in CABINET_SENSORS_JS
    assert '"/api/cabinet/sensors/latest"' in CABINET_SENSORS_JS
    assert "var POLL_MS = 1000;" in CABINET_SENSORS_JS
    assert "window.OrionHub" not in CABINET_SENSORS_JS
    assert '"POST"' not in CABINET_SENSORS_JS and "method: 'POST'" not in CABINET_SENSORS_JS


def test_cabinet_sensors_js_renders_status_grid_and_pressure_strip() -> None:
    for renderer in (
        "renderStatus",
        "renderSensorGrid",
        "renderPressureStrip",
        "renderPayload",
    ):
        assert f"function {renderer}(" in CABINET_SENSORS_JS, renderer
    for tile_key in (
        "environment",
        "uv",
        "magnetic",
        "particulate",
        "lidar",
        "imu",
    ):
        assert tile_key in CABINET_SENSORS_JS, tile_key
    for pressure_key in (
        "cabinet_climate_activity",
        "cabinet_particulate_activity",
        "cabinet_em_activity",
        "cabinet_uv_activity",
        "cabinet_vibration_activity",
        "cabinet_proximity_activity",
        "cabinet_sensor_staleness",
    ):
        assert pressure_key in CABINET_SENSORS_JS, pressure_key


def test_cabinet_sensors_js_absent_not_zero_contract() -> None:
    """Missing frame sub-objects must render as absent, never zero-filled."""
    assert "absent" in CABINET_SENSORS_JS
    assert "absent-not-zero" in CABINET_SENSORS_JS or "absent is not zero" in CABINET_SENSORS_JS.lower()
    assert "function formatAbsentOrValue(" in CABINET_SENSORS_JS or "function renderTile(" in CABINET_SENSORS_JS


def test_cabinet_sensors_js_keeps_last_good_on_poll_error() -> None:
    assert "poll error" in CABINET_SENSORS_JS.lower() or "poll-error" in CABINET_SENSORS_JS
    assert "state.lastPayload" in CABINET_SENSORS_JS
    # On fetch failure, do not blank previously good content.
    catch_region = CABINET_SENSORS_JS[
        CABINET_SENSORS_JS.index("catch (err)") : CABINET_SENSORS_JS.index("} finally {")
    ]
    assert "lastPayload" in catch_region or "poll" in catch_region.lower()


def test_cabinet_sensors_js_names_no_snapshot_service() -> None:
    assert "orion-cabinet-sensors.service" in CABINET_SENSORS_JS


def test_cabinet_sensors_js_guards_polling_on_real_visibility() -> None:
    assert 'els.panel.classList.contains("hidden")' in CABINET_SENSORS_JS
    guard_region = CABINET_SENSORS_JS[
        CABINET_SENSORS_JS.index("function startTimer") : CABINET_SENSORS_JS.index(
            "function activate"
        )
    ]
    assert 'contains("hidden")' in guard_region
    assert "deactivate();" in guard_region


def test_cabinet_sensors_js_does_not_grow_unbounded_client_state() -> None:
    assert "state.lastPayload =" in CABINET_SENSORS_JS
    assert ".push(" not in CABINET_SENSORS_JS
