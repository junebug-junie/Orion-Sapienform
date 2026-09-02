"""Biometrics view: Cognitive EKG toggle + deep-inspection modal, static-content
and wiring-contract assertions.

This repo's real UI-test convention (confirmed via test_reverie_visual_cockpit_ui.py
and test_cabinet_sensors_panel.py): string/regex assertions against the served
template and JS, not jsdom-driven interaction tests -- none exist anywhere in
this codebase, so this file doesn't introduce the first one.

Backend endpoints are covered by test_biometrics_preview_api.py and
test_biometrics_node_client.py; this file only checks that the page actually
wires up to them.
"""

from __future__ import annotations

from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]

INDEX_HTML = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (HUB_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
BIOMETRICS_VIEW_JS = (HUB_ROOT / "static" / "js" / "biometrics-view.js").read_text(
    encoding="utf-8"
)


# --------------------------------------------------------------------------
# Cognitive EKG card toggle
# --------------------------------------------------------------------------


def test_ekg_card_declares_toggle_and_both_view_containers() -> None:
    assert 'id="ekgViewToggle"' in INDEX_HTML
    assert 'id="stateVisualizerContainer"' in INDEX_HTML
    assert 'id="biometricsPreviewContainer"' in INDEX_HTML
    assert 'id="biometricsPreviewGrid"' in INDEX_HTML
    # Preview container starts hidden -- the Brain State iframe is the default view.
    preview_idx = INDEX_HTML.index('id="biometricsPreviewContainer"')
    tag_start = INDEX_HTML.rfind("<div", 0, preview_idx)
    tag_end = INDEX_HTML.index(">", preview_idx)
    assert "hidden" in INDEX_HTML[tag_start:tag_end]


def test_biometrics_view_js_wires_the_card_toggle() -> None:
    assert 'el("ekgViewToggle")' in BIOMETRICS_VIEW_JS
    assert "toggleCardView" in BIOMETRICS_VIEW_JS
    assert 'el("biometricsPreviewContainer")' in BIOMETRICS_VIEW_JS
    assert "openModal" in BIOMETRICS_VIEW_JS


def test_biometrics_view_js_only_polls_preview_while_shown() -> None:
    assert "CARD_POLL_MS" in BIOMETRICS_VIEW_JS
    assert "clearInterval(cardPollTimer)" in BIOMETRICS_VIEW_JS


def test_app_js_deactivates_biometrics_view_when_leaving_the_hub_tab() -> None:
    """Regression test (code review finding): without this, the EKG card's
    cardPollTimer -- started by the Biometrics preview toggle -- kept firing
    GET /api/biometrics/preview/snapshot every 10s forever after navigating
    to any other top-level tab, since hubTabPanel's hidden-toggle had no
    accompanying OrionBiometricsView.deactivate() call (every sibling panel,
    e.g. isFieldAttention/isReverie, already had this pair)."""
    toggle_idx = APP_JS.index('hubTabPanel.classList.toggle("hidden", !isHub);')
    nearby = APP_JS[toggle_idx : toggle_idx + 700]
    assert "window.OrionBiometricsView" in nearby
    assert "window.OrionBiometricsView.deactivate()" in nearby


def test_biometrics_view_js_resumes_preview_poll_on_reactivate() -> None:
    """Companion to the deactivate fix: coming back to the hub tab with the
    preview toggled on must resume polling, not silently strand the view."""
    activate_region = BIOMETRICS_VIEW_JS[
        BIOMETRICS_VIEW_JS.index("function activate() {") : BIOMETRICS_VIEW_JS.index(
            "function deactivate() {"
        )
    ]
    assert 'cardView === "biometrics"' in activate_region
    assert "showCardView(" in activate_region


# --------------------------------------------------------------------------
# Modal shell (app.js mechanics, matching every other Hub modal)
# --------------------------------------------------------------------------


def test_template_declares_modal_root_backdrop_dialog_near_fullscreen() -> None:
    assert 'id="biometricsModalRoot"' in INDEX_HTML
    assert 'id="biometricsModalBackdrop"' in INDEX_HTML
    assert 'id="biometricsModalDialog"' in INDEX_HTML
    assert 'id="biometricsModalClose"' in INDEX_HTML
    dialog_start = INDEX_HTML.index('id="biometricsModalDialog"')
    dialog_tag_end = INDEX_HTML.index(">", dialog_start)
    dialog_tag = INDEX_HTML[dialog_start:dialog_tag_end]
    assert "w-[92vw]" in dialog_tag
    assert "h-[92vh]" in dialog_tag


def test_template_declares_four_subtabs_and_subview_panels() -> None:
    for subtab_id in (
        "biometricsSubtabAthena",
        "biometricsSubtabCirce",
        "biometricsSubtabGpu",
        "biometricsSubtabCabinet",
    ):
        assert f'id="{subtab_id}"' in INDEX_HTML
    for panel_id in ("biometricsSubviewAthena", "biometricsSubviewCirce", "biometricsSubviewGpu"):
        assert f'id="{panel_id}"' in INDEX_HTML
    # Cabinet's own subview panel IS #cabinet -- no separate wrapper id.
    assert 'id="cabinet" data-panel="cabinet"' in INDEX_HTML


def test_app_js_wires_open_close_escape_backdrop_and_scroll_lock() -> None:
    assert "function openBiometricsModal()" in APP_JS
    assert "function closeBiometricsModal()" in APP_JS
    assert "biometricsModalBackdrop.addEventListener('click', closeBiometricsModal)" in APP_JS
    assert "biometricsModalRoot.addEventListener('click'" in APP_JS
    assert "biometricsModalClose.addEventListener('click', closeBiometricsModal)" in APP_JS
    assert (
        "event.key === 'Escape' && biometricsModalRoot && !biometricsModalRoot.classList.contains('hidden')"
        in APP_JS
    )
    assert "isModalVisible(biometricsModalRoot)" in APP_JS


def test_app_js_notifies_biometrics_view_of_open_and_close() -> None:
    """The generic modal shell lives in app.js; subview logic lives in
    biometrics-view.js. They must stay coupled via onModalOpen/onModalClose,
    not by biometrics-view.js polling modal visibility itself."""
    assert "window.OrionBiometricsView.onModalOpen()" in APP_JS
    assert "window.OrionBiometricsView.onModalClose()" in APP_JS


# --------------------------------------------------------------------------
# Modal subview switching (biometrics-view.js)
# --------------------------------------------------------------------------


def test_biometrics_view_js_declares_subview_switch_and_lazy_load_state() -> None:
    assert "function showModalSubview(" in BIOMETRICS_VIEW_JS
    for name in ("athena", "circe", "gpu", "cabinet"):
        assert f'"{name}"' in BIOMETRICS_VIEW_JS
    assert "loaded.athena" in BIOMETRICS_VIEW_JS
    assert "loaded.circe" in BIOMETRICS_VIEW_JS
    assert "loaded.gpu" in BIOMETRICS_VIEW_JS


def test_biometrics_view_js_has_wireonce_idempotency_guard() -> None:
    assert "var wired = false;" in BIOMETRICS_VIEW_JS
    assert "if (wired) return;" in BIOMETRICS_VIEW_JS
    assert "wired = true;" in BIOMETRICS_VIEW_JS


def test_biometrics_view_js_exposes_activate_deactivate_lifecycle() -> None:
    assert "window.OrionBiometricsView" in BIOMETRICS_VIEW_JS
    assert "activate," in BIOMETRICS_VIEW_JS
    assert "deactivate," in BIOMETRICS_VIEW_JS
    assert "onModalOpen," in BIOMETRICS_VIEW_JS
    assert "onModalClose," in BIOMETRICS_VIEW_JS


def test_biometrics_view_js_loads_node_detail_fetches_concurrently() -> None:
    """Regression test (code review finding): snapshot/history/induction are
    three independent reads. All three fetches must start before the first
    `await`, so total load time is the slowest single leg rather than their
    sum -- sequential awaiting on a slow node compounds three network round
    trips (each up to BIOMETRICS_NODE_CLIENT_TIMEOUT_SEC) into one wait."""
    fn_start = BIOMETRICS_VIEW_JS.index("async function loadNodeDetail(")
    fn_end = BIOMETRICS_VIEW_JS.index("\n  function cap(", fn_start)
    body = BIOMETRICS_VIEW_JS[fn_start:fn_end]
    first_await = body.index("await ")
    setup = body[:first_await]
    assert "snapshotPromise = fetchJson(" in setup
    assert "historiesPromise = fetchJson(" in setup
    assert "inductionPromise = fetchJson(" in setup


def test_biometrics_view_js_calls_only_the_preview_api_prefix() -> None:
    assert "/api/biometrics/preview/snapshot" in BIOMETRICS_VIEW_JS
    assert "/api/biometrics/preview/history" in BIOMETRICS_VIEW_JS
    assert "/api/biometrics/preview/induction" in BIOMETRICS_VIEW_JS
    assert "/api/biometrics/preview/gpu" in BIOMETRICS_VIEW_JS


def test_biometrics_view_js_only_polls_gpu_subview_while_open() -> None:
    assert "gpuPollTimer" in BIOMETRICS_VIEW_JS
    assert "clearInterval(gpuPollTimer)" in BIOMETRICS_VIEW_JS


# --------------------------------------------------------------------------
# Readability: status color, trend arrows, legend, channel coverage
# --------------------------------------------------------------------------
#
# Follow-up to the initial ship: cards had no good/bad/changing/important
# encoding (every tile looked the same neutral gray) and only 4 of ~14
# available channels were charted, plus the GPU trend sparkline read from a
# 5-sample buffer and looked nearly flat. Fixed by reusing this file's own
# established status-tone convention (cabinet-sensors.js's badge()) rather
# than inventing a second palette.


def test_biometrics_view_js_defines_a_reserved_status_tone_scale() -> None:
    assert "var TONE = {" in BIOMETRICS_VIEW_JS
    assert "good:" in BIOMETRICS_VIEW_JS
    assert "warning:" in BIOMETRICS_VIEW_JS
    assert "critical:" in BIOMETRICS_VIEW_JS
    assert "neutral:" in BIOMETRICS_VIEW_JS
    # emerald/amber/red on a dark surface -- this repo's own convention
    # (cabinet-sensors.js badge()), not a separately invented palette.
    assert "emerald" in BIOMETRICS_VIEW_JS
    assert "amber" in BIOMETRICS_VIEW_JS
    assert "text-red-200" in BIOMETRICS_VIEW_JS


def test_biometrics_view_js_status_never_ships_as_color_alone() -> None:
    """Every status tone in TONE carries an icon + label, per the dataviz
    status-color rule -- color never carries meaning unaided."""
    tone_block_start = BIOMETRICS_VIEW_JS.index("var TONE = {")
    tone_block_end = BIOMETRICS_VIEW_JS.index("};", tone_block_start)
    tone_block = BIOMETRICS_VIEW_JS[tone_block_start:tone_block_end]
    assert tone_block.count("icon:") >= 4
    assert tone_block.count("label:") >= 4


def test_biometrics_view_js_computes_tone_from_value_not_hardcoded() -> None:
    assert "function toneForPressure(" in BIOMETRICS_VIEW_JS
    assert "invert" in BIOMETRICS_VIEW_JS


def test_biometrics_view_js_shows_trend_arrows_from_induction_data() -> None:
    assert "function trendArrow(" in BIOMETRICS_VIEW_JS
    assert '"↑"' in BIOMETRICS_VIEW_JS or "'↑'" in BIOMETRICS_VIEW_JS
    assert '"↓"' in BIOMETRICS_VIEW_JS or "'↓'" in BIOMETRICS_VIEW_JS


def test_biometrics_view_js_sorts_snapshot_tiles_worst_first() -> None:
    assert "TONE_RANK" in BIOMETRICS_VIEW_JS
    assert "rows.sort(" in BIOMETRICS_VIEW_JS


def test_template_declares_a_color_legend_in_every_biometrics_panel() -> None:
    """A legend is mandatory once color carries meaning (dataviz rule) --
    one per surface: card preview, Athena, Circe, GPU."""
    assert INDEX_HTML.count("text-emerald-200\">● good</span>") >= 4


def test_biometrics_view_js_charts_more_than_the_original_four_channels() -> None:
    """Regression test: the original ship only charted strain/gpu_util/
    thermal/power. Channels absent from that original set must now be
    covered too, or this is the same sparse-trends complaint again."""
    assert "var ALL_CHANNELS = COMPOSITE_CHANNELS.concat(PRESSURE_CHANNELS);" in BIOMETRICS_VIEW_JS
    for channel in ("homeostasis", "stability", "cpu", "gpu_mem", "mem", "disk", "net", "fan"):
        assert f'"{channel}"' in BIOMETRICS_VIEW_JS, channel


def test_biometrics_view_js_requests_a_denser_gpu_trend_than_the_original_five() -> None:
    """Regression test: the original GPU sparkline read the endpoint's
    default limit=5, which looked nearly flat/empty."""
    assert "&limit=40" in BIOMETRICS_VIEW_JS


def test_gpu_endpoint_default_limit_is_not_the_original_sparse_five() -> None:
    source = (HUB_ROOT / "scripts" / "biometrics_preview_routes.py").read_text(encoding="utf-8")
    assert "limit: int = Query(5," not in source
    assert "limit: int = Query(40," in source


def test_unreachable_node_renders_critical_not_neutral() -> None:
    """Regression test (code review finding): an unreachable node's status
    tile originally rendered "neutral" (gray, identical to "no data yet"),
    which was the ONLY tile left visible once every value tile got filtered
    out for lack of summary data -- exactly invisible to an operator
    scanning for red tiles. Must be a distinguishable critical tone."""
    assert "function toneForNodeStatus(" in BIOMETRICS_VIEW_JS
    fn_start = BIOMETRICS_VIEW_JS.index("function toneForNodeStatus(")
    fn_end = BIOMETRICS_VIEW_JS.index("\n  }", fn_start)
    body = BIOMETRICS_VIEW_JS[fn_start:fn_end]
    assert 'return "critical"' in body
    assert "toneForNodeStatus(payload)" in BIOMETRICS_VIEW_JS
    assert "toneForNodeStatus(snapshot)" in BIOMETRICS_VIEW_JS
    # the old bug's exact literal must be gone, not just supplemented
    assert 'tone: payload.ok ? toneForPressure(strain) : "neutral"' not in BIOMETRICS_VIEW_JS
    assert 'tone: snapshot.ok ? "good" : "neutral"' not in BIOMETRICS_VIEW_JS


def test_history_uses_one_multi_channel_request_not_one_per_channel() -> None:
    """Regression test (code review finding): loadNodeDetail() originally
    fired one /history request PER channel (up to 14 concurrent, unpooled
    asyncpg connections per modal open -- this repo has live incident
    history with connection exhaustion, PR #2010). Must be a single request
    covering every channel."""
    assert "/api/biometrics/preview/history_multi" in BIOMETRICS_VIEW_JS
    fn_start = BIOMETRICS_VIEW_JS.index("async function loadNodeDetail(")
    fn_end = BIOMETRICS_VIEW_JS.index("\n  function cap(", fn_start)
    body = BIOMETRICS_VIEW_JS[fn_start:fn_end]
    # exactly one call site building the history request in this function
    assert body.count("/api/biometrics/preview/history") == 1
    assert "channels=" in body
