"""Inner Sky page wiring: the drawer loads it, its assets exist, and the retired /spark/ui page is gone."""
from __future__ import annotations

import re
from pathlib import Path

HUB = Path(__file__).resolve().parents[1]
REPO = HUB.parents[1]
INDEX = (HUB / "templates" / "index.html").read_text()
PAGE = (HUB / "static" / "inner-sky.html").read_text()


def test_drawer_embeds_the_inner_sky_not_the_retired_spark_page():
    assert 'src="/static/inner-sky.html?v={{HUB_UI_ASSET_VERSION}}"' in INDEX
    assert "/spark/ui" not in INDEX
    assert 'id="stateVisualizerContainer"' in INDEX  # biometrics toggle still swaps this container


def test_page_links_assets_that_exist_and_the_core_before_the_module():
    srcs = re.findall(r'<script[^>]+src="/static/([^"?]+)', PAGE)
    assert srcs == ["js/inner-sky-core.js", "js/inner-sky.js"]
    for rel in srcs:
        assert (HUB / "static" / rel).is_file(), rel


def test_page_reads_only_the_inner_sky_routes():
    js = (HUB / "static" / "js" / "inner-sky.js").read_text()
    assert set(re.findall(r"fetch\(`?'?(/api/[a-z-]+/[a-z-]+)", js)) == {"/api/inner-sky/events", "/api/inner-sky/state"}


def test_compact_mode_follows_width_and_widen_asks_the_hub_drawer():
    assert "drawerState !== 'wide'" in PAGE and "innerWidth < 1100" in PAGE and "body.embed #bar" in PAGE
    assert "ev.source !== window.parent" in PAGE
    # the overlay turns pointer events off for .hud; the widen link is itself .hud, so it must turn them back on
    assert re.search(r"#openFull \{[^}]*pointer-events: auto", PAGE)
    assert "type: 'orion-inner-sky-widen'" in PAGE
    drawer_js = (HUB / "static" / "js" / "hub-ekg-drawer.js").read_text()
    assert "var WIDEN_MESSAGE = 'orion-inner-sky-widen';" in drawer_js


def test_drawer_has_a_widen_button_and_wide_hides_chat():
    assert 'id="ekgDrawerWiden"' in INDEX
    css = (HUB / "static" / "css" / "style.css").read_text()
    md = css[css.index("@media (min-width: 768px)"):]
    md = md[: md.index("\n}\n")]
    assert ".hub-ekg-drawer.is-wide ~ .hub-chat-card { display: none; }" in md  # never on phones


def test_vector_host_no_longer_serves_the_old_page_or_its_socket():
    main = (REPO / "services" / "orion-vector-host" / "app" / "main.py").read_text()
    assert '@spark_router.get("/ui")' not in main
    assert "@spark_router.websocket" not in main and '@spark_router.post("/api/test-pulse")' not in main
    assert '"/api/tissue/state"' in main
    assert not (REPO / "services" / "orion-vector-host" / "app" / "static").exists()
