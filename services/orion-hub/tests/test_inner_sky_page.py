"""Inner Sky page wiring: the drawer loads it, its assets exist, and the retired /spark/ui page is gone."""
from __future__ import annotations

import re
from pathlib import Path

HUB = Path(__file__).resolve().parents[1]
REPO = HUB.parents[1]
INDEX = (HUB / "templates" / "index.html").read_text()
PAGE = (HUB / "static" / "inner-sky.html").read_text()


def test_drawer_embeds_the_inner_sky_not_the_retired_spark_page():
    assert 'src="/static/inner-sky.html?embed=1&v={{HUB_UI_ASSET_VERSION}}"' in INDEX
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


def test_embed_mode_hides_the_controls_and_links_to_the_full_page():
    assert "embed=1" in PAGE and "body.embed #bar" in PAGE
    assert 'href="/static/inner-sky.html" target="_top"' in PAGE


def test_vector_host_no_longer_serves_the_old_page_or_its_socket():
    main = (REPO / "services" / "orion-vector-host" / "app" / "main.py").read_text()
    assert '@spark_router.get("/ui")' not in main
    assert "@spark_router.websocket" not in main and '@spark_router.post("/api/test-pulse")' not in main
    assert '"/api/tissue/state"' in main
    assert not (REPO / "services" / "orion-vector-host" / "app" / "static").exists()
