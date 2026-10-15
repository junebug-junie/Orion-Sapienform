"""Browser smoke for "Walkway (live)" in the Vision panel (2026-10-15).

The REAL templates/index.html and the REAL static/js (app.js + vision-live-frame.js)
in Chromium, with every Hub API intercepted (no Hub process). The dropdown logic
lives inside app.js's boot closure, so this loads the whole page rather than a cut
panel. Covers the changed interaction, not just page load: choosing "Walkway (live)"
polls the walkway stream, shows the frame with a LIVE badge, and flips to a dimmed
STALE badge (picture kept) once the backend reports the stream stale.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

HUB = Path(__file__).resolve().parents[1]


def _jpeg() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (20, 120, 40)).save(buf, format="JPEG")
    return buf.getvalue()


def test_walkway_live_view_browser_smoke():
    pytest.importorskip("playwright.sync_api")
    pytest.importorskip("PIL")
    from playwright.sync_api import sync_playwright

    html = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{HUB_CFG}}", "{}")
    html = re.sub(r"\{\{[A-Z0-9_]+\}\}", "", html)
    jpeg = _jpeg()
    state = {"stale": False, "frame": 1}
    meta_streams: list[str] = []
    image_streams: list[str] = []

    def handle(route):
        url = route.request.url
        path = url.split("http://hub.test", 1)[1].split("?", 1)[0]
        if path == "/":
            return route.fulfill(body=html, content_type="text/html")
        if path.startswith("/static/"):
            f = HUB / path.lstrip("/")
            if f.is_file():
                ctype = "application/javascript" if f.suffix == ".js" else "text/css" if f.suffix == ".css" else "application/octet-stream"
                return route.fulfill(body=f.read_bytes(), content_type=ctype)
            return route.fulfill(status=404, body="")
        if path == "/api/vision/frames/latest":
            sid = re.search(r"stream_id=([^&]+)", url).group(1)
            meta_streams.append(sid)
            state["frame"] += 0 if state["stale"] else 1
            body = {
                "available": True, "stream_id": sid, "frame_id": f"walkway_walkway_{state['frame']}_ab.jpg",
                "age_sec": 42.0 if state["stale"] else 0.3, "stale": state["stale"], "stale_after_sec": 10.0,
                "source": "local",
            }
            return route.fulfill(body=json.dumps(body), content_type="application/json")
        if path == "/api/vision/frames/latest/image":
            image_streams.append(re.search(r"stream_id=([^&]+)", url).group(1))
            return route.fulfill(body=jpeg, content_type="image/jpeg")
        return route.fulfill(body="{}", content_type="application/json")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.route("http://hub.test/**", handle)
        page.route(re.compile(r"^(?!http://hub\.test).*"), lambda r: r.fulfill(status=204, body=""))
        page.goto("http://hub.test/")

        page.wait_for_function("() => !!window.OrionVisionLiveFrame")
        page.select_option("#visionSource", "walkway-live")

        badge = page.locator("#visionDockedContainer [data-live-frame-badge]")
        page.wait_for_function(
            "() => (document.querySelector('#visionDockedContainer [data-live-frame-badge]')||{}).textContent?.startsWith('LIVE')",
            timeout=10000,
        )
        img = page.locator('#visionDockedContainer [data-live-frame="walkway"] img')
        assert "/api/vision/frames/latest/image?stream_id=walkway" in (img.get_attribute("src") or "")
        assert "walkway" in meta_streams and "walkway" in image_streams
        assert "carbon" not in meta_streams

        state["stale"] = True
        page.wait_for_function(
            "() => (document.querySelector('#visionDockedContainer [data-live-frame-badge]')||{}).textContent?.startsWith('STALE')",
            timeout=10000,
        )
        assert "42s ago" in badge.inner_text()
        assert "opacity-40" in (img.get_attribute("class") or "")
        assert img.get_attribute("src"), "the last picture is kept, just marked stale"

        # Switching away stops the walkway poll.
        page.select_option("#visionSource", "none")
        before = len(meta_streams)
        page.wait_for_timeout(3500)
        assert len(meta_streams) == before
        browser.close()
