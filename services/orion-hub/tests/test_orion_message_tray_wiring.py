"""Static wiring for Orion's messages in the notification tray.

The behaviour itself is covered by static/js/orion-message-tray.test.js
(node:test) and evals/orion_message_tray_browser.py (real Chromium). These
checks catch the cheap breakages: the module not loaded, loaded after app.js,
the boot fetch dropped, or the Hub proxy not forwarding the tag filter.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

HUB = Path(__file__).resolve().parents[1]
INDEX = HUB / "templates" / "index.html"
APP_JS = HUB / "static" / "js" / "app.js"


def test_tray_module_loads_before_app_js() -> None:
    html = INDEX.read_text(encoding="utf-8")
    module = html.index('src="/static/js/orion-message-tray.js?v={{HUB_UI_ASSET_VERSION}}"')
    app = html.index('src="/static/js/app.js?v={{HUB_UI_ASSET_VERSION}}"')
    assert module < app
    assert (HUB / "static" / "js" / "orion-message-tray.js").is_file()


def test_badge_lives_in_the_existing_panel_header() -> None:
    html = INDEX.read_text(encoding="utf-8")
    panel = html.index('id="hubNotificationsPanel"')
    badge = html.index('id="orionMessagesBadge"')
    tray_list = html.index('id="notificationList"')
    assert panel < badge < tray_list


def test_boot_loads_open_orion_messages_from_the_durable_store() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    assert "await loadOrionMessages();" in js
    assert "/api/chat/messages?status=open&tag=endogenous_outreach" in js
    assert "replyToOrionMessage(m)" in js
    assert "dismissOrionMessage(m)" in js


def test_hub_proxy_forwards_status_and_tag(monkeypatch) -> None:
    from scripts import api_routes

    resp = MagicMock()
    resp.json.return_value = []
    resp.raise_for_status.return_value = None
    get = MagicMock(return_value=resp)
    monkeypatch.setattr(api_routes.requests, "get", get)
    monkeypatch.setattr(api_routes.settings, "NOTIFY_BASE_URL", "http://notify")

    assert api_routes.api_chat_messages(limit=100, status="open", tag="endogenous_outreach") == []
    params = get.call_args.kwargs["params"]
    assert params["status"] == "open"
    assert params["tag"] == "endogenous_outreach"
    assert get.call_args.args[0] == "http://notify/chat/messages"


def test_hub_proxy_failure_is_a_502_not_an_empty_list(monkeypatch) -> None:
    """[] would read as "nothing is open" and hide live Orion messages."""
    import pytest
    from fastapi import HTTPException

    from scripts import api_routes

    monkeypatch.setattr(api_routes.requests, "get", MagicMock(side_effect=RuntimeError("down")))
    monkeypatch.setattr(api_routes.settings, "NOTIFY_BASE_URL", "http://notify")
    with pytest.raises(HTTPException) as exc:
        api_routes.api_chat_messages(limit=100, status="open", tag="endogenous_outreach")
    assert exc.value.status_code == 502
