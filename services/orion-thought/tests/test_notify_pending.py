"""Shared open-page rule for orion-thought's health monitors (live 2026-11-03:
a pending recovery note shares its page's reason key and was read as open)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _settings():
    from app.settings import ThoughtSettings

    return ThoughtSettings(NOTIFY_BASE_URL="http://notify.test:7140", NOTIFY_API_TOKEN="")


def _item(reason, message, source="orion-thought"):
    return {"source_service": source, "reason": reason, "message": message}


def _resp(items):
    return MagicMock(status_code=200, json=lambda: items)


def test_newest_item_decides_open_state():
    from app.notify_pending import open_alert_reasons

    items = [  # newest first, as sql-writer returns them
        _item("a", "[Orion reverie] recovered: a"),
        _item("a", "[Orion reverie] a is broken"),
        _item("b", "[Orion reverie] b is broken"),
        _item("b", "[Orion reverie] recovered: b"),
        _item("c", "[Orion reverie] c is broken", source="other-service"),
    ]
    with patch("app.notify_pending.requests.get", return_value=_resp(items)):
        assert open_alert_reasons(_settings(), log_label="t") == ["b"]


def test_lookup_failure_fails_open():
    from app.notify_pending import open_alert_reasons

    with patch("app.notify_pending.requests.get", side_effect=RuntimeError("down")):
        assert open_alert_reasons(_settings(), log_label="t") == []


def test_reverie_monitor_ignores_answered_page():
    from app.reverie_health_monitor import ReverieMetacogHealthMonitor

    items = [
        _item("reverie_metacog_timeout", "[Orion reverie] recovered: reverie_metacog_timeout"),
        _item("reverie_metacog_timeout", "[Orion reverie] metacog timing out"),
    ]
    with patch("app.reverie_health_monitor.NotifyClient"), patch(
        "app.notify_pending.requests.get", return_value=_resp(items)
    ):
        assert ReverieMetacogHealthMonitor(_settings())._has_open_alert() is False


def test_visual_monitor_ignores_answered_page():
    from app.visual_chain_health_monitor import VisualChainHealthMonitor

    items = [
        _item("visual_chain_stale", "[Orion reverie] recovered: visual_chain_stale"),
        _item("visual_chain_stale", "[Orion reverie] visual chain stale"),
    ]
    with patch("app.visual_chain_health_monitor.NotifyClient"), patch(
        "app.notify_pending.requests.get", return_value=_resp(items)
    ):
        monitor = VisualChainHealthMonitor(_settings())
        assert monitor._has_open_alert("visual_chain_stale") is False
