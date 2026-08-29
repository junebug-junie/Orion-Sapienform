"""Reverie metacog-timeout health monitor. `app.*` imports are done inside each test
function (not module scope) because this service's conftest purges `app`/`app.*` from
`sys.modules` before every test -- a module-scope import would hold a stale reference that
`unittest.mock.patch("app....")` can't see.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _client_mock(client_cls, *, ok: bool = True):
    client_cls.return_value.attention_request.return_value = MagicMock(ok=ok)
    return client_cls.return_value.attention_request


def _no_pending_items():
    return MagicMock(status_code=200, json=lambda: [])


def _settings():
    from app.settings import ThoughtSettings

    return ThoughtSettings(NOTIFY_BASE_URL="http://notify.test:7140", NOTIFY_API_TOKEN="")


def test_monitor_does_not_alert_on_healthy_first_observation():
    with patch("app.reverie_health_monitor.NotifyClient") as client_cls, patch(
        "app.reverie_health_monitor.requests.get", return_value=_no_pending_items()
    ):
        from app.reverie_health_monitor import ReverieMetacogHealthMonitor

        monitor = ReverieMetacogHealthMonitor(settings_obj=_settings())
        monitor.record_tick(False)
        client_cls.return_value.attention_request.assert_not_called()


def test_monitor_alerts_only_on_timeout_transition_not_every_tick():
    with patch("app.reverie_health_monitor.NotifyClient") as client_cls, patch(
        "app.reverie_health_monitor.requests.get", return_value=_no_pending_items()
    ):
        from app.reverie_health_monitor import ReverieMetacogHealthMonitor

        _client_mock(client_cls)
        monitor = ReverieMetacogHealthMonitor(settings_obj=_settings())

        monitor.record_tick(False)  # healthy baseline
        monitor.record_tick(True)  # transition -> alert
        monitor.record_tick(True)  # still timing out -> no additional alert

        assert client_cls.return_value.attention_request.call_count == 1
        alert_kwargs = client_cls.return_value.attention_request.call_args.kwargs
        assert alert_kwargs["severity"] == "error"
        assert alert_kwargs["context"]["reason"] == "reverie_metacog_timeout"


def test_monitor_sends_recovery_note_on_healthy_transition():
    with patch("app.reverie_health_monitor.NotifyClient") as client_cls, patch(
        "app.reverie_health_monitor.requests.get", return_value=_no_pending_items()
    ):
        from app.reverie_health_monitor import ReverieMetacogHealthMonitor

        _client_mock(client_cls)
        monitor = ReverieMetacogHealthMonitor(settings_obj=_settings())

        monitor.record_tick(False)  # healthy baseline
        monitor.record_tick(True)  # timeout -> 1 alert
        monitor.record_tick(False)  # recovered

        assert client_cls.return_value.attention_request.call_count == 2
        recovery_kwargs = client_cls.return_value.attention_request.call_args_list[-1].kwargs
        assert recovery_kwargs["severity"] == "info"


def test_monitor_retries_until_notify_confirms_delivery():
    with patch("app.reverie_health_monitor.NotifyClient") as client_cls, patch(
        "app.reverie_health_monitor.requests.get", return_value=_no_pending_items()
    ):
        from app.reverie_health_monitor import ReverieMetacogHealthMonitor

        monitor = ReverieMetacogHealthMonitor(settings_obj=_settings())
        monitor.record_tick(False)  # healthy baseline

        _client_mock(client_cls, ok=False)
        monitor.record_tick(True)  # timeout, publish fails -> not committed
        monitor.record_tick(True)  # retried again

        assert client_cls.return_value.attention_request.call_count == 2

        _client_mock(client_cls, ok=True)
        monitor.record_tick(True)  # publish finally succeeds -> committed

        assert client_cls.return_value.attention_request.call_count == 3

        # Already committed unhealthy -- no further spurious alert.
        monitor.record_tick(True)
        assert client_cls.return_value.attention_request.call_count == 3


def test_monitor_suppresses_first_observation_when_notify_has_open_item():
    with patch("app.reverie_health_monitor.NotifyClient") as client_cls, patch(
        "app.reverie_health_monitor.requests.get"
    ) as mock_get:
        from app.reverie_health_monitor import ReverieMetacogHealthMonitor

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {"source_service": "orion-thought", "reason": "reverie_metacog_timeout"},
            ],
        )
        monitor = ReverieMetacogHealthMonitor(settings_obj=_settings())
        monitor.record_tick(True)  # first observation, already unhealthy

        client_cls.return_value.attention_request.assert_not_called()


def test_pending_lookup_failure_is_fail_open_not_fatal():
    with patch("app.reverie_health_monitor.NotifyClient") as client_cls, patch(
        "app.reverie_health_monitor.requests.get", side_effect=RuntimeError("network down")
    ):
        from app.reverie_health_monitor import ReverieMetacogHealthMonitor

        _client_mock(client_cls)
        monitor = ReverieMetacogHealthMonitor(settings_obj=_settings())
        monitor.record_tick(True)  # must not raise; falls through to _publish

        client_cls.return_value.attention_request.assert_called_once()


def test_check_reverie_metacog_timeout_module_singleton_never_raises():
    from app.reverie_health_monitor import (
        check_reverie_metacog_timeout,
        reset_monitor_for_tests,
    )

    reset_monitor_for_tests()
    with patch(
        "app.reverie_health_monitor.NotifyClient", side_effect=RuntimeError("boom")
    ):
        check_reverie_metacog_timeout(True)  # must not raise
    reset_monitor_for_tests()
