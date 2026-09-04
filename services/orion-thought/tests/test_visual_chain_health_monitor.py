"""Reverie visual-chain staleness health monitor. `app.*` imports are done inside each test
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

    return ThoughtSettings(
        NOTIFY_BASE_URL="http://notify.test:7140",
        NOTIFY_API_TOKEN="",
        ORION_VISUAL_CHAIN_STALENESS_THRESHOLD_MIN=45,
    )


def test_check_healthy_when_age_below_threshold():
    from app.visual_chain_health_monitor import _check

    result = _check(age_min=10.0, threshold_min=45.0)
    assert result.healthy is True
    assert result.message == ""


def test_check_unhealthy_when_age_past_threshold():
    from app.visual_chain_health_monitor import _check

    result = _check(age_min=90.0, threshold_min=45.0)
    assert result.healthy is False
    assert result.severity == "critical"
    assert "90.0" in result.message


def test_check_none_age_is_not_flagged_stale():
    """Table empty (never observed in practice) is NOT the same as stale --
    matches orion-field-digester/app/health_monitor.py's own precedent."""
    from app.visual_chain_health_monitor import _check

    result = _check(age_min=None, threshold_min=45.0)
    assert result.healthy is True


def test_monitor_does_not_alert_on_healthy_first_observation():
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get", return_value=_no_pending_items()
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        monitor = VisualChainHealthMonitor(settings_obj=_settings())
        monitor.record_check(age_min=10.0)
        client_cls.return_value.attention_request.assert_not_called()


def test_monitor_alerts_only_on_stale_transition_not_every_tick():
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get", return_value=_no_pending_items()
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        _client_mock(client_cls)
        monitor = VisualChainHealthMonitor(settings_obj=_settings())

        monitor.record_check(age_min=10.0)  # healthy baseline
        monitor.record_check(age_min=90.0)  # transition -> alert
        monitor.record_check(age_min=95.0)  # still stale -> no additional alert

        assert client_cls.return_value.attention_request.call_count == 1
        alert_kwargs = client_cls.return_value.attention_request.call_args.kwargs
        assert alert_kwargs["severity"] == "critical"
        assert alert_kwargs["context"]["reason"] == "visual_chain_stale"


def test_monitor_sends_recovery_note_on_healthy_transition():
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get", return_value=_no_pending_items()
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        _client_mock(client_cls)
        monitor = VisualChainHealthMonitor(settings_obj=_settings())

        monitor.record_check(age_min=10.0)  # healthy baseline
        monitor.record_check(age_min=90.0)  # stale -> 1 alert
        monitor.record_check(age_min=5.0)  # recovered

        assert client_cls.return_value.attention_request.call_count == 2
        recovery_kwargs = client_cls.return_value.attention_request.call_args_list[-1].kwargs
        assert recovery_kwargs["severity"] == "info"


def test_monitor_retries_until_notify_confirms_delivery():
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get", return_value=_no_pending_items()
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        monitor = VisualChainHealthMonitor(settings_obj=_settings())
        monitor.record_check(age_min=10.0)  # healthy baseline

        _client_mock(client_cls, ok=False)
        monitor.record_check(age_min=90.0)  # stale, publish fails -> not committed
        monitor.record_check(age_min=91.0)  # retried again

        assert client_cls.return_value.attention_request.call_count == 2

        _client_mock(client_cls, ok=True)
        monitor.record_check(age_min=92.0)  # publish finally succeeds -> committed

        assert client_cls.return_value.attention_request.call_count == 3

        # Already committed unhealthy -- no further spurious alert.
        monitor.record_check(age_min=93.0)
        assert client_cls.return_value.attention_request.call_count == 3


def test_monitor_suppresses_first_observation_when_notify_has_open_item():
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get"
    ) as mock_get:
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {"source_service": "orion-thought", "reason": "visual_chain_stale"},
            ],
        )
        monitor = VisualChainHealthMonitor(settings_obj=_settings())
        monitor.record_check(age_min=90.0)  # first observation, already unhealthy

        client_cls.return_value.attention_request.assert_not_called()


def test_pending_lookup_failure_is_fail_open_not_fatal():
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get", side_effect=RuntimeError("network down")
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        _client_mock(client_cls)
        monitor = VisualChainHealthMonitor(settings_obj=_settings())
        monitor.record_check(age_min=90.0)  # must not raise; falls through to _publish

        client_cls.return_value.attention_request.assert_called_once()


def test_check_visual_chain_staleness_module_singleton_never_raises():
    from app.visual_chain_health_monitor import (
        check_visual_chain_staleness,
        reset_monitor_for_tests,
    )

    reset_monitor_for_tests()
    with patch(
        "app.visual_chain_health_monitor.NotifyClient", side_effect=RuntimeError("boom")
    ):
        check_visual_chain_staleness(90.0)  # must not raise
    reset_monitor_for_tests()
