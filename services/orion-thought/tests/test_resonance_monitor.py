"""Phase H+ — resonance health monitor. `app.*` imports are done inside each
test function (not module scope) because this service's conftest purges
`app`/`app.*` from `sys.modules` before every test — a module-scope import
would hold a stale reference that `unittest.mock.patch("app....")` can't see.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _samples(*violation_counts, min_gap_sec=98.0, refractory_sec=900.0, occurrences=200):
    """Newest-first list of resonance-alert-row dicts, matching
    load_recent_resonance_alerts' return shape."""
    return [
        {
            "theme_key": "loop:t1",
            "violation_count": vc,
            "refractory_sec": refractory_sec,
            "min_gap_sec": min_gap_sec,
            "occurrences": occurrences,
        }
        for vc in violation_counts
    ]


def _client_mock(client_cls, *, ok: bool = True):
    client_cls.return_value.attention_request.return_value = MagicMock(ok=ok)
    return client_cls.return_value.attention_request


def _alert(theme_key="loop:t1"):
    return MagicMock(theme_key=theme_key)


def test_is_worsening_healthy_with_fewer_than_2_samples():
    from app.resonance_monitor import _is_worsening

    with patch("app.resonance_monitor.load_recent_resonance_alerts", return_value=_samples(3)):
        check = _is_worsening("loop:t1")
    assert check.healthy is True


def test_is_worsening_unhealthy_when_violation_count_increases():
    from app.resonance_monitor import _is_worsening

    # newest first: 5 (latest) vs 3 (previous) -> increasing -> worsening
    with patch(
        "app.resonance_monitor.load_recent_resonance_alerts", return_value=_samples(5, 3)
    ):
        check = _is_worsening("loop:t1")
    assert check.healthy is False
    assert "loop:t1" in check.message
    assert "3 -> 5" in check.message


def test_is_worsening_healthy_when_violation_count_decreasing_or_flat():
    from app.resonance_monitor import _is_worsening

    with patch(
        "app.resonance_monitor.load_recent_resonance_alerts", return_value=_samples(2, 6)
    ):
        check = _is_worsening("loop:t1")
    assert check.healthy is True

    with patch(
        "app.resonance_monitor.load_recent_resonance_alerts", return_value=_samples(4, 4)
    ):
        check = _is_worsening("loop:t1")
    assert check.healthy is True


def test_monitor_does_not_alert_on_healthy_first_observation():
    with patch("app.resonance_monitor.NotifyClient") as client_cls, patch(
        "app.resonance_monitor.load_recent_resonance_alerts", return_value=_samples(3)
    ):
        from app.resonance_monitor import ResonanceHealthMonitor

        monitor = ResonanceHealthMonitor(settings_obj=_settings())
        monitor.check(_alert())
        client_cls.return_value.attention_request.assert_not_called()


def test_monitor_alerts_only_on_worsening_transition_not_every_tick():
    with patch("app.resonance_monitor.NotifyClient") as client_cls, patch(
        "app.resonance_monitor.load_recent_resonance_alerts"
    ) as mock_load:
        from app.resonance_monitor import ResonanceHealthMonitor

        _client_mock(client_cls)
        monitor = ResonanceHealthMonitor(settings_obj=_settings())

        mock_load.return_value = _samples(3)  # healthy baseline
        monitor.check(_alert())

        mock_load.return_value = _samples(6, 3)  # worsening transition -> alert
        monitor.check(_alert())
        monitor.check(_alert())  # still worsening -> no additional alert

        assert client_cls.return_value.attention_request.call_count == 1


def test_monitor_sends_recovery_note_on_healthy_transition():
    with patch("app.resonance_monitor.NotifyClient") as client_cls, patch(
        "app.resonance_monitor.load_recent_resonance_alerts"
    ) as mock_load:
        from app.resonance_monitor import ResonanceHealthMonitor

        _client_mock(client_cls)
        monitor = ResonanceHealthMonitor(settings_obj=_settings())

        mock_load.return_value = _samples(3)
        monitor.check(_alert())  # healthy baseline

        mock_load.return_value = _samples(6, 3)
        monitor.check(_alert())  # worsening -> 1 alert

        mock_load.return_value = _samples(3, 6)  # decreasing again -> recovery
        monitor.check(_alert())

        assert client_cls.return_value.attention_request.call_count == 2
        recovery_kwargs = client_cls.return_value.attention_request.call_args_list[-1].kwargs
        assert recovery_kwargs["severity"] == "info"


def test_monitor_retries_until_notify_confirms_delivery():
    with patch("app.resonance_monitor.NotifyClient") as client_cls, patch(
        "app.resonance_monitor.load_recent_resonance_alerts"
    ) as mock_load:
        from app.resonance_monitor import ResonanceHealthMonitor

        monitor = ResonanceHealthMonitor(settings_obj=_settings())

        mock_load.return_value = _samples(3)
        monitor.check(_alert())  # healthy baseline

        mock_load.return_value = _samples(6, 3)
        _client_mock(client_cls, ok=False)
        monitor.check(_alert())  # worsening detected, publish fails -> not committed
        monitor.check(_alert())  # retried again

        assert client_cls.return_value.attention_request.call_count == 2

        _client_mock(client_cls, ok=True)
        monitor.check(_alert())  # publish finally succeeds -> committed

        assert client_cls.return_value.attention_request.call_count == 3

        monitor.check(_alert())  # committed as unhealthy -> no further calls
        assert client_cls.return_value.attention_request.call_count == 3


def test_monitor_suppresses_first_observation_when_notify_has_open_item():
    with patch("app.resonance_monitor.NotifyClient") as client_cls, patch(
        "app.resonance_monitor.requests.get"
    ) as mock_get, patch(
        "app.resonance_monitor.load_recent_resonance_alerts", return_value=_samples(6, 3)
    ):
        from app.resonance_monitor import ResonanceHealthMonitor

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {
                    "source_service": "orion-thought",
                    "reason": "reverie_resonance_worsening:loop:t1",
                }
            ],
        )
        monitor = ResonanceHealthMonitor(settings_obj=_settings())
        monitor.check(_alert())

        client_cls.return_value.attention_request.assert_not_called()


def test_monitor_recovers_previously_tracked_theme_even_when_alert_is_none():
    """A theme that stops being the current tick's most-severe theme should
    still get checked (and recovered) rather than staying silently open."""
    with patch("app.resonance_monitor.NotifyClient") as client_cls, patch(
        "app.resonance_monitor.load_recent_resonance_alerts"
    ) as mock_load:
        from app.resonance_monitor import ResonanceHealthMonitor

        _client_mock(client_cls)
        monitor = ResonanceHealthMonitor(settings_obj=_settings())

        mock_load.return_value = _samples(3)
        monitor.check(_alert("loop:t1"))  # healthy baseline for t1

        mock_load.return_value = _samples(6, 3)
        monitor.check(_alert("loop:t1"))  # t1 worsens -> alert, now tracked

        mock_load.return_value = _samples(3, 6)  # t1 has calmed
        monitor.check(None)  # this tick's most-severe theme is a different one (or none)

        assert client_cls.return_value.attention_request.call_count == 2
        recovery_kwargs = client_cls.return_value.attention_request.call_args_list[-1].kwargs
        assert recovery_kwargs["severity"] == "info"


def test_check_resonance_worsening_module_singleton_never_raises():
    from app.resonance_monitor import check_resonance_worsening, reset_monitor_for_tests

    reset_monitor_for_tests()
    with patch("app.resonance_monitor.NotifyClient", side_effect=RuntimeError("boom")):
        check_resonance_worsening(_alert())  # must not raise
    reset_monitor_for_tests()


def _settings():
    from app.settings import ThoughtSettings

    return ThoughtSettings(
        NOTIFY_BASE_URL="http://notify.test:7140",
        NOTIFY_API_TOKEN="",
    )
