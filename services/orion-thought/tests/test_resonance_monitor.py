"""Phase H+ — resonance health monitor. `app.*` imports are done inside each
test function (not module scope) because this service's conftest purges
`app`/`app.*` from `sys.modules` before every test — a module-scope import
would hold a stale reference that `unittest.mock.patch("app....")` can't see.

`ResonanceHealthMonitor.__init__` unconditionally bootstraps its tracked-theme
set from orion-notify's pending list, so every test that constructs one must
mock `app.resonance_monitor.requests.get` (via `_no_bootstrap_items()` below)
or it will attempt a real network call.
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


def _no_bootstrap_items():
    """A GET /attention response with nothing pending -- the common case for
    tests that don't specifically exercise the bootstrap-from-notify path."""
    return MagicMock(status_code=200, json=lambda: [])


def _settings():
    from app.settings import ThoughtSettings

    return ThoughtSettings(
        NOTIFY_BASE_URL="http://notify.test:7140",
        NOTIFY_API_TOKEN="",
    )


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
        "app.resonance_monitor.requests.get", return_value=_no_bootstrap_items()
    ), patch("app.resonance_monitor.load_recent_resonance_alerts", return_value=_samples(3)):
        from app.resonance_monitor import ResonanceHealthMonitor

        monitor = ResonanceHealthMonitor(settings_obj=_settings())
        monitor.check(_alert())
        client_cls.return_value.attention_request.assert_not_called()


def test_monitor_alerts_only_on_worsening_transition_not_every_tick():
    with patch("app.resonance_monitor.NotifyClient") as client_cls, patch(
        "app.resonance_monitor.requests.get", return_value=_no_bootstrap_items()
    ), patch("app.resonance_monitor.load_recent_resonance_alerts") as mock_load:
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
        "app.resonance_monitor.requests.get", return_value=_no_bootstrap_items()
    ), patch("app.resonance_monitor.load_recent_resonance_alerts") as mock_load:
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


def test_monitor_evicts_theme_after_confirmed_recovery():
    """Once a recovery note is confirmed delivered, the theme must stop being
    tracked -- otherwise every completed chain re-queries it forever."""
    with patch("app.resonance_monitor.NotifyClient") as client_cls, patch(
        "app.resonance_monitor.requests.get", return_value=_no_bootstrap_items()
    ), patch("app.resonance_monitor.load_recent_resonance_alerts") as mock_load:
        from app.resonance_monitor import ResonanceHealthMonitor

        _client_mock(client_cls)
        monitor = ResonanceHealthMonitor(settings_obj=_settings())

        mock_load.return_value = _samples(3)
        monitor.check(_alert())  # healthy baseline
        assert "loop:t1" in monitor._tracked_themes

        mock_load.return_value = _samples(6, 3)
        monitor.check(_alert())  # worsening -> tracked, alerted

        mock_load.return_value = _samples(3, 6)
        monitor.check(_alert())  # recovered -> evicted
        assert "loop:t1" not in monitor._tracked_themes
        assert "reverie_resonance_worsening:loop:t1" not in monitor._last_healthy

        # A later tick where t1 is no longer the most-severe theme (alert=None)
        # must not re-query it, since it's no longer tracked.
        mock_load.reset_mock()
        monitor.check(None)
        mock_load.assert_not_called()


def test_monitor_retries_until_notify_confirms_delivery():
    with patch("app.resonance_monitor.NotifyClient") as client_cls, patch(
        "app.resonance_monitor.requests.get", return_value=_no_bootstrap_items()
    ), patch("app.resonance_monitor.load_recent_resonance_alerts") as mock_load:
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
        monitor.check(_alert())  # publish finally succeeds -> worsening alert committed

        assert client_cls.return_value.attention_request.call_count == 3

        # Theme is still "unhealthy-committed" (not recovered), so it remains
        # tracked and re-checked -- confirm no further spurious alert.
        monitor.check(_alert())
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
        # Bootstrap itself will pick this theme up too -- that's exactly the
        # behavior under test in test_bootstrap_* below. Here we exercise the
        # per-check `_has_open_alert` path for a theme NOT already bootstrapped.
        monitor = ResonanceHealthMonitor(settings_obj=_settings())
        monitor._tracked_themes.clear()
        monitor._last_healthy.clear()

        monitor.check(_alert())

        client_cls.return_value.attention_request.assert_not_called()


def test_monitor_recovers_previously_tracked_theme_even_when_alert_is_none():
    """A theme that stops being the current tick's most-severe theme should
    still get checked (and recovered) rather than staying silently open."""
    with patch("app.resonance_monitor.NotifyClient") as client_cls, patch(
        "app.resonance_monitor.requests.get", return_value=_no_bootstrap_items()
    ), patch("app.resonance_monitor.load_recent_resonance_alerts") as mock_load:
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


def test_bootstrap_reconstructs_tracked_themes_from_notify_pending_list():
    """A fresh monitor (simulating a post-restart process) must reconstruct
    which themes were previously flagged worsening from orion-notify's own
    pending list -- otherwise a theme dropped by the restart could never be
    re-tracked, and its open Pending Attention item could never recover."""
    with patch("app.resonance_monitor.NotifyClient"), patch(
        "app.resonance_monitor.requests.get"
    ) as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {
                    "source_service": "orion-thought",
                    "reason": "reverie_resonance_worsening:loop:t1",
                },
                {  # a different service's pending item -- must be ignored
                    "source_service": "orion-field-digester",
                    "reason": "database_size_high",
                },
                {  # this service, but not a resonance-worsening reason -- ignored
                    "source_service": "orion-thought",
                    "reason": "some_other_check",
                },
            ],
        )
        from app.resonance_monitor import ResonanceHealthMonitor

        monitor = ResonanceHealthMonitor(settings_obj=_settings())

        assert monitor._tracked_themes == {"loop:t1"}
        assert monitor._last_healthy["reverie_resonance_worsening:loop:t1"] is False


def test_bootstrap_reconstructed_theme_can_recover_on_next_tick():
    """The whole point of the bootstrap: a theme reconstructed as unhealthy
    at startup must be able to actually recover (fire the recovery note) on
    the next real observation, rather than being silently dropped forever."""
    with patch("app.resonance_monitor.NotifyClient") as client_cls, patch(
        "app.resonance_monitor.requests.get"
    ) as mock_get, patch(
        "app.resonance_monitor.load_recent_resonance_alerts", return_value=_samples(3, 6)
    ):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {
                    "source_service": "orion-thought",
                    "reason": "reverie_resonance_worsening:loop:t1",
                }
            ],
        )
        _client_mock(client_cls)
        from app.resonance_monitor import ResonanceHealthMonitor

        monitor = ResonanceHealthMonitor(settings_obj=_settings())
        # t1 is no longer the most-severe theme this tick (alert=None), but it
        # must still be re-checked because bootstrap tracked it.
        monitor.check(None)

        client_cls.return_value.attention_request.assert_called_once()
        recovery_kwargs = client_cls.return_value.attention_request.call_args.kwargs
        assert recovery_kwargs["severity"] == "info"
        assert "loop:t1" not in monitor._tracked_themes  # evicted after recovery


def test_bootstrap_failure_is_fail_open_not_fatal():
    with patch("app.resonance_monitor.NotifyClient"), patch(
        "app.resonance_monitor.requests.get", side_effect=RuntimeError("network down")
    ):
        from app.resonance_monitor import ResonanceHealthMonitor

        monitor = ResonanceHealthMonitor(settings_obj=_settings())  # must not raise
        assert monitor._tracked_themes == set()


def test_check_resonance_worsening_module_singleton_never_raises():
    from app.resonance_monitor import check_resonance_worsening, reset_monitor_for_tests

    reset_monitor_for_tests()
    with patch("app.resonance_monitor.NotifyClient", side_effect=RuntimeError("boom")):
        check_resonance_worsening(_alert())  # must not raise
    reset_monitor_for_tests()
