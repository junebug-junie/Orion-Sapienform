from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.health_monitor import HealthMonitor, run_checks
from app.settings import Settings


def _settings(**overrides) -> Settings:
    base = dict(
        POSTGRES_URI="postgresql://unused/unused",
        ATTENTION_FRAME_RETENTION_HOURS="72.0",
        ATTENTION_FRAME_STALL_MULTIPLIER="1.5",
    )
    base.update(overrides)
    return Settings(**base)


def _store_with(age_hours: float | None) -> MagicMock:
    store = MagicMock()
    store.attention_frame_oldest_age_hours.return_value = age_hours
    return store


def _client_mock(client_cls, *, ok: bool = True):
    client_cls.return_value.attention_request.return_value = MagicMock(ok=ok)
    return client_cls.return_value.attention_request


def test_run_checks_calls_oldest_age_lookup_once():
    store = _store_with(10.0)

    run_checks(store, _settings())

    store.attention_frame_oldest_age_hours.assert_called_once_with()


def test_run_checks_flags_stalled_attention_frame_prune():
    store = _store_with(200.0)  # > 72 * 1.5

    checks = run_checks(store, _settings())

    stalled = next(c for c in checks if c.key == "attention_frame_prune_stalled")
    assert stalled.healthy is False
    assert stalled.message  # non-empty when unhealthy


def test_run_checks_healthy_when_within_bounds():
    store = _store_with(10.0)

    checks = run_checks(store, _settings())

    assert all(c.healthy for c in checks)
    assert all(c.message == "" for c in checks)


def test_run_checks_disabled_when_retention_zero():
    store = _store_with(200.0)

    checks = run_checks(store, _settings(ATTENTION_FRAME_RETENTION_HOURS="0"))

    assert checks == []


def test_health_monitor_does_not_alert_on_healthy_first_observation():
    store = _store_with(10.0)

    with patch("app.health_monitor.NotifyClient") as client_cls:
        monitor = HealthMonitor(store, _settings())
        monitor.run_tick()
        client_cls.return_value.attention_request.assert_not_called()


def test_health_monitor_alerts_only_on_unhealthy_transition_not_every_tick():
    store = MagicMock()

    with patch("app.health_monitor.NotifyClient") as client_cls:
        _client_mock(client_cls)
        monitor = HealthMonitor(store, _settings())

        store.attention_frame_oldest_age_hours.return_value = 10.0
        monitor.run_tick()  # healthy baseline observation, no alert

        store.attention_frame_oldest_age_hours.return_value = 200.0
        monitor.run_tick()  # transitions unhealthy -> alert
        monitor.run_tick()  # still unhealthy -> no additional alert

        assert client_cls.return_value.attention_request.call_count == 1


def test_health_monitor_sends_recovery_note_on_healthy_transition():
    store = MagicMock()

    with patch("app.health_monitor.NotifyClient") as client_cls:
        _client_mock(client_cls)
        monitor = HealthMonitor(store, _settings())

        store.attention_frame_oldest_age_hours.return_value = 10.0
        monitor.run_tick()  # healthy baseline, no alert

        store.attention_frame_oldest_age_hours.return_value = 200.0
        monitor.run_tick()  # unhealthy transition -> 1 alert

        store.attention_frame_oldest_age_hours.return_value = 10.0
        monitor.run_tick()  # healthy transition -> recovery note

        assert client_cls.return_value.attention_request.call_count == 2
        recovery_kwargs = client_cls.return_value.attention_request.call_args_list[-1].kwargs
        assert recovery_kwargs["severity"] == "info"


def test_health_monitor_retries_transition_alert_until_notify_confirms_delivery():
    # If orion-notify is unreachable (or returns ok=False) at the exact moment of
    # a transition, the alert must not be silently dropped -- it should retry on
    # every subsequent tick until delivery is actually confirmed.
    store = MagicMock()

    with patch("app.health_monitor.NotifyClient") as client_cls:
        monitor = HealthMonitor(store, _settings())

        store.attention_frame_oldest_age_hours.return_value = 10.0
        monitor.run_tick()  # healthy baseline, no alert

        store.attention_frame_oldest_age_hours.return_value = 200.0
        _client_mock(client_cls, ok=False)
        monitor.run_tick()  # transition detected, publish fails -> not committed
        monitor.run_tick()  # still not committed -> retried again

        assert client_cls.return_value.attention_request.call_count == 2

        _client_mock(client_cls, ok=True)
        monitor.run_tick()  # publish finally succeeds -> committed

        assert client_cls.return_value.attention_request.call_count == 3

        monitor.run_tick()  # now committed as unhealthy -> no further calls

        assert client_cls.return_value.attention_request.call_count == 3


def test_health_monitor_suppresses_first_observation_alert_when_notify_already_has_open_item():
    store = _store_with(200.0)  # already unhealthy at startup

    with patch("app.health_monitor.NotifyClient") as client_cls, patch(
        "app.health_monitor.requests.get"
    ) as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {
                    "source_service": "orion-attention-runtime",
                    "reason": "attention_frame_prune_stalled",
                }
            ],
        )
        monitor = HealthMonitor(store, _settings())
        monitor.run_tick()

        client_cls.return_value.attention_request.assert_not_called()


def test_health_monitor_fires_first_observation_alert_when_no_open_item_found():
    store = _store_with(200.0)  # already unhealthy at startup

    with patch("app.health_monitor.NotifyClient") as client_cls, patch(
        "app.health_monitor.requests.get"
    ) as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        _client_mock(client_cls)
        monitor = HealthMonitor(store, _settings())
        monitor.run_tick()

        client_cls.return_value.attention_request.assert_called_once()
