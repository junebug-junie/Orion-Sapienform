"""Unit tests for the substrate-runtime health monitor.

Note: this service's ``conftest.py`` deletes ``app.*`` from ``sys.modules``
before *every* test (autouse fixture, for cross-service test isolation), so
``app.health_monitor`` must be (re)imported fresh inside each test body --
importing it once at module import time (collection time, before the fixture
first runs) would bind names to a module object that ``unittest.mock.patch``
and the test body would then disagree about, since ``patch("app.health_monitor.X")``
re-resolves the module via ``sys.modules`` at call time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))


@pytest.fixture(autouse=True)
def _no_real_recheck_sleep():
    # Every fresh unhealthy transition now sleeps `health_recheck_delay_sec`
    # (default 15.0) before rechecking; keep the test suite fast by not
    # actually sleeping. Patched on the stdlib `time` module directly so it
    # applies regardless of when `app.health_monitor` gets (re)imported.
    with patch("time.sleep") as mock_sleep:
        yield mock_sleep


def _hm():
    import app.health_monitor as module

    return module


def _settings(**overrides):
    from app.settings import Settings

    base = dict(POSTGRES_URI="postgresql://unused/unused")
    base.update(overrides)
    return Settings(**base)


def _store() -> MagicMock:
    return MagicMock()


def _truth(
    *,
    degraded: bool,
    reasons: list[str] | None = None,
    cursor_positions: list[dict] | None = None,
) -> dict:
    reasons = reasons or []
    truth: dict = {"ok": not degraded, "degraded": degraded, "degraded_reasons": reasons}
    if cursor_positions is not None:
        truth["cursor_positions"] = cursor_positions
    return truth


def _client_mock(client_cls, *, ok: bool = True):
    client_cls.return_value.attention_request.return_value = MagicMock(ok=ok)
    return client_cls.return_value.attention_request


def test_run_checks_calls_build_substrate_grammar_truth_with_store():
    hm = _hm()
    store = _store()

    with patch(
        "app.health_monitor.build_substrate_grammar_truth", return_value=_truth(degraded=False)
    ) as truth_fn:
        hm.run_checks(store, _settings())

    truth_fn.assert_called_once_with(store)


def test_run_checks_healthy_when_not_degraded():
    hm = _hm()

    with patch(
        "app.health_monitor.build_substrate_grammar_truth", return_value=_truth(degraded=False)
    ):
        checks = hm.run_checks(_store(), _settings())

    assert len(checks) == 1
    check = checks[0]
    assert check.key == "substrate_grammar_degraded"
    assert check.healthy is True
    assert check.message == ""


def test_run_checks_unhealthy_includes_reasons_in_message():
    hm = _hm()
    reasons = ["cursor_lag:biometrics_grammar_consumer", "reducer_blocked:chat_grammar_consumer"]

    with patch(
        "app.health_monitor.build_substrate_grammar_truth",
        return_value=_truth(degraded=True, reasons=reasons),
    ):
        checks = hm.run_checks(_store(), _settings())

    check = checks[0]
    assert check.healthy is False
    assert check.severity == "critical"
    for reason in reasons:
        assert reason in check.message


def test_run_checks_annotates_cursor_lag_reason_with_local_denver_time():
    hm = _hm()
    reasons = ["cursor_lag:chat_grammar_consumer"]
    cursor_positions = [
        {
            "cursor_name": "chat_grammar_consumer",
            # 2026-07-13T07:52:47+00:00 is 2026-07-13 01:52 in America/Denver (MDT, UTC-6).
            "last_event_created_at": "2026-07-13T07:52:47+00:00",
        }
    ]

    with patch(
        "app.health_monitor.build_substrate_grammar_truth",
        return_value=_truth(degraded=True, reasons=reasons, cursor_positions=cursor_positions),
    ):
        checks = hm.run_checks(_store(), _settings())

    check = checks[0]
    assert "cursor_lag:chat_grammar_consumer" in check.message
    assert "2026-07-13 01:52 MDT" in check.message


def test_run_checks_leaves_reason_unannotated_without_cursor_position_data():
    hm = _hm()
    reasons = ["cursor_lag:chat_grammar_consumer"]

    with patch(
        "app.health_monitor.build_substrate_grammar_truth",
        return_value=_truth(degraded=True, reasons=reasons),
    ):
        checks = hm.run_checks(_store(), _settings())

    check = checks[0]
    assert check.message == "substrate-runtime grammar production degraded: cursor_lag:chat_grammar_consumer"


def test_health_monitor_does_not_alert_on_healthy_first_observation():
    hm = _hm()

    with (
        patch(
            "app.health_monitor.build_substrate_grammar_truth",
            return_value=_truth(degraded=False),
        ),
        patch("app.health_monitor.NotifyClient") as client_cls,
    ):
        monitor = hm.HealthMonitor(_store(), _settings())
        monitor.run_tick()
        client_cls.return_value.attention_request.assert_not_called()


def test_health_monitor_alerts_only_on_unhealthy_transition_not_every_tick():
    hm = _hm()

    with (
        patch("app.health_monitor.NotifyClient") as client_cls,
        patch("app.health_monitor.build_substrate_grammar_truth") as truth_fn,
    ):
        _client_mock(client_cls)
        monitor = hm.HealthMonitor(_store(), _settings())

        truth_fn.return_value = _truth(degraded=False)
        monitor.run_tick()  # healthy baseline observation, no alert

        truth_fn.return_value = _truth(degraded=True, reasons=["cursor_lag:biometrics"])
        monitor.run_tick()  # transitions unhealthy -> alert
        monitor.run_tick()  # still unhealthy -> no additional alert

        assert client_cls.return_value.attention_request.call_count == 1


def test_health_monitor_sends_recovery_note_on_healthy_transition():
    hm = _hm()

    with (
        patch("app.health_monitor.NotifyClient") as client_cls,
        patch("app.health_monitor.build_substrate_grammar_truth") as truth_fn,
    ):
        _client_mock(client_cls)
        monitor = hm.HealthMonitor(_store(), _settings())

        truth_fn.return_value = _truth(degraded=False)
        monitor.run_tick()  # healthy baseline, no alert

        truth_fn.return_value = _truth(degraded=True, reasons=["cursor_lag:biometrics"])
        monitor.run_tick()  # unhealthy transition -> 1 alert

        truth_fn.return_value = _truth(degraded=False)
        monitor.run_tick()  # healthy transition -> recovery note

        assert client_cls.return_value.attention_request.call_count == 2
        recovery_kwargs = client_cls.return_value.attention_request.call_args_list[-1].kwargs
        assert recovery_kwargs["severity"] == "info"


def test_health_monitor_retries_transition_alert_until_notify_confirms_delivery():
    # If orion-notify is unreachable (or returns ok=False) at the exact moment of
    # a transition, the alert must not be silently dropped -- it should retry on
    # every subsequent tick until delivery is actually confirmed.
    hm = _hm()

    with (
        patch("app.health_monitor.NotifyClient") as client_cls,
        patch("app.health_monitor.build_substrate_grammar_truth") as truth_fn,
    ):
        monitor = hm.HealthMonitor(_store(), _settings())

        truth_fn.return_value = _truth(degraded=False)
        monitor.run_tick()  # healthy baseline, no alert

        truth_fn.return_value = _truth(degraded=True, reasons=["cursor_lag:biometrics"])
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
    hm = _hm()

    with (
        patch("app.health_monitor.NotifyClient") as client_cls,
        patch(
            "app.health_monitor.build_substrate_grammar_truth",
            return_value=_truth(degraded=True, reasons=["cursor_lag:biometrics"]),
        ),
        patch("app.health_monitor.requests.get") as mock_get,
    ):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {
                    "source_service": "orion-substrate-runtime",
                    "reason": "substrate_grammar_degraded",
                }
            ],
        )
        monitor = hm.HealthMonitor(_store(), _settings())
        monitor.run_tick()

        client_cls.return_value.attention_request.assert_not_called()


def test_health_monitor_fires_first_observation_alert_when_no_open_item_found():
    hm = _hm()

    with (
        patch("app.health_monitor.NotifyClient") as client_cls,
        patch(
            "app.health_monitor.build_substrate_grammar_truth",
            return_value=_truth(degraded=True, reasons=["cursor_lag:biometrics"]),
        ),
        patch("app.health_monitor.requests.get") as mock_get,
    ):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        _client_mock(client_cls)
        monitor = hm.HealthMonitor(_store(), _settings())
        monitor.run_tick()

        client_cls.return_value.attention_request.assert_called_once()


def test_health_monitor_publish_exception_does_not_raise_out_of_run_tick():
    hm = _hm()

    with (
        patch("app.health_monitor.NotifyClient") as client_cls,
        patch(
            "app.health_monitor.build_substrate_grammar_truth",
            return_value=_truth(degraded=True, reasons=["cursor_lag:biometrics"]),
        ),
    ):
        client_cls.return_value.attention_request.side_effect = Exception("connection refused")
        monitor = hm.HealthMonitor(_store(), _settings())

        monitor.run_tick()  # must not raise

        assert client_cls.return_value.attention_request.call_count == 1


def test_health_monitor_has_open_alert_lookup_failure_does_not_raise():
    hm = _hm()

    with (
        patch("app.health_monitor.NotifyClient") as client_cls,
        patch(
            "app.health_monitor.build_substrate_grammar_truth",
            return_value=_truth(degraded=True, reasons=["cursor_lag:biometrics"]),
        ),
        patch("app.health_monitor.requests.get", side_effect=Exception("unreachable")),
    ):
        _client_mock(client_cls)
        monitor = hm.HealthMonitor(_store(), _settings())

        monitor.run_tick()  # must not raise; fails open into attempting _publish

        client_cls.return_value.attention_request.assert_called_once()


def test_health_monitor_suppresses_transient_blip_on_first_observation(_no_real_recheck_sleep):
    # First observation is already unhealthy but self-heals by the recheck --
    # must not page, and must leave state unset so the next tick evaluates fresh.
    hm = _hm()

    with (
        patch("app.health_monitor.NotifyClient") as client_cls,
        patch("app.health_monitor.build_substrate_grammar_truth") as truth_fn,
        patch("app.health_monitor.requests.get") as mock_get,
    ):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        truth_fn.side_effect = [
            _truth(degraded=True, reasons=["reducer_cursor_commit_failing:biometrics_grammar_consumer"]),
            _truth(degraded=False),  # recheck: self-healed
        ]
        monitor = hm.HealthMonitor(_store(), _settings())

        monitor.run_tick()

        client_cls.return_value.attention_request.assert_not_called()
        assert monitor._last_healthy.get("substrate_grammar_degraded") is None
        _no_real_recheck_sleep.assert_called_once_with(15.0)


def test_health_monitor_suppresses_transient_blip_on_healthy_to_unhealthy_transition():
    # Was healthy, one tick reports unhealthy, but recheck finds it recovered --
    # must not page, and must stay recorded healthy (no retry-alert next tick).
    hm = _hm()

    with (
        patch("app.health_monitor.NotifyClient") as client_cls,
        patch("app.health_monitor.build_substrate_grammar_truth") as truth_fn,
    ):
        _client_mock(client_cls)
        monitor = hm.HealthMonitor(_store(), _settings())

        truth_fn.return_value = _truth(degraded=False)
        monitor.run_tick()  # healthy baseline

        truth_fn.side_effect = [
            _truth(degraded=True, reasons=["reducer_cursor_commit_failing:biometrics_grammar_consumer"]),
            _truth(degraded=False),  # recheck: self-healed
        ]
        monitor.run_tick()  # blip absorbed, no alert

        assert client_cls.return_value.attention_request.call_count == 0
        assert monitor._last_healthy.get("substrate_grammar_degraded") is True


def test_health_monitor_pages_when_recheck_confirms_sustained_degradation():
    # Both the initial observation and the recheck report unhealthy -- a real,
    # sustained incident must still page exactly like before this change.
    hm = _hm()

    with (
        patch("app.health_monitor.NotifyClient") as client_cls,
        patch("app.health_monitor.build_substrate_grammar_truth") as truth_fn,
        patch("app.health_monitor.requests.get") as mock_get,
    ):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        _client_mock(client_cls)
        degraded = _truth(reasons=["reducer_cursor_commit_failing:biometrics_grammar_consumer"], degraded=True)
        truth_fn.side_effect = [degraded, degraded]
        monitor = hm.HealthMonitor(_store(), _settings())

        monitor.run_tick()

        client_cls.return_value.attention_request.assert_called_once()
        assert monitor._last_healthy.get("substrate_grammar_degraded") is False


def test_health_monitor_recheck_delay_is_configurable(_no_real_recheck_sleep):
    hm = _hm()

    with (
        patch("app.health_monitor.NotifyClient") as client_cls,
        patch(
            "app.health_monitor.build_substrate_grammar_truth",
            return_value=_truth(degraded=True, reasons=["cursor_lag:biometrics"]),
        ),
        patch("app.health_monitor.requests.get") as mock_get,
    ):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        _client_mock(client_cls)
        monitor = hm.HealthMonitor(_store(), _settings(SUBSTRATE_RUNTIME_HEALTH_RECHECK_DELAY_SEC=3.0))

        monitor.run_tick()

        _no_real_recheck_sleep.assert_called_once_with(3.0)


def test_health_monitor_does_not_resleep_on_retry_after_confirmed_transition(
    _no_real_recheck_sleep,
):
    # Once a transition has been recheck-confirmed, a publish failure must not
    # cause the next retry tick to pay the recheck delay (and a duplicate DB
    # read) again -- that would re-hammer the DB during exactly the kind of
    # sustained outage/pressure window the retry-until-delivered path exists
    # to survive gracefully.
    hm = _hm()
    degraded = _truth(reasons=["reducer_cursor_commit_failing:biometrics_grammar_consumer"], degraded=True)

    with (
        patch("app.health_monitor.NotifyClient") as client_cls,
        patch("app.health_monitor.build_substrate_grammar_truth") as truth_fn,
        patch("app.health_monitor.requests.get") as mock_get,
    ):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        _client_mock(client_cls, ok=False)  # orion-notify unreachable
        truth_fn.side_effect = [degraded, degraded]  # tick 1: check + recheck confirm
        monitor = hm.HealthMonitor(_store(), _settings())

        monitor.run_tick()  # transition confirmed via recheck, publish fails
        assert _no_real_recheck_sleep.call_count == 1
        assert client_cls.return_value.attention_request.call_count == 1

        truth_fn.side_effect = [degraded]  # tick 2: only the top-level check, no recheck
        monitor.run_tick()  # already recheck-confirmed -- retries publish directly

        assert _no_real_recheck_sleep.call_count == 1  # no additional sleep
        assert client_cls.return_value.attention_request.call_count == 2

        _client_mock(client_cls, ok=True)
        truth_fn.side_effect = [degraded]
        monitor.run_tick()  # publish finally succeeds

        assert _no_real_recheck_sleep.call_count == 1
        assert client_cls.return_value.attention_request.call_count == 3
        assert monitor._last_healthy.get("substrate_grammar_degraded") is False


def test_health_monitor_pages_when_recheck_itself_raises():
    # If the recheck's own DB read raises (plausible: the same DB pressure
    # that caused the degradation also breaks this query), that must not be
    # silently treated as "recovered" -- fail toward alerting instead.
    hm = _hm()

    with (
        patch("app.health_monitor.NotifyClient") as client_cls,
        patch("app.health_monitor.build_substrate_grammar_truth") as truth_fn,
        patch("app.health_monitor.requests.get") as mock_get,
    ):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        _client_mock(client_cls)
        truth_fn.side_effect = [
            _truth(degraded=True, reasons=["reducer_cursor_commit_failing:biometrics_grammar_consumer"]),
            RuntimeError("connection pool exhausted"),
        ]
        monitor = hm.HealthMonitor(_store(), _settings())

        monitor.run_tick()  # must not raise; recheck failure still pages

        client_cls.return_value.attention_request.assert_called_once()
        assert monitor._last_healthy.get("substrate_grammar_degraded") is False
