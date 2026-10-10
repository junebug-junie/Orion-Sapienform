"""Second watchdog check: `visual_painting_gap` (no produced painting in N hours).

Incident 2026-10-09/10: the GPU lane controller refused every swap for 27.6 h;
deferral rows kept `visual_chain_stale` green the whole time. These tests pin
that the gap check fires on its own, that the two checks keep independent
edge-triggered state, and that the open-alert lookup is per key. `app.*`
imports stay inside each test (conftest purges `app.*` between tests).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _settings():
    from app.settings import ThoughtSettings

    return ThoughtSettings(
        NOTIFY_BASE_URL="http://notify.test:7140",
        NOTIFY_API_TOKEN="",
        ORION_VISUAL_CHAIN_STALENESS_THRESHOLD_MIN=45,
        ORION_VISUAL_PAINTING_GAP_THRESHOLD_HOURS=12,
    )


def _pending(items):
    return MagicMock(status_code=200, json=lambda: items)


def _reasons(client_cls):
    return [
        (c.kwargs["context"]["reason"], c.kwargs["severity"])
        for c in client_cls.return_value.attention_request.call_args_list
    ]


def test_gap_threshold_default_is_12_hours():
    from app.settings import ThoughtSettings

    assert ThoughtSettings().visual_painting_gap_threshold_hours == 12.0


@pytest.mark.parametrize("hours,healthy", [(11.0, True), (12.0, True), (13.0, False)])
def test_gap_check_threshold(hours, healthy):
    from app.visual_chain_health_monitor import _painting_gap_check

    result = _painting_gap_check(age_hours=hours, threshold_hours=12.0)
    assert result.key == "visual_painting_gap"
    assert result.healthy is healthy
    assert result.severity == "error"


def test_gap_message_is_plain_english_with_hours_threshold_and_ordered_causes():
    from app.visual_chain_health_monitor import _painting_gap_check

    msg = _painting_gap_check(age_hours=27.6, threshold_hours=12.0).message
    assert "27.6 hours" in msg
    assert "threshold 12 h" in msg
    lane = msg.index("GPU lane controller")
    thermal = msg.index("thermal")
    wedged = msg.index("wedged")
    assert lane < thermal < wedged
    assert "scripts/gpu_pool_actuator_probe.py" in msg


def test_gap_none_age_is_not_flagged():
    from app.visual_chain_health_monitor import _painting_gap_check

    assert _painting_gap_check(age_hours=None, threshold_hours=12.0).healthy is True


def test_gap_alerts_once_on_transition_and_recovers_once():
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get", return_value=_pending([])
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        monitor = VisualChainHealthMonitor(settings_obj=_settings())
        monitor.record_painting_gap(age_hours=2.0)
        monitor.record_painting_gap(age_hours=13.0)  # transition -> 1 card
        monitor.record_painting_gap(age_hours=20.0)  # still bad -> nothing
        monitor.record_painting_gap(age_hours=27.6)
        monitor.record_painting_gap(age_hours=0.1)  # painting landed -> recovery
        monitor.record_painting_gap(age_hours=0.3)  # still fine -> nothing

        assert _reasons(client_cls) == [
            ("visual_painting_gap", "error"),
            ("visual_painting_gap", "info"),
        ]
        calls = client_cls.return_value.attention_request.call_args_list
        assert calls[0].kwargs["context"]["event_kind"] == (
            "orion.reverie.visual_painting_gap.health.attention.v1"
        )
        assert "a painting landed" in calls[1].kwargs["message"]


def test_checks_are_independent_gap_bad_while_staleness_green():
    """The incident shape: fresh deferral rows (staleness healthy) + no painting."""
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get", return_value=_pending([])
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        monitor = VisualChainHealthMonitor(settings_obj=_settings())
        for _ in range(3):
            monitor.record_check(age_min=5.0)
            monitor.record_painting_gap(age_hours=2.0)
        monitor.record_check(age_min=5.0)
        monitor.record_painting_gap(age_hours=13.0)
        monitor.record_check(age_min=5.0)
        monitor.record_painting_gap(age_hours=14.0)

        assert _reasons(client_cls) == [("visual_painting_gap", "error")]


def test_checks_are_independent_staleness_bad_while_gap_green():
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get", return_value=_pending([])
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        monitor = VisualChainHealthMonitor(settings_obj=_settings())
        monitor.record_check(age_min=5.0)
        monitor.record_painting_gap(age_hours=1.0)
        monitor.record_check(age_min=90.0)  # staleness alert
        monitor.record_painting_gap(age_hours=1.5)
        monitor.record_check(age_min=5.0)  # staleness recovery
        monitor.record_painting_gap(age_hours=2.0)

        assert _reasons(client_cls) == [
            ("visual_chain_stale", "critical"),
            ("visual_chain_stale", "info"),
        ]
        # The staleness recovery note keeps its original wording byte-identical.
        last = client_cls.return_value.attention_request.call_args_list[-1].kwargs
        assert last["message"] == "[Orion reverie] recovered: visual_chain_stale"
        assert last["context"]["event_kind"] == (
            "orion.reverie.visual_chain_stale.health.attention.v1"
        )


def test_open_alert_lookup_is_per_key_gap_suppressed_by_its_own_item():
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get",
        return_value=_pending([{"source_service": "orion-thought", "reason": "visual_painting_gap"}]),
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        monitor = VisualChainHealthMonitor(settings_obj=_settings())
        monitor.record_painting_gap(age_hours=20.0)  # first obs, already open
        client_cls.return_value.attention_request.assert_not_called()
        # ... but a stale-check first observation is NOT suppressed by it.
        monitor.record_check(age_min=90.0)
        assert _reasons(client_cls) == [("visual_chain_stale", "critical")]


def test_open_alert_lookup_is_per_key_staleness_item_does_not_suppress_gap():
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get",
        return_value=_pending([{"source_service": "orion-thought", "reason": "visual_chain_stale"}]),
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        monitor = VisualChainHealthMonitor(settings_obj=_settings())
        monitor.record_painting_gap(age_hours=20.0)
        assert _reasons(client_cls) == [("visual_painting_gap", "error")]


def test_gap_retries_until_notify_confirms():
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get", return_value=_pending([])
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        monitor = VisualChainHealthMonitor(settings_obj=_settings())
        monitor.record_painting_gap(age_hours=1.0)
        client_cls.return_value.attention_request.return_value = MagicMock(ok=False)
        monitor.record_painting_gap(age_hours=13.0)
        monitor.record_painting_gap(age_hours=13.2)
        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        monitor.record_painting_gap(age_hours=13.4)
        monitor.record_painting_gap(age_hours=13.6)
        assert client_cls.return_value.attention_request.call_count == 3


def test_check_visual_painting_gap_module_entrypoint_never_raises():
    from app.visual_chain_health_monitor import check_visual_painting_gap, reset_monitor_for_tests

    reset_monitor_for_tests()
    with patch("app.visual_chain_health_monitor.NotifyClient", side_effect=RuntimeError("boom")):
        check_visual_painting_gap(30.0)
    reset_monitor_for_tests()


class _FakeConn:
    def __init__(self, results, executed):
        self._results = list(results)
        self._executed = executed

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, params=None):
        self._executed.append((str(stmt), params))
        value = self._results.pop(0)
        return MagicMock(first=lambda: (value,))


def _fake_engine(results, executed):
    return MagicMock(connect=lambda: _FakeConn(results, executed))


def test_store_painting_age_uses_bounded_index_query_first():
    executed: list = []
    with patch("app.store._get_engine", return_value=_fake_engine([3.5], executed)):
        from app.store import visual_last_painting_age_hours

        assert visual_last_painting_age_hours() == 3.5
    assert len(executed) == 1
    sql, params = executed[0]
    assert "production_receipt" in sql and "produced_at" in sql
    assert "created_at > now()" in sql
    assert params == {"days": 7}


def test_store_painting_age_falls_back_to_unbounded_when_window_empty():
    executed: list = []
    with patch("app.store._get_engine", return_value=_fake_engine([None, 200.0], executed)):
        from app.store import visual_last_painting_age_hours

        assert visual_last_painting_age_hours() == 200.0
    assert len(executed) == 2
    assert "created_at > now()" not in executed[1][0]


def test_store_painting_age_none_when_never_painted_and_never_raises():
    with patch("app.store._get_engine", return_value=_fake_engine([None, None], [])):
        from app.store import visual_last_painting_age_hours

        assert visual_last_painting_age_hours() is None
    with patch("app.store._get_engine", side_effect=RuntimeError("db down")):
        from app.store import visual_last_painting_age_hours

        assert visual_last_painting_age_hours() is None
