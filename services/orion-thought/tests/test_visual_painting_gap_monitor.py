"""Second watchdog check: `visual_painting_gap` (no produced painting in N hours).

Incident 2026-10-09/10: the GPU lane controller refused every swap for 27.6 h
and nothing alerted -- the legacy staleness watchdog is gated on
visual_chain_enabled (off in production) and never ran. These tests pin that
the gap check fires on its own (own loop, own flag), that the two checks keep
independent edge-triggered state, that a failed read is no observation, and
that the open-alert lookup is per key. `app.*`
imports stay inside each test (conftest purges `app.*` between tests).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _no_live_deferral_read(monkeypatch, request):
    """A gap reads the baseline checkpoint (Postgres) for its cause; tests that want it patch it."""
    if request.node.name.startswith("test_store_reads_the_checkpoint_deferral"):
        return
    monkeypatch.setattr("app.store.visual_baseline_deferral", lambda: None)


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
    """Staleness healthy while no painting comes out: only the gap check fires."""
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


def test_db_read_failure_mid_gap_is_no_observation_not_recovery():
    """None (DB read failed) mid-gap must not send a recovery note and then
    re-alert on the next real reading."""
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get", return_value=_pending([])
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        monitor = VisualChainHealthMonitor(settings_obj=_settings())
        monitor.record_painting_gap(age_hours=1.0)
        monitor.record_painting_gap(age_hours=13.0)  # alert
        monitor.record_painting_gap(age_hours=None)  # read failed
        monitor.record_painting_gap(age_hours=None)
        monitor.record_painting_gap(age_hours=14.0)  # still the same gap
        assert _reasons(client_cls) == [("visual_painting_gap", "error")]


def test_none_before_any_reading_leaves_state_unset():
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get", return_value=_pending([])
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        monitor = VisualChainHealthMonitor(settings_obj=_settings())
        monitor.record_painting_gap(age_hours=None)
        assert "visual_painting_gap" not in monitor._last_healthy
        monitor.record_painting_gap(age_hours=13.0)  # first real obs, unhealthy
        assert _reasons(client_cls) == [("visual_painting_gap", "error")]


# --- run_visual_painting_gap_watchdog: own loop, not behind visual_chain_enabled ---


def _loop_settings(**overrides):
    from app.settings import ThoughtSettings

    values = dict(
        ORION_VISUAL_CHAIN_ENABLED=False,
        ORION_BUS_ENABLED=False,
        ORION_VISUAL_PAINTING_GAP_CHECK_ENABLED=True,
        ORION_VISUAL_PAINTING_GAP_CHECK_INTERVAL_SEC=0.01,
    )
    values.update(overrides)
    return ThoughtSettings(**values)


def test_gap_check_settings_defaults_on():
    from app.settings import ThoughtSettings

    s = ThoughtSettings()
    assert s.visual_painting_gap_check_enabled is True
    assert s.visual_painting_gap_check_interval_sec == 600.0


@pytest.mark.asyncio
async def test_gap_loop_runs_with_legacy_visual_chain_and_bus_disabled(monkeypatch):
    """Production shape: ORION_VISUAL_CHAIN_ENABLED=false. The legacy
    watchdog returns early there; this loop must still tick."""
    import asyncio

    from app import visual_chain_health_monitor as mon

    cfg = _loop_settings()
    assert cfg.visual_chain_enabled is False
    stop = asyncio.Event()
    gaps = iter([2.0, 13.0])
    reported = []

    def _fake_check(age_hours):
        reported.append(age_hours)
        if len(reported) >= 2:
            stop.set()

    monkeypatch.setattr("app.store.visual_last_painting_age_hours", lambda: next(gaps))
    monkeypatch.setattr(mon, "check_visual_painting_gap", _fake_check)
    await asyncio.wait_for(mon.run_visual_painting_gap_watchdog(stop, settings_obj=cfg), 2.0)
    assert reported == [2.0, 13.0]


@pytest.mark.asyncio
async def test_gap_loop_noop_when_its_own_flag_is_off(monkeypatch):
    from app import visual_chain_health_monitor as mon

    calls = []
    monkeypatch.setattr("app.store.visual_last_painting_age_hours", lambda: calls.append(1) or 1.0)
    import asyncio

    # Must return on its own; a loop that ignored the flag would never end
    # (stop is never set), so wait_for turns that into a fast failure.
    await asyncio.wait_for(
        mon.run_visual_painting_gap_watchdog(
            asyncio.Event(),
            settings_obj=_loop_settings(ORION_VISUAL_PAINTING_GAP_CHECK_ENABLED=False),
        ),
        1.0,
    )
    assert calls == []


@pytest.mark.asyncio
async def test_gap_loop_survives_a_failing_tick(monkeypatch):
    import asyncio

    from app import visual_chain_health_monitor as mon

    stop = asyncio.Event()
    n = {"v": 0}

    def _age():
        n["v"] += 1
        if n["v"] == 1:
            raise RuntimeError("boom")
        stop.set()
        return 1.0

    monkeypatch.setattr("app.store.visual_last_painting_age_hours", _age)
    monkeypatch.setattr(mon, "check_visual_painting_gap", lambda age_hours: None)
    await asyncio.wait_for(
        mon.run_visual_painting_gap_watchdog(stop, settings_obj=_loop_settings()), 2.0
    )
    assert n["v"] == 2


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_painting_gap_task(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    import app.main as main_module
    from app.settings import settings

    async def _noop(*_a, **_k):
        return None

    for name in (
        "run_bus_worker",
        "run_reverie_worker",
        "run_reverie_chain_worker",
        "run_reasoning_worker",
        "run_visual_chain_worker",
        "run_visual_chain_watchdog",
        "warm_pool",
    ):
        monkeypatch.setattr(main_module, name, _noop)
    monkeypatch.setattr(settings, "rpc_health_publish_enabled", False)
    monkeypatch.setattr(settings, "visual_chain_enabled", False)
    monkeypatch.setattr(
        main_module, "build_heartbeat_chassis", lambda: (_ for _ in ()).throw(RuntimeError("no hb"))
    )
    started = asyncio.Event()
    seen = {}

    async def _fake_gap_loop(stop_event):
        seen["stop"] = stop_event
        started.set()
        await stop_event.wait()

    monkeypatch.setattr(main_module, "run_visual_painting_gap_watchdog", _fake_gap_loop)
    app = SimpleNamespace(state=SimpleNamespace())
    async with main_module.lifespan(app):
        await asyncio.wait_for(started.wait(), 2.0)
        assert not app.state.visual_painting_gap_task.done()
    assert seen["stop"].is_set()
    assert app.state.visual_painting_gap_task.done()


# --- 2026-10-20: arousal-strained deferral named as the cause -----------------------------------
# Live 10-20: no painting since 10-18 12:07; arousal strained (cabinet_hot, gpu_queue_sustained) and
# every due baseline deferred. The alert's cause list named only the GPU controller / thermal / a
# wedged worker.
LIVE_DEFERRAL = {"reason": "arousal_strained", "since": "2026-10-18T12:37:30+00:00",
                 "last_at": "2026-10-20T18:01:26+00:00"}


def test_gap_message_names_arousal_deferral_first_when_recorded():
    from app.visual_chain_health_monitor import DeferralEvidence, _painting_gap_check

    ev = DeferralEvidence(reason="arousal_strained", since=LIVE_DEFERRAL["since"], last_at=LIVE_DEFERRAL["last_at"],
                          arousal_reasons=("cabinet_hot", "gpu_queue_sustained"))
    msg = _painting_gap_check(age_hours=54.0, threshold_hours=12.0, deferral=ev).message
    assert "arousal reads strained (cabinet_hot, gpu_queue_sustained)" in msg
    assert "since 2026-10-18T12:37:30+00:00" in msg and "cabinet is hot" in msg
    assert msg.index("arousal") < msg.index("GPU lane controller")
    assert "not a broken GPU path" in msg


def test_gap_message_without_deferral_keeps_gpu_first_wording():
    from app.visual_chain_health_monitor import _painting_gap_check

    msg = _painting_gap_check(age_hours=27.6, threshold_hours=12.0, deferral=None).message
    assert msg.index("GPU lane controller") < msg.index("wedged") and "arousal" not in msg


def _fresh_deferral(age_sec=60.0):
    from datetime import datetime, timedelta, timezone

    last = datetime.now(timezone.utc) - timedelta(seconds=age_sec)
    return {**LIVE_DEFERRAL, "last_at": last.isoformat()}


def _gap_message(deferral, view):
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get", return_value=_pending([])
    ), patch("app.store.visual_baseline_deferral", lambda: deferral), patch(
        "orion.regulation.arousal_read.read_arousal_sync", return_value=view
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        VisualChainHealthMonitor(_settings()).record_painting_gap(age_hours=54.0)
        return client_cls.return_value.attention_request.call_args.kwargs["message"]


STRAINED = ("cabinet_hot", "gpu_queue_sustained")


def test_monitor_reads_deferral_and_arousal_only_during_a_gap_and_sends_it():
    from orion.regulation.arousal_view import ArousalView

    reads = []

    def _deferral():
        reads.append(1)
        return _fresh_deferral()

    view = ArousalView(level="strained", reason="ok", reasons=STRAINED)
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get", return_value=_pending([])
    ), patch("app.store.visual_baseline_deferral", _deferral), patch(
        "orion.regulation.arousal_read.read_arousal_sync", return_value=view
    ):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        mon = VisualChainHealthMonitor(_settings())
        mon.record_painting_gap(age_hours=2.0)
        assert reads == []                       # healthy: no evidence read
        mon.record_painting_gap(age_hours=54.0)
        msg = client_cls.return_value.attention_request.call_args.kwargs["message"]
    assert reads == [1]
    assert "arousal reads strained (cabinet_hot, gpu_queue_sustained)" in msg


def test_stale_deferral_record_is_not_named_as_the_cause():
    """The checkpoint keeps the record on every early return (stale activity, active attempt ...):
    a deferral older than DEFERRAL_FRESH_SEC means arousal is no longer what holds the painting."""
    from app.visual_chain_health_monitor import DEFERRAL_FRESH_SEC
    from orion.regulation.arousal_view import ArousalView

    msg = _gap_message(_fresh_deferral(DEFERRAL_FRESH_SEC + 60),
                       ArousalView(level="strained", reason="ok", reasons=STRAINED))
    assert "arousal" not in msg and "GPU lane controller" in msg


@pytest.mark.parametrize("level", ["idle", "engaged", "unknown"])
def test_deferral_is_not_named_when_arousal_is_not_strained_now(level):
    from orion.regulation.arousal_view import ArousalView

    msg = _gap_message(_fresh_deferral(), ArousalView(level=level, reason="ok"))
    assert "arousal" not in msg and "GPU lane controller" in msg


def test_deferral_read_failure_falls_back_to_the_old_wording():
    with patch("app.visual_chain_health_monitor.NotifyClient") as client_cls, patch(
        "app.visual_chain_health_monitor.requests.get", return_value=_pending([])
    ), patch("app.store.visual_baseline_deferral", side_effect=RuntimeError("db down")):
        from app.visual_chain_health_monitor import VisualChainHealthMonitor

        client_cls.return_value.attention_request.return_value = MagicMock(ok=True)
        VisualChainHealthMonitor(_settings()).record_painting_gap(age_hours=54.0)
        msg = client_cls.return_value.attention_request.call_args.kwargs["message"]
    assert "GPU lane controller" in msg and "arousal" not in msg


class _DeferralConn(_FakeConn):
    def execute(self, stmt, params=None):
        self._executed.append((str(stmt), params))
        value = self._results.pop(0)
        return MagicMock(first=lambda: (value,), scalar=lambda: value)


def test_store_reads_the_checkpoint_deferral_and_tolerates_absence():
    executed: list = []
    engine = MagicMock(connect=lambda: _DeferralConn(["visual_baseline_checkpoint", dict(LIVE_DEFERRAL)], executed))
    with patch("app.store._get_engine", return_value=engine):
        from app.store import visual_baseline_deferral

        assert visual_baseline_deferral() == LIVE_DEFERRAL
    assert "state->'deferral'" in executed[1][0]
    engine = MagicMock(connect=lambda: _DeferralConn(["visual_baseline_checkpoint", None], []))
    with patch("app.store._get_engine", return_value=engine):
        from app.store import visual_baseline_deferral

        assert visual_baseline_deferral() is None     # today's checkpoint: no deferral key yet
    with patch("app.store._get_engine", side_effect=RuntimeError("db down")):
        from app.store import visual_baseline_deferral

        assert visual_baseline_deferral() is None
