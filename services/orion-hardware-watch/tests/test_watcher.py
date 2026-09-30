"""The watcher end to end against the in-memory store: alert before investigation, one incident per
subject, restart safety, shed latch + refresh, operator resolve + snooze, kill switches."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.hardware_watch.rules import CoolingPoint, TempPoint
from orion.schemas.curiosity_urgent import URGENT_REQUEST_CHANNEL, CuriosityUrgentRequestV1
from orion.schemas.hardware_watch import HARDWARE_WATCH_INCIDENT_CHANNEL, HardwareWatchIncidentV1

from app.settings import Settings
from app.store import MemoryStore
from app.watcher import Watcher

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


class Recorder:
    """Records publishes and notifies in ONE list, so order is checkable."""

    def __init__(self, notify_ok=True, publish_fails=False):
        self.log: list[tuple[str, object]] = []
        self.notify_ok = notify_ok
        self.publish_fails = publish_fails

    async def publish(self, channel, env):
        if self.publish_fails:
            raise ConnectionError("bus down")
        self.log.append((channel, env.payload))

    def notify(self, req):
        self.log.append(("notify", req))
        return SimpleNamespace(ok=self.notify_ok, detail=None if self.notify_ok else "refused")

    def on(self, channel):
        return [p for c, p in self.log if c == channel]


def settings(**kw):
    base = dict(ORION_BUS_URL="redis://x", POSTGRES_URI="x", HARDWARE_WATCH_ENABLED=True,
                HARDWARE_WATCH_SHED_ENABLED=True, HARDWARE_WATCH_URGENT_ENABLED=True,
                HARDWARE_WATCH_HEAT_NODES="athena", HARDWARE_WATCH_GPU_NODES="circe")
    base.update(kw)
    return Settings(**base)


async def inline(fn, *a, **kw):
    return fn(*a, **kw)


def make(rec=None, store=None, **kw):
    clock = Clock()
    rec = rec or Recorder()
    store = store or MemoryStore()
    w = Watcher(settings=settings(**kw), store=store, publish=rec.publish, notify=rec.notify, clock=clock,
                run_sync=inline)
    return w, store, rec, clock


def feed_cooling(store, start, end, watts, step=5, **kw):
    t = start
    while t <= end:
        w = watts(t) if callable(watts) else watts
        store.cooling.append(CoolingPoint(T0 + timedelta(seconds=t), w, kw.get("stale", False),
                                          kw.get("online", True), True))
        t += step


def feed_temp(store, node, key, start, end, value, step=30):
    t = start
    while t <= end:
        v = value(t) if callable(value) else value
        store.temps.setdefault((node, key), []).append(TempPoint(T0 + timedelta(seconds=t), v))
        t += step


def varying(t):
    return 850.0 + (int(t) // 60) % 7


def at(clock, sec):
    clock.t = T0 + timedelta(seconds=sec)


def go(coro):
    return asyncio.run(coro)


def cabinet(store, start, end, temp):
    feed_temp(store, "athena", "cabinet_temp_c", start, end, temp)


# --- AC --------------------------------------------------------------------------------------

def test_ac_failure_alerts_first_then_investigates_then_announces():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 0, varying)
    feed_cooling(store, 5, 190, 33.1)
    cabinet(store, -1000, 190, 30.2)
    at(clock, 190)
    go(w.tick())
    kinds = [c for c, _ in rec.log]
    assert kinds[:3] == ["notify", URGENT_REQUEST_CHANNEL, HARDWARE_WATCH_INCIDENT_CHANNEL]
    alert = rec.on("notify")[0]
    assert alert.severity == "critical" and alert.channels_requested == ["in_app", "email"]
    [row] = store.open_incidents()
    assert alert.dedupe_key == f"cooling:{row['incident_id']}:alert"
    urgent = CuriosityUrgentRequestV1.model_validate(rec.on(URGENT_REQUEST_CHANNEL)[0])
    assert urgent.trigger == "cooling" and urgent.subject == "cabinet_ac" and urgent.incident_id == row["incident_id"]
    ev = HardwareWatchIncidentV1.model_validate(rec.on(HARDWARE_WATCH_INCIDENT_CHANNEL)[0])
    assert ev.transition == "opened" and ev.open_reason == "low_power"
    assert ev.shed.requested and ev.shed.reason == "cabinet_elevated"
    assert ev.shed.valid_until == T0 + timedelta(seconds=190 + 300)
    assert row["alert_sent_at"] and row["urgent_requested_at"]


def test_second_tick_does_not_refire_and_restart_does_not_refire():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 0, varying)
    feed_cooling(store, 5, 400, 33.1)
    cabinet(store, -1000, 400, 30.2)
    at(clock, 190)
    go(w.tick())
    at(clock, 220)
    go(w.tick())
    # a new process: fresh Watcher, same store
    w2 = Watcher(settings=settings(), store=store, publish=rec.publish, notify=rec.notify, clock=clock, run_sync=inline)
    at(clock, 250)
    go(w2.tick())
    assert len(rec.on("notify")) == 1 and len(rec.on(URGENT_REQUEST_CHANNEL)) == 1
    assert len(store.incidents) == 1


def test_refresh_every_minute_keeps_the_pool_signal_alive():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 0, varying)
    feed_cooling(store, 5, 400, 33.1)
    cabinet(store, -1000, 400, 30.2)
    for sec in (190, 220, 250, 280):
        at(clock, sec)
        go(w.tick())
    evs = [HardwareWatchIncidentV1.model_validate(p) for p in rec.on(HARDWARE_WATCH_INCIDENT_CHANNEL)]
    assert [e.transition for e in evs] == ["opened", "refresh"]      # 190, then 250 (60 s later)
    assert evs[1].shed.valid_until == T0 + timedelta(seconds=250 + 300)


def test_recovery_resolves_and_the_pool_is_told():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 0, varying)
    feed_cooling(store, 5, 190, 33.1)
    cabinet(store, -1000, 1000, 30.2)
    at(clock, 190)
    go(w.tick())
    feed_cooling(store, 195, 800, varying)
    at(clock, 800)
    go(w.tick())
    [row] = store.list_incidents()
    assert row["status"] == "resolved" and row["resolve_reason"] == "recovered"
    last = HardwareWatchIncidentV1.model_validate(rec.on(HARDWARE_WATCH_INCIDENT_CHANNEL)[-1])
    assert last.transition == "resolved" and last.shed is not None and not last.shed.requested
    assert rec.on("notify")[-1].dedupe_key.endswith(":resolved")


def test_shed_waits_for_a_warm_cabinet_then_latches():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 0, varying)
    feed_cooling(store, 5, 2000, 33.1)
    cabinet(store, -1000, 200, 26.0)          # cool and flat: no shed at open
    at(clock, 190)
    go(w.tick())
    assert not store.open_incidents()[0]["shed_requested"]
    feed_temp(store, "athena", "cabinet_temp_c", 230, 1100, lambda t: 26.0 + (t - 200) / 900)  # +1 C / 15 min
    at(clock, 1100)
    go(w.tick())
    row = store.open_incidents()[0]
    assert row["shed_requested"] and row["shed_reason"] == "cabinet_rising"
    feed_temp(store, "athena", "cabinet_temp_c", 1130, 1500, 25.0)   # cools again: latch holds
    at(clock, 1500)
    go(w.tick())
    assert store.open_incidents()[0]["shed_requested"]


def test_shed_kill_switch_never_requests():
    w, store, rec, clock = make(HARDWARE_WATCH_SHED_ENABLED=False)
    feed_cooling(store, -4000, 0, varying)
    feed_cooling(store, 5, 190, 33.1)
    cabinet(store, -1000, 190, 31.0)
    at(clock, 190)
    go(w.tick())
    ev = HardwareWatchIncidentV1.model_validate(rec.on(HARDWARE_WATCH_INCIDENT_CHANNEL)[0])
    assert not ev.shed.requested and ev.shed.reason == "disabled"


def test_urgent_kill_switch_alerts_without_investigating():
    w, store, rec, clock = make(HARDWARE_WATCH_URGENT_ENABLED=False)
    feed_cooling(store, -4000, 0, varying)
    feed_cooling(store, 5, 190, 33.1)
    cabinet(store, -1000, 190, 30.0)
    at(clock, 190)
    go(w.tick())
    assert rec.on("notify") and not rec.on(URGENT_REQUEST_CHANNEL)


def test_watcher_off_does_nothing():
    w, store, rec, clock = make(HARDWARE_WATCH_ENABLED=False)
    at(clock, 190)
    go(w.tick())
    assert rec.log == [] and store.incidents == {}


def test_failed_alert_is_retried_next_tick_and_not_after_success():
    rec = Recorder(notify_ok=False)
    w, store, rec, clock = make(rec=rec)
    feed_cooling(store, -4000, 0, varying)
    feed_cooling(store, 5, 400, 33.1)
    cabinet(store, -1000, 400, 30.0)
    at(clock, 190)
    go(w.tick())
    assert store.open_incidents()[0]["alert_error"] == "refused"
    rec.notify_ok = True
    at(clock, 220)
    go(w.tick())
    at(clock, 250)
    go(w.tick())
    alerts = [r for r in rec.on("notify") if r.event_kind == "hardware.watch.cooling.alert"]
    row = store.open_incidents()[0]
    assert row["alert_sent_at"] == T0 + timedelta(seconds=220)
    assert len(alerts) == row["alert_attempts"]   # stopped once it succeeded
    assert rec.log.index(("notify", alerts[0])) < [c for c, _ in rec.log].index(URGENT_REQUEST_CHANNEL)


def test_operator_resolve_clears_and_snoozes_reopen():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 0, varying)
    feed_cooling(store, 5, 2000, 33.1)          # still broken after the resolve
    cabinet(store, -1000, 2000, 30.0)
    at(clock, 190)
    go(w.tick())
    inc = store.open_incidents()[0]["incident_id"]
    at(clock, 300)
    row = go(w.resolve_by_operator(inc))
    assert row["status"] == "resolved" and row["resolve_reason"] == "operator"
    at(clock, 330)
    go(w.tick())
    assert store.open_incidents() == []          # snoozed for an hour
    at(clock, 300 + 3600 + 1)
    feed_cooling(store, 2005, 3905, 33.1)
    cabinet(store, 2005, 3905, 30.0)
    go(w.tick())
    assert len(store.open_incidents()) == 1


def test_simulated_incident_only_ends_by_operator():
    w, store, rec, clock = make(HARDWARE_WATCH_TEST_HOOK_ENABLED=True)
    feed_cooling(store, -4000, 1000, varying)   # the AC is fine
    cabinet(store, -1000, 1000, 30.0)
    at(clock, 100)
    row = go(w.simulate_cooling())
    assert row["open_reason"] == "simulated" and rec.on("notify")[0].title.startswith("SIMULATED")
    at(clock, 1000)
    go(w.tick())
    assert store.open_incidents()[0]["incident_id"] == row["incident_id"]
    assert go(w.simulate_cooling()) is None      # one open per subject


def test_bus_down_urgent_request_is_retried():
    rec = Recorder(publish_fails=True)
    w, store, rec, clock = make(rec=rec)
    feed_cooling(store, -4000, 0, varying)
    feed_cooling(store, 5, 400, 33.1)
    cabinet(store, -1000, 400, 30.0)
    at(clock, 190)
    go(w.tick())
    assert store.open_incidents()[0]["urgent_error"]
    rec.publish_fails = False
    at(clock, 220)
    go(w.tick())
    assert len(rec.on(URGENT_REQUEST_CHANNEL)) == 1 and store.open_incidents()[0]["urgent_requested_at"]


# --- heat ------------------------------------------------------------------------------------

def test_cpu_heat_opens_one_incident_and_resolves_below_p75():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 1500, varying)
    feed_temp(store, "athena", "temp_c_max", -3 * 86400, 0, lambda t: 55.0 + (int(t) // 30) % 20)  # 55..74
    feed_temp(store, "athena", "temp_c_max", 30, 700, 80.0)
    at(clock, 700)
    go(w.tick())
    [row] = store.open_incidents()
    assert (row["rule"], row["subject"], row["open_reason"]) == ("cpu_heat", "athena", "above_p95")
    urgent = CuriosityUrgentRequestV1.model_validate(rec.on(URGENT_REQUEST_CHANNEL)[0])
    assert urgent.trigger == "heat" and "p95" in urgent.question
    assert not rec.on("notify")                      # heat: the investigation's report is the notice
    feed_temp(store, "athena", "temp_c_max", 730, 1400, 80.0)
    at(clock, 1400)
    go(w.tick())
    assert len(store.list_incidents()) == 1          # feedback guard: still one incident
    feed_temp(store, "athena", "temp_c_max", 1430, 1460, 56.0)
    at(clock, 1460)
    go(w.tick())
    assert store.list_incidents()[0]["status"] == "resolved"


def test_gpu_ceiling_fires_without_history_and_p95_waits_for_three_days():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 300, varying)
    feed_temp(store, "circe", "gpu3_temp_c", -3600, 0, 60.0)
    feed_temp(store, "circe", "gpu3_temp_c", 30, 300, 86.0)
    feed_temp(store, "circe", "gpu1_temp_c", -3600, 300, 80.0)   # hot vs its short history, below ceiling
    at(clock, 300)
    go(w.tick())
    rows = store.open_incidents()
    assert [(r["subject"], r["open_reason"]) for r in rows] == [("circe/gpu3", "above_ceiling")]
    assert w.last.verdicts["gpu_heat:circe/gpu1"]["armed"] is False
