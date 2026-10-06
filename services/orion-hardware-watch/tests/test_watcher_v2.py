"""Thermal controller v2 through the real Watcher (in-memory store, fake clock).

Spec: docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md -- D1 (one cabinet read,
grace), D2 (per-tick reflex signal, no latch), D5 (AC power is a diagnosis), D6 (dedupe per
rule+subject window), D7 (ceilings, sensor_lost), D9 (snooze per reason)."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from orion.schemas.curiosity_urgent import URGENT_REQUEST_CHANNEL
from orion.schemas.hardware_watch import (
    HARDWARE_WATCH_INCIDENT_CHANNEL, HARDWARE_WATCH_REFLEX_SHED_CHANNEL, HardwareWatchIncidentV1,
    HardwareWatchReflexShedV1,
)

from app.store import MemoryStore
from tests.test_watcher import Recorder, T0, at, feed_cooling, feed_temp, go, settings, varying
from app.watcher import Watcher


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


async def inline(fn, *a, **kw):
    return fn(*a, **kw)


class ClockedStore(MemoryStore):
    """MemoryStore that never shows a reading from the fake clock's future (like the replay store)."""

    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def cooling_points(self, since):
        return [p for p in super().cooling_points(since) if p.ts <= self.clock()]

    def temp_points(self, node, key, since):
        return [p for p in super().temp_points(node, key, since) if p.ts <= self.clock()]


def make(store=None, rec=None, **kw):
    clock = Clock()
    rec = rec or Recorder()
    store = store or ClockedStore(clock)
    w = Watcher(settings=settings(HARDWARE_WATCH_HEAT_CONTROLLER="v2", **kw), store=store, publish=rec.publish,
                notify=rec.notify, clock=clock, run_sync=inline)
    return w, store, rec, clock


def cycling_cool_night(t):
    """A healthy AC on a cool night: ~60 s compressor (750 W), ~190 s fan-only (104 W) -> ~280 W mean."""
    return 750.0 if int(t) % 250 < 60 else 104.0


def dead_ac(t):
    """A dead AC: ~100 W fan-only, with the plug's small jitter (a constant would read as frozen)."""
    return 100.0 + (int(t) // 5) % 3 * 0.3


def reflex(rec):
    return [HardwareWatchReflexShedV1.model_validate(p) for p in rec.on(HARDWARE_WATCH_REFLEX_SHED_CHANNEL)]


def cab(store, start, end, temp):
    feed_temp(store, "athena", "cabinet_temp_c", start, end, temp)


# --- D5: cool night ---------------------------------------------------------------------------

def test_cool_night_duty_cycle_opens_nothing_and_sheds_nothing():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 3600, cycling_cool_night)
    cab(store, -4000, 3600, lambda t: 25.6 + (1.0 if t > 1800 else 0.0))   # the 10-06 "rise" (C2)
    for sec in range(0, 3601, 30):
        at(clock, sec)
        go(w.tick())
    assert store.list_incidents() == [] and not rec.on("notify") and reflex(rec) == []
    assert w.last.verdicts["cooling"]["ac_low"] is False


def test_ac_dead_on_a_cool_night_still_opens_nothing():
    """AC power alone never opens (C1): the cabinet is the hazard."""
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 1200, dead_ac)
    cab(store, -4000, 1200, 26.0)
    at(clock, 1200)
    go(w.tick())
    assert store.list_incidents() == [] and w.last.verdicts["cooling"]["ac_low"] is True


def test_ac_low_with_a_warm_cabinet_opens_an_alert_only_incident():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 1200, dead_ac)
    cab(store, -4000, 1200, 30.2)
    at(clock, 1200)
    go(w.tick())
    [row] = store.open_incidents()
    assert row["open_reason"] == "low_power" and not row["shed_requested"]
    [ev] = [HardwareWatchIncidentV1.model_validate(p) for p in rec.on(HARDWARE_WATCH_INCIDENT_CHANNEL)]
    assert ev.shed is None                                  # v2 incidents never shed
    [alert] = [n for n in rec.on("notify") if n.severity == "critical"]
    assert "15-minute mean" in alert.body_text and "34 C" in alert.body_text
    assert reflex(rec) == []                                # 30.2 C: below critical, no reflex


def test_low_power_resolves_only_at_one_and_a_half_times_the_floor():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 1200, dead_ac)
    cab(store, -4000, 4000, 30.2)
    at(clock, 1200)
    go(w.tick())
    feed_cooling(store, 1205, 2200, 180.0)        # above the 140 W floor, below 210 W
    at(clock, 2200)
    go(w.tick())
    assert store.open_incidents(), "must not resolve between floor and 1.5x floor (C4)"
    feed_cooling(store, 2205, 3200, 400.0)
    at(clock, 3200)
    go(w.tick())
    assert store.open_incidents() == []


# --- D1/D2: the reflex -------------------------------------------------------------------------

def test_critical_sheds_every_tick_and_clears_once_below_rearm():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 1000, varying)
    cab(store, -1800, 0, 33.5)
    for sec in (0, 300, 330, 600, 630, 660, 690):
        if sec == 300:
            cab(store, 30, 300, 34.2)
        elif sec == 330:
            cab(store, 330, 330, 33.4)   # held: above the 33 C re-arm
        elif sec == 600:
            cab(store, 360, 600, 33.4)
        elif sec >= 630:
            cab(store, sec, sec, 32.5)   # below re-arm (still hot by the 32 C line: no reflex)
        at(clock, sec)
        go(w.tick())
    sigs = reflex(rec)
    assert [(s.active, s.reason) for s in sigs] == [(True, "cabinet_hot")] * 3 + [(False, None)]
    assert all(s.valid_until == s.emitted_at + timedelta(seconds=90) for s in sigs if s.active)
    assert w.reflex_snapshot()["active"] is False and w.reflex_snapshot()["cabinet"]["thermal_state"] == "hot"


def test_32_to_34_is_not_a_reflex_shed():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 0, varying)
    cab(store, -1800, 0, 33.9)
    at(clock, 0)
    go(w.tick())
    assert reflex(rec) == [] and w.reflex_snapshot()["cabinet"]["thermal_state"] == "hot"


def test_one_failed_cabinet_query_is_not_unknown_but_grace_expiry_is():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 1000, varying)
    cab(store, -1800, 0, 30.0)
    at(clock, 0)
    go(w.tick())
    real = store.temp_points
    store.temp_points = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("pg down"))
    at(clock, 60)
    go(w.tick())
    assert w.last.verdicts["cabinet"]["thermal_state"] == "elevated" and reflex(rec) == []
    assert "cabinet_read" in w.last.errors
    at(clock, 400)                        # 400 s since the last reading: past the 300 s grace
    go(w.tick())
    [sig] = reflex(rec)
    assert sig.active and sig.reason == "cabinet_unknown"
    store.temp_points = real
    cab(store, 410, 500, 30.0)            # readings resume
    at(clock, 500)
    go(w.tick())
    assert [(s.active, s.reason) for s in reflex(rec)][-1] == (False, None)


def test_sensor_dead_and_ac_low_escalates_to_cabinet_hot():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 1200, dead_ac)
    cab(store, -1800, 0, 29.0)
    at(clock, 1200)
    go(w.tick())
    [sig] = reflex(rec)
    assert sig.reason == "cabinet_hot"
    [row] = store.open_incidents()                       # unknown counts as elevated: AC low opens
    assert row["open_reason"] == "low_power"


def test_monitoring_outage_is_cabinet_unknown_not_hot():
    """10-03: AC plug and cabinet both silent -> alert (no_samples) + background-only shed."""
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 0, varying)
    cab(store, -1800, 0, 25.0)
    at(clock, 400)
    go(w.tick())
    [sig] = reflex(rec)
    assert sig.reason == "cabinet_unknown"
    [row] = store.open_incidents()
    assert row["open_reason"] == "no_samples"
    assert [n.severity for n in rec.on("notify")] == ["critical"]


def test_shed_switch_off_publishes_nothing_but_says_what_it_would_do():
    w, store, rec, clock = make(HARDWARE_WATCH_SHED_ENABLED=False)
    feed_cooling(store, -4000, 0, varying)
    cab(store, -1800, 0, 34.5)
    at(clock, 0)
    go(w.tick())
    assert reflex(rec) == [] and w.reflex_snapshot()["would_shed"] == "cabinet_hot"


def test_reflex_schema_refuses_an_active_signal_without_reason_or_expiry():
    with pytest.raises(ValidationError):
        HardwareWatchReflexShedV1(source_id="s", active=True, reason=None, valid_until=T0)
    with pytest.raises(ValidationError):
        HardwareWatchReflexShedV1(source_id="s", active=True, reason="cabinet_hot")
    with pytest.raises(ValidationError):
        HardwareWatchReflexShedV1(source_id="s", active=True, reason="cooling_incident", valid_until=T0)


# --- D6: dedupe per rule+subject across a sliding window ----------------------------------------

def _open_and_close(w, store, clock, start, warm=True):
    feed_cooling(store, start - 4000, start + 1200, dead_ac)
    cab(store, start - 4000, start + 3000, 30.2 if warm else 26.0)
    at(clock, start + 1200)
    go(w.tick())
    feed_cooling(store, start + 1205, start + 2400, 600.0)
    at(clock, start + 2400)
    go(w.tick())


def test_second_incident_within_six_hours_sends_no_email_and_no_investigation():
    w, store, rec, clock = make()
    _open_and_close(w, store, clock, 0)
    _open_and_close(w, store, clock, 4 * 3600)
    incs = store.list_incidents()
    assert len(incs) == 2
    assert len([n for n in rec.on("notify") if n.severity == "critical"]) == 1
    assert len(rec.on(URGENT_REQUEST_CHANNEL)) == 1
    second = max(incs, key=lambda r: r["opened_at"])
    assert second["alert_error"].startswith("deduped:") and second["urgent_error"].startswith("deduped:")
    # no "closed" notice for the deduped incident either
    assert len([n for n in rec.on("notify") if n.severity == "info"]) == 1
    _open_and_close(w, store, clock, 7 * 3600)          # 7 h after the first alert: a new one
    assert len([n for n in rec.on("notify") if n.severity == "critical"]) == 2


# --- D9: snooze only the reason resolved ----------------------------------------------------

def test_operator_resolving_low_power_does_not_silence_device_offline():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 1200, dead_ac)
    cab(store, -4000, 6000, 30.2)
    at(clock, 1200)
    go(w.tick())
    [row] = store.open_incidents()
    go(w.resolve_by_operator(row["incident_id"]))
    at(clock, 1300)
    go(w.tick())
    assert store.open_incidents() == []                  # low_power is snoozed
    feed_cooling(store, 1205, 1800, dead_ac, online=False)
    at(clock, 1800)
    go(w.tick())
    [row2] = store.open_incidents()
    assert row2["open_reason"] == "device_offline"


# --- D7: heat --------------------------------------------------------------------------------

def test_gpu_75c_far_above_its_p95_is_no_incident_and_85c_is():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 1000, varying)
    feed_temp(store, "circe", "gpu3_temp_c", -4 * 86400, 0, lambda t: 50.0 + (int(t) // 30) % 8)
    feed_temp(store, "circe", "gpu3_temp_c", 30, 700, 78.0)
    at(clock, 700)
    go(w.tick())
    assert store.open_incidents() == [] and w.last.verdicts["gpu_heat:circe/gpu3"]["above_p95_c"] > 15


def test_heat_incident_with_a_silent_sensor_resolves_sensor_lost():
    w, store, rec, clock = make()
    feed_cooling(store, -4000, 3000, varying)
    feed_temp(store, "circe", "gpu3_temp_c", -3600, 300, 87.0)
    at(clock, 300)
    go(w.tick())
    [row] = store.open_incidents()
    assert row["open_reason"] == "above_ceiling"
    at(clock, 300 + 600)
    go(w.tick())
    assert store.open_incidents()                         # 10 min silent: still open
    at(clock, 300 + 960)
    go(w.tick())
    assert store.open_incidents() == []
    assert store.get_incident(row["incident_id"])["resolve_reason"] == "sensor_lost"
