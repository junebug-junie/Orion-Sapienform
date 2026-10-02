"""Replay of the 2026-10-02 incident through the stability checks.

Numbers are the real ones read off the live stack that day: orion-bus-mirror
RestartCount 576 -> 581 at its observed ~20-minute crash cadence (restart
timestamps from `docker logs`), Redis client_output_buffer_limit_disconnections
rising in lockstep, FalkorDB's BGSAVE child running 136,812s, and 118,112
Channel nodes in orion_bus_synapse against a 332-entry catalog. The healthy
baseline is the same stack measured after the fix (03:58Z).

Measures, per check: does it fire on the incident, how many 60s cycles until
it does, and does it stay silent on the healthy baseline. The roster probes
that existed before this patch fired zero cards across the 8-day incident.
"""
from __future__ import annotations

from app.stability import (
    AlertGate,
    CounterRiseTracker,
    CrashLoopTracker,
    graph_inflation_alert,
    slow_consumer_alert,
    snapshot_alerts,
)

T0 = 1_790_904_000.0  # ~2026-10-02T01:00Z
CYCLE = 60.0
# Restart moments from the mirror's logs (minutes after T0): 01:09, 01:31, 01:56, 02:17, 02:39, 02:59.
RESTART_MINUTES = [9, 31, 56, 77, 99, 119]
BASE_RESTARTS = 575

STUCK_FALKOR = {
    "rdb_bgsave_in_progress": 1,
    "rdb_current_bgsave_time_sec": 136_812,
    "rdb_last_bgsave_status": "ok",
    "rdb_last_save_time": 1_790_774_214,
    "rdb_changes_since_last_save": 5_000_000,
}
HEALTHY_PERSISTENCE = {
    "rdb_bgsave_in_progress": 0,
    "rdb_current_bgsave_time_sec": -1,
    "rdb_last_bgsave_status": "ok",
    "rdb_changes_since_last_save": 7363,
}
SAVE_CONFIG = "3600 1 300 100 60 10000"


def _replay(*, incident: bool, cycles: int = 180) -> dict[str, int | None]:
    """Returns, per alert kind, the first cycle index at which it fired (None = never)."""
    crash = CrashLoopTracker()
    disconnects = CounterRiseTracker()
    gate = AlertGate()
    first: dict[str, int | None] = {
        "crash_loop": None,
        "slow_consumer_kill": None,
        "snapshot_stuck": None,
        "graph_inflation": None,
    }
    for cycle in range(cycles):
        now = T0 + cycle * CYCLE
        minute = cycle * CYCLE / 60.0
        if incident:
            restarts = BASE_RESTARTS + sum(1 for m in RESTART_MINUTES if m <= minute)
            persistence = {**STUCK_FALKOR, "rdb_current_bgsave_time_sec": 136_812 + int(cycle * CYCLE)}
            channels = 118_112
        else:
            restarts = 5
            persistence = {**HEALTHY_PERSISTENCE, "rdb_last_save_time": now - 70}
            channels = 242
        alerts = []
        alerts += crash.observe({"orion-athena-bus-mirror": restarts, "orion-athena-hub": 2}, now)
        rose = disconnects.observe(restarts)  # disconnections tracked restarts 1:1 that day
        alerts += slow_consumer_alert("bus-redis", rose, restarts)
        alerts += snapshot_alerts("falkordb", persistence, save_config=SAVE_CONFIG, now=now)
        alerts += graph_inflation_alert("orion_bus_synapse", channels, 332)
        for alert in gate.admit(alerts, now):
            if first[alert.kind] is None:
                first[alert.kind] = cycle
    return first


def test_every_check_fires_on_the_incident() -> None:
    first = _replay(incident=True)
    report = {kind: (f"cycle {c} ({c} min)" if c is not None else "NEVER") for kind, c in first.items()}
    print("\n10-02 incident replay, first alert per check:", report)
    assert all(c is not None for c in first.values()), report


def test_state_checks_fire_on_the_first_cycle() -> None:
    first = _replay(incident=True)
    # Snapshot and graph state are visible immediately; no history needed.
    assert first["snapshot_stuck"] == 0
    assert first["graph_inflation"] == 0


def test_event_checks_fire_on_the_first_event_after_baseline() -> None:
    first = _replay(incident=True)
    # First disconnection after the guardian's baseline: 01:09 -> cycle 9.
    assert first["slow_consumer_kill"] == 9
    # Third restart inside the 2h window: 01:56 -> cycle 56.
    assert first["crash_loop"] == 56


def test_healthy_baseline_is_silent() -> None:
    first = _replay(incident=False)
    assert all(c is None for c in first.values()), first
