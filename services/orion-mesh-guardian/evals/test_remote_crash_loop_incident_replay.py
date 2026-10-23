"""Replay of the 2026-10-14..23 circe incident through the remote crash-loop path.

orion-circe-affectgpt-worker crash-looped on circe (torch CUDA OOM at model load; GPU index "1" had
become the agent llama-server's card) from 2026-10-14T03:00Z until it was stopped
2026-10-23T22:27Z: RestartCount 601, exit code 3. Nothing alerted: the guardian's crash-loop check
read only athena's docker socket.

The restart cadence is replayed two ways, because the two numbers we have disagree: the operator
report says "every ~22 s", while 601 restarts over 235.45 h averages one per ~23.5 min. The slow
cadence is the conservative case (fewest restarts per window), so it bounds time-to-card.

The path replayed is the real one: circe's gpu-lane-controller reads RestartCount every 30 s
(ContainerRestartsV1 on its heartbeat) -> the equilibrium snapshot carries that heartbeat's details
-> the guardian's 60 s stability cycle reads the latest snapshot. Each hop's lag is modelled by the
age of the reading the guardian sees (census age up to 30 s + snapshot 15 s).

Measures: cycles until the first crash_loop card on the circe-qualified key, that it lands within
one CRASH_LOOP_WINDOW_SEC, how many cards the whole 9.8 days would have raised (AlertGate), that the
healthy baseline is silent, and that circe going silent mid-incident is a blind card, not calm.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from app.stability import (
    CRASH_LOOP_WINDOW_SEC,
    AlertGate,
    CrashLoopTracker,
    qualify,
    remote_restart_counts,
)
from orion.schemas.telemetry.system_health import CONTAINER_RESTARTS_DETAIL_KEY

T0 = datetime(2026, 10, 14, 3, 0, tzinfo=timezone.utc).timestamp()
STOPPED = datetime(2026, 10, 23, 22, 27, tzinfo=timezone.utc).timestamp()
TOTAL_RESTARTS = 601
CYCLE = 60.0
CENSUS_EVERY = 30.0
SNAPSHOT_LAG = 15.0
KEY = "orion-circe-affectgpt-worker@5e1f0c3a9b2d"
SLOW_GAP = (STOPPED - T0) / TOTAL_RESTARTS  # ~1410 s
FAST_GAP = 22.0


def _snapshot(now: float, count: int, *, frozen_at: float | None = None) -> SimpleNamespace:
    """What the guardian sees at ``now``: the newest census, CENSUS_EVERY-quantized, SNAPSHOT_LAG late.
    ``frozen_at``: the reporter stopped heartbeating then; equilibrium keeps serving its last
    record (and that record's details) until retention purges it, so the reading just ages."""
    read_at = (now - SNAPSHOT_LAG) // CENSUS_EVERY * CENSUS_EVERY
    if frozen_at is not None:
        read_at = min(read_at, (frozen_at - SNAPSHOT_LAG) // CENSUS_EVERY * CENSUS_EVERY)
    details = {CONTAINER_RESTARTS_DETAIL_KEY: {
        "read_at": datetime.fromtimestamp(read_at, tz=timezone.utc).isoformat(),
        "counts": {KEY: count, "orion-circe-biometrics@aaaaaaaaaaaa": 2},
    }}
    return SimpleNamespace(services=[SimpleNamespace(service="gpu-lane-controller", node="circe", details=details)])


def _replay(*, gap: float | None, cycles: int, silent_from: int | None = None) -> dict:
    tracker, gate = CrashLoopTracker(), AlertGate()
    first: dict[str, int | None] = {"crash_loop": None, "container_watch_blind": None}
    cards: list[tuple[int, str]] = []
    for cycle in range(cycles):
        now = T0 + cycle * CYCLE
        seen_at = (now - SNAPSHOT_LAG) // CENSUS_EVERY * CENSUS_EVERY
        count = 0 if gap is None else min(TOTAL_RESTARTS, int(max(seen_at - T0, 0) // gap))
        frozen_at = None if silent_from is None else T0 + silent_from * CYCLE
        census = remote_restart_counts(
            _snapshot(now, count, frozen_at=frozen_at).services,
            expected_hosts=["circe"], local_node="athena", now=now, watching_since=T0 - 3600,
        )
        local = qualify("athena", {"orion-athena-hub@1234567890ab": 2})
        alerts = census.alerts + tracker.observe(
            {**local, **census.counts}, now, read_hosts={"athena", *census.fresh_hosts}
        )
        for alert in gate.admit(alerts, now):
            cards.append((cycle, alert.key))
            if first[alert.kind] is None:
                first[alert.kind] = cycle
    return {"first": first, "cards": cards}


def test_conservative_cadence_alerts_within_one_window() -> None:
    out = _replay(gap=SLOW_GAP, cycles=int(CRASH_LOOP_WINDOW_SEC / CYCLE) + 60)
    first = out["first"]["crash_loop"]
    print(f"\n10-14 circe replay, one restart per {SLOW_GAP / 60:.1f} min: first card at cycle {first} ({first} min)")
    assert first is not None
    assert first * CYCLE <= CRASH_LOOP_WINDOW_SEC
    assert out["cards"][0][1] == "crash_loop:circe/orion-circe-affectgpt-worker"


def test_reported_fast_cadence_alerts_within_three_cycles() -> None:
    first = _replay(gap=FAST_GAP, cycles=30)["first"]["crash_loop"]
    print(f"\n10-14 circe replay, one restart per {FAST_GAP:.0f} s: first card at cycle {first}")
    assert first is not None and first <= 3


def test_whole_incident_is_one_card_per_realert_window_not_silence() -> None:
    # 9.8 days at 60 s cycles; AlertGate re-raises every 6 h while it persists.
    cycles = int((STOPPED - T0) / CYCLE)
    out = _replay(gap=SLOW_GAP, cycles=cycles)
    loop_cards = [c for c in out["cards"] if c[1].startswith("crash_loop:")]
    print(f"\n10-14 circe replay, whole incident: {len(loop_cards)} cards over {cycles / 60:.0f} h (was 0)")
    assert 30 <= len(loop_cards) <= 45
    assert {k for _, k in loop_cards} == {"crash_loop:circe/orion-circe-affectgpt-worker"}


def test_healthy_baseline_is_silent() -> None:
    out = _replay(gap=None, cycles=600)
    assert out["cards"] == []


def test_circe_going_silent_is_a_blind_card_not_calm() -> None:
    out = _replay(gap=None, cycles=60, silent_from=10)
    blind = out["first"]["container_watch_blind"]
    print(f"\ncirce silent from cycle 10: blind card at cycle {blind}")
    # Frozen reading is already ~30-45 s old at cycle 10, stale past 300 s: card by cycle 15.
    assert blind is not None and 10 < blind <= 15
    assert out["first"]["crash_loop"] is None
