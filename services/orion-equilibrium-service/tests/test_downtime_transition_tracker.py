from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.schemas.telemetry.system_health import EquilibriumServiceState
from app.downtime_transition_tracker import DowntimeTransitionTracker


def _state(service: str, status: str, *, node: str | None = "athena") -> EquilibriumServiceState:
    now = datetime.now(timezone.utc)
    return EquilibriumServiceState(
        service=service,
        node=node,
        status=status,
        last_seen_ts=now,
        heartbeat_interval_sec=10.0,
        down_for_ms=0,
        uptime_pct={"60": 1.0},
    )


def test_first_observation_seeds_state_without_firing() -> None:
    """A service seen for the first time must not fire a transition -- there
    is no real "from" status to report, and treating first-sight as a
    transition would spuriously fire on every tracked service the moment the
    tracker (re)boots.
    """
    tracker = DowntimeTransitionTracker()
    now = datetime.now(timezone.utc)

    transitions = tracker.detect(
        [_state("recall", "ok")],
        now=now,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )

    assert transitions == []


def test_repeated_same_status_does_not_fire() -> None:
    """The critical off-by-one/duplicate-firing case: two consecutive ticks
    reporting the same status for the same service must not produce a
    transition event, even though detect() is called on both ticks.
    """
    tracker = DowntimeTransitionTracker()
    now = datetime.now(timezone.utc)

    tracker.detect(
        [_state("recall", "ok")],
        now=now,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )
    transitions = tracker.detect(
        [_state("recall", "ok")],
        now=now + timedelta(seconds=15),
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )

    assert transitions == []


def test_ok_to_down_transition_fires_and_starts_down_since() -> None:
    tracker = DowntimeTransitionTracker()
    t0 = datetime.now(timezone.utc)

    tracker.detect(
        [_state("recall", "ok")],
        now=t0,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )
    t1 = t0 + timedelta(seconds=15)
    transitions = tracker.detect(
        [_state("recall", "down")],
        now=t1,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )

    assert len(transitions) == 1
    ev = transitions[0]
    assert ev.from_status == "ok"
    assert ev.to_status == "down"
    assert ev.transitioned_at == t1
    # Not a recovery -- no duration/since fields populated yet.
    assert ev.down_since_ts is None
    assert ev.down_duration_ms is None


def test_down_to_ok_transition_fires_with_duration_and_since() -> None:
    tracker = DowntimeTransitionTracker()
    t0 = datetime.now(timezone.utc)

    tracker.detect(
        [_state("recall", "ok")],
        now=t0,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )
    t1 = t0 + timedelta(seconds=15)
    tracker.detect(
        [_state("recall", "down")],
        now=t1,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )
    t2 = t1 + timedelta(seconds=45)
    transitions = tracker.detect(
        [_state("recall", "ok")],
        now=t2,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )

    assert len(transitions) == 1
    ev = transitions[0]
    assert ev.from_status == "down"
    assert ev.to_status == "ok"
    assert ev.down_since_ts == t1
    assert ev.down_duration_ms == 45_000


def test_ok_degraded_ok_fires_twice_with_no_down_fields() -> None:
    """ok -> degraded -> ok: two real transitions, neither one a downtime
    episode (degraded is never "down"), so neither carries down_since_ts/
    down_duration_ms.
    """
    tracker = DowntimeTransitionTracker()
    t0 = datetime.now(timezone.utc)

    tracker.detect(
        [_state("recall", "ok")],
        now=t0,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )
    t1 = t0 + timedelta(seconds=15)
    first = tracker.detect(
        [_state("recall", "degraded")],
        now=t1,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )
    t2 = t1 + timedelta(seconds=15)
    second = tracker.detect(
        [_state("recall", "ok")],
        now=t2,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )

    assert len(first) == 1
    assert first[0].from_status == "ok"
    assert first[0].to_status == "degraded"
    assert first[0].down_duration_ms is None

    assert len(second) == 1
    assert second[0].from_status == "degraded"
    assert second[0].to_status == "ok"
    assert second[0].down_duration_ms is None


def test_transitions_are_tracked_independently_per_service_and_node() -> None:
    """Two services (and the same service on two different nodes) must not
    share tracking state -- a status change in one must not fire or suppress
    a transition for the other.
    """
    tracker = DowntimeTransitionTracker()
    t0 = datetime.now(timezone.utc)

    tracker.detect(
        [
            _state("recall", "ok", node="athena"),
            _state("recall", "ok", node="atlas"),
            _state("sql-writer", "ok", node="athena"),
        ],
        now=t0,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )

    t1 = t0 + timedelta(seconds=15)
    transitions = tracker.detect(
        [
            _state("recall", "down", node="athena"),
            _state("recall", "ok", node="atlas"),
            _state("sql-writer", "ok", node="athena"),
        ],
        now=t1,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )

    assert len(transitions) == 1
    assert transitions[0].service == "recall"
    assert transitions[0].node == "athena"
    assert transitions[0].to_status == "down"


def test_service_missing_from_a_tick_then_reappearing_uses_real_down_since() -> None:
    """A service can drop out of `states` entirely for one or more ticks
    (e.g. purged by service.py's state_retention_sec pruning, and not in
    EQUILIBRIUM_EXPECTED_SERVICES so it is not force-re-added). The tracker
    holds no per-tick "who did I see" bookkeeping beyond `_last_status`/
    `_down_since`, so a gap tick must be a pure no-op for that key -- and
    when the service reappears, the transition it fires (if any) must still
    be computed against the *original* down_since, not something reset by
    the gap.
    """
    tracker = DowntimeTransitionTracker()
    t0 = datetime.now(timezone.utc)

    tracker.detect(
        [_state("recall", "ok")],
        now=t0,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )
    t1 = t0 + timedelta(seconds=15)
    tracker.detect(
        [_state("recall", "down")],
        now=t1,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )

    # recall drops out of the states list entirely for a few ticks (purged
    # upstream) -- must not raise, and must not touch recall's tracked state.
    t2 = t1 + timedelta(seconds=15)
    gap_transitions = tracker.detect(
        [],
        now=t2,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )
    assert gap_transitions == []

    # recall reappears, recovered -- duration must span the FULL real gap
    # (t1 -> t3), including the untracked window, not just since t2.
    t3 = t2 + timedelta(seconds=30)
    recovery = tracker.detect(
        [_state("recall", "ok")],
        now=t3,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )

    assert len(recovery) == 1
    assert recovery[0].from_status == "down"
    assert recovery[0].to_status == "ok"
    assert recovery[0].down_since_ts == t1
    assert recovery[0].down_duration_ms == int((t3 - t1).total_seconds() * 1000)


def test_transition_id_is_unique_per_event() -> None:
    tracker = DowntimeTransitionTracker()
    t0 = datetime.now(timezone.utc)

    tracker.detect(
        [_state("recall", "ok")],
        now=t0,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )
    t1 = t0 + timedelta(seconds=15)
    a = tracker.detect(
        [_state("recall", "down")],
        now=t1,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )
    t2 = t1 + timedelta(seconds=15)
    b = tracker.detect(
        [_state("recall", "ok")],
        now=t2,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )

    assert a[0].transition_id != b[0].transition_id
