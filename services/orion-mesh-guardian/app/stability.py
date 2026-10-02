"""Host-wide stability checks the per-service probes cannot see.

The roster probes ask "is this service answering right now?". On 2026-10-02
that question stayed "yes" for 8 days while orion-bus-mirror restarted 581
times (it was back up within seconds every ~20 min), Redis disconnected it as
a slow pub/sub consumer each time (dropping its queued messages), and
FalkorDB's snapshot child sat deadlocked for 38h with nothing persisted. Each
check below watches a signal that moved during that incident and is flat on a
healthy stack:

- crash loop: Docker ``RestartCount`` rising, any container
- slow-consumer kill: Redis ``client_output_buffer_limit_disconnections`` rising
- snapshot: Redis/FalkorDB ``BGSAVE`` stuck, failed, or long overdue
- graph inflation: bus-synapse Channel nodes far above the channel catalog

Evaluation is pure (``*Tracker.observe`` / ``*_alerts``) so the incident can be
replayed in tests; collection is the thin async layer in ``service.py``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

# Fixed thresholds, chosen against the 10-02 incident and the healthy baseline
# measured the same day (highest RestartCount on the host: 5, flat for days).
CRASH_LOOP_RESTARTS = 3
CRASH_LOOP_WINDOW_SEC = 2 * 3600
SNAPSHOT_STUCK_SEC = 15 * 60
SNAPSHOT_STALE_SEC = 6 * 3600
GRAPH_INFLATION_RATIO = 2.0
REALERT_AFTER_SEC = 6 * 3600


@dataclass(frozen=True)
class StabilityAlert:
    key: str  # dedup identity: same key re-alerts only after REALERT_AFTER_SEC
    subject: str  # what the card is about (container / redis instance / graph)
    kind: str
    severity: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


class CrashLoopTracker:
    """Alerts when a container's RestartCount rises by CRASH_LOOP_RESTARTS
    within CRASH_LOOP_WINDOW_SEC. A count that goes down means the container
    was recreated (counter reset), so its history restarts from there."""

    def __init__(self) -> None:
        self._samples: dict[str, deque[tuple[float, int]]] = {}

    def observe(self, restart_counts: dict[str, int], now: float) -> list[StabilityAlert]:
        alerts: list[StabilityAlert] = []
        for name, count in restart_counts.items():
            samples = self._samples.setdefault(name, deque())
            if samples and count < samples[-1][1]:
                samples.clear()
            samples.append((now, count))
            while samples and samples[0][0] < now - CRASH_LOOP_WINDOW_SEC:
                samples.popleft()
            rose = count - samples[0][1]
            if rose >= CRASH_LOOP_RESTARTS:
                minutes = max((now - samples[0][0]) / 60.0, 1.0)
                alerts.append(
                    StabilityAlert(
                        key=f"crash_loop:{name}",
                        subject=name,
                        kind="crash_loop",
                        severity="error",
                        message=(
                            f"{name} restarted {rose} times in the last {minutes:.0f} min "
                            f"(total {count}). It looks 'Up' between restarts; check "
                            f"`docker logs {name}` for the exit reason."
                        ),
                        context={"restart_count": count, "restarts_in_window": rose},
                    )
                )
        for gone in set(self._samples) - set(restart_counts):
            del self._samples[gone]
        return alerts


class CounterRiseTracker:
    """Alerts when a monotonic counter rises between observations. The first
    observation only sets the baseline; a drop (server restart) re-baselines."""

    def __init__(self) -> None:
        self._last: int | None = None

    def observe(self, value: int) -> int:
        """Returns how much the counter rose since the last observation (0 if not)."""
        last, self._last = self._last, value
        if last is None or value < last:
            return 0
        return value - last


def slow_consumer_alert(instance: str, rose: int, total: int) -> list[StabilityAlert]:
    if rose <= 0:
        return []
    return [
        StabilityAlert(
            key=f"slow_consumer:{instance}",
            subject=instance,
            kind="slow_consumer_kill",
            severity="error",
            message=(
                f"Redis {instance} disconnected {rose} pub/sub client(s) for falling behind "
                f"(client-output-buffer-limit; {total} total since Redis start). Their queued "
                f"messages were dropped. A consumer is too slow for its traffic -- look for a "
                f"service that is crash-looping or whose logs show 'Connection closed by server'."
            ),
            context={"disconnections_rose": rose, "disconnections_total": total},
        )
    ]


def snapshot_alerts(
    instance: str, persistence: dict[str, Any], *, save_config: str, now: float
) -> list[StabilityAlert]:
    """``persistence`` is Redis ``INFO persistence``. ``save_config`` is
    ``CONFIG GET save``; empty means RDB snapshots are off, so "overdue" does
    not apply (a stuck or failed save still does)."""
    alerts: list[StabilityAlert] = []
    running_sec = int(persistence.get("rdb_current_bgsave_time_sec", -1) or -1)
    if int(persistence.get("rdb_bgsave_in_progress", 0) or 0) and running_sec >= SNAPSHOT_STUCK_SEC:
        alerts.append(
            StabilityAlert(
                key=f"snapshot_stuck:{instance}",
                subject=instance,
                kind="snapshot_stuck",
                severity="critical",
                message=(
                    f"{instance}'s background save has been running for {running_sec / 3600:.1f}h "
                    f"(normally ~1s). Nothing new is persisted to disk; a restart would lose "
                    f"everything since the last good save. The save child is likely deadlocked: "
                    f"`kill -9` the redis-rdb-bgsave process inside the container, then BGSAVE."
                ),
                context={"bgsave_running_sec": running_sec},
            )
        )
    status = str(persistence.get("rdb_last_bgsave_status", "ok"))
    if status != "ok":
        alerts.append(
            StabilityAlert(
                key=f"snapshot_failed:{instance}",
                subject=instance,
                kind="snapshot_failed",
                severity="critical",
                message=f"{instance}'s last background save failed (status={status}). Check its logs and disk.",
                context={"rdb_last_bgsave_status": status},
            )
        )
    last_save = float(persistence.get("rdb_last_save_time", 0) or 0)
    changes = int(persistence.get("rdb_changes_since_last_save", 0) or 0)
    age = now - last_save
    if save_config.strip() and changes > 0 and age >= SNAPSHOT_STALE_SEC and not alerts:
        alerts.append(
            StabilityAlert(
                key=f"snapshot_overdue:{instance}",
                subject=instance,
                kind="snapshot_overdue",
                severity="error",
                message=(
                    f"{instance} has not saved to disk for {age / 3600:.1f}h with {changes} "
                    f"unsaved changes, although snapshots are configured ({save_config})."
                ),
                context={"last_save_age_sec": int(age), "changes_since_last_save": changes},
            )
        )
    return alerts


def graph_inflation_alert(graph: str, channel_nodes: int, catalog_size: int) -> list[StabilityAlert]:
    if catalog_size <= 0 or channel_nodes <= catalog_size * GRAPH_INFLATION_RATIO:
        return []
    return [
        StabilityAlert(
            key=f"graph_inflation:{graph}",
            subject=graph,
            kind="graph_inflation",
            severity="error",
            message=(
                f"Graph {graph} has {channel_nodes} Channel nodes against a {catalog_size}-entry "
                f"channel catalog. Per-request reply channels are not being collapsed to their "
                f"wildcard -- usually orion-bus-mirror running an image older than the catalog "
                f"entry. Rebuild bus-mirror, then run its scripts/stale_channel_node_cleanup.py."
            ),
            context={"channel_nodes": channel_nodes, "catalog_size": catalog_size},
        )
    ]


class AlertGate:
    """One card per alert key per REALERT_AFTER_SEC while the condition persists."""

    def __init__(self) -> None:
        self._last_sent: dict[str, float] = {}

    def admit(self, alerts: list[StabilityAlert], now: float) -> list[StabilityAlert]:
        out = []
        for alert in alerts:
            last = self._last_sent.get(alert.key)
            if last is None or now - last >= REALERT_AFTER_SEC:
                self._last_sent[alert.key] = now
                out.append(alert)
        return out
