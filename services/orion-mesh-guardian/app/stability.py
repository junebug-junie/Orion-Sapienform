"""Host-wide stability checks the per-service probes cannot see.

The roster probes ask "is this service answering right now?". On 2026-10-02
that question stayed "yes" for 8 days while orion-bus-mirror restarted 581
times (it was back up within seconds every ~20 min), Redis disconnected it as
a slow pub/sub consumer each time (dropping its queued messages), and
FalkorDB's snapshot child sat deadlocked for 38h with nothing persisted. Each
check below watches a signal that moved during that incident and is flat on a
healthy stack:

- crash loop: Docker ``RestartCount`` rising, any container -- athena's own docker socket, plus
  remote hosts' censuses carried on their heartbeats (``remote_restart_counts``; added after
  circe's affectgpt worker restarted 601 times unseen, 2026-10-14..23)
- slow-consumer kill: Redis ``client_output_buffer_limit_disconnections`` rising
- snapshot: Redis/FalkorDB ``BGSAVE`` stuck, failed, or long overdue
- graph inflation: bus-synapse Channel nodes far above the channel catalog
- stopped container: a container that should run (restart policy ``always``/``unless-stopped``)
  sitting ``exited``/``created`` past a grace period, or a half-finished compose recreate
  (``<12 hex>_<name>`` in ``created``). Local docker only: the remote census carries counts, not
  state. Added after 2026-10-29: a killed post-pull rebuild left orion-athena-harness-governor
  stopped and its replacement "Created" for ~9.5 h; the crash-loop check saw nothing, because a
  container that is stopped cleanly never restarts.

Evaluation is pure (``*Tracker.observe`` / ``*_alerts``) so the incident can be
replayed in tests; collection is the thin async layer in ``service.py``.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import ValidationError

from orion.schemas.telemetry.system_health import CONTAINER_RESTARTS_DETAIL_KEY, ContainerRestartsV1

# Fixed thresholds, chosen against the 10-02 incident and the healthy baseline
# measured the same day (highest RestartCount on the host: 5, flat for days).
CRASH_LOOP_RESTARTS = 3
CRASH_LOOP_WINDOW_SEC = 2 * 3600
SNAPSHOT_STUCK_SEC = 15 * 60
SNAPSHOT_STALE_SEC = 6 * 3600
GRAPH_INFLATION_RATIO = 2.0
REALERT_AFTER_SEC = 6 * 3600
# A remote census older than this is not evidence of anything: the host reads as blind, not calm.
# 10x the producer's 30 s refresh, so one late heartbeat or equilibrium tick is not a card.
REMOTE_CENSUS_STALE_SEC = 300
# A read_at further in the future than this is clock skew, not a fresh reading. (A remote clock
# more than REMOTE_CENSUS_STALE_SEC behind reads blind: loud, and the card shows the age.)
CLOCK_SKEW_TOLERANCE_SEC = 60


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

    def observe(
        self, restart_counts: dict[str, int], now: float, *, read_hosts: set[str] | None = None
    ) -> list[StabilityAlert]:
        """``read_hosts``: when given, a container missing from ``restart_counts`` loses its history
        only if its host ("<host>/..." key prefix) was read this cycle (it was removed); a host
        that could not be read keeps its histories. None = every key is in scope (one host)."""
        alerts: list[StabilityAlert] = []
        for key, count in restart_counts.items():
            name = key.split("@", 1)[0]  # "<host>/<container>" or a bare local name
            host, _, container = name.rpartition("/")
            samples = self._samples.setdefault(key, deque())
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
                            f"`docker logs {container}`{f' on {host}' if host else ''} for the exit reason."
                        ),
                        context={"restart_count": count, "restarts_in_window": rose, "host": host or None},
                    )
                )
        for gone in set(self._samples) - set(restart_counts):
            if read_hosts is None or gone.split("@", 1)[0].rpartition("/")[0] in read_hosts:
                del self._samples[gone]
        return alerts


def qualify(host: str, counts: dict[str, int]) -> dict[str, int]:
    """{"<name>@<id12>": n} -> {"<host>/<name>@<id12>": n}: one host's containers never share a
    history or a card with another host's container of the same name."""
    return {f"{host}/{key}": count for key, count in counts.items()}


@dataclass
class RemoteCensus:
    counts: dict[str, int]  # host-qualified, from fresh readings only
    fresh_hosts: dict[str, float]  # host -> age (s) of the reading used
    alerts: list[StabilityAlert]  # one container_watch_blind per expected host without a fresh reading


def remote_restart_counts(
    services: Iterable[Any],
    *,
    expected_hosts: Iterable[str],
    local_node: str,
    now: float,
    watching_since: float,
    stale_after_sec: float = REMOTE_CENSUS_STALE_SEC,
) -> RemoteCensus:
    """Remote hosts' RestartCount censuses out of the equilibrium snapshot's per-service heartbeat
    details (``services``: anything with ``.node``, ``.service`` and ``.details``).

    Absence is not calm. An expected host with no reading, only a stale one, or only a failed one
    yields a ``container_watch_blind`` alert instead of zero restarts -- after a boot grace of
    ``stale_after_sec`` (the guardian has not had time to receive one yet). Hosts not in
    ``expected_hosts`` are still watched when they report; they just never read as blind."""
    counts: dict[str, int] = {}
    fresh: dict[str, float] = {}
    last_seen: dict[str, tuple[float, str, str | None]] = {}  # host -> (age, reporter, error)
    best: dict[str, tuple[float, dict[str, int]]] = {}  # host -> freshest usable (age, counts)
    for svc in services:
        host = str(getattr(svc, "node", None) or "")
        details = getattr(svc, "details", None) or {}
        raw = details.get(CONTAINER_RESTARTS_DETAIL_KEY) if isinstance(details, dict) else None
        if not host or host == local_node or raw is None:
            continue
        reporter = str(getattr(svc, "service", "?"))
        try:
            # Unknown fields are dropped first: the model forbids extras, and a newer producer
            # adding one must not blind an older guardian.
            known = {k: v for k, v in raw.items() if k in ContainerRestartsV1.model_fields} if isinstance(raw, dict) else raw
            reading = ContainerRestartsV1.model_validate(known)
        except ValidationError as exc:
            last_seen.setdefault(host, (float("inf"), reporter, f"malformed census: {exc.errors()[0]['msg']}"))
            continue
        age = now - reading.read_at.timestamp()
        error = reading.error
        if error is None and age < -CLOCK_SKEW_TOLERANCE_SEC:
            # read_at in the future: the hosts' clocks disagree, so age means nothing.
            error = f"census read_at is {-age:.0f}s in the future (clock skew between {host} and this host)"
        prev = last_seen.get(host)
        if prev is None or age < prev[0]:
            last_seen[host] = (age, reporter, error)
        if error is None and age <= stale_after_sec:
            # Freshest error-free reading per host wins outright: merging two reporters could let
            # an older, lower count read as a container recreate and reset the loop history.
            if host not in best or age < best[host][0]:
                best[host] = (age, reading.counts)
    for host, (age, host_counts) in best.items():
        counts.update(qualify(host, host_counts))
        fresh[host] = max(age, 0.0)
    alerts: list[StabilityAlert] = []
    if now - watching_since >= stale_after_sec:
        for host in expected_hosts:
            if host in fresh or host == local_node:
                continue
            seen = last_seen.get(host)
            if seen is None:
                why = "no container census from it has reached the equilibrium snapshot"
            elif seen[2] is not None:
                when = "" if seen[0] == float("inf") else f", {seen[0]:.0f}s ago"
                why = f"its last census ({seen[1]}{when}) is unusable: {seen[2]}"
            else:
                why = f"its newest census ({seen[1]}) is {seen[0] / 60:.0f} min old"
            alerts.append(
                StabilityAlert(
                    key=f"container_watch_blind:{host}",
                    subject=host,
                    kind="container_watch_blind",
                    severity="warning",
                    message=(
                        f"The crash-loop check cannot see {host}'s containers: {why}. A container "
                        f"there could be restarting right now and nothing would say so. The census "
                        f"rides a {host} service's heartbeat ({seen[1] if seen else 'on circe: gpu-lane-controller'}) "
                        f"through orion-equilibrium-service: check both."
                    ),
                    context={
                        "host": host,
                        "last_census_age_sec": None if seen is None or seen[0] == float("inf") else int(seen[0]),
                        "last_census_error": None if seen is None else seen[2],
                    },
                )
            )
    return RemoteCensus(counts=counts, fresh_hosts=fresh, alerts=alerts)


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


SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2, "critical": 3}


class AlertGate:
    """One card per alert key per REALERT_AFTER_SEC while the condition persists. A higher
    severity on the same key is admitted at once (an early error must not hide a critical)."""

    def __init__(self) -> None:
        self._last_sent: dict[str, tuple[float, int]] = {}

    def admit(self, alerts: list[StabilityAlert], now: float) -> list[StabilityAlert]:
        out = []
        for alert in alerts:
            rank = SEVERITY_RANK.get(alert.severity, 2)
            last = self._last_sent.get(alert.key)
            if last is None or now - last[0] >= REALERT_AFTER_SEC or rank > last[1]:
                self._last_sent[alert.key] = (now, rank)
                out.append(alert)
        return out


# --- stopped containers ----------------------------------------------------------------------

# Docker restarts a container with these policies by itself, so one that is not running was
# stopped (by `docker stop`, or by compose in the middle of a recreate) and nothing started it.
SHOULD_RUN_POLICIES = frozenset({"always", "unless-stopped"})
STOPPED_STATES = frozenset({"exited", "created"})
LIVE_STATES = frozenset({"running", "restarting"})
# compose v2 creates the replacement as "<old id[:12]>_<name>", stops and removes the old one,
# then renames and starts the new one. A run killed in between leaves the temp name "Created".
TEMP_RECREATE_NAME = re.compile(r"^[0-9a-f]{12}_(.+)$")


@dataclass(frozen=True)
class ContainerState:
    id: str
    name: str
    status: str  # docker State.Status: running, exited, created, restarting, ...
    restart_policy: str = ""
    since: float | None = None  # exited: State.FinishedAt; created: Created (epoch seconds)
    compose_service: str | None = None  # "<project>/<service>" from the compose labels
    compose_dir: str | None = None  # basename of com.docker.compose.project.working_dir


def _group_of(c: ContainerState) -> str:
    if c.compose_service:
        return c.compose_service
    m = TEMP_RECREATE_NAME.match(c.name)
    return m.group(1) if m else c.name


def stopped_containers(
    containers: Iterable[ContainerState], now: float, *, grace_sec: float
) -> dict[str, list[tuple[ContainerState, str]]]:
    """Containers that should be running and are not, grouped by compose service (or bare name).

    A group is dropped when another container of the same compose service is live: a stale
    leftover beside a running replacement is junk, not an outage."""
    containers = list(containers)
    live = {_group_of(c) for c in containers if c.status in LIVE_STATES}
    out: dict[str, list[tuple[ContainerState, str]]] = {}
    for c in containers:
        if c.status not in STOPPED_STATES or c.since is None or now - c.since < grace_sec:
            continue
        if c.status == "created" and TEMP_RECREATE_NAME.match(c.name):
            why = "half-finished compose recreate"
        elif c.restart_policy in SHOULD_RUN_POLICIES:
            why = f"restart={c.restart_policy}"
        else:
            continue
        group = _group_of(c)
        if group not in live:
            out.setdefault(group, []).append((c, why))
    return out


class StoppedContainerTracker:
    """One card per stop episode. An episode is (container id, stopped-since), grouped by compose
    service: a group cards when none of its stopped containers has carded yet. The same container
    stopping again later is a new episode, and staying stopped never re-cards -- so a container
    someone stopped on purpose costs one card, not one every 6 h. Episodes live in memory: a
    guardian restart re-cards each still-stopped container once."""

    def __init__(self) -> None:
        self._carded: set[tuple[str, float]] = set()

    def observe(self, containers: Iterable[ContainerState], now: float, *, grace_sec: float) -> list[StabilityAlert]:
        groups = stopped_containers(containers, now, grace_sec=grace_sec)
        seen = {(c.id, c.since) for members in groups.values() for c, _ in members}
        self._carded &= seen  # a container that started or was removed ends its episode
        alerts: list[StabilityAlert] = []
        for group, members in sorted(groups.items()):
            episode = {(c.id, c.since) for c, _ in members}
            already = bool(episode & self._carded)
            self._carded |= episode
            if already:  # one outage, one card: a container joining a carded group adds none
                continue
            oldest = min(c.since for c, _ in members if c.since is not None)
            hours = (now - oldest) / 3600.0
            listing = ", ".join(f"{c.name} ({c.status}, {why})" for c, why in members)
            compose_dir = next((c.compose_dir for c, _ in members if c.compose_dir), None)
            service = group.split("/", 1)[1] if "/" in group else ""
            start = (
                f"`scripts/safe_docker_build.sh {compose_dir} up -d{(' ' + service) if service else ''}` "
                f"from the primary main checkout"
                if compose_dir
                else "`docker start <name>`"
            )
            names = " ".join(c.name for c, _ in members)
            alerts.append(
                StabilityAlert(
                    key=f"stopped_container:{group}:" + ",".join(sorted(f"{c.id[:12]}@{c.since:.0f}" for c, _ in members)),
                    subject=group,
                    kind="stopped_container",
                    severity="error",
                    message=(
                        f"Nothing of {group} is running, and it should be: {listing}. Stopped for "
                        f"{hours:.1f} h. Docker will not start it again by itself. Anything that "
                        f"waits on this service waits until someone does. If a rebuild was "
                        f"interrupted, look for a START with no END in ~/.orion/post_pull_rebuild.log. "
                        f"Start it: {start}. If it was stopped on purpose, remove it "
                        f"(`docker rm {names}`) so this check stops counting it."
                    ),
                    context={
                        "containers": [c.name for c, _ in members],
                        "states": [c.status for c, _ in members],
                        "stopped_hours": round(hours, 2),
                        "compose_dir": compose_dir,
                    },
                )
            )
        return alerts
