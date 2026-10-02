from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pathlib import Path

from orion.bus.census import load_channel_catalog_names
from orion.core.bus.async_service import OrionBusAsync

from .attention import AttentionPublisher
from .equilibrium_watch import equilibrium_status_for_service, watch_equilibrium
from .probe import run_probe
from .remediator import execute_remediation
from .roster import NEVER_REMEDIATE_IDS, RosterDocument, RosterEntry, load_roster, validate_roster
from .settings import Settings
from .stability import (
    AlertGate,
    CounterRiseTracker,
    CrashLoopTracker,
    StabilityAlert,
    graph_inflation_alert,
    slow_consumer_alert,
    snapshot_alerts,
)
from .state_machine import ServiceState, TransitionInput, transition
from .state_store import load_all, save_one

logger = logging.getLogger("orion.mesh.guardian")


class MeshGuardianService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bus = OrionBusAsync(url=settings.orion_bus_url)
        self.attention = AttentionPublisher(settings)
        self.roster: RosterDocument | None = None
        self.states: dict[str, ServiceState] = {}
        self.latest_snapshot = None
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._equilibrium_queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        self._equilibrium_task_alive = False
        self._crash_loops = CrashLoopTracker()
        self._bus_disconnects = CounterRiseTracker()
        self._alert_gate = AlertGate()
        self._stability_cycles = 0
        self._falkordb = None

    async def start(self) -> None:
        if not self.settings.enabled:
            logger.info("mesh guardian disabled")
            return
        self.roster = load_roster(
            self.settings.roster_path,
            project=self.settings.project,
            node_name=self.settings.node_name,
        )
        roster_errors = validate_roster(self.roster)
        if roster_errors:
            raise ValueError("invalid mesh guardian roster: " + "; ".join(roster_errors))
        await self._connect_bus_with_retry()
        if self.bus.redis is not None:
            self.states = await load_all(self.bus.redis)
        for entry in self.roster.services:
            self.states.setdefault(entry.id, ServiceState())
        self._stop.clear()
        self._tasks = [
            asyncio.create_task(self._probe_loop(), name="mesh-guardian-probe"),
            asyncio.create_task(self._equilibrium_loop(), name="mesh-guardian-equilibrium"),
        ]
        if self.settings.stability_enabled:
            self._tasks.append(asyncio.create_task(self._stability_loop(), name="mesh-guardian-stability"))

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._falkordb is not None:
            await self._falkordb.aclose()
            self._falkordb = None
        await self.bus.close()

    def equilibrium_subscriber_alive(self) -> bool:
        return self._equilibrium_task_alive

    async def _connect_bus_with_retry(
        self,
        *,
        max_wait_sec: float = 60.0,
        initial_backoff_sec: float = 1.0,
    ) -> None:
        """Connect to mesh Redis; retry through transient bring-up races (batched compose up)."""
        deadline = time.monotonic() + max_wait_sec
        backoff = initial_backoff_sec
        while True:
            try:
                await self.bus.connect()
                return
            except Exception as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                wait = min(backoff, remaining)
                logger.warning(
                    "mesh guardian bus connect failed url=%s err=%s; retry in %.1fs",
                    self.settings.orion_bus_url,
                    exc,
                    wait,
                )
                await self.bus.close()
                await asyncio.sleep(wait)
                backoff = min(backoff * 2.0, 15.0)

    async def _equilibrium_loop(self) -> None:
        self._equilibrium_task_alive = True
        try:
            await watch_equilibrium(
                self.bus,
                self.settings.channel_equilibrium_snapshot,
                self._equilibrium_queue,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("equilibrium watch failed")
        finally:
            self._equilibrium_task_alive = False

    async def _drain_equilibrium(self) -> None:
        while not self._equilibrium_queue.empty():
            snapshot = await self._equilibrium_queue.get()
            self.latest_snapshot = snapshot

    async def _apply_transition(self, entry: RosterEntry, *, probe_status: str, equilibrium_bad: bool) -> None:
        assert self.roster is not None
        state = self.states.get(entry.id, ServiceState())
        now = time.time()
        out = transition(
            state,
            TransitionInput(
                equilibrium_bad=equilibrium_bad,
                probe_status=probe_status,  # type: ignore[arg-type]
                auto_remediate=self.settings.auto_remediate and entry.auto_remediate,
                now=now,
                cooldown_sec=self.settings.remediation_cooldown_sec,
                max_attempts_per_hour=self.settings.max_attempts_per_hour,
                consecutive_probe_fails_threshold=self.settings.consecutive_probe_fails,
                post_grace_sec=self.settings.post_remediate_grace_sec,
            ),
            service_id=entry.id,
        )
        self.states[entry.id] = out.new_state
        if self.bus.redis is not None:
            await save_one(self.bus.redis, entry.id, out.new_state)

        for event in out.attention_events:
            await asyncio.to_thread(
                self.attention.publish_transition,
                service_id=entry.id,
                heartbeat_name=entry.heartbeat_name,
                event=event,
            )

        if entry.id in NEVER_REMEDIATE_IDS or not entry.auto_remediate:
            return
        if not self.settings.auto_remediate:
            return

        if out.should_remediate_tier1 or out.should_remediate_tier2:
            tier = 2 if out.should_remediate_tier2 else 1
            result = await execute_remediation(entry, repo_root=self.settings.orion_repo_root, tier=tier)
            if not result.ok:
                await asyncio.to_thread(
                    self.attention.publish_transition,
                    service_id=entry.id,
                    heartbeat_name=entry.heartbeat_name,
                    event={
                        "severity": "error",
                        "message": f"mesh health: remediation tier-{tier} failed for {entry.id}",
                        "context": {"stderr_tail": result.stderr_tail, "command": result.command},
                    },
                )
                return
            post = transition(
                out.new_state,
                TransitionInput(
                    equilibrium_bad=equilibrium_bad,
                    probe_status=probe_status,  # type: ignore[arg-type]
                    auto_remediate=True,
                    now=time.time(),
                    cooldown_sec=self.settings.remediation_cooldown_sec,
                    max_attempts_per_hour=self.settings.max_attempts_per_hour,
                    consecutive_probe_fails_threshold=self.settings.consecutive_probe_fails,
                    post_grace_sec=self.settings.post_remediate_grace_sec,
                ),
                service_id=entry.id,
            )
            self.states[entry.id] = post.new_state
            if self.bus.redis is not None:
                await save_one(self.bus.redis, entry.id, post.new_state)

    async def _probe_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._drain_equilibrium()
                if self.roster is None or self.bus.redis is None:
                    await asyncio.sleep(self.settings.probe_interval_sec)
                    continue
                for entry in self.roster.services:
                    if entry.probe.mode.value == "redis" and not entry.probe.intake_channels:
                        eq_bad, _ = equilibrium_status_for_service(
                            self.latest_snapshot,
                            heartbeat_name=entry.heartbeat_name,
                            grace_sec=float(self.settings.equilibrium_grace_sec),
                        )
                        await self._apply_transition(entry, probe_status="probe_ok", equilibrium_bad=eq_bad)
                        continue
                    probe = await run_probe(redis=self.bus.redis, entry_probe=entry.probe)
                    eq_bad, _ = equilibrium_status_for_service(
                        self.latest_snapshot,
                        heartbeat_name=entry.heartbeat_name,
                        grace_sec=float(self.settings.equilibrium_grace_sec),
                    )
                    await self._apply_transition(entry, probe_status=probe.status, equilibrium_bad=eq_bad)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("probe loop error")
            await asyncio.sleep(self.settings.probe_interval_sec)

    # --- host-wide stability checks (see app/stability.py) -------------------

    async def _stability_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.run_stability_checks(time.time())
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("stability check cycle error")
            await asyncio.sleep(self.settings.stability_interval_sec)

    async def run_stability_checks(self, now: float) -> list[StabilityAlert]:
        """One cycle. Each check is isolated: one unreachable source (FalkorDB
        down, docker socket missing) must not blind the others."""
        alerts: list[StabilityAlert] = []
        summary: dict[str, object] = {}
        checks = (
            ("containers", self._check_crash_loops),
            ("bus_redis", self._check_bus_redis),
            ("falkordb", self._check_falkordb),
        )
        for name, check in checks:
            try:
                found, info = await check(now)
                alerts.extend(found)
                summary[name] = info
            except Exception as exc:
                summary[name] = f"error:{type(exc).__name__}"
                logger.warning("stability check %s failed: %s", name, exc)
        # First cycle, then hourly at the default interval: live evidence the
        # loop is running and what it reads, without a line every minute.
        if self._stability_cycles % 60 == 0 or alerts:
            logger.info("stability cycle %s alerts=%d", summary, len(alerts))
        self._stability_cycles += 1
        for alert in self._alert_gate.admit(alerts, now):
            logger.warning("stability alert kind=%s subject=%s: %s", alert.kind, alert.subject, alert.message)
            await asyncio.to_thread(
                self.attention.publish_transition,
                service_id=alert.subject,
                heartbeat_name="stability",
                event={
                    "severity": alert.severity,
                    "message": alert.message,
                    "context": {"event": alert.kind, **alert.context},
                },
            )
        return alerts

    async def _check_crash_loops(self, now: float) -> tuple[list[StabilityAlert], object]:
        counts = await _docker_restart_counts()
        info = {"containers": len(counts), "max_restarts": max(counts.values(), default=0)}
        return self._crash_loops.observe(counts, now), info

    async def _check_bus_redis(self, now: float) -> tuple[list[StabilityAlert], object]:
        redis = self.bus.redis  # raises RuntimeError when not connected
        stats = await redis.info("stats")
        total = int(stats.get("client_output_buffer_limit_disconnections", 0))
        alerts = slow_consumer_alert("bus-redis", self._bus_disconnects.observe(total), total)
        persistence = await redis.info("persistence")
        alerts += snapshot_alerts("bus-redis", persistence, save_config=await _save_config(redis), now=now)
        return alerts, {"disconnections": total, "bgsave_sec": persistence.get("rdb_current_bgsave_time_sec")}

    async def _check_falkordb(self, now: float) -> tuple[list[StabilityAlert], object]:
        if self._falkordb is None:
            import redis.asyncio as aioredis

            self._falkordb = aioredis.Redis.from_url(
                self.settings.falkordb_uri,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                decode_responses=True,
            )
        persistence = await self._falkordb.info("persistence")
        alerts = snapshot_alerts(
            "falkordb", persistence, save_config=await _save_config(self._falkordb), now=now
        )
        info: dict[str, object] = {"bgsave_sec": persistence.get("rdb_current_bgsave_time_sec")}
        # Own try: a missing/renamed graph or a query timeout must not discard
        # the snapshot alerts above (the stuck-save detector is the critical one).
        try:
            result = await self._falkordb.execute_command(
                "GRAPH.RO_QUERY", self.settings.falkordb_bus_graph, "MATCH (c:Channel) RETURN count(c)"
            )
            channel_nodes = int(result[1][0][0]) if result and len(result) > 1 and result[1] else 0
            # The live repo checkout (mounted at /repo), not the copy baked into
            # this image: a stale baked catalog is the exact failure this detects.
            repo_catalog = Path(self.settings.orion_repo_root) / "orion" / "bus" / "channels.yaml"
            catalog_size = len(
                await asyncio.to_thread(load_channel_catalog_names, repo_catalog if repo_catalog.is_file() else None)
            )
            alerts += graph_inflation_alert(self.settings.falkordb_bus_graph, channel_nodes, catalog_size)
            info.update(channel_nodes=channel_nodes, catalog=catalog_size)
        except Exception as exc:
            info["graph"] = f"error:{type(exc).__name__}"
            logger.warning("stability graph-inflation check failed: %s", exc)
        return alerts, info


async def _save_config(redis) -> str:
    cfg = await redis.config_get("save")
    value = cfg.get("save", cfg.get(b"save", ""))
    return value.decode() if isinstance(value, bytes) else str(value)


async def _docker_restart_counts(socket_path: str = "/var/run/docker.sock", transport=None) -> dict[str, int]:
    """Docker Engine API over the mounted socket, not the docker CLI: the CLI
    is not in this image (Debian's docker.io package no longer ships it)."""
    import httpx

    transport = transport or httpx.AsyncHTTPTransport(uds=socket_path)
    async with httpx.AsyncClient(transport=transport, base_url="http://docker", timeout=10.0) as client:
        resp = await client.get("/containers/json", params={"all": "true"})
        resp.raise_for_status()
        lines = []
        for container in resp.json():
            detail = await client.get(f"/containers/{container['Id']}/json")
            if detail.status_code != 200:
                continue  # removed between list and inspect
            body = detail.json()
            lines.append(f"{body['Id']} {body['Name']} {body['RestartCount']}")
    return _parse_restart_counts("\n".join(lines))


def _parse_restart_counts(text: str) -> dict[str, int]:
    """``<id> /<name> <count>`` lines -> {"<name>@<id12>": count}. Keyed by id
    too, so a container recreated under the same name starts a new history."""
    counts: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2].isdigit():
            counts[f"{parts[1].lstrip('/')}@{parts[0][:12]}"] = int(parts[2])
    return counts
