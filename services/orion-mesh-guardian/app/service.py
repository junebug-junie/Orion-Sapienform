from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from orion.core.bus.async_service import OrionBusAsync

from .attention import AttentionPublisher
from .equilibrium_watch import equilibrium_status_for_service, watch_equilibrium
from .probe import run_probe
from .remediator import execute_remediation
from .roster import NEVER_REMEDIATE_IDS, RosterDocument, RosterEntry, load_roster
from .settings import Settings
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

    async def start(self) -> None:
        if not self.settings.enabled:
            logger.info("mesh guardian disabled")
            return
        self.roster = load_roster(
            self.settings.roster_path,
            project=self.settings.project,
            node_name=self.settings.node_name,
        )
        await self.bus.connect()
        if self.bus.redis is not None:
            self.states = await load_all(self.bus.redis)
        for entry in self.roster.services:
            self.states.setdefault(entry.id, ServiceState())
        self._stop.clear()
        self._tasks = [
            asyncio.create_task(self._probe_loop(), name="mesh-guardian-probe"),
            asyncio.create_task(self._equilibrium_loop(), name="mesh-guardian-equilibrium"),
        ]

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self.bus.close()

    def equilibrium_subscriber_alive(self) -> bool:
        return self._equilibrium_task_alive

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
            self.attention.publish_transition(
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
                self.attention.publish_transition(
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
                now = time.time()
                for entry in self.roster.services:
                    if entry.probe.mode.value == "redis" and not entry.probe.intake_channels:
                        eq_bad, _ = equilibrium_status_for_service(
                            self.latest_snapshot,
                            heartbeat_name=entry.heartbeat_name,
                            grace_sec=float(self.settings.equilibrium_grace_sec),
                            now_ts=now,
                        )
                        await self._apply_transition(entry, probe_status="probe_ok", equilibrium_bad=eq_bad)
                        continue
                    probe = await run_probe(redis=self.bus.redis, entry_probe=entry.probe)
                    eq_bad, _ = equilibrium_status_for_service(
                        self.latest_snapshot,
                        heartbeat_name=entry.heartbeat_name,
                        grace_sec=float(self.settings.equilibrium_grace_sec),
                        now_ts=now,
                    )
                    await self._apply_transition(entry, probe_status=probe.status, equilibrium_bad=eq_bad)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("probe loop error")
            await asyncio.sleep(self.settings.probe_interval_sec)
