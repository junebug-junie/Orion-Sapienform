"""AI Town runs only in sessions (Juniper, 2026-10-30): Orion chooses them, capped at 2 hours a day.

Contract: orion/schemas/aitown_session.py. Arithmetic: orion/embodiment/town_sessions.py.

What "Orion chooses" means, mechanically (ladder retirement PR 3, 2026-11-05; until then the
proposal arena offered ``aitown_session``, the motor allocator admitted it, and dispatch sent one
``orion:aitown_session:start:request`` RPC here):

1. embodiment's own timer (``app/town_self_start.py``) ticks every ``EMBODIMENT_TOWN_SELF_START_TICK_SEC``.
   It holds the self-monitor act gate dispatch held, and the controller's last refusal for its own
   ``retry_after_sec`` (dispatch's refusal hold).
2. It places a SHADOW bid for the ``aitown.session`` want (logged, not obeyed until Patch D), BEFORE
   the gate, then calls ``start_session``. This controller runs its own gate: the kill switch, the
   daily cap, the gap since the last session, and no session already open. If it passes, it
   unpauses the town and opens the ``play`` act.

The hard cap does not depend on any of that: every tick (``EMBODIMENT_TOWN_TICK_SEC``) the
controller pauses the town when the open session reaches ``EMBODIMENT_TOWN_SESSION_MAX_SEC`` or the
local day's Orion time reaches ``EMBODIMENT_TOWN_DAILY_CAP_SEC``.

Outside an Orion session the controller never keeps the town awake. If the town is running anyway,
Juniper opened it (Hub's Unfreeze button; or, only while the town is idle rather than paused, her
open page waking it): that is a ``manual`` session, logged, never paused here, and not counted
toward Orion's cap. After the controller pauses the town (``testing:stop``), a page no longer wakes
it: only Unfreeze does.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional
from uuid import NAMESPACE_URL, uuid4, uuid5

from orion.embodiment import town_sessions as ts
from orion.schemas.aitown_session import (
    AITOWN_SESSION_FAMILY,
    AITownSessionRowV1,
    aitown_session_object_ref,
)

logger = logging.getLogger("orion.embodiment.town_sessions")

#: The Convex env var the patched NPC LLM client sends as X-Orion-Correlation-Id
#: (services/orion-ai-town/patches/orion-town-correlation-id.patch).
CORRELATION_ENV = "ORION_TOWN_CORRELATION_ID"
#: Hold for a refusal with no clock of its own (the dispatch runtime's default is the same).
DEFAULT_HOLD_SEC = 300.0
#: Persist last_seen_at at most this often. A crash is closed at the last PERSISTED tick, so it can
#: under-charge by up to this much, plus the time the town ran while embodiment was down (bounded by
#: AI Town's own ~5-10 min idle stop when no page is open).
LAST_SEEN_WRITE_SEC = 15.0
ACT_VERB = "play"
#: Spare time after the session cap for the close to land before the act's horizon.
ACT_HORIZON_SLACK_SEC = 600.0


def session_id_for(act_id: str) -> str:
    """The session's one id: also the play act's correlation id, the shadow want id, and the NPC
    calls' X-Orion-Correlation-Id. Deterministic from the act id (``aitown_self_start:<uuid>``; a
    dispatch id before 2026-11-05)."""
    return str(uuid5(NAMESPACE_URL, f"orion:aitown_session:{act_id}"))


@dataclass
class StartOutcome:
    """One start attempt's answer (the dispatch era's AITownSessionStartResultV1, now in-process).
    ``refusal`` is set only for ``state=refused``: ``disabled`` | ``daily_cap`` | ``cooldown`` |
    ``session_open`` | ``manual_session_open`` | ``ledger_unknown`` | ``town_unreachable`` |
    ``resume_failed`` | ``already_ran``. ``retry_after_sec``: how long the same refusal will hold, from
    the controller's own clock (the next local midnight for ``daily_cap``); None = no clock of its own."""

    act_id: str
    state: str
    session_id: Optional[str] = None
    object_ref: Optional[str] = None
    refusal: Optional[str] = None
    retry_after_sec: Optional[float] = None
    gates: dict = field(default_factory=dict)


@dataclass
class TownConfig:
    enabled: bool = True
    session_max_sec: float = 1800.0
    daily_cap_sec: float = 7200.0
    min_gap_sec: float = 3600.0
    min_session_sec: float = 300.0
    tz: str = ts.DEFAULT_TZ
    correlation_env_enabled: bool = True
    shadow_bid_enabled: bool = True

    @classmethod
    def from_settings(cls, s: Any) -> "TownConfig":
        return cls(
            enabled=bool(getattr(s, "town_sessions_enabled", True)),
            session_max_sec=float(getattr(s, "town_session_max_sec", 1800.0)),
            daily_cap_sec=float(getattr(s, "town_daily_cap_sec", 7200.0)),
            min_gap_sec=float(getattr(s, "town_min_gap_sec", 3600.0)),
            min_session_sec=float(getattr(s, "town_min_session_sec", 300.0)),
            tz=str(getattr(s, "town_tz", ts.DEFAULT_TZ) or ts.DEFAULT_TZ),
            correlation_env_enabled=bool(getattr(s, "town_correlation_env_enabled", True)),
            shadow_bid_enabled=bool(getattr(s, "physiology_shadow_bid_enabled", True)),
        )


class ConvexTown:
    """The four town calls the controller makes (sync; run in a thread). Swapped out in tests."""

    def engine_running(self, world_id: Optional[str]) -> bool:
        from orion.embodiment import aitown_client

        return aitown_client.engine_running(world_id)

    def resume(self) -> None:
        from orion.embodiment import aitown_client

        aitown_client.resume_world()

    def stop(self) -> None:
        from orion.embodiment import aitown_client

        aitown_client.stop_world()

    def heartbeat(self, world_id: Optional[str]) -> None:
        from orion.embodiment import aitown_client

        aitown_client.heartbeat_world(world_id)

    def set_correlation(self, value: Optional[str]) -> None:
        from orion.embodiment import aitown_client

        aitown_client.set_deployment_env(CORRELATION_ENV, value)


def _act(cfg: TownConfig, act_id: str, session_id: str):
    from orion.grammar.acts import build_act

    return build_act(
        actor=AITOWN_SESSION_FAMILY, verb=ACT_VERB, key=str(act_id), signal_id="artifact_produced",
        horizon_sec=cfg.session_max_sec + ACT_HORIZON_SLACK_SEC,
        basis=(f"artifact_produced (Orion spoke at least one line in AI Town) by {cfg.session_max_sec:.0f}s "
               f"session cap + {ACT_HORIZON_SLACK_SEC:.0f}s for the close; not yet measured (no session has run)"),
        object_ref=f"aitown_session:{session_id}", test_before=aitown_session_object_ref(act_id),
    )


async def publish_act(bus: Any, event: Any) -> None:
    from orion.grammar.publish import publish_grammar_event

    await publish_grammar_event(bus, event, source_name="orion-embodiment")


class TownSessionController:
    def __init__(self, cfg: TownConfig, *, redis_getter: Callable[[], Any], world_id_getter: Callable[[], Optional[str]],
                 town: Optional[ConvexTown] = None, bus_getter: Callable[[], Any] = lambda: None,
                 act_publisher: Callable[[Any, Any], Awaitable[None]] = publish_act,
                 clock: Callable[[], float] = time.time) -> None:
        self.cfg = cfg
        self._redis = redis_getter
        self._world_id = world_id_getter
        self.town = town or ConvexTown()
        self._bus = bus_getter
        self._publish_act = act_publisher
        self.clock = clock
        self._lock = asyncio.Lock()
        self._booted = False
        self.orion: Optional[AITownSessionRowV1] = None
        self.manual: Optional[AITownSessionRowV1] = None
        self._rows: list[AITownSessionRowV1] = []
        self._pending_stop = False
        self._last_write: dict[str, float] = {}

    # --- state ------------------------------------------------------------------------------

    @property
    def heartbeat_wanted(self) -> bool:
        """Keep the town awake only while Orion's own session is open."""
        return self.orion is not None

    def act_attempt_rows(self, secs: float) -> list[dict]:
        """The ``play`` acts this controller opened in the last ``secs``, in the act ledger's shape
        (object_ref, test_before, opened_at): the self-monitor gate's "last real attempt" read. Every
        Orion row is an act this controller opened, and it is the ledger's only writer. Raises before
        the ledger is loaded (the gate reads that as unreadable, never as a refusal)."""
        from datetime import datetime, timezone

        if not self._booted:
            raise RuntimeError("aitown_session_ledger_not_loaded")
        rows = list(self._rows)
        cutoff = self.clock() - float(secs)   # after the copy: never older than the rows it ages
        return [{"object_ref": f"aitown_session:{r.session_id}",
                 "test_before": aitown_session_object_ref(r.dispatch_id or ""),
                 "opened_at": datetime.fromtimestamp(r.started_at, tz=timezone.utc)}
                for r in rows if r.kind == "orion" and r.started_at > cutoff]

    def note_utterance(self) -> None:
        """Orion spoke a line in town (the play act's artifact)."""
        for row in (self.orion, self.manual):
            if row is not None:
                row.utterances += 1

    async def _save(self, row: AITownSessionRowV1, *, force: bool = True) -> bool:
        now = self.clock()
        if not force and now - self._last_write.get(row.session_id, 0.0) < LAST_SEEN_WRITE_SEC:
            return True
        try:
            await ts.save_row(self._redis(), row)
            self._last_write[row.session_id] = now
            self._rows = [r for r in self._rows if r.session_id != row.session_id] + [row]
            return True
        except Exception:  # noqa: BLE001
            logger.warning("aitown_session_ledger_write_failed session=%s", row.session_id, exc_info=True)
            return False

    async def _set_correlation(self, value: Optional[str]) -> None:
        if not self.cfg.correlation_env_enabled:
            return
        try:
            await asyncio.to_thread(self.town.set_correlation, value)
        except Exception:  # noqa: BLE001
            logger.info("aitown_session_correlation_env_failed value=%s", value, exc_info=True)

    async def _act_event(self, row: AITownSessionRowV1, *, closed: bool) -> None:
        if not row.dispatch_id:
            return
        try:
            from datetime import datetime, timezone

            from orion.grammar.acts import act_closed_event, act_opened_event

            act = _act(self.cfg, row.dispatch_id, row.session_id)
            if closed:
                event = act_closed_event(
                    act, produced=row.utterances > 0, source_service="orion-embodiment",
                    source_component="town_sessions.act", correlation_id=row.session_id,
                    note=f"{row.stop_reason}; {row.utterances} line(s) spoken",
                    at=datetime.fromtimestamp(row.ended_at or self.clock(), tz=timezone.utc))
            else:
                event = act_opened_event(
                    act, source_service="orion-embodiment", source_component="town_sessions.act",
                    correlation_id=row.session_id, at=datetime.fromtimestamp(row.started_at, tz=timezone.utc))
            await self._publish_act(self._bus(), event)
        except Exception:  # noqa: BLE001
            logger.warning("aitown_session_act_failed session=%s closed=%s", row.session_id, closed, exc_info=True)

    # --- boot -------------------------------------------------------------------------------

    async def _recover(self) -> None:
        """A session left open by a restart is closed at its last tick, and the town is paused."""
        rows = await ts.load_rows(self._redis())
        self._rows = list(rows)
        crashed = ts.open_row(rows, "orion")
        if crashed is not None:
            crashed.ended_at = crashed.last_seen_at
            crashed.stop_reason = "embodiment_restart"
            await self._save(crashed)
            await self._act_event(crashed, closed=True)
            await self._release(crashed.session_id)
            # With sessions off (the rollback path) nothing is paused: a stoppedByDeveloper world
            # cannot be woken by the legacy heartbeat.
            self._pending_stop = self.cfg.enabled
            logger.info("aitown_session_closed session=%s reason=embodiment_restart", crashed.session_id)
        self.manual = ts.open_row(rows, "manual")
        if crashed is None and self.manual is None and self.cfg.enabled:
            # Paused by default: a town running at boot with no session on the ledger was left on by
            # the old always-on heartbeat (the first deploy) or opened while this process was down.
            # Pause it once; Juniper's own open session is on the ledger and is left alone.
            try:
                if await asyncio.to_thread(self.town.engine_running, self._world_id()):
                    self._pending_stop = True
                    logger.info("aitown_session_boot_pause (town running with no session on the ledger)")
            except Exception:  # noqa: BLE001
                logger.info("aitown_session_boot_engine_unreadable", exc_info=True)
        try:
            from orion.schemas.aitown_session import AITownSessionConfigV1

            await ts.save_config(self._redis(), AITownSessionConfigV1(
                enabled=self.cfg.enabled, session_max_sec=self.cfg.session_max_sec,
                daily_cap_sec=self.cfg.daily_cap_sec, min_gap_sec=self.cfg.min_gap_sec, tz=self.cfg.tz,
                written_at=self.clock()))
            await ts.prune(self._redis(), rows, self.clock())
        except Exception:  # noqa: BLE001
            logger.info("aitown_session_config_or_prune_failed", exc_info=True)
        self._booted = True

    # --- the tick ---------------------------------------------------------------------------

    async def tick(self) -> None:
        async with self._lock:
            if not self._booted:
                try:
                    await self._recover()
                except Exception:  # noqa: BLE001
                    logger.warning("aitown_session_recover_failed (ledger unreadable; retry next tick)", exc_info=True)
                    return
            now = self.clock()
            if self._pending_stop:
                try:
                    await asyncio.to_thread(self.town.stop)
                    self._pending_stop = False
                    await self._set_correlation(None)
                except Exception:  # noqa: BLE001
                    logger.warning("aitown_session_stop_failed (retry next tick)", exc_info=True)
                    return
            if self.orion is not None:
                # The caps are checked BEFORE the engine read: a failing read must never let a
                # session run past them (a page open on the town keeps it awake meanwhile).
                reason = ts.stop_reason(self.orion, self._rows, now, enabled=self.cfg.enabled,
                                        session_max_sec=self.cfg.session_max_sec,
                                        daily_cap_sec=self.cfg.daily_cap_sec, tz=self.cfg.tz)
                if reason is not None:
                    await self._end_orion(reason, now, stop_town=True)
                    return
            try:
                running = await asyncio.to_thread(self.town.engine_running, self._world_id())
            except Exception:  # noqa: BLE001
                logger.info("aitown_session_engine_unreadable", exc_info=True)
                return
            if self.orion is not None:
                await self._tick_orion(now, running)
            else:
                await self._tick_manual(now, running)

    async def _release(self, want_id: str) -> None:
        if not self.cfg.shadow_bid_enabled:
            return
        from app.physiology_shadow import release

        await release(want_id)

    async def _tick_orion(self, now: float, running: bool) -> None:
        row = self.orion
        assert row is not None
        if not running:
            await self._end_orion("town_paused", now, stop_town=False)
            return
        row.last_seen_at = now
        await self._save(row, force=False)
        try:
            await asyncio.to_thread(self.town.heartbeat, self._world_id())
        except Exception:  # noqa: BLE001
            logger.info("aitown_session_heartbeat_failed", exc_info=True)

    async def _end_orion(self, reason: str, now: float, *, stop_town: bool) -> None:
        row = self.orion
        assert row is not None
        if stop_town:
            try:
                await asyncio.to_thread(self.town.stop)
                self._stop_failures = 0
            except Exception:  # noqa: BLE001
                # Keep the session OPEN and counting until the pause lands: any overrun is on the
                # ledger, and the next tick retries (the stop reason still holds).
                self._stop_failures = getattr(self, "_stop_failures", 0) + 1
                logger.warning("aitown_session_stop_failed session=%s reason=%s consecutive=%d (retry next tick)",
                               row.session_id, reason, self._stop_failures, exc_info=True)
                row.last_seen_at = now
                await self._save(row)
                return
        row.last_seen_at = now
        row.ended_at = now
        row.stop_reason = reason  # type: ignore[assignment]
        await self._save(row)
        self.orion = None
        await self._set_correlation(None)
        await self._act_event(row, closed=True)
        await self._release(row.session_id)
        logger.info("aitown_session_closed session=%s kind=orion reason=%s seconds=%.0f utterances=%d",
                    row.session_id, reason, now - row.started_at, row.utterances)

    async def _tick_manual(self, now: float, running: bool) -> None:
        if running and self.manual is None:
            self.manual = AITownSessionRowV1(session_id=str(uuid4()), kind="manual", started_at=now,
                                             last_seen_at=now, start_reason="juniper_override")
            await self._save(self.manual)
            await self._set_correlation(self.manual.session_id)
            logger.info("aitown_session_opened session=%s kind=manual reason=juniper_override", self.manual.session_id)
        elif running and self.manual is not None:
            self.manual.last_seen_at = now
            await self._save(self.manual, force=False)
        elif not running and self.manual is not None:
            row, self.manual = self.manual, None
            row.ended_at = now
            row.last_seen_at = now
            row.stop_reason = "town_paused"
            await self._save(row)
            await self._set_correlation(None)
            logger.info("aitown_session_closed session=%s kind=manual reason=town_paused seconds=%.0f",
                        row.session_id, now - row.started_at)

    # --- the start (called by app/town_self_start.py) ---------------------------------------

    def _refused(self, act_id: str, reason: str, retry: Optional[float] = None,
                 gates: Optional[dict] = None) -> StartOutcome:
        return StartOutcome(act_id=act_id, state="refused", refusal=reason, retry_after_sec=retry, gates=gates or {})

    async def start_session(self, act_id: str) -> StartOutcome:
        """One start attempt under every cap. The caller placed the shadow bid already."""
        async with self._lock:
            return await self._start(act_id, session_id_for(act_id))

    async def _start(self, act_id: str, session_id: str) -> StartOutcome:
        rows: Optional[list[AITownSessionRowV1]]
        try:
            rows = await ts.load_rows(self._redis())
        except Exception:  # noqa: BLE001
            logger.warning("aitown_session_ledger_read_failed", exc_info=True)
            rows = None
        # The clock is taken AFTER the ledger read: every age below (the gap since the last session,
        # today's seconds) is measured against a time no earlier than the rows it reads (#3134).
        now = self.clock()
        if rows is not None:
            self._rows = list(rows)
        if not self._booted:
            return self._refused(act_id, "ledger_unknown", DEFAULT_HOLD_SEC)
        if self._pending_stop:
            return self._refused(act_id, "town_unreachable", DEFAULT_HOLD_SEC)
        # The live in-memory sessions are the truth for "open" (the ledger can lag one write).
        live = [r for r in (rows or []) if r.session_id not in {x.session_id for x in (self.orion, self.manual) if x}]
        live += [x for x in (self.orion, self.manual) if x is not None]
        reason = ts.start_block_reason(None if rows is None else live, now, enabled=self.cfg.enabled,
                                       daily_cap_sec=self.cfg.daily_cap_sec,
                                       min_session_sec=self.cfg.min_session_sec,
                                       min_gap_sec=self.cfg.min_gap_sec, tz=self.cfg.tz)
        used = ts.seconds_today(live, now, kind="orion", tz=self.cfg.tz) if rows is not None else None
        gates = {"orion_seconds_today": None if used is None else round(used, 1),
                 "daily_cap_sec": self.cfg.daily_cap_sec}
        if reason is None and any(r.session_id == session_id for r in live):
            # Never a second row under the same id (it would overwrite the first one's counted seconds
            # and reopen its act). Act ids are minted fresh per attempt, so this is only a guard.
            reason = "already_ran"
        if reason is not None:
            retry = ts.retry_after_sec(reason, live, now, min_gap_sec=self.cfg.min_gap_sec, tz=self.cfg.tz)
            if reason == "session_open" and self.orion is not None:
                retry = max(1.0, self.cfg.session_max_sec - (now - self.orion.started_at))
            logger.info("aitown_session_refused act_id=%s refusal=%s", act_id, reason)
            return self._refused(act_id, reason, retry, gates)
        # The town running with no Orion session is Juniper's, even if the tick has not seen it yet:
        # never absorb her time into Orion's session (or pause her at Orion's cap).
        try:
            running = await asyncio.to_thread(self.town.engine_running, self._world_id())
        except Exception as exc:  # noqa: BLE001
            return self._refused(act_id, "town_unreachable", DEFAULT_HOLD_SEC, {**gates, "error": type(exc).__name__})
        if running:
            logger.info("aitown_session_refused act_id=%s refusal=manual_session_open (not yet ticked)", act_id)
            return self._refused(act_id, "manual_session_open", None, gates)
        try:
            await asyncio.to_thread(self.town.resume)
        except Exception as exc:  # noqa: BLE001
            logger.warning("aitown_session_resume_failed act_id=%s", act_id, exc_info=True)
            return self._refused(act_id, "resume_failed", DEFAULT_HOLD_SEC, {**gates, "error": type(exc).__name__})
        # The session starts when the town is really resumed (after two Convex calls that can take
        # 15-25 s each under contention), not at the gate's clock.
        started = self.clock()
        row = AITownSessionRowV1(session_id=session_id, kind="orion", started_at=started, last_seen_at=started,
                                 start_reason="want_funded", dispatch_id=act_id)
        if not await self._save(row):
            # No ledger row = no cap accounting: do not run uncounted.
            try:
                await asyncio.to_thread(self.town.stop)
            except Exception:  # noqa: BLE001
                self._pending_stop = True
            return self._refused(act_id, "ledger_unknown", DEFAULT_HOLD_SEC, gates)
        self.orion = row
        await self._set_correlation(session_id)
        await self._act_event(row, closed=False)
        logger.info("aitown_session_opened session=%s kind=orion reason=want_funded act_id=%s used_today=%s",
                    session_id, act_id, gates["orion_seconds_today"])
        return StartOutcome(act_id=act_id, state="started", session_id=session_id,
                            object_ref=aitown_session_object_ref(act_id), gates=gates)


__all__ = [
    "CORRELATION_ENV",
    "ConvexTown",
    "StartOutcome",
    "TownConfig",
    "TownSessionController",
    "session_id_for",
]
