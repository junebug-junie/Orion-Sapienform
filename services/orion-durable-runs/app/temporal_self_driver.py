"""Single writer for the ``temporal_self.update`` thread (Temporal Self rev 4, order 3).

Modelled on ``situation_driver.SituationDriver`` (one step at a time, bursts coalesced, one
checkpoint per step with ``durability="exit"``). Feeds the graph from:

* a clock tick every ``TEMPORAL_SELF_TICK_SEC`` (120 s);
* ``orion:chat:history:turn`` -- an immediate step on a JUNIPER turn (Orion's own outreach does
  not wake it: ``orion.regulation.juniper_turns.is_juniper_turn``).

Threads are local-date buckets, ``temporal_self:orion:<YYYY-MM-DD>`` in ORION_SITUATION_TIMEZONE.
The first step of a day seeds the new thread from the previous day's last state (so ``since`` and
the strain latch carry over midnight), and threads past TEMPORAL_SELF_RETENTION_DAYS are deleted
with the saver's own ``adelete_thread``.

A step that starts a new level episode (level change, or restart after a gap) is logged (``arousal_transition``) and recorded as one row
in ``temporal_self_event`` by the graph. It does NOT publish a ``DurableRunStateV1`` row: that
model's ``workflow`` is a closed Literal read by sql-writer and Hub, and widening it would need a
consumer-first rollout for a trace the event table already holds.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from orion.regulation.juniper_turns import is_juniper_turn
from orion.schemas.regulation import TEMPORAL_SELF_THREAD_PREFIX, TEMPORAL_SELF_WORKFLOW, RegulationStateV1

from app.temporal_self_graph import TemporalSelfDeps, build_temporal_self_graph

logger = logging.getLogger("orion-durable-runs.temporal_self_driver")

_PRIORITY = {"chat_turn": 3, "tick": 1, "boot": 0}
RETIRE_LOOKBACK_DAYS = 31
CARRIED_KEYS = ("regulation", "last_inputs", "last_juniper_turn_at")


def day_id_for(at: datetime, tz: ZoneInfo) -> str:
    return at.astimezone(tz).date().isoformat()


def thread_for(day_id: str) -> str:
    return f"{TEMPORAL_SELF_THREAD_PREFIX}:{day_id}"


def event_from_chat_turn(payload: dict, now: datetime) -> Optional[dict]:
    """Only Juniper's turns wake the regulate step; the bus event carries no timestamp, so the
    receive time stands in for when she spoke (sql-writer's row lands moments later)."""
    if not is_juniper_turn(payload):
        return None
    corr = str((payload or {}).get("correlation_id") or (payload or {}).get("id") or now.isoformat())
    return {"event_id": f"chat:{corr}", "kind": "chat_turn", "correlation_id": corr,
            "at": now.isoformat(), "juniper": True}


def coalesce(events: list[dict]) -> dict:
    lead = dict(max(events, key=lambda e: _PRIORITY.get(e.get("kind"), 0)))
    turns = [e for e in events if e.get("kind") == "chat_turn"]
    if turns:
        lead = dict(max(turns, key=lambda e: str(e.get("at"))))
    lead["coalesced"] = len(events)
    return lead


class TemporalSelfDriver:
    def __init__(
        self,
        *,
        checkpointer: Any,
        deps: TemporalSelfDeps,
        timezone_name: str = "America/Denver",
        tick_sec: float = 120.0,
        retention_days: int = 2,
    ) -> None:
        self._saver = checkpointer
        self._deps = deps
        self._graph = build_temporal_self_graph(deps, checkpointer)
        self._tz = ZoneInfo(timezone_name)
        self._tick_sec = tick_sec
        self._retention_days = retention_days
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=1000)
        self._tasks: list[asyncio.Task] = []
        self.steps = 0
        self.latest: Optional[RegulationStateV1] = None
        self.last_error: Optional[str] = None
        self._retired_for: Optional[str] = None

    # --- intake -------------------------------------------------------------------------------

    def offer(self, event: Optional[dict]) -> None:
        if event is None:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("temporal_self_queue_full dropped=%s", event.get("kind"))

    def offer_chat_turn(self, payload: dict) -> None:
        self.offer(event_from_chat_turn(payload, self._deps.now()))

    def offer_tick(self) -> None:
        self.offer({"event_id": f"tick:{self._deps.now().isoformat()}", "kind": "tick"})

    # --- lifecycle ----------------------------------------------------------------------------

    async def start(self, stop: asyncio.Event) -> None:
        self.offer({"event_id": f"boot:{self._deps.now().isoformat()}", "kind": "boot"})
        self._tasks = [asyncio.create_task(self._consume(stop)), asyncio.create_task(self._tick(stop))]

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _tick(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._tick_sec)
            except asyncio.TimeoutError:
                self.offer_tick()

    async def _consume(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            first = await self._queue.get()
            burst = [first]
            while not self._queue.empty():
                burst.append(self._queue.get_nowait())
            try:
                await self.step(coalesce(burst))
            except Exception as exc:  # noqa: BLE001 - one bad step must not stop the writer
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("temporal_self_step_failed kind=%s", first.get("kind"))

    # --- one step -----------------------------------------------------------------------------

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    async def _seed(self, thread_id: str, now: datetime) -> dict:
        """The most recent earlier day's carried state, for a day thread with no checkpoint yet."""
        snap = await self._graph.aget_state(self._config(thread_id))
        if snap and snap.values:
            return {}
        for days in range(1, self._retention_days + 1):
            prev = await self._graph.aget_state(self._config(thread_for(day_id_for(now - timedelta(days=days), self._tz))))
            if prev and prev.values and prev.values.get("regulation"):
                return {k: prev.values[k] for k in CARRIED_KEYS if prev.values.get(k) is not None}
        return {}

    async def _retire_old(self, now: datetime) -> bool:
        ok = True
        for days in range(self._retention_days + 1, self._retention_days + RETIRE_LOOKBACK_DAYS):
            try:
                await self._saver.adelete_thread(thread_for(day_id_for(now - timedelta(days=days), self._tz)))
            except Exception:  # noqa: BLE001
                ok = False
                logger.warning("temporal_self_thread_delete_failed days=%s", days, exc_info=True)
        return ok

    async def step(self, event: dict) -> dict:
        now = self._deps.now()
        day_id = day_id_for(now, self._tz)
        thread_id = thread_for(day_id)
        inputs: dict = {"event": event, "thread_id": thread_id, "day_id": day_id, "workflow": TEMPORAL_SELF_WORKFLOW}
        inputs.update(await self._seed(thread_id, now))
        if self._retired_for != thread_id and await self._retire_old(now):
            self._retired_for = thread_id
        out = await self._graph.ainvoke(inputs, self._config(thread_id), durability="exit")
        self.steps += 1
        self.latest = RegulationStateV1.model_validate(out["regulation"])
        transition = out.get("transition")
        if transition:
            a = self.latest.arousal
            logger.info("arousal_transition thread=%s from=%s to=%s since=%s reasons=%s warnings=%s", thread_id,
                        transition.get("from"), transition.get("to"), a.since.isoformat(), ",".join(a.reasons),
                        ",".join(self.latest.warnings))
        return out

    def health(self) -> dict:
        a = self.latest.arousal if self.latest else None
        return {"steps": self.steps, "arousal_level": a.arousal_level if a else None,
                "since": a.since.isoformat() if a else None, "queued": self._queue.qsize(),
                "last_error": self.last_error}
