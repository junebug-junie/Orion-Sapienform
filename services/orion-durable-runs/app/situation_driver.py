"""Single writer for Orion's situation (spec 2026-10-07-situation-graph-design.md, step 2, SHADOW).

Feeds the ``situation.update`` graph from three sources and never runs two steps at once:

* ``orion:chat:history:turn``  -- a chat turn finished (its prompt cues recall);
* ``orion:durable:run:state``  -- a ``memory.episode_distill`` run finished (new facts landed);
* a clock tick every ``SITUATION_TICK_SEC`` -- end dates come due with nobody talking;
* ``orion:vision:identity:sighting`` -- a home camera matched Juniper's face (whereabouts).

Events that arrive while a step runs are coalesced into the next one (the newest chat text wins),
because every step re-derives the facts anyway.

Threads are day buckets, ``situation:juniper:<UTC date>``: the first step of a day seeds the new
thread from yesterday's last state, and threads older than ``SITUATION_RETENTION_DAYS`` are deleted
with the saver's own ``adelete_thread``. With one checkpoint per step (``durability="exit"``) that
keeps the runner's resume sweep, which walks every checkpoint, from growing without bound.

A step that changed the situation (or failed to prime) publishes one ``DurableRunStateV1`` row
(workflow ``situation.update``), so the run views show the revision history; a changed revision is
also projected to Redis and the bus. Quiet steps leave no row.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from orion.schemas.chat_history import ChatHistoryTurnV1
from orion.schemas.durable_run import DurableRunStateV1
from orion.schemas.memory_episode import MEMORY_EPISODE_DISTILL_WORKFLOW
from orion.schemas.situation_state import SITUATION_THREAD_PREFIX, SITUATION_WORKFLOW, SituationStateV1

from app.situation_graph import NODES, SituationDeps, build_situation_graph

logger = logging.getLogger("orion-durable-runs.situation_driver")

# Coalescing order: the event that names the step when several were pending.
_PRIORITY = {"chat_turn": 3, "sighting": 2, "episode_distilled": 2, "tick": 1, "boot": 0}
TURN_TEXT_CHARS = 4000
RETIRE_LOOKBACK_DAYS = 31


def thread_for(day: datetime) -> str:
    return f"{SITUATION_THREAD_PREFIX}:{day.astimezone(timezone.utc).date().isoformat()}"


def event_from_chat_turn(payload: dict) -> Optional[dict]:
    try:
        turn = ChatHistoryTurnV1.model_validate(payload or {})
    except Exception:  # noqa: BLE001
        return None
    corr = turn.correlation_id or turn.id or uuid.uuid4().hex
    return {"event_id": f"chat:{corr}", "kind": "chat_turn", "correlation_id": corr,
            "text": (turn.prompt or "")[:TURN_TEXT_CHARS]}


def event_from_run_state(payload: dict) -> Optional[dict]:
    """Only a finished episode distill is news; everything else on the channel is ignored
    (including this graph's own rows)."""
    p = payload or {}
    if p.get("workflow") != MEMORY_EPISODE_DISTILL_WORKFLOW or p.get("status") != "completed":
        return None
    run_id = str(p.get("run_id") or "")
    return {"event_id": f"distill:{run_id}:{p.get('entry_id') or ''}", "kind": "episode_distilled",
            "correlation_id": p.get("correlation_id"), "text": ""}


def event_from_sighting(payload: dict) -> Optional[dict]:
    """A home-camera face match (IdentitySightingV1) for the enrolled subject."""
    from orion.schemas.vision_sighting import IdentitySightingV1

    try:
        s = IdentitySightingV1.model_validate(payload or {})
    except Exception:  # noqa: BLE001
        return None
    if s.subject != "juniper" or s.place != "home":
        return None
    return {"event_id": f"sighting:{s.correlation_id}", "kind": "sighting", "correlation_id": s.correlation_id,
            "text": "", "sighting": {"stream_id": s.stream_id, "seen_at": s.seen_at.isoformat(),
                                     "similarity": s.similarity, "correlation_id": s.correlation_id}}


def coalesce(events: list[dict]) -> dict:
    """One step for a burst: named by the highest-priority kind, carrying the newest chat text."""
    lead = max(events, key=lambda e: _PRIORITY.get(e.get("kind"), 0))
    texts = [e.get("text") for e in events if e.get("kind") == "chat_turn" and e.get("text")]
    merged = dict(lead)
    merged["text"] = texts[-1] if texts else lead.get("text", "")
    sightings = [e["sighting"] for e in events if e.get("sighting")]
    if sightings:
        merged["sighting"] = max(sightings, key=lambda x: str(x.get("seen_at")))
    merged["coalesced"] = len(events)
    return merged


class SituationDriver:
    def __init__(
        self,
        *,
        checkpointer: Any,
        deps: SituationDeps,
        publish_state: Callable[[DurableRunStateV1], Awaitable[Any]],
        tick_sec: float = 900.0,
        retention_days: int = 2,
    ) -> None:
        self._saver = checkpointer
        self._deps = deps
        self._graph = build_situation_graph(deps, checkpointer)
        self._publish_state = publish_state
        self._tick_sec = tick_sec
        self._retention_days = retention_days
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=1000)
        self._tasks: list[asyncio.Task] = []
        self.steps = 0
        self.last_revision: Optional[int] = None
        self.last_error: Optional[str] = None
        self._retired_for: Optional[str] = None

    # --- intake -------------------------------------------------------------------------------

    def offer(self, event: Optional[dict]) -> None:
        if event is None:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("situation_queue_full dropped=%s", event.get("kind"))

    def offer_tick(self) -> None:
        now = self._deps.now()
        self.offer({"event_id": f"tick:{now.isoformat()}", "kind": "tick", "text": ""})

    # --- lifecycle ----------------------------------------------------------------------------

    async def start(self, stop: asyncio.Event) -> None:
        self.offer({"event_id": f"boot:{self._deps.now().isoformat()}", "kind": "boot", "text": ""})
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
                logger.exception("situation_step_failed kind=%s", first.get("kind"))

    # --- one step -----------------------------------------------------------------------------

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    async def _seed(self, thread_id: str, now: datetime) -> Optional[dict]:
        """The most recent earlier day's last situation, for a day thread with no checkpoint yet.
        Looks back past gap days (downtime) as far as threads are retained."""
        snap = await self._graph.aget_state(self._config(thread_id))
        if snap and snap.values:
            return None
        for days in range(1, self._retention_days + 1):
            prev = await self._graph.aget_state(self._config(thread_for(now - timedelta(days=days))))
            if prev and prev.values and prev.values.get("situation"):
                seeded = dict(prev.values["situation"])
                seeded["thread_id"] = thread_id
                return seeded
        return None

    async def _retire_old(self, now: datetime) -> bool:
        """Delete day threads past retention (a month back, so downtime gaps are covered).
        True when every delete succeeded; the driver retries on the next step otherwise."""
        ok = True
        for days in range(self._retention_days + 1, self._retention_days + RETIRE_LOOKBACK_DAYS):
            try:
                await self._saver.adelete_thread(thread_for(now - timedelta(days=days)))
            except Exception:  # noqa: BLE001
                ok = False
                logger.warning("situation_thread_delete_failed days=%s", days, exc_info=True)
        return ok

    async def step(self, event: dict) -> dict:
        now = self._deps.now()
        thread_id = thread_for(now)
        inputs: dict = {"event": event, "thread_id": thread_id, "workflow": SITUATION_WORKFLOW}
        seed = await self._seed(thread_id, now)
        if seed is not None:
            inputs["situation"] = seed
        if self._retired_for != thread_id and await self._retire_old(now):
            self._retired_for = thread_id      # once per day thread, retried until it succeeds
        out = await self._graph.ainvoke(inputs, self._config(thread_id), durability="exit")
        self.steps += 1
        sit = SituationStateV1.model_validate(out["situation"])
        self.last_revision = sit.revision
        if out.get("skipped") or (not out.get("changed") and not out.get("prime_error")):
            # Quiet steps (most ticks) leave no row: the run views and Hub's activity surface show
            # revisions, not heartbeats. /health still counts every step.
            return out
        await self._publish_state(DurableRunStateV1(
            run_id=thread_id, workflow=SITUATION_WORKFLOW, thread_id=thread_id, node=NODES[-1],
            status="completed", correlation_id=str(event.get("correlation_id") or event.get("event_id") or thread_id),
            detail={
                "event_kind": event.get("kind"), "coalesced": event.get("coalesced", 1),
                "skipped": bool(out.get("skipped")), "changed": bool(out.get("changed")),
                "revision": sit.revision,
                "whereabouts": sit.juniper.whereabouts.memory_id if sit.juniper.whereabouts else None,
                "doing": len(sit.juniper.doing), "waiting_on": len(sit.juniper.waiting_on),
                "recent": len(sit.juniper.recent),
                "cues": len(sit.recall.cues), "primed": len(sit.recall.primed),
                "primed_revision": sit.recall.primed_revision,
                "prime_ms": out.get("prime_ms"), "prime_error": out.get("prime_error"),
                "lapsed": len(sit.lapsed),
            },
        ))
        return out

    def health(self) -> dict:
        return {"steps": self.steps, "revision": self.last_revision, "queued": self._queue.qsize(),
                "last_error": self.last_error}
