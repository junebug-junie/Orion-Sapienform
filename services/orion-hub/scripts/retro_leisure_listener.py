"""Bus RPC consumer for the retro leisure start (orion/schemas/retro_leisure.py).

Orion plays Zelda, section E. The motor allocator (orion-execution-dispatch-runtime, route
``retro_leisure_session``) asks; Lane B's leisure runner (``scripts/retro_leisure.py``) decides and
plays. This is the "retro_leisure_session dispatch consumer" that module's docstring names. Same
trusted internal bus boundary and reply-subject rule as ``scripts/world_pulse_read_pull_listener.py``.

The runner interface read here, and nothing else:

- ``should_start(config)`` (async) -> None when a session may start, else the gate's named reason
  (``disabled``, ``cabinet_hot``, ``arousal_strained``, ``browser_connected``, ``human_recent``,
  ``cooldown``, ``daily_cap``, ...).
- ``run_leisure_session(config, act_id=)`` (async) -> ``LeisureOutcome(started, refusal, ended)``.
  It re-checks the gate itself and runs the whole session (up to its 1800 s cap). ``ended`` is a
  ``scripts.retro_session_end.RetroSessionEnded``.

Per admitted dispatch:

1. no runner importable -> refused ``runner_unavailable`` (the dispatch runtime holds 300 s), never a
   fake session; a session already running here -> refused ``leisure_already_running``;
2. ``should_start`` says no -> refused with its reason;
3. otherwise start ``run_leisure_session(act_id=<dispatch_id>)`` and wait up to START_GRACE_SEC: a
   refusal from its own re-check in that time is answered as a refusal (no act opened);
4. still running (or finished) -> open the ``play`` act and answer ``started``; when the session ends,
   ``scripts.retro_memory_export.export_leisure_outcome`` writes the memory and closes the act.

Gate refusals carry no retry clock (the runner does not report one), so the dispatch runtime holds
the route for its own 300 s default: one RPC per five minutes while Orion may not play.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections import OrderedDict
from contextlib import suppress
from typing import Any, Callable

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.retro_leisure import (
    RETRO_LEISURE_START_REPLY_PREFIX,
    RETRO_LEISURE_START_REQUEST_CHANNEL,
    RETRO_LEISURE_START_REQUEST_KIND,
    RETRO_LEISURE_START_RESULT_KIND,
    RetroLeisureStartRequestV1,
    RetroLeisureStartResultV1,
    retro_leisure_object_ref,
)

logger = logging.getLogger("orion-hub.retro_leisure_listener")

UNAVAILABLE_HOLD_SEC = 300.0
# How long to wait for the runner's own start re-check before answering. Its refusals (gate,
# lock, save read, Retro start) all return before the first planner call; under the dispatch
# runtime's 15 s RPC timeout.
START_GRACE_SEC = 5.0
# A replayed dispatch id is answered with its first result, never a second session.
REPLAY_CACHE = 256


def import_runner() -> Any:
    """Lane B's module, or None when it is not deployed in this Hub (an older image)."""
    try:
        return importlib.import_module("scripts.retro_leisure")
    except ImportError:
        logger.warning("retro_leisure_runner_missing", exc_info=True)
        return None


def gate_verdict(raw: Any) -> tuple[bool, str | None]:
    """``should_start``'s answer: None (or True) = may start; a string (or False) = refused."""
    if raw is None or raw is True:
        return True, None
    if raw is False or not str(raw).strip():
        return False, "unnamed_refusal"
    return False, str(raw)[:200]


class RetroLeisureListener:
    def __init__(
        self,
        source_ref: Any,
        *,
        config_provider: Callable[[], Any],
        pool_provider: Callable[[], Any],
        runner_provider: Callable[[], Any] = import_runner,
    ) -> None:
        self.source_ref = source_ref
        self.config_provider = config_provider
        self.pool_provider = pool_provider
        self.runner_provider = runner_provider
        self.bus: Any = None
        self.task: asyncio.Task | None = None
        self.session_task: asyncio.Task | None = None
        self._followers: set[asyncio.Task] = set()
        self._seen: OrderedDict[str, RetroLeisureStartResultV1] = OrderedDict()
        self._lock = asyncio.Lock()

    @staticmethod
    def _refused(req: RetroLeisureStartRequestV1, reason: str, retry: float | None = None,
                 gates: dict | None = None) -> RetroLeisureStartResultV1:
        return RetroLeisureStartResultV1(dispatch_id=req.dispatch_id, state="refused", refusal=reason,
                                         retry_after_sec=retry, gates=gates or {})

    async def start(self, req: RetroLeisureStartRequestV1) -> RetroLeisureStartResultV1:
        async with self._lock:  # one decision at a time: two dispatches never both pass the gate
            prior = self._seen.get(req.dispatch_id)
            if prior is not None:
                return prior
            out = await self._start(req)
            if out.state == "started":
                self._seen[req.dispatch_id] = out
                while len(self._seen) > REPLAY_CACHE:
                    self._seen.popitem(last=False)
            return out

    async def _start(self, req: RetroLeisureStartRequestV1) -> RetroLeisureStartResultV1:
        runner = self.runner_provider()
        if runner is None or not hasattr(runner, "should_start") or not hasattr(runner, "run_leisure_session"):
            return self._refused(req, "runner_unavailable", UNAVAILABLE_HOLD_SEC)
        if self.session_task is not None and not self.session_task.done():
            return self._refused(req, "leisure_already_running")
        config = self.config_provider()
        try:
            ok, reason = gate_verdict(await runner.should_start(config))
        except Exception as exc:  # noqa: BLE001
            logger.warning("retro_leisure_gate_failed dispatch_id=%s", req.dispatch_id, exc_info=True)
            return self._refused(req, "gate_failed", UNAVAILABLE_HOLD_SEC, {"error": type(exc).__name__})
        if not ok:
            logger.info("retro_leisure_refused dispatch_id=%s refusal=%s", req.dispatch_id, reason)
            return self._refused(req, reason or "unnamed_refusal")

        session = asyncio.create_task(runner.run_leisure_session(config, act_id=req.dispatch_id))
        done, _ = await asyncio.wait({session}, timeout=START_GRACE_SEC)
        if session in done:
            try:
                outcome = session.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("retro_leisure_run_failed dispatch_id=%s", req.dispatch_id, exc_info=True)
                return self._refused(req, "start_failed", UNAVAILABLE_HOLD_SEC, {"error": type(exc).__name__})
            if not getattr(outcome, "started", False):
                reason = getattr(outcome, "refusal", None) or "unnamed_refusal"
                logger.info("retro_leisure_refused dispatch_id=%s refusal=%s at=runner", req.dispatch_id, reason)
                return self._refused(req, str(reason)[:200])
        self.session_task = session
        ended = getattr(session.result(), "ended", None) if session in done else None
        session_id = getattr(ended, "session_id", None) or None
        from scripts.act_emit import publish_play_opened

        await publish_play_opened(self.bus, dispatch_id=req.dispatch_id, session_id=session_id)
        follower = asyncio.create_task(self._follow(req.dispatch_id, session))
        self._followers.add(follower)
        follower.add_done_callback(self._followers.discard)
        logger.info("retro_leisure_started dispatch_id=%s session_id=%s", req.dispatch_id, session_id)
        return RetroLeisureStartResultV1(dispatch_id=req.dispatch_id, state="started", session_id=session_id,
                                         object_ref=retro_leisure_object_ref(req.dispatch_id))

    async def _follow(self, dispatch_id: str, session: asyncio.Task) -> None:
        """Wait for the session to end, then export it (which closes the play act)."""
        from scripts.act_emit import publish_play_closed
        from scripts.retro_memory_export import export_leisure_outcome

        try:
            outcome = await session
        except asyncio.CancelledError:
            await publish_play_closed(self.bus, dispatch_id=dispatch_id, session_id=None, produced=False,
                                      note="hub_stopped")
            raise
        except Exception:  # noqa: BLE001
            logger.warning("retro_leisure_run_failed dispatch_id=%s", dispatch_id, exc_info=True)
            await publish_play_closed(self.bus, dispatch_id=dispatch_id, session_id=None, produced=False,
                                      note="runner_failed")
            return
        ended = getattr(outcome, "ended", None)
        if ended is None:
            await publish_play_closed(self.bus, dispatch_id=dispatch_id, session_id=None, produced=False,
                                      note=f"no_session:{getattr(outcome, 'refusal', None)}"[:200])
            return
        result = await export_leisure_outcome(pool=self.pool_provider(), bus=self.bus, ended=ended)
        logger.info("retro_leisure_exported dispatch_id=%s session_id=%s state=%s card_id=%s",
                    dispatch_id, result.session_id, result.state, result.card_id)

    async def handle(self, envelope: BaseEnvelope) -> None:
        reply = str(envelope.reply_to or "")
        # Never answer on a subject outside the start reply prefix.
        if envelope.kind != RETRO_LEISURE_START_REQUEST_KIND or not reply.startswith(RETRO_LEISURE_START_REPLY_PREFIX):
            return
        try:
            req = RetroLeisureStartRequestV1.model_validate(envelope.payload)
        except ValueError:
            logger.warning("retro_leisure_request_invalid correlation_id=%s", envelope.correlation_id, exc_info=True)
            return
        result = await self.start(req)
        out = BaseEnvelope(kind=RETRO_LEISURE_START_RESULT_KIND, source=self.source_ref,
                           correlation_id=envelope.correlation_id, payload=result.model_dump(mode="json"))
        await self.bus.publish(reply, out)

    async def _run(self) -> None:
        while True:
            try:
                async with self.bus.subscribe(RETRO_LEISURE_START_REQUEST_CHANNEL) as pubsub:
                    async for raw in self.bus.iter_messages(pubsub):
                        try:
                            decoded = self.bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                # Its own task: a start waits up to START_GRACE_SEC, and the
                                # subscription must keep reading meanwhile.
                                t = asyncio.create_task(self.handle(decoded.envelope))
                                self._followers.add(t)
                                t.add_done_callback(self._followers.discard)
                        except Exception:  # noqa: BLE001
                            logger.warning("retro_leisure_message_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("retro_leisure_listener_reconnecting", exc_info=True)
                await asyncio.sleep(1)

    async def begin(self, bus: Any) -> None:
        self.bus = bus
        self.task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        tasks = [t for t in (self.task, self.session_task, *self._followers) if t is not None]
        for t in tasks:
            t.cancel()
        for t in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await t
        self.task = self.session_task = None


__all__ = ["RetroLeisureListener", "START_GRACE_SEC", "UNAVAILABLE_HOLD_SEC", "gate_verdict", "import_runner"]
