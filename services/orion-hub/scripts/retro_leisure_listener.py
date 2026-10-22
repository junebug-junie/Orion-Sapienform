"""Bus RPC adapter for the retro leisure start (orion/schemas/retro_leisure.py).

Orion plays Zelda, section E. The motor allocator (orion-execution-dispatch-runtime, route
``retro_leisure_session``) asks; the leisure runner decides and plays. This class validates the
envelope, asks the runner's own gate, starts the session, opens the ``play`` act, and answers. Same
trusted internal bus boundary and reply-subject rule as ``scripts/world_pulse_read_pull_listener.py``.

The runner is Lane B's ``scripts/retro_leisure.py`` object, found through ``runner_provider`` (Hub
main: ``app.state.retro_leisure_runner``). The interface read here, and nothing else:

- ``should_start()`` (sync or async) -> the gate verdict, as any of: a bool; a tuple
  ``(ok, reason[, retry_after_sec])``; or an object/dict with ``ok`` (or ``allowed``), ``reason`` and
  ``retry_after_sec``. A refusal must carry a named reason; an unnamed one is ``unnamed_refusal``.
- ``start(dispatch_id=..., correlation=...)`` (sync or async) -> the session id, as a str or an
  object/dict with ``session_id``. It raises when the session cannot start.

No runner (Hub booted without it) -> ``runner_unavailable`` with a 300 s hold, so a missing Lane B
costs the allocator one refused RPC per hold, never a fake session.
"""

from __future__ import annotations

import asyncio
import inspect
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
# A replayed dispatch id is answered with its first result, never a second session.
REPLAY_CACHE = 256


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _field(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def gate_verdict(raw: Any) -> tuple[bool, str | None, float | None]:
    """Normalise the runner's ``should_start()`` answer to (ok, reason, retry_after_sec)."""
    if isinstance(raw, bool):
        ok, reason, retry = raw, None, None
    elif isinstance(raw, tuple):
        ok = bool(raw[0])
        reason = raw[1] if len(raw) > 1 else None
        retry = raw[2] if len(raw) > 2 else None
    else:
        ok_value = _field(raw, "ok", "allowed")
        ok = bool(ok_value)
        reason = _field(raw, "reason", "refusal")
        retry = _field(raw, "retry_after_sec")
    retry_f = float(retry) if isinstance(retry, (int, float)) and retry >= 0 else None
    if ok:
        return True, None, None
    return False, (str(reason)[:200] if reason else "unnamed_refusal"), retry_f


class RetroLeisureListener:
    def __init__(self, runner_provider: Callable[[], Any], source_ref: Any) -> None:
        self.runner_provider = runner_provider
        self.source_ref = source_ref
        self.bus: Any = None
        self.task: asyncio.Task | None = None
        self._seen: OrderedDict[str, RetroLeisureStartResultV1] = OrderedDict()
        self._lock = asyncio.Lock()

    def _refused(self, req: RetroLeisureStartRequestV1, reason: str, retry: float | None,
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
        if runner is None or not hasattr(runner, "should_start") or not hasattr(runner, "start"):
            return self._refused(req, "runner_unavailable", UNAVAILABLE_HOLD_SEC)
        try:
            ok, reason, retry = gate_verdict(await _maybe_await(runner.should_start()))
        except Exception as exc:  # noqa: BLE001
            logger.warning("retro_leisure_gate_failed dispatch_id=%s", req.dispatch_id, exc_info=True)
            return self._refused(req, "gate_failed", UNAVAILABLE_HOLD_SEC, {"error": type(exc).__name__})
        if not ok:
            logger.info("retro_leisure_refused dispatch_id=%s refusal=%s retry_after=%s",
                        req.dispatch_id, reason, retry)
            return self._refused(req, reason or "unnamed_refusal", retry)
        try:
            started = await _maybe_await(runner.start(dispatch_id=req.dispatch_id, correlation=dict(req.correlation)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("retro_leisure_start_failed dispatch_id=%s", req.dispatch_id, exc_info=True)
            return self._refused(req, "start_failed", UNAVAILABLE_HOLD_SEC, {"error": type(exc).__name__})
        session_id = started if isinstance(started, str) else _field(started, "session_id")
        if not session_id:
            return self._refused(req, "start_failed", UNAVAILABLE_HOLD_SEC, {"error": "no_session_id"})
        from scripts.act_emit import publish_play_opened

        await publish_play_opened(self.bus, dispatch_id=req.dispatch_id, session_id=str(session_id))
        logger.info("retro_leisure_started dispatch_id=%s session_id=%s", req.dispatch_id, session_id)
        return RetroLeisureStartResultV1(dispatch_id=req.dispatch_id, state="started", session_id=str(session_id),
                                         object_ref=retro_leisure_object_ref(req.dispatch_id))

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
                                await self.handle(decoded.envelope)
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
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
            self.task = None


__all__ = ["RetroLeisureListener", "UNAVAILABLE_HOLD_SEC", "gate_verdict"]
