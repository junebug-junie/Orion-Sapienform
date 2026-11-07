"""The disk rule: ask cortex-exec to prune dangling docker images when the docker volume is full.

Ladder retirement PR 4 (docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md).
Before this, only the proposal -> policy -> dispatch ladder asked for ``skills.runtime.image_prune.v1``
(the ``prune_dangling_images`` template). Build cache and stopped containers are no longer asked for
automatically; they stay on the hand-run skill catalogue (Juniper, decision 4).

One tick (the watcher's own 30 s loop):

1. Read the docker data volume (``shutil.disk_usage`` on the read-only bind mount, the same
   filesystem and the same formula as the skill's own ``_builder_prune_disk_usage``).
2. Take the clock AFTER that read, and after reading this rule's own state.
3. ``orion.hardware_watch.rules.image_prune_verdict`` decides: one ask at a time, a start delay, never
   on an unreadable disk, never below the skill's own 70 % floor, then the cooldown after an answered
   ask (12 h) or the retry wait after an unanswered one (30 min).
4. An ask runs in the background: one bus RPC to cortex-exec on ``orion:cortex:exec:request:background``,
   the same request dispatch sent (``PlanExecutionRequest`` for the verb, ``skill_args.mode=execute``).
   The skill then runs its own gate (disk >= 70 % AND >= 100 dangling images) and its own prune.

``GET /health`` -> ``image_prune`` shows the last tick's reading and verdict, the ask in flight, the last
answer, and counts of every verdict and outcome.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from orion.hardware_watch.rules import (
    DiskReading, ImagePruneRuleConfig, ImagePruneState, ImagePruneVerdict, image_prune_verdict,
)

logger = logging.getLogger("orion-hardware-watch.image_prune")

IMAGE_PRUNE_VERB = "skills.runtime.image_prune.v1"
ORIGIN = "hardware_watch.disk_rule"
#: A disk read that takes longer than this reads as unreadable, so a stuck mount cannot stall the
#: heat and AC rules that share the tick loop.
DISK_READ_TIMEOUT_SEC = 5.0

#: (request) -> reply payload dict. Raises on a transport failure or timeout.
Ask = Callable[[str], Awaitable[dict]]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def read_docker_volume(path: str) -> DiskReading:
    """``shutil.disk_usage`` on the bind mount. A path that is not a mount point reads as an error:
    Docker creates an empty directory when a bind source is missing, and its usage would be the
    container's own overlay filesystem, a plausible but wrong number (same check as
    orion-biometrics' ``collect_disk_capacity``)."""
    try:
        if not path or not os.path.isdir(path):
            return DiskReading(None, "not_mounted")
        if not os.path.ismount(path):
            return DiskReading(None, "not_a_mount_point")
        total, used, _free = shutil.disk_usage(path)
        if total <= 0:
            return DiskReading(None, "zero_total_bytes")
        return DiskReading(round(100.0 * used / total, 3))
    except Exception as exc:  # noqa: BLE001 -- an unreadable disk never asks
        return DiskReading(None, f"{type(exc).__name__}: {exc}"[:200])


def skill_answer(payload: dict) -> dict[str, Any]:
    """The skill's own result from a cortex-exec reply: ``result.final_text`` is the skill's JSON
    (``verb_adapters._skill_result_output``; cortex-exec sets it for a single local skill step, which is
    what dispatch's stored results show too). It does not fall back to ``steps[*].result``. Returns
    ``{"outcome": <decision>, ...}``, or ``{"outcome": "unreadable_reply"}`` when the reply carries no
    skill result."""
    if isinstance(payload, dict) and payload.get("ok") is False:
        return {"outcome": "exec_error", "error": str(payload.get("error") or "")[:300]}
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        return {"outcome": "unreadable_reply"}
    text = result.get("final_text")
    try:
        data = json.loads(text) if isinstance(text, str) and text.strip() else None
    except (json.JSONDecodeError, ValueError):
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("decision"), str):
        return {"outcome": "unreadable_reply", "plan_status": result.get("status")}
    dangling = data.get("dangling_before") if isinstance(data.get("dangling_before"), dict) else {}
    disk = data.get("disk_before") if isinstance(data.get("disk_before"), dict) else {}
    return {
        "outcome": data["decision"],
        "plan_status": result.get("status"),
        "acted": data.get("acted"),
        "reason": data.get("reason"),
        "skill_disk_used_pct": disk.get("used_pct"),
        "dangling_before": dangling.get("dangling_count"),
        "images_removed": data.get("images_removed"),
        "bytes_reclaimed": data.get("bytes_reclaimed"),
    }


class ImagePruneRule:
    def __init__(self, *, enabled: bool, mount_path: str, cfg: ImagePruneRuleConfig, ask: Ask,
                 read_disk: Callable[[str], DiskReading] = read_docker_volume,
                 clock: Callable[[], datetime] = _utcnow):
        self.enabled = bool(enabled)
        self.mount_path = mount_path
        self.cfg = cfg
        self._ask = ask
        self._read_disk = read_disk
        self.clock = clock
        self.started_at = clock()
        self._in_flight: asyncio.Task | None = None
        self._last_finished_at: datetime | None = None
        self._last_outcome: str | None = None
        self.last_tick: dict[str, Any] | None = None
        self.last_ask: dict[str, Any] | None = None
        self.asks = 0
        self.verdicts: Counter[str] = Counter()
        self.outcomes: Counter[str] = Counter()

    def _state(self) -> ImagePruneState:
        return ImagePruneState(started_at=self.started_at,
                               in_flight=self._in_flight is not None and not self._in_flight.done(),
                               last_finished_at=self._last_finished_at, last_outcome=self._last_outcome)

    async def tick(self) -> ImagePruneVerdict | None:
        """One evaluation. Returns the verdict (None when the rule is switched off)."""
        if not self.enabled:
            return None
        try:
            disk = await asyncio.wait_for(asyncio.to_thread(self._read_disk, self.mount_path),
                                          timeout=DISK_READ_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            disk = DiskReading(None, "read_timeout")
        state = self._state()
        now = self.clock()   # after both reads: an ask that just finished is age 0, never negative
        verdict = image_prune_verdict(disk, now, state, self.cfg)
        self.verdicts[verdict.reason] += 1
        self.last_tick = {"at": now.isoformat(), "disk_used_pct": disk.used_pct, "disk_error": disk.error,
                          "verdict": verdict.reason, "wait_sec": round(verdict.wait_sec, 1)}
        if verdict.ask:
            self.asks += 1
            attempt_id = f"image_prune_disk_rule:{uuid.uuid4()}"
            self.last_ask = {"attempt_id": attempt_id, "started_at": now.isoformat(),
                             "disk_used_pct": disk.used_pct, "finished_at": None, "outcome": None}
            logger.info("image_prune_ask attempt=%s disk_used_pct=%s", attempt_id, disk.used_pct)
            self._in_flight = asyncio.create_task(self._run(attempt_id))
        return verdict

    async def _run(self, attempt_id: str) -> None:
        try:
            answer = skill_answer(await self._ask(attempt_id))
        except Exception as exc:  # noqa: BLE001 -- no reply is an unanswered ask, never a crash
            answer = {"outcome": "rpc_error", "error": f"{type(exc).__name__}: {exc}"[:300]}
        finished = self.clock()
        self._last_finished_at = finished
        self._last_outcome = str(answer.get("outcome"))
        self.outcomes[self._last_outcome] += 1
        if self.last_ask is not None and self.last_ask.get("attempt_id") == attempt_id:
            self.last_ask.update(answer, finished_at=finished.isoformat())
        if self._last_outcome == "would_act":
            # The rule always asks in execute mode: a preview answer means that mode was lost between
            # here and the skill. Retried after retry_sec, not held for the cooldown, and said loudly.
            logger.warning("image_prune_preview_despite_execute attempt=%s reply=%s", attempt_id, answer)
        logger.info("image_prune_answer attempt=%s outcome=%s dangling_before=%s bytes_reclaimed=%s err=%s",
                    attempt_id, answer.get("outcome"), answer.get("dangling_before"),
                    answer.get("bytes_reclaimed"), answer.get("error"))

    async def stop(self) -> None:
        if self._in_flight is not None and not self._in_flight.done():
            self._in_flight.cancel()
            await asyncio.gather(self._in_flight, return_exceptions=True)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "verb": IMAGE_PRUNE_VERB,
            "mount_path": self.mount_path,
            "min_disk_pct": self.cfg.min_disk_pct,
            "cooldown_sec": self.cfg.cooldown_sec,
            "retry_sec": self.cfg.retry_sec,
            "start_delay_sec": self.cfg.start_delay_sec,
            "started_at": self.started_at.isoformat(),
            "in_flight": self._state().in_flight,
            "last_tick": self.last_tick,
            "last_ask": self.last_ask,
            "asks": self.asks,
            "verdicts": dict(self.verdicts),
            "outcomes": dict(self.outcomes),
        }


def build_bus_ask(bus: Any, *, request_channel: str, result_prefix: str, source: Any,
                  timeout_sec: float) -> Ask:
    """The ask: one RPC to cortex-exec, the request dispatch's cortex client sent for this verb.
    ``require_listener``: when cortex-exec is down (mid-restart) the ask fails at once instead of
    waiting the whole 720 s budget."""
    from orion.cognition.plan_loader import build_plan_for_verb
    from orion.core.bus.bus_schemas import BaseEnvelope
    from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest

    async def ask(attempt_id: str) -> dict:
        req = PlanExecutionRequest(
            plan=build_plan_for_verb(IMAGE_PRUNE_VERB, mode="brain"),
            args=PlanExecutionArgs(request_id=attempt_id, trigger_source=source.name,
                                   extra={"mode": "brain", "origin": ORIGIN, "attempt_id": attempt_id}),
            # ctx["skill_args"] is read by cortex-exec's executor (the path dispatch used). `execute`:
            # the skill's own measured gate is the safety, as it was behind dispatch.
            context={"skill_args": {"mode": "execute"}, "origin": ORIGIN, "attempt_id": attempt_id},
        )
        corr = str(uuid.uuid4())
        reply = f"{result_prefix}:{corr}"
        env = BaseEnvelope(kind=req.kind, source=source, correlation_id=corr, reply_to=reply,
                           payload=req.model_dump(mode="json"))
        msg = await bus.rpc_request(request_channel, env, reply_channel=reply, timeout_sec=timeout_sec,
                                    require_listener=True)
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"cortex-exec reply undecodable: {decoded.error}")
        payload = decoded.envelope.payload
        if not isinstance(payload, dict):
            raise RuntimeError("cortex-exec reply payload is not an object")
        return payload

    return ask
