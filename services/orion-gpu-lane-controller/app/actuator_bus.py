"""Stage 4.2 bridge: the GPU pool's actuation requests, executed by gpu2.transition().

Subscribes to ``orion:gpu_pool:actuate:request`` (GpuActuateV1) and answers on
``orion:gpu_pool:actuate:result`` (GpuActuateResultV1): ``accepted``, then one ``progress`` per
phase, then exactly one terminal ``succeeded`` | ``failed`` | ``refused``. The drain, stop, start,
readiness and rollback steps are gpu2.transition()'s, unchanged; only who asks, and the fence,
differ (app/pool_fence.py).

Nothing moves unless GPU2_AUTHORITY=pool. Under the default ``durable`` every request addressed to
this actuator is refused ``authority_durable`` (so a pool that sends early gets a clear answer, not
a timeout), and durable-runs keeps the HTTP activate route exactly as before.

Spec: docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from loguru import logger
from pydantic import ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.gpu_pool import (
    GPU_ACTUATE_RESULT_KIND,
    GPU_POOL_ACTUATE_RESULT_CHANNEL,
    GpuActuateResultV1,
    GpuActuateV1,
)
from orion.schemas.gpu_slot import GpuSlotRequestV1

from . import gpu2, pool_fence
from .settings import settings

Publish = Callable[[GpuActuateResultV1, Any], Awaitable[None]]

_admit_lock = asyncio.Lock()   # validate + persist + start is one step; two requests never interleave it
_task: asyncio.Task | None = None
_current: dict[str, Any] | None = None   # the in-flight request, for replays and `status`


def _observed_state(snap: dict[str, Any]) -> str:
    if snap.get("error") or snap.get("state") == "unknown":
        return "unknown"
    if any(r.get("state") == "running" for r in snap.get("containers") or []):
        return "running"
    if snap.get("state") == "absent":
        return "absent"
    if snap.get("state") in {"exited", "dead", "created"}:
        return "exited"
    return "unknown"


async def observe() -> dict[str, str]:
    """Role -> container state on gpu2, for the pool to reconcile from. Never raises."""
    try:
        snaps = await asyncio.to_thread(gpu2.snapshots)
    except Exception:  # noqa: BLE001
        return {role: "unknown" for role in pool_fence.TARGET_ROLES.values()}
    return {pool_fence.TARGET_ROLES[t]: _observed_state(snap) for t, snap in snaps.items()
            if t in pool_fence.TARGET_ROLES}


def _result(req: GpuActuateV1, status: str, *, started: float | None = None, **fields) -> GpuActuateResultV1:
    elapsed = None if started is None else round((time.monotonic() - started) * 1000.0, 1)
    return GpuActuateResultV1(action_id=req.action_id, generation=req.generation, role=req.role,
                              action=req.action, status=status, elapsed_ms=elapsed, **fields)


def _salvage(payload: Any) -> GpuActuateV1 | None:
    """Enough of an invalid request to address a refusal to it, or None (unanswerable). Only a
    request explicitly addressed to this actuator is answered -- never on another host's behalf."""
    if not isinstance(payload, dict) or payload.get("actuator") != settings.GPU_POOL_ACTUATOR_NAME:
        return None
    try:
        salvaged = GpuActuateV1.model_construct(
            action_id=str(payload["action_id"])[:128], generation=max(1, int(payload["generation"])),
            role=str(payload["role"])[:64] or "unknown", action=payload["action"])
        _result(salvaged, "refused")  # must itself be a valid result, or stay silent
        return salvaged
    except Exception:  # noqa: BLE001
        return None


def _recorded(row: dict[str, Any]) -> GpuActuateResultV1 | None:
    try:
        return GpuActuateResultV1.model_validate(_strip(row))
    except Exception:  # noqa: BLE001 -- a row from an older schema: answer from `status` instead
        return None


async def handle(payload: Any, publish: Publish, corr: Any = None) -> None:
    """One GpuActuateV1. Every path that is addressed to this actuator publishes a result."""
    try:
        req = GpuActuateV1.model_validate(payload)
    except ValidationError as exc:
        salvaged = _salvage(payload)
        if salvaged is None or salvaged.action not in ("load", "unload", "status"):
            logger.warning("gpu_actuate_unanswerable errors={}", exc.error_count())
            return
        await publish(_result(salvaged, "refused", reason=f"invalid_request:{exc.errors()[0]['loc']}"[:2000]), corr)
        return
    if req.actuator != settings.GPU_POOL_ACTUATOR_NAME:
        return  # another host's actuator; not ours to answer
    if req.deadline_at.tzinfo is None:
        # A naive datetime cannot be compared with UTC now; refuse rather than crash unanswered.
        return await _refuse(req, "invalid_request:deadline_at_naive", publish, corr)
    await _dispatch(req, publish, corr)


async def _refuse(req: GpuActuateV1, reason: str, publish: Publish, corr: Any) -> None:
    logger.info("gpu_actuate_refused action_id={} role={} action={} generation={} reason={}",
                req.action_id, req.role, req.action, req.generation, reason)
    await publish(_result(req, "refused", reason=reason), corr)


async def _dispatch(req: GpuActuateV1, publish: Publish, corr: Any) -> None:
    global _task, _current
    if not gpu2.pool_authority():
        return await _refuse(req, "authority_durable", publish, corr)
    if not settings.GPU2_ENABLED:
        return await _refuse(req, "gpu2_disabled", publish, corr)
    status_view = None
    async with _admit_lock:
        try:
            state = await asyncio.to_thread(pool_fence.read_state)
        except Exception as exc:  # noqa: BLE001 -- unreadable fence: act on nothing
            return await _refuse(req, f"fence_state_unreadable:{type(exc).__name__}", publish, corr)
        recorded = state["actions"].get(req.action_id)
        if recorded is not None and (replay := _recorded(recorded)) is not None:
            # Idempotent replay: the recorded terminal result, never a second transition.
            return await publish(replay, corr)
        if _current is not None and _current["action_id"] == req.action_id:
            return await publish(_result(req, "progress", phase=_current.get("phase")), corr)
        if req.action == "status":
            status_view = _status_view(req, state)
    if status_view is not None:
        # Outside the admit lock: observe() shells out to docker (up to ~60s) and must not starve
        # a concurrent load's `accepted` past the pool's actuate_ack_sec.
        return await _status(req, status_view, publish, corr)
    async with _admit_lock:
        if req.deadline_at <= datetime.now(timezone.utc):
            return await _refuse(req, "deadline_passed", publish, corr)
        if req.profile is not None:
            return await _refuse(req, "profile_unsupported", publish, corr)  # stage 4 never sends one
        try:
            cfg = await asyncio.to_thread(pool_fence.load_config)
            target = pool_fence.resolve(cfg, role=req.role, action=req.action, cards=req.cards,
                                        digest=req.launch_digest)
        except pool_fence.Refusal as exc:
            return await _refuse(req, str(exc), publish, corr)
        except Exception as exc:  # noqa: BLE001 -- unparseable own config: act on nothing
            return await _refuse(req, f"config_unloadable:{type(exc).__name__}", publish, corr)
        if (_task is not None and not _task.done()) or gpu2._lock.locked():
            return await _refuse(req, "busy", publish, corr)
        try:
            # Re-read after every await above: a transition that finished meanwhile recorded its
            # result in the file, and writing an older snapshot back would erase it.
            state = await asyncio.to_thread(pool_fence.read_state)
        except Exception as exc:  # noqa: BLE001
            return await _refuse(req, f"fence_state_unreadable:{type(exc).__name__}", publish, corr)
        if req.generation <= pool_fence.last_generation(state, req.cards):
            return await _refuse(req, "stale_generation", publish, corr)
        flight = {"action_id": req.action_id, "generation": req.generation, "role": req.role,
                  "action": req.action, "cards": sorted(req.cards), "launch_digest": req.launch_digest}
        state["generations"][pool_fence.card_set(req.cards)] = req.generation
        state["in_flight"] = flight
        try:
            # Durable before any container is touched: a restart can never re-admit this generation.
            await asyncio.to_thread(pool_fence.write_state, state)
        except Exception as exc:  # noqa: BLE001
            return await _refuse(req, f"fence_state_unwritable:{type(exc).__name__}", publish, corr)
        _current = {**flight, "phase": None}
        started = time.monotonic()
        # The generation is spent; the action must run even if the ack cannot be published (the
        # pool then reconciles via `status`), or it would be stranded in flight forever.
        _task = asyncio.create_task(_run(req, target, started, publish, corr))
        try:
            await publish(_result(req, "accepted", started=started), corr)
        except Exception:  # noqa: BLE001
            logger.warning("gpu_actuate_accepted_publish_failed action_id={}", req.action_id)


async def _run(req: GpuActuateV1, target: str, started: float, publish: Publish, corr: Any) -> None:
    global _current

    async def progress(phase: str) -> None:
        if _current is not None:
            _current["phase"] = phase
        await publish(_result(req, "progress", started=started, phase=phase), corr)

    slot_req = GpuSlotRequestV1(slot="circe-gpu2", target=target, operation_id=req.action_id,
                                generation=req.generation)
    token = gpu2.progress_hook.set(progress)
    try:
        outcome = await gpu2.transition(slot_req)
    except Exception as exc:  # noqa: BLE001 -- transition() catches its own; this is belt and braces
        outcome = {"status": "failed", "error": f"transition_crashed:{type(exc).__name__}"}
    finally:
        gpu2.progress_hook.reset(token)
    observed = await observe()
    if outcome.get("status") in {"success", "noop"}:
        final = _result(req, "succeeded", started=started, observed=observed,
                        reason="noop" if outcome["status"] == "noop" else None)
    else:
        # restored: only meaningful (and only allowed by the schema) on a failed load. None means
        # no rollback ran -- nothing had been evicted yet; `observed` shows what is on the card.
        restored = outcome.get("restored") if req.action == "load" else None
        final = _result(req, "failed", started=started, observed=observed, restored=restored,
                        reason=str(outcome.get("error") or "transition_failed")[:2000])
    try:
        await asyncio.to_thread(_finish, req.action_id,
                                {**final.model_dump(mode="json"), "cards": sorted(req.cards)})
    except Exception:  # noqa: BLE001 -- still publish; a replay then re-derives from `status`
        logger.exception("gpu_actuate_record_failed action_id={}", req.action_id)
    _current = None
    await publish(final, corr)


def _finish(action_id: str, result: dict[str, Any]) -> None:
    state = pool_fence.read_state()
    pool_fence.record(state, action_id, result)
    if (state.get("in_flight") or {}).get("action_id") == action_id:
        state["in_flight"] = None
    pool_fence.write_state(state)


def _status_view(req: GpuActuateV1, state: dict[str, Any]) -> dict[str, Any]:
    """What `status` reports, copied under the admit lock so it is one consistent snapshot."""
    cards = pool_fence.card_set(req.cards)
    last = None
    for action_id in reversed(state["order"]):
        row = state["actions"].get(action_id) or {}
        if pool_fence.card_set(row.get("cards") or []) == cards:
            last = row
            break
    return {"last": last, "last_generation": pool_fence.last_generation(state, req.cards),
            "in_flight": _current["action_id"] if _current else None,
            "phase": _current.get("phase") if _current else None}


async def _status(req: GpuActuateV1, view: dict[str, Any], publish: Publish, corr: Any) -> None:
    """Read-only: re-publish the last recorded result for the card set (so a restarted pool adopts
    it by its own action_id), then answer this request with the observed containers."""
    try:
        cfg = await asyncio.to_thread(pool_fence.load_config)
        pool_fence.resolve(cfg, role=req.role, action="status", cards=req.cards, digest=None)
    except pool_fence.Refusal as exc:
        return await _refuse(req, str(exc), publish, corr)
    except Exception as exc:  # noqa: BLE001
        return await _refuse(req, f"config_unloadable:{type(exc).__name__}", publish, corr)
    last = view["last"]
    if last is not None and view["in_flight"] is None and (replay := _recorded(last)) is not None:
        await publish(replay, corr)
    # State travels in the structured status-only fields (schema, stage 4.3); `reason` is for humans.
    last_id = (last or {}).get("action_id")
    reason = (f"last generation {view['last_generation']}, last action {last_id or 'none'}, "
              f"in flight {view['in_flight'] or 'none'}")
    await publish(_result(req, "succeeded", observed=await observe(), reason=reason, phase=view["phase"],
                          in_flight=view["in_flight"] is not None, last_action_id=last_id), corr)


def _strip(row: dict[str, Any]) -> dict[str, Any]:
    """Recorded rows may carry fence-only keys (cards, launch_digest) the result schema forbids."""
    return {k: v for k, v in row.items() if k in GpuActuateResultV1.model_fields}


def bus_publisher(bus) -> Publish:
    async def publish(result: GpuActuateResultV1, corr: Any) -> None:
        try:
            cid = uuid.UUID(str(corr)) if corr else uuid.uuid4()
        except ValueError:
            cid = uuid.uuid4()
        await bus.publish(GPU_POOL_ACTUATE_RESULT_CHANNEL, BaseEnvelope(
            kind=GPU_ACTUATE_RESULT_KIND, correlation_id=cid, payload=result.model_dump(mode="json"),
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION,
                              node=settings.NODE_NAME)))
        logger.info("gpu_actuate_result action_id={} status={} phase={} reason={}",
                    result.action_id, result.status, result.phase, result.reason)
    return publish
