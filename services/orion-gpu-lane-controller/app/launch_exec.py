"""Stage 5.2: run any role's ``launch:`` block -- the generic "load this role on this card" actuator.

Spec: docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md
(Decision 2). Executes a ``pool_fence.LaunchPlan``; every compose file, service, profile and env var
*name* comes from this checkout's config/gpu_pool.yaml, fenced by ``launch_digest``. The only values
set are the card index (``cuda_env``) and an allow-listed profile (``profile_var``). No free-form
Docker: ``docker`` is the only binary, the subcommands are ``ps``/``stop``/``up -d --no-build
--no-deps``, and each names exactly one compose service.

The steps (the stage-4 gpu2 bridge's, per role instead of per fixed target; the bridge itself was
deleted in stage 5.6):

- **load**: for each evicted role that is running: drain (if it has ``drain``) -> stop. Then ``up``
  the seat with its env, un-drain it if it has ``drain``, and wait for ``ready`` up to
  ``timeout_sec``. On failure after touching an evicted role: stop the seat, restart the stopped
  evicted roles in reverse order (each waits for its ``ready``), un-drain any drained but not
  stopped. ``restored`` says whether that worked.
- **unload**: a running ``kind: llm`` seat must have every llama.cpp ``/slots`` idle; drain (if it
  has ``drain``) -> stop the seat, then start every evicted role and wait for ready.
- Generation + digest fence (``pool_fence.authority``) before every mutation; rollback checks
  identity and generation only, so a checkout edited mid-load cannot strand the card.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
import time
import urllib.request
from typing import Any, Callable, NamedTuple

from loguru import logger

from . import compose as dc
from . import pool_fence
from .pool_fence import LaunchPlan, RolePlan
from .settings import settings

lock = asyncio.Lock()
# actuator_bus sets this for the action it runs, so each step is published as a progress phase.
progress_hook = contextvars.ContextVar("launch_exec_progress_hook", default=None)

# Container states it is safe to act around (anything else -- paused, restarting, unknown -- stops).
SAFE_STATES = {"running", "absent", "exited", "dead", "created"}
STOPPED_STATES = {"absent", "exited", "dead", "created"}
READ_TIMEOUT_SEC = 30.0
POLL_SEC = 1.0   # drain/ready/resume poll interval


class Intent(NamedTuple):
    """What pool_fence.authority checks: the in-flight action id and its generation."""
    operation_id: str
    generation: int


def runner(timeout_sec: float) -> dc.SafeCommandRunner:
    """Only ``docker``, never through a shell. Replaced by a fake in tests."""
    return dc.SafeCommandRunner(allowed_commands={"docker"}, timeout_sec=timeout_sec)


def _http(url: str, payload: Any = None) -> Any:
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)


async def request(url: str, payload: Any = None) -> Any:
    return await asyncio.to_thread(_http, url, payload)


async def phase(name: str) -> None:
    hook = progress_hook.get()
    if hook is None:
        return
    try:
        # Bounded: a hung bus publish must not stall an action mid-drain.
        await asyncio.wait_for(hook(name), timeout=5)
    except Exception:  # noqa: BLE001 -- progress telemetry never changes an action's outcome
        logger.warning("launch_exec_progress_publish_failed phase={}", name)


def _target(plan: RolePlan) -> dc.ComposeTarget:
    return dc.ComposeTarget(key=plan.role, compose_relpath=plan.compose, env_relpath=plan.env_file or "",
                         compose_service=plan.service, profile=plan.compose_profile, extra_env=dict(plan.env))


def _snapshot_sync(plan: RolePlan) -> dict[str, Any]:
    return dc.snapshot(runner(READ_TIMEOUT_SEC), dc.repo_root(), _target(plan))


async def snapshot(plan: RolePlan) -> dict[str, Any]:
    return await asyncio.to_thread(_snapshot_sync, plan)


def observed_state(snap: dict[str, Any]) -> str:
    if snap.get("error") or snap.get("state") == "unknown":
        return "unknown"
    if any(r.get("state") == "running" for r in snap.get("containers") or []):
        return "running"
    if snap.get("state") == "absent":
        return "absent"
    if snap.get("state") in {"exited", "dead", "created"}:
        return "exited"
    return "unknown"


def _running(snap: dict[str, Any]) -> bool:
    return any(r.get("state") == "running" for r in snap.get("containers") or [])


async def observe(plans: dict[str, RolePlan]) -> dict[str, str]:
    """Role -> running|exited|absent|unknown for every launch role on this actuator. Never raises."""
    out: dict[str, str] = {}
    for role, plan in plans.items():
        try:
            out[role] = observed_state(await snapshot(plan))
        except Exception:  # noqa: BLE001
            out[role] = "unknown"
    return out


async def ready(plan: RolePlan) -> bool:
    """Ready = the ``ready`` path answers 2xx with ``{"ready": true}`` (diffusion-host /ready) or
    ``{"status": "ok"}`` (llama.cpp /health). Anything else, including an error, is not ready."""
    try:
        body = await request(plan.base_url + plan.ready)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(body, dict) and (body.get("ready") is True or body.get("status") == "ok")


async def wait_ready(plan: RolePlan, deadline: float | None = None) -> None:
    deadline = time.monotonic() + plan.timeout_sec if deadline is None else deadline
    while not await ready(plan):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"model_readiness_timeout:{plan.role}")
        await asyncio.sleep(POLL_SEC)


async def drain(plan: RolePlan) -> None:
    assert plan.drain is not None
    await request(plan.base_url + plan.drain.set_path, {"draining": True})
    # launch.drain has no timeout key: one controller-wide max wait for in-flight work to finish
    # (stage 5.6 renamed it from the gpu2-era GPU2_DRAIN_TIMEOUT_SEC; same meaning, same default).
    deadline = time.monotonic() + settings.GPU_LANE_DRAIN_TIMEOUT_SEC
    while True:
        state = await request(plan.base_url + plan.drain.status)
        if isinstance(state, dict) and state.get("draining") is True and state.get("in_flight") is False:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"drain_timeout:{plan.role}")
        await asyncio.sleep(POLL_SEC)


async def undrain(plan: RolePlan, *, deadline: float | None = None) -> None:
    """Resume a drained role; retried until ``deadline`` (monotonic) while it is still booting."""
    assert plan.drain is not None
    deadline = time.monotonic() if deadline is None else deadline
    while True:
        try:
            await request(plan.base_url + plan.drain.set_path, {"draining": False})
            return
        except Exception:  # noqa: BLE001
            if time.monotonic() >= deadline:
                raise RuntimeError(f"resume_failed:{plan.role}")
            await asyncio.sleep(POLL_SEC)


async def llm_idle(plan: RolePlan) -> bool:
    slots = await request(plan.base_url + "/slots")
    return isinstance(slots, list) and bool(slots) and all(
        isinstance(s, dict) and s.get("is_processing") is False for s in slots)


def _compose(plan: RolePlan, *args: str):
    root = dc.repo_root()
    target = _target(plan)
    env = {k: v for k, v in os.environ.items() if k not in plan.unset}
    env.update(plan.env)   # process env wins over --env-file for compose interpolation
    cmd = [*dc.base_cmd(target, root), *args, target.compose_service]
    return runner(settings.GPU_LANE_COMMAND_TIMEOUT_SEC).run(cmd, cwd=str(root), env=env)


async def stop(plan: RolePlan) -> None:
    try:
        proc = await asyncio.to_thread(_compose, plan, "stop")
    except Exception:  # noqa: BLE001
        raise RuntimeError(f"stop_failed:{plan.role}")
    if proc.returncode:
        raise RuntimeError(f"stop_failed:{plan.role}")
    snap = await snapshot(plan)
    if snap.get("error") or snap.get("state") not in STOPPED_STATES:
        raise RuntimeError(f"stop_unconfirmed:{plan.role}")


async def start(plan: RolePlan) -> None:
    logger.info("launch_exec up role={} service={} env={}", plan.role, plan.service, plan.env_text())
    # One readiness budget per role (resume + ready together): the pool sizes its stuck-actuator
    # timer from launch.timeout_sec, so the two waits must not each get the full budget.
    deadline = time.monotonic() + plan.timeout_sec
    try:
        proc = await asyncio.to_thread(_compose, plan, "up", "-d", "--no-build", "--no-deps")
    except Exception:  # noqa: BLE001
        raise RuntimeError(f"startup_failed:{plan.role}")
    if proc.returncode:
        raise RuntimeError(f"startup_failed:{plan.role}")
    await phase("ready_wait")
    if plan.drain is not None:
        # A drained service may come back still draining; resume it once it answers.
        await undrain(plan, deadline=deadline)
    await wait_ready(plan, deadline)


Authority = Callable[..., Any]


async def execute(plan: LaunchPlan, action: str, intent: Intent, *,
                  authority: Authority | None = None) -> dict[str, Any]:
    """Run one accepted ``load``/``unload``. Returns ``{"status": success|noop|failed, "error"?,
    "restored"?}``; actuator_bus maps it onto a GpuActuateResultV1."""
    authority = authority or pool_fence.authority
    async with lock:
        started = time.monotonic()
        seat = plan.seat
        logger.info("launch_exec {} role={} service={} env={} profile={} evicts={} action_id={} generation={}",
                    action, seat.role, seat.service, seat.env_text(), plan.profile,
                    ",".join(p.role for p in plan.evicts) or "-", intent.operation_id, intent.generation)
        try:
            await authority(intent, require_drained=False)
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": str(exc) if isinstance(exc, RuntimeError) else "stale_or_unknown_intent"}
        outcome: dict[str, Any] = {}
        try:
            if action == "load":
                outcome = await _load(plan, intent, authority)
            elif action == "unload":
                outcome = await _unload(plan, intent, authority)
            else:
                outcome = {"status": "failed", "error": f"unsupported_action:{action}"}
            return outcome
        finally:
            logger.info("launch_exec {} role={} status={} error={} restored={} seconds={:.2f}", action, seat.role,
                        outcome.get("status"), outcome.get("error"), outcome.get("restored"),
                        time.monotonic() - started)


async def _observe_all(plan: LaunchPlan) -> dict[str, dict[str, Any]]:
    snaps: dict[str, dict[str, Any]] = {}
    for rp in (plan.seat, *plan.evicts):
        snap = await snapshot(rp)
        if snap.get("error") or snap.get("state") not in SAFE_STATES:
            # Unknown Docker state never authorizes starting something else on the card.
            raise RuntimeError(f"container_state_not_safe:{rp.role}")
        snaps[rp.role] = snap
    return snaps


def _error(exc: BaseException) -> str:
    return str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__


async def _load(plan: LaunchPlan, intent: Intent, authority: Authority) -> dict[str, Any]:
    seat = plan.seat
    drained: list[RolePlan] = []
    stopped: list[RolePlan] = []
    try:
        snaps = await _observe_all(plan)
        running = {r for r, s in snaps.items() if _running(s)}
        if seat.role in running:
            if running & {p.role for p in plan.evicts}:
                raise RuntimeError("seat_and_evicted_both_running")
            if await ready(seat):
                return {"status": "noop"}
        await authority(intent)
        for rp in plan.evicts:
            if rp.role not in running:
                continue
            if rp.drain is not None:
                await phase("draining")
                drained.append(rp)
                await drain(rp)
                await authority(intent)
            await phase("stopping")
            await stop(rp)
            stopped.append(rp)
        await authority(intent)
        await phase("starting")
        await start(seat)
        return {"status": "success"}
    except Exception as exc:  # noqa: BLE001 -- no model output or subprocess stderr in evidence
        error = _error(exc)
        out: dict[str, Any] = {"status": "failed", "error": error}
        if drained or stopped:
            try:
                await phase("rolling_back")
                # Identity and generation only: a checkout edited mid-load must not block returning
                # the card to its previous residents.
                await authority(intent, require_drained=False)
                if stopped:
                    await stop(seat)
                for rp in reversed(stopped):
                    await start(rp)
                for rp in drained:
                    if rp not in stopped:
                        await undrain(rp)
                out["restored"] = True
            except Exception:  # noqa: BLE001
                out["restored"] = False
                out["error"] = error + ":restoration_failed"
        logger.error("launch_exec_load_failed role={} action_id={} reason={}", seat.role,
                     intent.operation_id, out["error"])
        return out


async def _unload(plan: LaunchPlan, intent: Intent, authority: Authority) -> dict[str, Any]:
    seat = plan.seat
    seat_drained = False
    seat_stopped = False
    try:
        snaps = await _observe_all(plan)
        running = {r for r, s in snaps.items() if _running(s)}
        evicted_running = [p for p in plan.evicts if p.role in running]
        if seat.role in running and evicted_running:
            raise RuntimeError("seat_and_evicted_both_running")
        if seat.role not in running and len(evicted_running) == len(plan.evicts):
            if all([await ready(p) for p in plan.evicts]):
                return {"status": "noop"}
        await authority(intent)
        if seat.role in running:
            if seat.kind == "llm":
                try:
                    idle = await llm_idle(seat)
                except Exception:  # noqa: BLE001
                    idle = False
                if not idle:
                    raise RuntimeError(f"upstream_not_idle:{seat.role}")
            if seat.drain is not None:
                await phase("draining")
                seat_drained = True
                await drain(seat)
            await authority(intent)   # still the newest generation, right before the stop
            await phase("stopping")
            await stop(seat)
            seat_stopped = True
        await authority(intent)
        await phase("starting")
        for rp in plan.evicts:
            await start(rp)
        return {"status": "success"}
    except Exception as exc:  # noqa: BLE001
        error = _error(exc)
        if seat_drained and not seat_stopped:
            try:
                await undrain(seat)   # the seat keeps serving; never leave it silently draining
            except Exception:  # noqa: BLE001
                error += ":resume_failed"
        logger.error("launch_exec_unload_failed role={} action_id={} reason={}", seat.role,
                     intent.operation_id, error)
        return {"status": "failed", "error": error}
