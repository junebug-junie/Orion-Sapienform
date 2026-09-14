"""Second fixed slot; authority intent fences HTTP retries across restarts.

Single uvicorn process, independent of GPU1's lock. Every Docker invocation
names one of two locally fixed services. No ordinary transition builds images.
"""
import asyncio
import json
import time
import urllib.request
from loguru import logger
from . import lane_control as lc
from .settings import settings

_lock = asyncio.Lock()
_job = None
_state = {"state": "neither", "error": None}


def targets():
    return {
        "diffusion": lc.LaneTarget("diffusion", "services/orion-diffusion-host/docker-compose.yml",
            "services/orion-diffusion-host/.env", "diffusion-host", extra_env={"CUDA_VISIBLE_DEVICES": "2"}),
        "agent-burst": lc.LaneTarget("agent-burst", "services/orion-llamacpp-host/docker-compose.atlas-workers.yml",
            "services/orion-llamacpp-host/.env", "atlas-agent-burst", profile="agent-burst"),
    }


def http(url, payload=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)


async def request(url, payload=None):
    return await asyncio.to_thread(http, url, payload)


def snapshots():
    runner = lc.SafeCommandRunner(allowed_commands={"docker"}, timeout_sec=30)
    return {key: lc._snapshot(runner, lc._repo_root(), value) for key, value in targets().items()}


async def status():
    observed = await asyncio.to_thread(snapshots)
    # Container presence and readiness are different, especially while loading.
    live = [key for key, snap in observed.items() if any(r["state"] == "running" for r in snap["containers"])]
    active = "both" if len(live) == 2 else live[0] if live else "neither"
    if _state["state"] == "neither" and settings.GPU2_ENABLED:
        try:
            durable = await request(settings.GPU2_AUTHORITY_URL.rstrip("/")+"/elastic/status")
            if durable.get("state") == "failed":
                _state.update(state="failed", error=durable.get("detail", {}).get("error", "transition_failed"))
        except Exception:
            pass  # Docker observations stay honest; mutation separately fails closed.
    return {"slot": "circe-gpu2", "enabled": settings.GPU2_ENABLED,
            "active": active, "targets": observed, **_state}


async def authority(req, *, require_drained=True):
    row = await request(settings.GPU2_AUTHORITY_URL.rstrip("/")+"/elastic/status")
    if (row.get("operation_id") != req.operation_id or row.get("generation") != req.generation
            or row.get("desired_target") != req.target):
        raise RuntimeError("stale_or_unknown_intent")
    if require_drained and not row.get("can_transition"):
        raise RuntimeError("authority_ownership_not_drained")
    if require_drained and req.target == "agent-burst" and not row.get("activation_eligible"):
        raise RuntimeError("activation_eligibility_suppressed")
    return row


async def drain_diffusion():
    base = settings.GPU2_DIFFUSION_URL.rstrip("/")
    await request(base+"/v1/lifecycle/drain", {"draining": True})
    deadline = time.monotonic()+settings.GPU2_DRAIN_TIMEOUT_SEC
    while True:
        state = await request(base+"/v1/lifecycle/status")
        if state.get("draining") is True and state.get("in_flight") is False:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("diffusion_drain_timeout")
        await asyncio.sleep(1)


async def model_ready(target):
    url = (settings.GPU2_DIFFUSION_URL.rstrip("/")+"/ready" if target == "diffusion"
           else settings.GPU2_AGENT_URL.rstrip("/")+"/health")
    try:
        body = await request(url)
        return body.get("ready") is True if target == "diffusion" else body.get("status") == "ok"
    except Exception:
        return False


async def wait_ready(target):
    deadline = time.monotonic()+settings.GPU2_MODEL_READY_TIMEOUT_SEC
    while not await model_ready(target):
        if time.monotonic() >= deadline:
            raise RuntimeError("model_readiness_timeout")
        await asyncio.sleep(1)


async def stop(target):
    runner = lc.SafeCommandRunner(allowed_commands={"docker"}, timeout_sec=settings.GPU_LANE_COMMAND_TIMEOUT_SEC)
    result = await asyncio.to_thread(lc._stop, runner, lc._repo_root(), targets()[target])
    if not result["ok"]:
        raise RuntimeError("stop_failed")
    snap = (await asyncio.to_thread(snapshots))[target]
    if snap.get("error") or snap["state"] not in {"absent", "exited", "dead", "created"}:
        raise RuntimeError("stop_unconfirmed")


async def start(target):
    runner = lc.SafeCommandRunner(allowed_commands={"docker"}, timeout_sec=settings.GPU_LANE_COMMAND_TIMEOUT_SEC)
    result = await asyncio.to_thread(lc._run_compose, runner, lc._repo_root(), targets()[target], "up", "-d", "--no-build", "--no-deps")
    if result.returncode:
        raise RuntimeError("startup_failed")
    if target == "diffusion":
        deadline = time.monotonic()+settings.GPU2_MODEL_READY_TIMEOUT_SEC
        while True:
            try:
                await request(settings.GPU2_DIFFUSION_URL.rstrip("/")+"/v1/lifecycle/drain",
                              {"draining": False})
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError("diffusion_resume_failed")
                await asyncio.sleep(1)
    await wait_ready(target)


async def transition(req):
    async with _lock:
        started = time.monotonic()
        # Reject obsolete requests before changing observable transition state.
        try:
            await authority(req, require_drained=False)
        except Exception:
            return {"status": "failed", "error": "stale_or_unknown_intent"}
        _state.clear()
        _state.update(state="draining", error=None, operation_id=req.operation_id, generation=req.generation)
        touched = False
        diffusion_stopped = False
        try:
            snap = await status()
            for observed in snap["targets"].values():
                if observed.get("error") or observed["state"] not in {"running", "absent", "exited", "dead", "created"}:
                    raise RuntimeError("container_state_not_safe")
            if snap["active"] == "both":
                raise RuntimeError("invalid_both")
            if snap["active"] == req.target and await model_ready(req.target):
                _state.update(state="ready", error=None, operation_id=req.operation_id, generation=req.generation)
                return {"status": "noop", **_state}
            await authority(req)
            _state.update(state="draining", error=None, operation_id=req.operation_id, generation=req.generation)
            if req.target == "agent-burst":
                # If diffusion is absent after a controller crash, the durable
                # intent still authorizes recovery. Unknown Docker state does not.
                diffusion = snap["targets"]["diffusion"]
                if diffusion.get("error") or diffusion["state"] == "unknown":
                    raise RuntimeError("diffusion_state_unknown")
                if any(r["state"] == "running" for r in diffusion["containers"]):
                    touched = True
                    await drain_diffusion()
                    await authority(req)
                    await stop("diffusion")
                    diffusion_stopped = True
            else:
                burst = snap["targets"]["agent-burst"]
                if burst.get("error") or burst["state"] == "unknown":
                    raise RuntimeError("burst_state_unknown")
                if any(r["state"] == "running" for r in burst["containers"]):
                    slots = await request(settings.GPU2_AGENT_URL.rstrip("/")+"/slots")
                    if not isinstance(slots, list) or not slots or not all(
                            isinstance(s, dict) and s.get("is_processing") is False for s in slots):
                        raise RuntimeError("burst_upstream_not_idle")
                    await authority(req)  # closed admissions + no permits/leases
                    await stop("agent-burst")
            _state.update(state="activating")
            await authority(req)
            cold = time.monotonic()
            await start(req.target)
            _state.update(state="ready", cold_start_seconds=time.monotonic()-cold)
            return {"status": "success", **_state}
        except Exception as exc:
            error = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
            # No model output, prompt or subprocess stderr in operational evidence.
            if req.target == "agent-burst" and touched:
                try:
                    # Authority has never opened new admissions for this intent.
                    row = await authority(req, require_drained=False)
                    if not row.get("can_transition"):
                        raise RuntimeError("rollback_ownership_not_drained")
                    if diffusion_stopped:
                        await stop("agent-burst")
                        await start("diffusion")
                    else:
                        await request(settings.GPU2_DIFFUSION_URL.rstrip("/")+"/v1/lifecycle/drain",
                                      {"draining": False})
                    _state["restored"] = True
                except Exception:
                    _state["restored"] = False
                    error += ":restoration_failed"
            _state.update(state="failed", error=error)
            logger.error("gpu2_transition_failed operation={} reason={}", req.operation_id, error)
            return {"status": "failed", **_state}
        finally:
            _state["transition_seconds"] = time.monotonic()-started
            logger.info("gpu2_transition operation={} state={} seconds={:.2f}", req.operation_id, _state["state"], _state["transition_seconds"])


async def flip(req):
    global _job
    if not settings.GPU2_ENABLED:
        return {"status": "disabled"}
    if _job is not None and not _job.done():
        return {"status": "busy", **_state}
    _job = asyncio.create_task(transition(req))
    result = await asyncio.shield(_job)
    return {**result, "transition_seconds": _state.get("transition_seconds")}
