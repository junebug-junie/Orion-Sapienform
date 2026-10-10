"""Ask a GPU pool host actuator two read-only questions over the bus and classify its answer.

Shared by the operator CLI (``scripts/gpu_pool_actuator_probe.py``) and orion-mesh-guardian's
always-on GPU watch (``services/orion-mesh-guardian/app/gpu_watch.py``).

- ``status``: is the controller up, can it parse its own checkout's config/gpu_pool.yaml
  (``config_unloadable:*`` = image not rebuilt after a pull), what does it observe on the card.
  A read: the controller answers from its fence file and ``docker compose ps``.
- ``digest``: a ``load`` naming PROBE_PROFILE, which is never in any allow-list. The controller
  checks deadline, config, launch digest and profile (pool_fence.resolve) before the busy check,
  the generation fence or any docker call, so the answer is always a refusal:
  ``profile_not_allowed`` = digests agree; ``launch_digest_mismatch`` = the two checkouts differ.

Why a foreign action_id is harmless (read from code, 2026-10-10):

- controller (services/orion-gpu-lane-controller/app/actuator_bus.py): ``status`` runs
  ``_status`` outside the admit lock and writes no fence state; the digest ``load`` raises
  ``pool_fence.Refusal("profile_not_allowed")`` (or ``launch_digest_mismatch`` /
  ``config_unloadable``) inside ``resolve`` before ``write_state`` or ``launch_exec``. Pinned by
  services/orion-gpu-pool/tests/test_stage5_3_cutover_e2e.py (no docker call, no generation).
- pool (services/orion-gpu-pool/app/runtime.py ``on_actuate_result``): a result whose action_id
  is not the seat's pending action is logged ``*_stale`` and dropped, so a probe's refusal never
  becomes an ``actuate_refused`` event.
- EXCEPT ``status``: it makes the controller re-publish the last result it recorded for the card
  set under the POOL's action_id. Outside the pool's own reconcile, that replay reaches the
  ``late_result`` branch and can flip the pool's belief back to an old action's outcome (e.g.
  load, pause, hand unload, resume-reconcile, then a status probe replays the old load). So
  ``status`` is for an operator by hand only; anything periodic must use ``digest``, which
  replays nothing, writes nothing, and still answers ``config_unloadable:*`` (the load path
  parses the controller's config before ``resolve``) -- the incident's 155 real refusals were
  exactly that path.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from orion.gpu_pool.config import PoolConfig, launch_digest
from orion.schemas.gpu_pool import (
    GPU_ACTUATE_KIND,
    GPU_POOL_ACTUATE_REQUEST_CHANNEL,
    GPU_POOL_ACTUATE_RESULT_CHANNEL,
    GpuActuateV1,
)

# Never a config/llm_profiles.yaml name, so never in a launch.profiles allow-list.
PROBE_PROFILE = "gpu-pool-actuator-probe-not-a-profile"
DEFAULT_SOURCE = "operator:gpu-pool-actuator-probe"
TERMINAL = ("succeeded", "failed", "refused")

Check = Literal["status", "digest"]
VerdictKind = Literal["ok", "config_unloadable", "digest_mismatch", "no_answer", "other_refusal"]


@dataclass(frozen=True)
class Verdict:
    check: str
    role: str
    kind: VerdictKind
    reason: str          # the controller's terminal reason ("" when none / no answer)
    status: str | None   # the terminal status, None on no answer
    line: str            # one line a human can act on (the CLI prints this)
    launch_digest: str = ""
    results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.kind == "ok"


def build_request(cfg: PoolConfig, role: str, check: str, *, now: datetime | None = None) -> GpuActuateV1:
    """The request for one check. ``status`` carries profile None; ``digest`` is a ``load`` with
    PROBE_PROFILE and THIS checkout's launch digest. ValueError for a role with no launch block."""
    spec = cfg.roles[role]
    if spec.launch is None:
        raise ValueError(f"{role} has no launch block: no actuator to ask")
    now = now or datetime.now(timezone.utc)
    return GpuActuateV1(
        action_id=f"probe-{check}:{role}:{uuid.uuid4().hex[:8]}", generation=1, actuator=spec.launch.actuator,
        role=role, action="status" if check == "status" else "load", cards=list(spec.cards),
        profile=None if check == "status" else PROBE_PROFILE, launch_digest=launch_digest(cfg, role),
        deadline_at=now + timedelta(seconds=120), reason="operator_probe")


def classify(check: str, results: list[dict], *, role: str = "", digest: str = "") -> Verdict:
    """The verdict from the results carrying this probe's action_id (the last terminal one wins)."""
    final = [r for r in results if r.get("status") in TERMINAL]

    def v(kind: VerdictKind, line: str, reason: str = "", status: str | None = None) -> Verdict:
        return Verdict(check=check, role=role, kind=kind, reason=reason, status=status, line=line,
                       launch_digest=digest, results=list(results))

    if not final:
        return v("no_answer", "NO ANSWER (controller down, wrong actuator name, or bus unreachable)")
    last = final[-1]
    status = last.get("status")
    reason = last.get("reason") or ""
    unloadable = reason.startswith("config_unloadable:")
    if check == "digest":
        if reason == "profile_not_allowed":
            return v("ok", "OK: launch digests agree", reason, status)
        if reason == "launch_digest_mismatch":
            return v("digest_mismatch", "MISMATCH: controller checkout/image is not on this commit", reason, status)
        return v("config_unloadable" if unloadable else "other_refusal",
                 f"UNEXPECTED: {status} {reason}", reason, status)
    if status == "succeeded":
        return v("ok", (f"OK: observed={last.get('observed')} in_flight={last.get('in_flight')} "
                        f"last_action_id={last.get('last_action_id')}"), reason, status)
    return v("config_unloadable" if unloadable else "other_refusal", f"NOT OK: {status} {reason}", reason, status)


def _payload(message: Any) -> dict | None:
    data = message.get("data") if isinstance(message, dict) else None
    if data is None:
        return None
    try:
        env = json.loads(data)
    except (TypeError, ValueError):
        return None
    payload = env.get("payload") if isinstance(env, dict) else None
    return payload if isinstance(payload, dict) else None


async def probe(bus, cfg: PoolConfig, role: str, check: str, wait_sec: float,
                source: str = DEFAULT_SOURCE) -> Verdict:
    """Subscribe to the result channel, publish one probe request, wait up to ``wait_sec`` for its
    terminal answer, and classify it. ``bus`` is a connected OrionBusAsync (or a fake with the same
    ``subscribe`` / ``publish``)."""
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    msg = build_request(cfg, role, check)
    got: list[dict] = []
    loop = asyncio.get_running_loop()
    async with bus.subscribe(GPU_POOL_ACTUATE_RESULT_CHANNEL) as pubsub:
        await bus.publish(GPU_POOL_ACTUATE_REQUEST_CHANNEL, BaseEnvelope(
            kind=GPU_ACTUATE_KIND, source=ServiceRef(name=source), correlation_id=uuid.uuid4(),
            payload=msg.model_dump(mode="json")))
        deadline = loop.time() + wait_sec
        while loop.time() < deadline:
            m = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            payload = _payload(m) if m else None
            if payload is not None and payload.get("action_id") == msg.action_id:
                got.append(payload)
                if payload.get("status") in TERMINAL:
                    break
    return classify(check, got, role=role, digest=msg.launch_digest)
