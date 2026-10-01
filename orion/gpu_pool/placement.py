"""Which model is serving a turn, told from the lease the turn holds -- not from the route table.

Spec: docs/superpowers/specs/2026-09-24-gpu-pool-design.md, "Transport-metric and reader
impacts" item 5. Before the pool, a route named one worker, so the gateway's ``/routes`` answer
("route ``agent`` runs model X") was also the answer to "what am I running on". With the pool a
call is placed per lease: an ``agent`` call can land on ``agent-gpu2`` or on ``chat`` (live,
2026-09-29: 1463 of the ``http:anthropic`` agent-class leases in four days were served by
``chat``). Reading ``/routes`` then tells Orion something false about itself.

Two honest shapes, and only two:

- ``from_lease``: the turn holds a pool lease (a durable run's hold, ``GpuLeaseRefV1``). Every
  call under a hold runs on the hold's role, so the role is a fact and the discovered profile of
  that role (pool state) is the model.
- ``route_default``: no hold. Each call is placed separately, so before the turn the only true
  statement is "this route's default model, and a busy moment can place a call elsewhere".
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.gpu_pool import (
    GPU_POOL_STATE_REPLY_PREFIX, GPU_POOL_STATE_REQUEST_CHANNEL, GPU_POOL_STATE_REQUEST_KIND,
    DiscoveredRoleV1, GpuPoolStateRequestV1,
)

logger = logging.getLogger("orion.gpu_pool.placement")

# RPC-health label for the state read below: the pool answers state at once, so this is real
# transport (not queue wait) and stays in the baseline under its own key.
STATE_RPC_HEALTH_LABEL = "gpu_pool_self_placement"


def _model_name(value: str | None) -> str | None:
    """``/models/gguf/Qwen3.8-27B.gguf`` -> ``Qwen3.8-27B.gguf``: the file is the identity."""
    text = str(value or "").strip()
    if not text:
        return None
    return text.rsplit("/", 1)[-1] or None


@dataclass(frozen=True)
class ServingPlacement:
    """What Orion may truthfully say about the model serving this turn."""

    source: str                      # "from_lease" | "route_default"
    model: str | None = None
    role: str | None = None          # pool role (from_lease only)
    profile: str | None = None       # llm_profiles.yaml profile the pool discovered on that role
    route: str | None = None         # the route asked for (route_default only)
    role_status: str | None = None   # pool discovery status of `role` (from_lease only)

    def self_line(self) -> str | None:
        """One prompt line, or None when there is nothing true to say (never a placeholder)."""
        if self.source == "from_lease" and self.role:
            if not self.model:
                if self.role_status:
                    why = (f"the pool reports that role as {self.role_status}, not a confirmed model"
                           if self.role_status not in ("confirmed", "static")
                           else "the pool did not report a model file for it")
                else:
                    why = "the pool's state could not be read"
                return (f"This turn holds the GPU pool's {self.role} role; {why}, so do not name one.")
            profile = f", profile {self.profile}" if self.profile else ""
            return (f"Backend model serving this turn: {self.model} (GPU pool role {self.role}{profile}; "
                    "this turn holds that role, so every call runs there).")
        if self.source == "route_default" and self.model:
            route = f" for route {self.route}" if self.route else ""
            return (f"Default backend model{route}: {self.model}. The GPU pool places each call on "
                    "its own, and a busy moment can put a call on a different model, so this is "
                    "the expected model, not a confirmed one.")
        return None


def discovered_role(state: Mapping[str, Any] | None, role: str) -> DiscoveredRoleV1 | None:
    """The pool's discovered entry for ``role`` out of a ``GpuPoolStateV1`` payload (dict)."""
    if not isinstance(state, Mapping):
        return None
    for raw in state.get("roles") or []:
        if isinstance(raw, Mapping) and raw.get("role") == role:
            try:
                return DiscoveredRoleV1.model_validate(dict(raw))
            except Exception:  # noqa: BLE001 -- a newer pool's row shape must not break a turn
                logger.debug("discovered_role_unparseable role=%s", role, exc_info=True)
                return None
    return None


def placement_from_lease(role: str, found: DiscoveredRoleV1 | None) -> ServingPlacement:
    model = None
    profile = None
    if found is not None and found.status in ("confirmed", "static"):
        model = _model_name(found.model_file) or _model_name(found.model_path)
        profile = found.profile_name
    return ServingPlacement(source="from_lease", role=role, model=model, profile=profile,
                            role_status=found.status if found is not None else None)


def placement_from_route_default(route: str | None, model: str | None) -> ServingPlacement:
    return ServingPlacement(source="route_default", route=route, model=_model_name(model))


async def fetch_pool_state(bus: Any, *, source: str, timeout_sec: float = 2.0,
                           include_config: bool = False) -> dict[str, Any] | None:
    """One ``orion:gpu_pool:state:request`` RPC without leases. Fails open to None.

    ``include_config``: also return the pool's parsed ``config/gpu_pool.yaml`` (about 5 KB), which
    ``orion.gpu_pool.route_view`` needs to map a route to the role it lands on."""
    reply_channel = f"{GPU_POOL_STATE_REPLY_PREFIX}{uuid.uuid4().hex}"
    env = BaseEnvelope(
        kind=GPU_POOL_STATE_REQUEST_KIND, source=ServiceRef(name=source), correlation_id=uuid.uuid4(),
        reply_to=reply_channel,
        payload=GpuPoolStateRequestV1(include_leases=False, include_config=include_config).model_dump(
            mode="json", exclude_defaults=True),
    )
    try:
        raw = await bus.rpc_request(GPU_POOL_STATE_REQUEST_CHANNEL, env, reply_channel=reply_channel,
                                    timeout_sec=timeout_sec, health_label=STATE_RPC_HEALTH_LABEL)
        decoded = bus.codec.decode(raw["data"])  # rpc_request returns the raw pubsub message
    except Exception:  # noqa: BLE001 -- self-context must never fail a turn
        logger.warning("gpu_pool_state_read_failed source=%s", source, exc_info=True)
        return None
    payload = decoded.envelope.payload if getattr(decoded, "ok", False) else None
    return payload if isinstance(payload, dict) else None
