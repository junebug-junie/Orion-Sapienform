"""Wire form of a GPU pool lease reference (``X-Orion-Gpu-Lease`` / bus ``options.gpu_lease``).

The pool is its fencing authority, reached by the gateway's ``attach``. The durable-runs lease token
(``X-Orion-Resource-Lease`` / ``ResourceLeaseV1``) that used to share this module was deleted in GPU
pool stage 4.6; a pool hold ref is the only run lease."""
from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from orion.schemas.gpu_pool import GpuLeaseRefV1

MAX_LEASE_HEADER_BYTES = 8192
GPU_LEASE_HEADER = "X-Orion-Gpu-Lease"      # HTTP carrier of GpuLeaseRefV1
GPU_LEASE_OPTION = "gpu_lease"              # bus carrier: options.gpu_lease
# The route a call under a GPU pool hold names when its caller has no route of its own. Placement
# ignores it (``attach`` runs the call on the hold's role); it only names the work class the pool
# records for the child, and durable-run holds are agent class (stage 4 spec, Decision 1 rule 1).
# A hold's role (e.g. "agent-gpu2") is not a route name, so it can never stand in for one.
GPU_LEASE_ROUTE = "agent"


class ResourceLeaseRejected(RuntimeError):
    pass


def held_call_route(
    gpu_lease: GpuLeaseRefV1 | dict[str, Any] | None,
    default_route: str,
) -> tuple[str, dict[str, Any] | None]:
    """Where an LLM call made for a turn goes: ``(route, gpu_lease_option)``.

    One rule (Juniper, 2026-11-12): a turn that holds a GPU pool seat runs ALL of its LLM work on
    that seat. World-pulse reads are durable runs, and every reading turn carries its hold
    (``ReadingTurnRequestV1.gpu_lease`` is required). Their pre-turn check and their Mind phases
    used to ignore the hold and ride ``quick_turn``/``metacog``: Qwen3-8B with a 4,096-token slot,
    against 21k-39k-character reading prompts. Those calls waited in the pool to their deadline
    (stage-2 checks failed 194 of 373 in the 7 days to 11-12).

    With a hold: the call names the hold's work-class route (``agent``) and carries the hold ref as
    the bus ``options.gpu_lease``; the gateway attaches it to the hold's role whatever the route.
    Without one (Juniper's own chat turns): ``default_route``, unchanged. There is deliberately no
    fallback from a held call to ``default_route``: a refused attach is a failed call, never a
    quiet trip back to the small lanes."""
    if gpu_lease is None:
        return default_route, None
    return GPU_LEASE_ROUTE, GpuLeaseRefV1.model_validate(gpu_lease).model_dump(mode="json")


def encode_gpu_lease_header(ref: GpuLeaseRefV1 | dict[str, Any]) -> str:
    payload = GpuLeaseRefV1.model_validate(ref).model_dump(mode="json")
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    if len(encoded) > MAX_LEASE_HEADER_BYTES:
        raise ValueError("gpu lease header too large")
    return encoded


def decode_gpu_lease_header(value: str) -> GpuLeaseRefV1:
    try:
        if not value or len(value) > MAX_LEASE_HEADER_BYTES:
            raise ValueError("invalid gpu lease header length")
        # Some proxies strip base64 '=' padding; restore it before strict decoding.
        padded = value + "=" * (-len(value) % 4)
        raw = base64.b64decode(padded.encode(), altchars=b"-_", validate=True)
        return GpuLeaseRefV1.model_validate_json(raw)
    except (ValueError, binascii.Error, UnicodeError) as exc:
        raise ResourceLeaseRejected("malformed_gpu_lease") from exc
