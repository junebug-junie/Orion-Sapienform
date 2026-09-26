"""Transport for a broker-issued lease; the broker remains the fencing authority.

Also the wire form of a GPU pool lease reference (``X-Orion-Gpu-Lease``, stage 4.1): the pool is its
fencing authority, reached by the gateway's ``attach``. The durable-runs half of this module is
deleted in stage 4.6; the pool half stays."""
from __future__ import annotations

import base64
import binascii
import json
from typing import Any

import httpx

from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.resource_admission import ResourceLeaseV1

LEASE_HEADER = "X-Orion-Resource-Lease"
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


def encode_lease_header(lease: dict[str, Any]) -> str:
    payload = ResourceLeaseV1.model_validate(lease).model_dump(mode="json")
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    if len(encoded) > MAX_LEASE_HEADER_BYTES:
        raise ValueError("resource lease header too large")
    return encoded


def decode_lease_header(value: str) -> dict[str, Any]:
    try:
        if not value or len(value) > MAX_LEASE_HEADER_BYTES:
            raise ValueError("invalid lease header length")
        raw = base64.b64decode(value.encode(), altchars=b"-_", validate=True)
        return ResourceLeaseV1.model_validate_json(raw).model_dump(mode="json")
    except (ValueError, binascii.Error, UnicodeError) as exc:
        raise ResourceLeaseRejected("malformed_resource_lease") from exc


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


async def validate_resource_lease(
    lease: dict[str, Any], *, lane: str, backend_key: str,
    validation_url: str, timeout_sec: float = 2.0,
) -> None:
    """Fail closed for protected requests, without interpreting stale token expiry.

    Renewals update Postgres while the original request token remains unchanged.
    Only the broker can decide whether its generation is still current.
    """
    try:
        parsed = ResourceLeaseV1.model_validate(lease)
    except ValueError as exc:
        raise ResourceLeaseRejected("malformed_resource_lease") from exc
    if parsed.lane != lane or parsed.backend_key.rstrip("/") != backend_key.rstrip("/"):
        raise ResourceLeaseRejected("resource_lease_route_mismatch")
    if parsed.status != "active":
        raise ResourceLeaseRejected("resource_lease_inactive")
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.post(validation_url, json={
                "lease": parsed.model_dump(mode="json"), "lane": lane, "backend_key": parsed.backend_key,
            })
            response.raise_for_status()
            result = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ResourceLeaseRejected("resource_lease_validation_unavailable") from exc
    if not isinstance(result, dict) or result.get("valid") is not True:
        reason = result.get("reason") if isinstance(result, dict) else None
        raise ResourceLeaseRejected(str(reason or "resource_lease_stale"))
