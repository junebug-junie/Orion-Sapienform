"""Transport for a broker-issued lease; the broker remains the fencing authority."""
from __future__ import annotations

import base64
import binascii
import json
from typing import Any

import httpx

from orion.schemas.resource_admission import ResourceLeaseV1

LEASE_HEADER = "X-Orion-Resource-Lease"
MAX_LEASE_HEADER_BYTES = 8192


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
