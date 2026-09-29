"""Parsing of a GPU pool lease ref (bus ``options.gpu_lease``, HTTP ``X-Orion-Gpu-Lease``).

A GPU lease ref is not validated here: the pool is its fencing authority, reached by ``attach`` in
pool_placement (a stale or unknown hold makes attach refuse, and the call gets
``gpu_pool_unavailable``). The durable-run lease token (``options.resource_lease`` /
``X-Orion-Resource-Lease``) and its broker validation (``LeaseGuard``) were deleted in GPU pool
stage 4.6; a caller that still sends one is not fenced by it.
"""
from __future__ import annotations

from typing import Any, Optional

from orion.llm.resource_lease import (
    GPU_LEASE_HEADER, GPU_LEASE_OPTION, ResourceLeaseRejected, decode_gpu_lease_header,
)
from orion.schemas.gpu_pool import GpuLeaseRefV1


def gpu_lease_from_options(options: Any) -> Optional[GpuLeaseRefV1]:
    """``options.gpu_lease`` (a GpuLeaseRefV1 dict) or None. Malformed -> ResourceLeaseRejected,
    never None: a call that meant to run under a hold must not silently take a lease of its own."""
    value = (options or {}).get(GPU_LEASE_OPTION) if isinstance(options, dict) else None
    if value is None:
        return None
    try:
        return GpuLeaseRefV1.model_validate(value)
    except ValueError as exc:
        raise ResourceLeaseRejected("malformed_gpu_lease") from exc


def gpu_lease_from_headers(headers: Any) -> Optional[GpuLeaseRefV1]:
    value = headers.get(GPU_LEASE_HEADER)
    return decode_gpu_lease_header(value) if value is not None else None


def lease_error(reason: str) -> dict[str, Any]:
    return {"error": {"type": "resource_lease_rejected", "message": reason}}
