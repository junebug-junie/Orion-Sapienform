"""Outbound-HTTP hop timing for orion-durable-runs (mesh transport coverage).

Every httpx client this service opens toward the llm-gateway (``/routes``), a lane
upstream (``/slots``), the cabinet thermal feed, orion-thought's visual activity, and the
GPU2 elastic controller records its round trip into the long-lived ``rpc_bus`` aggregator
as hop ``http:<host>[:<port>]<path>`` (ids collapsed by ``normalize_id_path``), which
``RpcHealthPublisher`` publishes in ``RpcHealthSnapshotV1.channel_latency``.

All of these clients are opened inside async coroutines on the FastAPI event loop (none
go through ``run_in_executor``/``to_thread``), which is what the lock-free aggregator
requires.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import httpx

from orion.core.bus.http_health import AsyncHopTimingTransport, normalize_id_path

RecorderGetter = Callable[[], Optional[Any]]


def hop_client_kwargs(recorder_getter: Optional[RecorderGetter]) -> dict:
    """``httpx.AsyncClient(**hop_client_kwargs(getter), timeout=...)``. Empty when no
    recorder is wired (bus disabled, or a test runtime), so the client is exactly the
    pre-instrumentation one."""
    if recorder_getter is None:
        return {}
    return {
        "transport": AsyncHopTimingTransport(
            httpx.AsyncHTTPTransport(),
            recorder_getter=recorder_getter,
            path_normalizer=normalize_id_path,
        )
    }
