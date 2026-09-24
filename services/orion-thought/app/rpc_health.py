"""RPC-health wiring for orion-thought (mesh transport coverage, A0 of
docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-baseline-design.md).

orion-thought has no single long-lived RPC bus: the per-request handler
(bus_listener._run_bus_message_handler) and /visual-chain/run-once open a bus per call,
and the reverie / reverie-chain / visual-chain workers each hold their own. Every one of
them folds its RPC-health window into the process-wide ``RPC_HEALTH_SINK`` (per tick for
the workers, in ``finally`` for per-call buses), and the mind HTTP client records
straight into the sink. main.py's lifespan runs one ``RpcHealthPublisher`` on a
dedicated long-lived bus that drains the sink every interval. All of these run on the one
FastAPI event loop; the sink is lock-guarded anyway.

Hop keys published (``RpcHealthSnapshotV1.channel_latency``):

- ``<CHANNEL_CORTEX_EXEC_REQUEST>`` (and lane channels) -- CortexExecClient.execute_plan
  (reverie / stance / visual legacy exec)
- ``<CHANNEL_VISION_HOST_REQUEST>`` -- visual_chain.request_caption (vision-host RPC)
- ``http:<mind host[:port]>/v1/mind/run`` -- mind_enrichment.run_mind_for_thought
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from orion.core.bus.bus_schemas import ServiceRef
from orion.core.bus.rpc_health import SharedRpcHealthSink
from orion.core.bus.rpc_health_publish import RpcHealthPublisher

RPC_HEALTH_SINK = SharedRpcHealthSink()


def fold_bus(bus: Any) -> None:
    """Move ``bus``'s accumulated RPC-health window into the process sink. Never raises."""
    if bus is not None:
        RPC_HEALTH_SINK.absorb_bus(bus)


def hop_recorder() -> SharedRpcHealthSink:
    return RPC_HEALTH_SINK


def build_publisher(settings: Any, bus_getter: Callable[[], Optional[Any]]) -> RpcHealthPublisher:
    return RpcHealthPublisher(
        enabled=bool(settings.rpc_health_publish_enabled and settings.orion_bus_enabled),
        bus_getter=bus_getter,
        service=settings.service_name,
        node=settings.node_name,
        instance="main",
        source=ServiceRef(
            name=settings.service_name, version=settings.service_version, node=settings.node_name
        ),
        interval_sec=settings.rpc_health_publish_interval_sec,
        include_channel_latency=settings.rpc_health_channel_latency_enabled,
        sinks=[RPC_HEALTH_SINK],
    )
