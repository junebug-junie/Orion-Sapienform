"""RPC delivery bridge wiring in the substrate-runtime worker: settings default,
bus snapshot -> window -> receipt, and "nothing called" writes nothing."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, SUBSTRATE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.worker import BiometricsSubstrateWorker
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.substrate.rpc_delivery import (
    RPC_DELIVERY_CHANNEL,
    RPC_DELIVERY_NODE_ID,
    RPC_DELIVERY_TARGET_KIND,
    RpcDeliveryConfig,
    RpcDeliveryWindow,
)


def _worker(enabled: bool = True) -> BiometricsSubstrateWorker:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()
    worker._settings.enable_rpc_delivery_bridge = enabled
    worker._store = MagicMock()
    worker._rpc_delivery_window = RpcDeliveryWindow(RpcDeliveryConfig())
    worker._bus = MagicMock()
    worker._bus.codec = OrionCodec()
    return worker


def _raw_message(window_end: datetime, hops: dict) -> dict:
    """The exact wire shape rpc_health_publish emits (an encoded envelope)."""
    env = BaseEnvelope(
        kind="rpc_health.snapshot.v1",
        source=ServiceRef(name="cortex-exec", node="athena"),
        payload={
            "service": "cortex-exec",
            "node": "athena",
            "instance": "background",
            "window_start": (window_end - timedelta(seconds=30)).isoformat(),
            "window_end": window_end.isoformat(),
            "success_count": 0,
            "timeout_count": 0,
            "channel_counts": {},
            "truncated": False,
            "channel_latency": {
                hop: {
                    "success_count": s,
                    "timeout_count": t,
                    "log_ms_sum": 0.0,
                    "log_ms_sumsq": 0.0,
                    "max_ms": None,
                }
                for hop, (s, t) in hops.items()
            },
        },
    )
    return {"type": "message", "data": OrionCodec().encode(env)}


def test_settings_default_is_off(monkeypatch):
    import app.settings as settings_mod

    monkeypatch.setenv("POSTGRES_URI", "postgresql://u:p@unused/db")
    for key in (
        "SUBSTRATE_RPC_DELIVERY_BRIDGE_ENABLED",
        "SUBSTRATE_RPC_DELIVERY_WINDOW_SEC",
        "SUBSTRATE_RPC_DELIVERY_MIN_DENOMINATOR",
        "SUBSTRATE_RPC_DELIVERY_EXCLUDE_LABELS",
    ):
        monkeypatch.delenv(key, raising=False)
    s = settings_mod.Settings()
    assert s.enable_rpc_delivery_bridge is False
    assert s.rpc_delivery_window_sec == 600.0
    assert s.rpc_delivery_min_denominator == 10
    assert "current_turn_probe" in s.rpc_delivery_exclude_labels_raw
    assert s.rpc_health_snapshot_channel == "orion:rpc_health:snapshot"


def test_bus_snapshot_to_receipt_names_the_worst_hop():
    worker = _worker()
    now = datetime.now(timezone.utc)
    msg = _raw_message(
        now - timedelta(seconds=5),
        {
            "orion:exec:request:LLMGatewayService": (18, 2),
            "orion:exec:request:RecallService": (9, 0),
            # excluded: the fail-open probe and a non-bus hop
            "orion:exec:request:LLMGatewayService#current_turn_probe": (0, 9),
            "http:llm-gateway:8210/routes": (0, 40),
        },
    )
    assert worker._handle_rpc_health_message(msg) is True

    worker._rpc_delivery_tick()

    receipt = worker._store.save_receipt.call_args.args[0]
    (delta,) = receipt.state_deltas
    assert delta.target_kind == RPC_DELIVERY_TARGET_KIND
    assert delta.target_id == RPC_DELIVERY_NODE_ID
    assert delta.after["pressure_hints"] == {RPC_DELIVERY_CHANNEL: 0.1}
    assert delta.after["reading"]["worst_hop"] == "orion:exec:request:LLMGatewayService"
    assert delta.after["reading"]["total_calls"] == 29
    json.dumps(receipt.model_dump(mode="json"))  # storable


def test_replayed_snapshot_is_not_double_counted():
    worker = _worker()
    msg = _raw_message(
        datetime.now(timezone.utc) - timedelta(seconds=5),
        {"orion:state:request": (5, 5)},
    )
    assert worker._handle_rpc_health_message(msg) is True
    assert worker._handle_rpc_health_message(msg) is False
    worker._rpc_delivery_tick()
    reading = worker._store.save_receipt.call_args.args[0].state_deltas[0].after["reading"]
    assert reading["total_calls"] == 10


def test_no_counted_calls_writes_nothing():
    worker = _worker()
    worker._handle_rpc_health_message(
        _raw_message(
            datetime.now(timezone.utc) - timedelta(seconds=5),
            {"http:100.92.216.81:8080/api/cabinet/sensors/latest": (4, 1)},
        )
    )
    worker._rpc_delivery_tick()
    worker._store.save_receipt.assert_not_called()


def test_flag_off_tick_writes_nothing():
    worker = _worker(enabled=False)
    worker._handle_rpc_health_message(
        _raw_message(datetime.now(timezone.utc), {"orion:state:request": (1, 1)})
    )
    worker._rpc_delivery_tick()
    worker._store.save_receipt.assert_not_called()


def test_tick_never_raises_on_store_failure():
    worker = _worker()
    worker._handle_rpc_health_message(
        _raw_message(datetime.now(timezone.utc) - timedelta(seconds=1), {"orion:state:request": (1, 0)})
    )
    worker._store.save_receipt.side_effect = RuntimeError("db down")
    worker._rpc_delivery_tick()  # logged, not raised
