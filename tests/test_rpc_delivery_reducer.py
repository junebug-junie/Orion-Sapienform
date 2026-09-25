"""orion/substrate/rpc_delivery.py: worst-hop RPC timeout ratio with a
denominator floor, over a rolling window of RpcHealthSnapshotV1 payloads."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.schemas.telemetry.rpc_health import RpcHealthSnapshotV1
from orion.substrate.rpc_delivery import (
    RPC_DELIVERY_CHANNEL,
    RpcDeliveryConfig,
    RpcDeliveryWindow,
    counted_hop,
    hop_pressure,
    parse_exclude_labels,
    rpc_delivery_receipt,
)

T0 = datetime(2026, 9, 25, 5, 0, tzinfo=timezone.utc)


def _snap(end: datetime, hops: dict, *, service="cortex-exec", instance="background") -> dict:
    payload = {
        "service": service,
        "node": "athena",
        "instance": instance,
        "window_start": (end - timedelta(seconds=30)).isoformat(),
        "window_end": end.isoformat(),
        "success_count": 0,
        "timeout_count": 0,
        "channel_counts": {},
        "truncated": False,
        "channel_latency": {
            hop: {"success_count": s, "timeout_count": t, "log_ms_sum": 0.0, "log_ms_sumsq": 0.0, "max_ms": None}
            for hop, (s, t) in hops.items()
        },
    }
    # Every fixture is a valid wire payload, so the reducer is tested on the
    # shape producers actually publish.
    RpcHealthSnapshotV1.model_validate(payload)
    return payload


def _reading(win: RpcDeliveryWindow, at: datetime):
    return win.reading(at.timestamp())


def test_one_timeout_in_one_call_does_not_read_one():
    assert hop_pressure(1, 0, 10) == 0.1
    win = RpcDeliveryWindow()
    win.fold(_snap(T0, {"orion:state:request": (0, 1)}))
    assert _reading(win, T0).pressure == 0.1


def test_full_outage_reaches_one_once_the_floor_is_met():
    win = RpcDeliveryWindow()
    for i in range(10):
        win.fold(_snap(T0 + timedelta(seconds=30 * i), {"orion:state:request": (0, 1)}))
    r = _reading(win, T0 + timedelta(seconds=270))
    assert r.pressure == 1.0
    assert (r.worst_timeouts, r.worst_calls) == (10, 10)


def test_calm_traffic_reads_exactly_zero():
    win = RpcDeliveryWindow()
    win.fold(_snap(T0, {"orion:exec:request:RecallService": (40, 0), "orion:state:request": (12, 0)}))
    r = _reading(win, T0)
    assert r.pressure == 0.0
    assert r.total_calls == 52 and r.total_timeouts == 0


def test_worst_hop_wins_rather_than_being_pooled_away():
    win = RpcDeliveryWindow()
    win.fold(
        _snap(
            T0,
            {
                "orion:exec:request:RecallService": (500, 0),
                "orion:cortex:request": (0, 6),
            },
        )
    )
    r = _reading(win, T0)
    # pooled would be 6/506 ~= 0.012; the dead hop reads 6/max(6,10) = 0.6
    assert r.pressure == 0.6
    assert r.worst_hop == "orion:cortex:request"


def test_same_hop_is_summed_across_producers():
    win = RpcDeliveryWindow()
    win.fold(_snap(T0, {"orion:state:request": (8, 1)}, service="cortex-exec"))
    win.fold(_snap(T0, {"orion:state:request": (10, 1)}, service="cortex-orch", instance="main"))
    r = _reading(win, T0)
    assert r.worst_calls == 20 and r.worst_timeouts == 2
    assert r.pressure == 0.1
    assert r.producers == 2


def test_only_bus_rpc_hops_count():
    assert counted_hop("orion:exec:request:LLMGatewayService", ())
    assert counted_hop("orion:gpu_pool:lease:request#gpu_pool_lease", ())
    for hop in (
        "http:llm-gateway:8210/routes",
        "gpu_pool:fast#gpu_pool_wait",
        "fcc:claude-opus",
        "verb:self_study.reflect",
        "governor:agent",
    ):
        assert not counted_hop(hop, ()), hop


def test_default_exclusions_drop_probe_metacog_and_queue_wait():
    labels = RpcDeliveryConfig().exclude_labels
    assert not counted_hop("orion:exec:request:LLMGatewayService#current_turn_probe", labels)
    assert not counted_hop("orion:cortex:exec:request:background#log_orion_metacognition", labels)
    assert counted_hop("orion:exec:request:LLMGatewayService", labels)
    win = RpcDeliveryWindow()
    win.fold(
        _snap(
            T0,
            {
                "orion:exec:request:LLMGatewayService#current_turn_probe": (0, 9),
                "orion:exec:request:LLMGatewayService": (30, 0),
            },
        )
    )
    assert _reading(win, T0).pressure == 0.0


def test_no_counted_calls_is_unmeasured_not_calm():
    win = RpcDeliveryWindow()
    assert _reading(win, T0) is None
    win.fold(_snap(T0, {"http:100.92.216.81:8080/api/cabinet/sensors/latest": (4, 1)}))
    assert _reading(win, T0) is None
    win.fold(_snap(T0 + timedelta(seconds=30), {}))
    assert _reading(win, T0 + timedelta(seconds=30)) is None


def test_snapshot_without_channel_latency_is_ignored():
    win = RpcDeliveryWindow()
    old = _snap(T0, {"orion:state:request": (1, 1)})
    old.pop("channel_latency")
    assert win.fold(old) is False
    assert _reading(win, T0) is None


def test_replay_and_out_of_order_are_ignored_per_producer():
    win = RpcDeliveryWindow()
    s = _snap(T0, {"orion:state:request": (1, 1)})
    assert win.fold(s) is True
    assert win.fold(s) is False
    assert win.fold(_snap(T0 - timedelta(seconds=30), {"orion:state:request": (0, 5)})) is False
    assert _reading(win, T0).worst_calls == 2


def test_window_expiry_returns_to_unmeasured_and_calm_recovers():
    win = RpcDeliveryWindow(RpcDeliveryConfig(window_s=600.0))
    win.fold(_snap(T0, {"orion:state:request": (0, 10)}))
    assert _reading(win, T0).pressure == 1.0
    later = T0 + timedelta(seconds=601)
    assert _reading(win, later) is None
    win.fold(_snap(later, {"orion:state:request": (12, 0)}))
    assert _reading(win, later).pressure == 0.0


def test_tie_break_is_deterministic():
    win = RpcDeliveryWindow()
    win.fold(_snap(T0, {"orion:b:request": (0, 1), "orion:a:request": (0, 1), "orion:c:request": (8, 2)}))
    r = _reading(win, T0)
    # a and b: 0.1 with 1 timeout; c: 2/10 = 0.2 wins outright
    assert r.worst_hop == "orion:c:request" and r.pressure == 0.2
    win2 = RpcDeliveryWindow()
    win2.fold(_snap(T0, {"orion:b:request": (0, 1), "orion:a:request": (0, 1)}))
    assert _reading(win2, T0).worst_hop == "orion:b:request"


def test_parse_exclude_labels():
    assert parse_exclude_labels(" a, ,b ") == ("a", "b")
    assert parse_exclude_labels("") == ()
    assert "current_turn_probe" in parse_exclude_labels(None)


def test_receipt_shape_is_what_the_field_digester_reads():
    win = RpcDeliveryWindow()
    win.fold(_snap(T0, {"orion:state:request": (9, 1)}))
    receipt = rpc_delivery_receipt(_reading(win, T0), now=T0)
    (delta,) = receipt.state_deltas
    assert delta.target_kind == "rpc_delivery"
    assert delta.after["node_id"] == "node:substrate.rpc_delivery"
    assert delta.after["pressure_hints"] == {RPC_DELIVERY_CHANNEL: 0.1}
    assert "orion:state:request" in delta.explanation
