"""Durable hourly per-hop summaries of the transport baseline gate."""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import EquilibriumService, settings
from app.transport_baseline_gate import TransportBaselineGate
from app.transport_baseline_hourly import TransportBaselineHourly, _pct, hour_start_of
from orion.metacog.transport_baseline import TransportBaselineConfig
from orion.schemas.telemetry.transport_baseline_hourly import (
    TRANSPORT_BASELINE_HOURLY_KIND,
    TransportBaselineHourlyV1,
)

T0 = datetime(2026, 9, 29, 8, 0, tzinfo=timezone.utc)  # 02:00 MDT, a quiet hour
LLM = "orion:exec:request:LLMGatewayService"
METACOG_HOP = "orion:cortex:exec:request:background#log_orion_metacognition"


def _stats(lat: list[float], timeouts: int = 0) -> dict:
    logs = [math.log(x) for x in lat]
    return {
        "success_count": len(lat), "timeout_count": timeouts,
        "log_ms_sum": sum(logs), "log_ms_sumsq": sum(v * v for v in logs),
        "max_ms": max(lat) if lat else None,
    }


def _snap(i: int, channel_latency: dict) -> dict:
    start = T0 + timedelta(seconds=30 * i)
    return {
        "service": "cortex-exec", "node": "athena", "instance": "chat",
        "window_start": start.isoformat(), "window_end": (start + timedelta(seconds=30)).isoformat(),
        "success_count": 0, "timeout_count": 0, "channel_counts": {},
        "channel_latency": channel_latency,
    }


def _we(i: int) -> float:
    return (T0 + timedelta(seconds=30 * (i + 1))).timestamp()


def _run(gate, acc, windows: range, lat_fn, extra=None):
    for i in windows:
        cl = {LLM: _stats(lat_fn(i))}
        if extra:
            cl.update(extra(i))
        res, _ = gate.process(_snap(i, cl), zen_state="zen", pressure=0.0, recall_enabled=True)
        acc.observe(res, window_end_ts=_we(i))


def test_percentile_helper():
    assert _pct([], 0.5) is None
    assert _pct([3.0], 0.9) == 3.0
    assert _pct([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert hour_start_of(T0.timestamp() + 1799) == T0.timestamp()


def test_calm_hour_summarises_to_rest_state_and_validates():
    gate = TransportBaselineGate(TransportBaselineConfig(), ["log_orion_metacognition"])
    acc = TransportBaselineHourly(config_fingerprint=gate.config.fingerprint())
    # windows 0..118 end inside 08:00-09:00 (the 120th ends exactly at 09:00)
    _run(gate, acc, range(119), lambda i: [1000.0 * (1.0 + 0.05 * ((i + k) % 3 - 1)) for k in range(5)])
    assert acc.flush_due(T0.timestamp() + 3600 + 30, emit_effective=False) == []  # inside grace
    rows = acc.flush_due(T0.timestamp() + 3600 + 91, emit_effective=False)
    assert len(rows) == 1
    r = rows[0]
    TransportBaselineHourlyV1.model_validate(r.model_dump(mode="json"))
    assert (r.service, r.instance, r.key, r.flush_reason) == ("cortex-exec", "chat", LLM, "hour_end")
    assert r.hour_start == T0
    assert r.windows_seen == 119 and r.windows_evaluated == 119
    assert r.success_count == 119 * 5 and r.timeout_count == 0
    assert abs(r.z_p50) < 0.5 and r.z_p90 is not None
    assert 0.8 <= r.saturation_ratio_p50 <= 1.3
    assert r.warm and not r.excluded
    assert r.floor_ms_start is not None and r.floor_ms is not None and r.baseline_ms is not None
    assert r.calls_per_min_mean == pytest.approx(10.0)
    assert r.would_emit_by_condition == {} and r.conditions_opened == {}
    assert r.config_fingerprint == gate.config.fingerprint()
    assert acc.bucket_count == 0


def test_would_emit_counts_exclude_excluded_hops_but_opened_counts_them():
    gate = TransportBaselineGate(TransportBaselineConfig(), ["log_orion_metacognition"])
    acc = TransportBaselineHourly(config_fingerprint="f")
    _run(
        gate, acc, range(4), lambda i: [900.0] * 5,
        extra=lambda i: {METACOG_HOP: _stats([], timeouts=1)} if i == 1 else {},
    )
    # a real timeout on the LLM hop
    res, _ = gate.process(_snap(4, {LLM: _stats([], timeouts=2)}), zen_state="zen", pressure=0.0, recall_enabled=True)
    acc.observe(res, window_end_ts=_we(4))
    rows = {r.key: r for r in acc.flush_all(T0.timestamp() + 200, emit_effective=True)}
    assert rows[METACOG_HOP].excluded
    assert rows[METACOG_HOP].conditions_opened == {"timeout": 1}
    assert rows[METACOG_HOP].would_emit_by_condition == {}
    assert rows[LLM].would_emit_by_condition == {"zero_success:open": 1}
    assert "zero_success" in rows[LLM].open_at_hour_end
    assert rows[LLM].flush_reason == "shutdown" and rows[LLM].emit_effective


def test_windows_split_across_hour_boundary_land_in_their_own_hour():
    gate = TransportBaselineGate(TransportBaselineConfig(), [])
    acc = TransportBaselineHourly(config_fingerprint="f")
    _run(gate, acc, range(118, 122), lambda i: [1000.0] * 5)
    rows = acc.flush_all(T0.timestamp() + 7300, emit_effective=False)
    assert [(r.hour_start - T0).total_seconds() for r in rows] == [0.0, 3600.0]
    assert [r.windows_seen for r in rows] == [1, 3]


# ------------------------------------------------------------ service wiring


def _service(monkeypatch, *, publish_enable=True) -> EquilibriumService:
    monkeypatch.setattr(settings, "transport_baseline_enable", True)
    monkeypatch.setattr(settings, "transport_baseline_emit", False)
    monkeypatch.setattr(settings, "metacog_transport_trigger_enable", True)
    monkeypatch.setattr(settings, "transport_baseline_hourly_publish_enable", publish_enable)
    svc = EquilibriumService()
    svc.bus = MagicMock()
    svc.bus.publish = AsyncMock()
    svc.bus.redis = MagicMock()
    svc.bus.redis.set = AsyncMock()
    return svc


def _hourly_published(svc):
    return [c.args[1] for c in svc.bus.publish.call_args_list
            if c.args[0] == settings.channel_transport_baseline_hourly]


@pytest.mark.asyncio
async def test_service_publishes_hourly_rows_on_the_hourly_channel(monkeypatch):
    svc = _service(monkeypatch)
    for i in range(3):
        await svc._handle_rpc_health_snapshot(_snap(i, {LLM: _stats([1000.0] * 5)}), zen=0.9, distress=0.1)
    await svc._transport_housekeeping_once(T0.timestamp() + 600)
    assert _hourly_published(svc) == []
    await svc._transport_housekeeping_once(T0.timestamp() + 3600 + 91)
    envs = _hourly_published(svc)
    assert len(envs) == 1 and envs[0].kind == TRANSPORT_BASELINE_HOURLY_KIND
    assert envs[0].payload["windows_seen"] == 3 and envs[0].payload["key"] == LLM


@pytest.mark.asyncio
async def test_failed_publish_is_retried_not_lost(monkeypatch):
    svc = _service(monkeypatch)
    await svc._handle_rpc_health_snapshot(_snap(0, {LLM: _stats([1000.0] * 5)}), zen=0.9, distress=0.1)
    svc.bus.publish = AsyncMock(side_effect=RuntimeError("bus down"))
    await svc._transport_housekeeping_once(T0.timestamp() + 3600 + 91)
    assert len(svc._hourly_outbox) == 1
    svc.bus.publish = AsyncMock()
    await svc._transport_housekeeping_once(T0.timestamp() + 3600 + 101)
    assert len(_hourly_published(svc)) == 1 and svc._hourly_outbox == []


@pytest.mark.asyncio
async def test_shutdown_flushes_the_partial_hour(monkeypatch):
    svc = _service(monkeypatch)
    await svc._handle_rpc_health_snapshot(_snap(0, {LLM: _stats([1000.0] * 5)}), zen=0.9, distress=0.1)
    await svc._transport_shutdown_flush()
    envs = _hourly_published(svc)
    assert len(envs) == 1 and envs[0].payload["flush_reason"] == "shutdown"


@pytest.mark.asyncio
async def test_publish_flag_off_keeps_no_accumulator(monkeypatch):
    svc = _service(monkeypatch, publish_enable=False)
    assert svc._transport_hourly is None
    await svc._handle_rpc_health_snapshot(_snap(0, {LLM: _stats([1000.0] * 5)}), zen=0.9, distress=0.1)
    await svc._transport_shutdown_flush()
    assert _hourly_published(svc) == []


def test_a_fold_for_an_already_flushed_hour_is_a_late_row_not_a_reopen():
    gate = TransportBaselineGate(TransportBaselineConfig(), [])
    acc = TransportBaselineHourly(config_fingerprint="f")
    _run(gate, acc, range(0, 3), lambda i: [1000.0] * 5)
    assert [r.flush_reason for r in acc.flush_due(T0.timestamp() + 3700, emit_effective=False)] == ["hour_end"]
    _run(gate, acc, range(3, 4), lambda i: [1000.0] * 5)  # delayed snapshot, same hour
    rows = acc.flush_due(T0.timestamp() + 3800, emit_effective=False)
    assert [(r.flush_reason, r.windows_seen) for r in rows] == [("late", 1)]


def test_warm_at_start_records_whether_the_hour_began_warm():
    gate = TransportBaselineGate(TransportBaselineConfig(), [])
    acc = TransportBaselineHourly(config_fingerprint="f")
    _run(gate, acc, range(0, 30), lambda i: [1000.0] * 5)
    r = acc.flush_all(T0.timestamp() + 3700, emit_effective=False)[0]
    assert r.warm and not r.warm_at_start
