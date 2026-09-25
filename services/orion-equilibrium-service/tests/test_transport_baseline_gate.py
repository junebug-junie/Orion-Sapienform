"""Service wiring for the transport baseline gate (spec 2026-09-24, A1-A4)."""

from __future__ import annotations

import json
import logging
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
from app.transport_baseline_gate import (
    TransportBaselineGate,
    build_transport_baseline_trigger,
    config_from_settings,
)
from orion.metacog.transport_baseline import TransportBaselineConfig, TransportConditionEvent

T0 = datetime(2026, 9, 24, tzinfo=timezone.utc)
METACOG_HOP = "orion:cortex:exec:request:background#log_orion_metacognition"
LLM_HOP = "orion:exec:request:LLMGatewayService"

CONTRACT_KEYS = {
    "evidence_source", "condition", "phase", "service", "instance", "key", "z",
    "saturation_ratio", "baseline_ms", "floor_ms", "window_mean_ms", "calls_per_min",
    "calls_per_min_usual", "duration_s", "peak_ms", "timeout_count",
}


def _stats(lat: list[float], timeouts: int = 0) -> dict:
    logs = [math.log(x) for x in lat]
    return {
        "success_count": len(lat),
        "timeout_count": timeouts,
        "log_ms_sum": sum(logs),
        "log_ms_sumsq": sum(v * v for v in logs),
        "max_ms": max(lat) if lat else None,
    }


def _snap(i: int, channel_latency: dict | None, *, success=0, timeouts=0) -> dict:
    start = T0 + timedelta(seconds=30 * i)
    p = {
        "service": "cortex-orch",
        "node": "athena",
        "instance": None,
        "window_start": start.isoformat(),
        "window_end": (start + timedelta(seconds=30)).isoformat(),
        "success_count": success,
        "timeout_count": timeouts,
        "success_latency_ms_p95": 17000.0 if success else None,
        "channel_counts": {},
    }
    if channel_latency is not None:
        p["channel_latency"] = channel_latency
    return p


def _event(**kw) -> TransportConditionEvent:
    base = dict(
        condition="timeout", phase="open", service="cortex-exec", instance="chat",
        key=LLM_HOP, excluded=False, z=None, saturation_ratio=None, baseline_ms=None,
        floor_ms=None, window_mean_ms=None, calls_per_min=4.0, calls_per_min_usual=6.0,
        duration_s=None, peak_ms=None, timeout_count=2,
    )
    base.update(kw)
    return TransportConditionEvent(**base)


# ------------------------------------------------------------ output contract


def test_trigger_matches_output_contract_exactly():
    trig = build_transport_baseline_trigger(
        _event(condition="spike", phase="escalate", z=6.2, saturation_ratio=1.4,
               baseline_ms=9000.0, floor_ms=8800.0, window_mean_ms=31000.0,
               duration_s=120.0, peak_ms=40000.0, timeout_count=0),
        zen_state="zen", pressure=0.1, recall_enabled=True,
    )
    assert trig.trigger_kind == "transport"
    assert set(trig.upstream) == CONTRACT_KEYS
    assert trig.upstream["evidence_source"] == "transport_baseline"
    assert trig.reason.startswith(f"transport:spike:escalate:cortex-exec:{LLM_HOP}")
    assert trig.upstream["z"] == 6.2 and trig.upstream["timeout_count"] == 0


def test_excluded_event_never_becomes_a_trigger():
    assert build_transport_baseline_trigger(
        _event(key=METACOG_HOP, excluded=True), zen_state="zen", pressure=0.0, recall_enabled=True
    ) is None


def test_config_from_settings_maps_tunables(monkeypatch):
    monkeypatch.setattr(settings, "transport_baseline_min_calls", 7)
    monkeypatch.setattr(settings, "transport_baseline_regime_after_sec", 60.0)
    cfg = config_from_settings(settings)
    assert cfg.min_calls == 7 and cfg.regime_after_s == 60.0
    assert cfg.fingerprint() != TransportBaselineConfig().fingerprint()


def test_default_exclude_labels_cover_metacog_self_loop():
    # Was pinned to ["log_orion_metacognition"] and already stale on main once
    # gpu_pool_wait joined the default; assert membership of each required label.
    labels = settings.transport_exclude_labels()
    assert {"log_orion_metacognition", "gpu_pool_wait", "current_turn_probe"} <= set(labels)


def test_gate_load_logs_cold_start_on_config_change(caplog):
    a = TransportBaselineGate(TransportBaselineConfig(), ())
    a.process(_snap(0, {LLM_HOP: _stats([1000.0] * 5)}), zen_state="zen", pressure=0.0, recall_enabled=True)
    dumped = a.dump()
    b = TransportBaselineGate(TransportBaselineConfig(spike_z=4.0), ())
    with caplog.at_level(logging.WARNING):
        reason = b.load(dumped)
    assert reason.startswith("config_fingerprint_mismatch")
    assert "transport_baseline cold_start" in caplog.text
    assert b.state.keys == {}
    c = TransportBaselineGate(TransportBaselineConfig(), ())
    assert c.load(dumped) is None and len(c.state.keys) == 1
    assert TransportBaselineGate(TransportBaselineConfig(), ()).load(b"{not json") is not None


# ------------------------------------------------------------ service wiring


def _service(monkeypatch, *, emit: bool) -> EquilibriumService:
    monkeypatch.setattr(settings, "transport_baseline_enable", True)
    monkeypatch.setattr(settings, "transport_baseline_emit", emit)
    monkeypatch.setattr(settings, "metacog_transport_trigger_enable", True)
    monkeypatch.setattr(settings, "metacog_transport_cooldown_sec", 30.0)
    svc = EquilibriumService()
    svc.bus = MagicMock()
    svc.bus.publish = AsyncMock()
    svc.bus.redis = MagicMock()
    svc.bus.redis.set = AsyncMock()
    svc.bus.redis.get = AsyncMock(return_value=None)
    return svc


def _published(svc) -> list[dict]:
    return [c.args[1].payload for c in svc.bus.publish.call_args_list]


@pytest.mark.asyncio
async def test_log_only_mode_publishes_nothing_but_logs_and_persists(monkeypatch, caplog):
    svc = _service(monkeypatch, emit=False)
    with caplog.at_level(logging.INFO, logger="orion.equilibrium.transport_baseline_gate"):
        await svc._handle_rpc_health_snapshot(
            _snap(0, {LLM_HOP: _stats([], timeouts=1)}, timeouts=1), zen=0.9, distress=0.1
        )
    # legacy timeout branch still owns publishing while EMIT is off
    payloads = _published(svc)
    assert [p["upstream"]["evidence_source"] for p in payloads] == ["rpc_health_snapshot"]
    assert "transport_baseline_event emit=False" in caplog.text
    assert "transport_baseline_obs" in caplog.text
    svc.bus.redis.set.assert_awaited()
    key, blob = svc.bus.redis.set.await_args.args
    assert key == settings.transport_baseline_state_key
    assert json.loads(blob)["keys"]


@pytest.mark.asyncio
async def test_emit_mode_replaces_legacy_timeout_branch_no_double_firing(monkeypatch):
    svc = _service(monkeypatch, emit=True)
    await svc._handle_rpc_health_snapshot(
        _snap(0, {LLM_HOP: _stats([], timeouts=2)}, timeouts=2), zen=0.9, distress=0.1
    )
    payloads = _published(svc)
    assert len(payloads) == 1
    up = payloads[0]["upstream"]
    assert up["evidence_source"] == "transport_baseline"
    assert (up["condition"], up["phase"], up["timeout_count"]) == ("timeout", "open", 2)


@pytest.mark.asyncio
async def test_emit_mode_episode_rows_are_not_eaten_by_transport_cooldown(monkeypatch):
    svc = _service(monkeypatch, emit=True)
    hop_b = "orion:state:request"
    await svc._handle_rpc_health_snapshot(
        _snap(0, {LLM_HOP: _stats([], timeouts=1), hop_b: _stats([], timeouts=1)}, timeouts=2),
        zen=0.9, distress=0.1,
    )
    # two opens in the same second: both published despite the 30s lane
    assert len(_published(svc)) == 2
    # and they did not consume the lane for the other transport sources
    assert "transport" not in svc._last_trigger_ts_by_kind


@pytest.mark.asyncio
async def test_metacog_self_loop_never_triggers_even_when_emitting(monkeypatch):
    """Acceptance check 2 in miniature: metacog's own slow background draft,
    with a timeout, is measured but produces no trigger."""
    svc = _service(monkeypatch, emit=True)
    for i in range(40):
        await svc._handle_rpc_health_snapshot(
            _snap(i, {METACOG_HOP: _stats([17000.0] * 5, timeouts=1 if i == 30 else 0)}, success=5),
            zen=0.9, distress=0.1,
        )
    assert _published(svc) == []
    ks = next(iter(svc._transport_baseline_gate.state.keys.values()))
    assert ks.hop == METACOG_HOP and ks.fast_count > 0


@pytest.mark.asyncio
async def test_old_producer_without_channel_latency_is_skipped_quietly(monkeypatch):
    svc = _service(monkeypatch, emit=True)
    await svc._handle_rpc_health_snapshot(_snap(0, None, success=2), zen=0.9, distress=0.1)
    assert _published(svc) == []
    svc.bus.redis.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_baseline_disabled_keeps_legacy_timeout_branch(monkeypatch):
    monkeypatch.setattr(settings, "transport_baseline_enable", False)
    monkeypatch.setattr(settings, "transport_baseline_emit", True)
    monkeypatch.setattr(settings, "metacog_transport_trigger_enable", True)
    svc = EquilibriumService()
    svc.bus = MagicMock()
    svc.bus.publish = AsyncMock()
    assert svc._transport_baseline_gate is None
    assert settings.transport_baseline_emit_effective() is False
    await svc._handle_rpc_health_snapshot(_snap(0, None, timeouts=1), zen=0.9, distress=0.1)
    assert [p["upstream"]["evidence_source"] for p in _published(svc)] == ["rpc_health_snapshot"]


@pytest.mark.asyncio
async def test_load_state_from_redis_on_boot(monkeypatch):
    svc = _service(monkeypatch, emit=False)
    other = TransportBaselineGate(config_from_settings(settings), settings.transport_exclude_labels())
    other.process(_snap(0, {LLM_HOP: _stats([1000.0] * 5)}), zen_state="zen", pressure=0.0, recall_enabled=True)
    svc.bus.redis.get = AsyncMock(return_value=other.dump().encode())
    await svc._load_transport_baseline_state()
    assert len(svc._transport_baseline_gate.state.keys) == 1


# ------------------------------------------------------------ review fixes


def test_admit_enforces_hourly_budget():
    gate = TransportBaselineGate(TransportBaselineConfig(), (), max_triggers_per_hour=3)
    assert [gate.admit(t) for t in (0, 1, 2, 3)] == [True, True, True, False]
    assert gate.admit(3600.5)  # oldest aged out
    assert TransportBaselineGate(TransportBaselineConfig(), (), max_triggers_per_hour=0).admit(0)


@pytest.mark.asyncio
async def test_mesh_wide_outage_is_capped_by_budget(monkeypatch, caplog):
    monkeypatch.setattr(settings, "transport_baseline_max_triggers_per_hour", 2)
    svc = _service(monkeypatch, emit=True)
    hops = {f"orion:hop:{n}": _stats([], timeouts=1) for n in range(6)}
    with caplog.at_level(logging.WARNING):
        await svc._handle_rpc_health_snapshot(_snap(0, hops, timeouts=6), zen=0.9, distress=0.1)
    assert len(_published(svc)) == 2
    assert caplog.text.count("transport_baseline_suppressed") == 4


@pytest.mark.asyncio
async def test_fold_exception_cold_starts_and_keeps_legacy_branch(monkeypatch, caplog):
    svc = _service(monkeypatch, emit=False)
    gate = svc._transport_baseline_gate
    gate.process(_snap(0, {LLM_HOP: _stats([1000.0] * 5)}), zen_state="zen", pressure=0.0, recall_enabled=True)
    assert gate.state.keys

    def boom(*a, **k):
        raise RuntimeError("fold bug")

    monkeypatch.setattr("app.transport_baseline_gate.fold_snapshot", boom)
    with caplog.at_level(logging.ERROR):
        await svc._handle_rpc_health_snapshot(_snap(1, {}, timeouts=1), zen=0.9, distress=0.1)
    assert gate.state.keys == {}
    assert "cold_start reason=fold_failed:RuntimeError" in caplog.text
    assert [p["upstream"]["evidence_source"] for p in _published(svc)] == ["rpc_health_snapshot"]


def test_skipped_snapshots_are_logged_rate_limited(caplog):
    gate = TransportBaselineGate(TransportBaselineConfig(), ())
    with caplog.at_level(logging.INFO, logger="orion.equilibrium.transport_baseline_gate"):
        for i in range(250):
            gate.process(_snap(i, None), zen_state="zen", pressure=0.0, recall_enabled=True)
    lines = [r.getMessage() for r in caplog.records if "transport_baseline_skip" in r.getMessage()]
    assert lines == [
        "transport_baseline_skip reason=no_channel_latency count=1 service=cortex-orch",
        "transport_baseline_skip reason=no_channel_latency count=100 service=cortex-orch",
        "transport_baseline_skip reason=no_channel_latency count=200 service=cortex-orch",
    ]


@pytest.mark.asyncio
async def test_legacy_timeout_branch_ignores_new_publishers(monkeypatch):
    monkeypatch.setattr(settings, "transport_baseline_enable", False)
    monkeypatch.setattr(settings, "metacog_transport_trigger_enable", True)
    svc = EquilibriumService()
    svc.bus = MagicMock()
    svc.bus.publish = AsyncMock()
    snap = _snap(0, None, timeouts=3)
    snap["service"] = "orion-durable-runs"
    await svc._handle_rpc_health_snapshot(snap, zen=0.9, distress=0.1)
    assert _published(svc) == []
