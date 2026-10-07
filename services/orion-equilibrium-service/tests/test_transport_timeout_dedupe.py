"""One real RPC timeout -> one transport trigger, in both gate modes.

Log-only (EMIT off): the per-call rpc_transport_timeout atom is the only owner;
an rpc_health snapshot that saw the same timeout publishes nothing.
EMIT on: a gate-folded window that saw the timeout owns it and the atom is
dropped; an atom no window claims (a service that does not publish rpc_health)
still fires after the grace period -- coverage fails open, never closed.
"""

from __future__ import annotations

import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import EquilibriumService, settings
from app.transport_timeout_owner import PendingAtom, TimeoutAtomOwner, request_channel_of

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
LLM = "orion:exec:request:LLMGatewayService"
METACOG_HOP = "orion:cortex:exec:request:background#log_orion_metacognition"
UNCOVERED = "orion:topic_foundry:request"  # a caller that does not publish rpc_health


def _stats(lat: list[float], timeouts: int = 0) -> dict:
    logs = [math.log(x) for x in lat]
    return {
        "success_count": len(lat), "timeout_count": timeouts,
        "log_ms_sum": sum(logs), "log_ms_sumsq": sum(v * v for v in logs),
        "max_ms": max(lat) if lat else None,
    }


def _snap(i: int, channel_latency: dict | None, *, service="cortex-exec", instance="chat") -> dict:
    start = T0 + timedelta(seconds=30 * i)
    p = {
        "service": service, "node": "athena", "instance": instance,
        "window_start": start.isoformat(),
        "window_end": (start + timedelta(seconds=30)).isoformat(),
        "success_count": 0, "timeout_count": 0, "channel_counts": {},
    }
    if channel_latency is not None:
        p["channel_latency"] = channel_latency
    return p


def _atom_event(channel: str, *, at: datetime, corr: str = "c1") -> dict:
    """The GrammarEventV1 payload shape _emit_rpc_timeout_grammar publishes."""
    return {
        "event_id": f"bus.transport:rpc_timeout:{corr}:x",
        "event_kind": "atom_emitted",
        "trace_id": f"bus.transport:rpc_timeout:{corr}",
        "correlation_id": corr,
        "emitted_at": at.isoformat(),
        "atom": {
            "semantic_role": "rpc_transport_timeout",
            "text_value": channel,
            "summary": f"RPC timeout: {channel} -> reply after 60.0s",
        },
        "provenance": {"source_service": "orion-bus", "source_component": "rpc_request_timeout"},
    }


def _service(monkeypatch, *, emit: bool) -> EquilibriumService:
    monkeypatch.setattr(settings, "transport_baseline_enable", True)
    monkeypatch.setattr(settings, "transport_baseline_emit", emit)
    monkeypatch.setattr(settings, "metacog_transport_trigger_enable", True)
    monkeypatch.setattr(settings, "metacog_transport_cooldown_sec", 0.0)
    monkeypatch.setattr(settings, "transport_timeout_atom_grace_sec", 75.0)
    svc = EquilibriumService()
    svc.bus = MagicMock()
    svc.bus.publish = AsyncMock()
    svc.bus.redis = MagicMock()
    svc.bus.redis.set = AsyncMock()
    svc.bus.redis.get = AsyncMock(return_value=None)
    return svc


def _triggers(svc) -> list[dict]:
    return [
        c.args[1].payload for c in svc.bus.publish.call_args_list
        if c.args[0] == settings.channel_metacog_trigger
    ]


def _sources(svc) -> list[str]:
    return [p["upstream"]["evidence_source"] for p in _triggers(svc)]


async def _atom(svc, ev: dict) -> None:
    await svc._handle_rpc_timeout_atom(ev["atom"], ev, zen=0.9, distress=0.1)


async def _after_grace(svc) -> None:
    await svc._transport_housekeeping_once(time.time() + settings.transport_timeout_atom_grace_sec + 1)


# ------------------------------------------------------------------ log-only


@pytest.mark.asyncio
async def test_log_only_one_timeout_is_one_trigger_from_the_atom(monkeypatch):
    svc = _service(monkeypatch, emit=False)
    assert svc._timeout_owner is None
    await svc._handle_rpc_health_snapshot(_snap(0, {LLM: _stats([], timeouts=1)}), zen=0.9, distress=0.1)
    await _atom(svc, _atom_event(LLM, at=T0 + timedelta(seconds=10)))
    await _after_grace(svc)
    assert _sources(svc) == ["rpc_transport_timeout_grammar"]


@pytest.mark.asyncio
async def test_log_only_uncovered_service_atom_fires_immediately(monkeypatch):
    svc = _service(monkeypatch, emit=False)
    await _atom(svc, _atom_event(UNCOVERED, at=T0))
    assert _sources(svc) == ["rpc_transport_timeout_grammar"]


# ------------------------------------------------------------------ EMIT on


@pytest.mark.asyncio
async def test_emit_atom_before_snapshot_is_owned_by_the_gate(monkeypatch):
    svc = _service(monkeypatch, emit=True)
    await _atom(svc, _atom_event(LLM, at=T0 + timedelta(seconds=10)))
    assert _triggers(svc) == []  # held, not fired
    await svc._handle_rpc_health_snapshot(_snap(0, {LLM: _stats([], timeouts=1)}), zen=0.9, distress=0.1)
    await _after_grace(svc)
    assert _sources(svc) == ["transport_baseline"]
    assert svc._timeout_owner.pending_count == 0


@pytest.mark.asyncio
async def test_emit_atom_after_snapshot_is_dropped_on_arrival(monkeypatch):
    svc = _service(monkeypatch, emit=True)
    await svc._handle_rpc_health_snapshot(_snap(0, {LLM: _stats([], timeouts=1)}), zen=0.9, distress=0.1)
    await _atom(svc, _atom_event(LLM, at=T0 + timedelta(seconds=29)))
    assert svc._timeout_owner.pending_count == 0
    await _after_grace(svc)
    assert _sources(svc) == ["transport_baseline"]


@pytest.mark.asyncio
async def test_emit_uncovered_service_atom_still_fires_after_grace(monkeypatch):
    svc = _service(monkeypatch, emit=True)
    # the gate is busy with other hops, but nobody reports this channel
    await svc._handle_rpc_health_snapshot(_snap(0, {LLM: _stats([900.0] * 5)}), zen=0.9, distress=0.1)
    await _atom(svc, _atom_event(UNCOVERED, at=T0 + timedelta(seconds=5)))
    await svc._transport_housekeeping_once(time.time() + 1)  # inside grace: nothing yet
    assert _triggers(svc) == []
    await _after_grace(svc)
    assert _sources(svc) == ["rpc_transport_timeout_grammar"]
    assert _triggers(svc)[0]["upstream"]["request_channel"] == UNCOVERED


@pytest.mark.asyncio
async def test_emit_two_callers_one_gate_timeout_is_count_conserving(monkeypatch):
    """Two services time out on the same channel in one window; only one of
    them publishes rpc_health. One gate row + one atom row, not 1 and not 3."""
    svc = _service(monkeypatch, emit=True)
    await _atom(svc, _atom_event(LLM, at=T0 + timedelta(seconds=10), corr="a"))
    await _atom(svc, _atom_event(LLM, at=T0 + timedelta(seconds=12), corr="b"))
    await svc._handle_rpc_health_snapshot(_snap(0, {LLM: _stats([], timeouts=1)}), zen=0.9, distress=0.1)
    await _after_grace(svc)
    assert sorted(_sources(svc)) == ["rpc_transport_timeout_grammar", "transport_baseline"]


@pytest.mark.asyncio
async def test_emit_atom_outside_the_window_is_not_absorbed(monkeypatch):
    svc = _service(monkeypatch, emit=True)
    await svc._handle_rpc_health_snapshot(_snap(0, {LLM: _stats([], timeouts=1)}), zen=0.9, distress=0.1)
    await _atom(svc, _atom_event(LLM, at=T0 + timedelta(seconds=120)))
    await _after_grace(svc)
    assert sorted(_sources(svc)) == ["rpc_transport_timeout_grammar", "transport_baseline"]


@pytest.mark.asyncio
async def test_emit_labelled_hop_matches_its_bare_request_channel(monkeypatch):
    svc = _service(monkeypatch, emit=True)
    hop = "orion:gpu_pool:lease:request#gpu_pool_lease"
    await svc._handle_rpc_health_snapshot(
        _snap(0, {hop: _stats([], timeouts=1)}, service="orion-durable-runs", instance="main"),
        zen=0.9, distress=0.1,
    )
    await _atom(svc, _atom_event("orion:gpu_pool:lease:request", at=T0 + timedelta(seconds=3)))
    await _after_grace(svc)
    assert _sources(svc) == ["transport_baseline"]


@pytest.mark.asyncio
async def test_emit_excluded_hop_owns_its_atom_so_metacog_cannot_loop(monkeypatch):
    svc = _service(monkeypatch, emit=True)
    await svc._handle_rpc_health_snapshot(
        _snap(0, {METACOG_HOP: _stats([], timeouts=1)}, service="cortex-orch", instance="main"),
        zen=0.9, distress=0.1,
    )
    await _atom(svc, _atom_event("orion:cortex:exec:request:background", at=T0 + timedelta(seconds=8)))
    await _after_grace(svc)
    assert _triggers(svc) == []


@pytest.mark.asyncio
async def test_emit_skipped_snapshot_leaves_atom_to_fire(monkeypatch):
    """An old producer (no channel_latency) is skipped by the gate, so it
    grants no credit and the atom keeps coverage."""
    svc = _service(monkeypatch, emit=True)
    await _atom(svc, _atom_event(LLM, at=T0 + timedelta(seconds=10)))
    await svc._handle_rpc_health_snapshot(_snap(0, None), zen=0.9, distress=0.1)
    await _after_grace(svc)
    assert _sources(svc) == ["rpc_transport_timeout_grammar"]


@pytest.mark.asyncio
async def test_emit_shutdown_fires_held_atoms_instead_of_losing_them(monkeypatch):
    svc = _service(monkeypatch, emit=True)
    await _atom(svc, _atom_event(UNCOVERED, at=T0))
    await svc._transport_shutdown_flush()
    assert _sources(svc) == ["rpc_transport_timeout_grammar"]


@pytest.mark.asyncio
async def test_atom_path_still_respects_the_transport_cooldown_lane(monkeypatch):
    svc = _service(monkeypatch, emit=False)
    monkeypatch.setattr(settings, "metacog_transport_cooldown_sec", 30.0)
    await _atom(svc, _atom_event(UNCOVERED, at=T0, corr="a"))
    await _atom(svc, _atom_event(UNCOVERED, at=T0, corr="b"))
    assert len(_triggers(svc)) == 1


# ------------------------------------------------------------ pure owner unit


def _pending(channel: str, emitted: float, received: float = 0.0) -> PendingAtom:
    return PendingAtom(atom={}, correlation_id="c", request_channel=channel,
                       emitted_ts=emitted, received_ts=received, zen_state="zen", pressure=0.0)


class _Ob:
    def __init__(self, key, timeouts, excluded=False):
        self.key, self.timeout_count, self.excluded = key, timeouts, excluded
        self.service, self.instance = "svc", "i"


def test_request_channel_of_strips_the_health_label():
    assert request_channel_of("orion:x#lbl") == "orion:x"
    assert request_channel_of("orion:x") == "orion:x"
    assert request_channel_of("verb:chat_general") == "verb:chat_general"


def test_owner_prefers_a_non_excluded_credit_for_an_ambiguous_atom():
    o = TimeoutAtomOwner()
    o.add_credits([_Ob("orion:bg#log_orion_metacognition", 1, True), _Ob("orion:bg", 1)],
                  window_start_ts=0.0, window_end_ts=30.0, now=0.0)
    assert o.offer_atom(_pending("orion:bg", 10.0))
    # the excluded credit is what is left
    excl, plain = o._credits["orion:bg"]
    assert excl.excluded and excl.remaining == 1
    assert plain.remaining == 0


def test_owner_credits_age_out():
    o = TimeoutAtomOwner(credit_ttl_s=300.0)
    o.add_credits([_Ob("orion:x", 2)], window_start_ts=0.0, window_end_ts=30.0, now=0.0)
    assert o.credit_count == 2
    o.expire(301.0)
    assert o.credit_count == 0


def test_owner_matches_a_timeout_absorbed_into_a_later_window():
    """Short-lived buses (orion-mind, dispatch runtime, thought) fold into the
    publisher via absorb(), which keeps the absorbing window's start: a
    timeout 20 s before window_start is still that window's timeout."""
    o = TimeoutAtomOwner()
    o.add_credits([_Ob("orion:exec:request:LLMGatewayService", 1)], window_start_ts=100.0,
                  window_end_ts=130.0, now=0.0)
    assert o.offer_atom(_pending("orion:exec:request:LLMGatewayService", 80.0))
    # but not arbitrarily early
    o.add_credits([_Ob("orion:x", 1)], window_start_ts=100.0, window_end_ts=130.0, now=0.0)
    assert not o.offer_atom(_pending("orion:x", 100.0 - o.lookback_s - 1))


def test_owner_pending_overflow_fires_oldest_instead_of_dropping():
    o = TimeoutAtomOwner(max_pending=2)
    for i in range(3):
        assert not o.offer_atom(_pending("orion:a" if i % 2 else "orion:b", 0.0, received=float(i)))
    assert o.pending_count == 2
    fired = o.expire(10.0)  # inside grace: only the overflow fires
    assert [p.received_ts for p in fired] == [0.0]
    assert len(o.drain()) == 2


@pytest.mark.asyncio
async def test_real_shutdown_path_fires_held_atoms_before_the_bus_closes(monkeypatch):
    """The chassis cancels _run while it is parked in iter_messages(), so the
    flush must live in _shutdown and run before bus.close()."""
    svc = _service(monkeypatch, emit=True)
    order: list[str] = []
    svc.bus.publish = AsyncMock(side_effect=lambda ch, env: order.append(f"publish:{ch}"))
    svc.bus.close = AsyncMock(side_effect=lambda: order.append("close"))
    await _atom(svc, _atom_event(UNCOVERED, at=T0))
    await svc._handle_rpc_health_snapshot(_snap(0, {LLM: _stats([900.0] * 5)}), zen=0.9, distress=0.1)
    await svc._shutdown()
    assert order[-1] == "close"
    assert f"publish:{settings.channel_metacog_trigger}" in order
    assert f"publish:{settings.channel_transport_baseline_hourly}" in order


@pytest.mark.asyncio
async def test_housekeeping_task_is_created_once_across_run_restarts(monkeypatch):
    svc = _service(monkeypatch, emit=True)
    svc._ensure_transport_housekeeping()
    first = svc._transport_housekeeping_task
    svc._ensure_transport_housekeeping()
    assert svc._transport_housekeeping_task is first
    first.cancel()
