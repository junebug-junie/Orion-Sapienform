"""generate_visual_bytes on the GPU pool (stage 5.4; replaced the durable-runs /capacity permit).

- With the durable run's hold: the hold is the grant, no second gate.
- Without one (the /visual-chain/run-once route, the legacy worker): one `diffusion` request lease
  around the call. Refused, late, or an unreachable pool is a deferral (resource_deferred), never
  an ungated diffusion call; a failure of the diffusion call itself stays a generation failure.

The pool side (a world lease really waits behind this) is
services/orion-gpu-pool/tests/test_world_diffusion_serialization.py against the real pool.
"""
from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock

import pytest

from orion.gpu_pool import client
from orion.gpu_pool.client import LeaseUnavailable, PoolRpcTimeout
from orion.schemas.gpu_pool import GpuLeaseRefV1

HOLD = GpuLeaseRefV1(lease_id="h1", generation=1, role="diffusion", holder="durable-runs:run-1")


def _refusing(error: BaseException, calls: list):
    @contextlib.asynccontextmanager
    async def lease(bus, **kw):
        calls.append(kw)
        raise error
        yield  # pragma: no cover
    return lease


def _diffusion(monkeypatch, vc, seen: list, pool=None, result=b"png"):
    def fake(prompt, *, base_url, timeout_sec):
        seen.append(pool.held if pool is not None else None)
        if isinstance(result, BaseException):
            raise result
        return result
    monkeypatch.setattr(vc, "call_diffusion_generate", fake)


def test_hold_is_the_grant_no_pool_call(monkeypatch, gpu_pool):
    from app import visual_chain as vc
    seen: list = []
    _diffusion(monkeypatch, vc, seen)
    assert asyncio.run(vc.generate_visual_bytes("p", correlation_id="c", hold=HOLD)) == b"png"
    assert seen == [None] and gpu_pool.calls == []


def test_no_hold_takes_one_diffusion_lease_around_the_call(monkeypatch, gpu_pool):
    from app import visual_chain as vc
    monkeypatch.setattr(vc.settings, "visual_chain_gpu_lease_deadline_sec", 180.0)
    seen: list = []
    _diffusion(monkeypatch, vc, seen, pool=gpu_pool)
    bus = object()
    assert asyncio.run(vc.generate_visual_bytes("p", correlation_id="chain-1", bus=bus)) == b"png"
    assert seen == [1], "diffusion must run while the lease is held"
    assert gpu_pool.held == 0
    [call] = gpu_pool.calls
    assert call["bus"] is bus
    assert (call["work_class"], call["priority"], call["deadline_sec"], call["turn_correlation_id"]) == \
        ("diffusion", "background", 180.0, "chain-1")


@pytest.mark.parametrize("error, reason", [
    (LeaseUnavailable("deadline", "l1"), "gpu_pool:deadline"),
    (LeaseUnavailable("role_down:diffusion", "l1"), "gpu_pool:role_down:diffusion"),
    (PoolRpcTimeout(full=True), "gpu_pool_unreachable"),
    (ConnectionError("redis down"), "gpu_pool_unreachable:ConnectionError"),
])
def test_refused_or_unreachable_pool_is_a_deferral_and_never_calls_diffusion(monkeypatch, error, reason):
    from app import visual_chain as vc
    calls: list = []
    monkeypatch.setattr(client, "gpu_lease", _refusing(error, calls))
    seen: list = []
    _diffusion(monkeypatch, vc, seen)
    with pytest.raises(vc.DiffusionResourceDeferred) as exc:
        asyncio.run(vc.generate_visual_bytes("p", correlation_id="c", bus=object()))
    assert str(exc.value) == reason
    assert seen == [] and len(calls) == 1


def test_no_bus_and_no_hold_defers_instead_of_running_ungated(monkeypatch, gpu_pool):
    from app import visual_chain as vc
    seen: list = []
    _diffusion(monkeypatch, vc, seen)
    with pytest.raises(vc.DiffusionResourceDeferred, match="gpu_pool_unreachable:no_bus"):
        asyncio.run(vc.generate_visual_bytes("p", correlation_id="c"))
    assert seen == [] and gpu_pool.calls == []


def test_diffusion_failure_inside_the_lease_stays_a_generation_failure(monkeypatch, gpu_pool):
    from app import visual_chain as vc
    seen: list = []
    _diffusion(monkeypatch, vc, seen, pool=gpu_pool, result=vc.DiffusionGenerationError("HTTP 500"))
    with pytest.raises(vc.DiffusionGenerationError):
        asyncio.run(vc.generate_visual_bytes("p", correlation_id="c", bus=object()))
    assert gpu_pool.held == 0


@pytest.mark.asyncio
async def test_run_once_pool_refusal_is_a_resource_deferred_chain(tmp_path, monkeypatch):
    """The visual chain's deferral path, end to end through run_visual_chain_once: the pool says no,
    the chain is persisted as resource_deferred with the pool's reason, no image, no diffusion call."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(visual_chain.settings, "thermal_gate_enabled", False)
    monkeypatch.setattr(visual_chain, "load_latest_visual_chain_continuity_state", lambda **kw: ("old", 0, 0))
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)
    calls: list = []
    monkeypatch.setattr(client, "gpu_lease", _refusing(LeaseUnavailable("deadline", "l1"), calls))

    def never(prompt, **kw):
        raise AssertionError("diffusion must not run without a grant")
    monkeypatch.setattr(visual_chain, "call_diffusion_generate", never)
    persisted: list = []
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: persisted.append(c) or True)

    chain = await visual_chain.run_visual_chain_once(AsyncMock())

    assert chain is not None and chain.terminal_reason == "resource_deferred"
    assert chain.chain_json["resource_gate"]["reason"] == "gpu_pool:deadline"
    assert len(persisted) == 1 and len(calls) == 1
    assert calls[0]["work_class"] == "diffusion"


def test_no_permit_or_elastic_code_left():
    """Kill means kill: nothing in orion-thought reaches the durable-runs /capacity permit or the
    gpu-lane-controller slot-status pre-check any more, and their settings are gone."""
    from pathlib import Path
    from app.settings import ThoughtSettings

    app_dir = Path(__file__).resolve().parents[1] / "app"
    for path in app_dir.rglob("*.py"):
        text = path.read_text()
        for banned in ("capacity_client", "GpuCapacityPermit", "capacity_url", ":8121/capacity", "/v1/gpu-slots"):
            assert banned not in text, f"{path.name} still references {banned}"
    fields = set(ThoughtSettings.model_fields)
    assert not any(f.startswith(("visual_chain_gpu2_capacity", "visual_elastic")) for f in fields)
