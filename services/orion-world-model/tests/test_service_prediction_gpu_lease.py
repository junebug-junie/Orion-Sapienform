"""GPU2 mutex with orion-diffusion-host on the forward-pass call site, via an orion-gpu-pool lease
(stage 5.4; replaced the durable-runs /capacity permit).

The two services share circe's gpu2 with no OS-level arbitration (two "CUDA-capable device(s) is/are
busy or unavailable" failures, 2026-09-24). These tests pin the service-side contract: a CUDA forward
pass runs only inside a granted `world` lease, a refused/late lease fails fast with gpu_contended and
never runs the forward pass, an unreachable pool is its own code (never an ungated run), and CPU
fallback never asks. The pool-side half (the lease really waits behind a diffusion hold) is
services/orion-gpu-pool/tests/test_world_diffusion_serialization.py, against the real pool.
"""
from __future__ import annotations

import asyncio
import contextlib

import pytest
import torch

from app import main
from app.main import WorldModelService
from app.model import FeatureGroupDims, WorldModel
from orion.gpu_pool.client import LeaseUnavailable, PoolRpcTimeout
from orion.schemas.world_model import WorldModelFeatureGroupV1, WorldModelTaskRequestPayload, WorldModelTrajectoryStepV1


def _dims() -> FeatureGroupDims:
    return FeatureGroupDims(
        biometrics=4, affect=2, execution_context=2, memory_pointers=4, temporal=1, vision_embedding=8
    )


def _step(ts: float, dims: FeatureGroupDims) -> WorldModelTrajectoryStepV1:
    def grp(dim: int) -> WorldModelFeatureGroupV1:
        return WorldModelFeatureGroupV1(dim=dim, vector=[0.1] * dim)

    return WorldModelTrajectoryStepV1(
        ts=ts, biometrics=grp(dims.biometrics), affect=grp(dims.affect),
        execution_context=grp(dims.execution_context), memory_pointers=grp(dims.memory_pointers),
        temporal=grp(dims.temporal), vision_embedding=grp(dims.vision_embedding),
    )


def _service(*, device: str, bus: object | None = "bus") -> tuple[WorldModelService, WorldModelTaskRequestPayload]:
    service = WorldModelService()
    dims = _dims()
    service.dims = dims
    service.model = WorldModel(
        dims, fusion_dim=16, d_model=16, nhead=2, num_layers=1, dim_feedforward=32, max_window=8, state_dim=16
    )
    service.model.eval()
    service.device = device
    service.bus = bus  # type: ignore[assignment]
    service._inflight_sem = asyncio.Semaphore(2)
    steps = [_step(float(i), dims) for i in range(3)]
    return service, WorldModelTaskRequestPayload(task_type="predict_next_state", trajectory=steps)


class FakeLease:
    def __init__(self):
        self.release_outcome = None
        self.release_detail = None


class FakePool:
    """Stands in for orion.gpu_pool.client.gpu_lease; records every call and its lifecycle."""

    def __init__(self, error: BaseException | None = None):
        self.error = error
        self.calls: list[dict] = []
        self.held = False
        self.released: list[FakeLease] = []

    @contextlib.asynccontextmanager
    async def __call__(self, bus, **kw):
        self.calls.append({"bus": bus, **kw})
        if self.error is not None:
            raise self.error
        lease = FakeLease()
        self.held = True
        try:
            yield lease
        finally:
            self.held = False
            self.released.append(lease)


def _stub_forward(service: WorldModelService, seen: list | None = None, pool: FakePool | None = None):
    async def run(payload):
        if seen is not None:
            seen.append(pool.held if pool else None)
        return torch.zeros(1, 4), torch.zeros(1, 4)
    service._run_forward = run  # type: ignore[method-assign]


def test_forward_pass_runs_inside_a_world_lease_and_releases_after(monkeypatch):
    pool = FakePool()
    monkeypatch.setattr(main, "gpu_lease", pool)
    monkeypatch.setattr(main.settings, "WM_GPU_LEASE_DEADLINE_SEC", 2.0)
    service, payload = _service(device="cuda:0")
    seen: list = []
    _stub_forward(service, seen, pool)

    result = asyncio.run(service.run_prediction_task(payload))

    assert result.ok is True
    assert seen == [True], "the forward pass must run while the lease is held"
    assert not pool.held and len(pool.released) == 1
    call = pool.calls[0]
    assert call["bus"] == "bus"
    assert (call["work_class"], call["priority"], call["deadline_sec"]) == ("world", "system", 2.0)
    assert call["holder"] == main.settings.SERVICE_NAME


@pytest.mark.parametrize("reason", ["deadline", "serialized:diffusion", "role_down:world"])
def test_refused_lease_is_gpu_contended_and_never_runs_the_forward_pass(monkeypatch, reason):
    monkeypatch.setattr(main, "gpu_lease", FakePool(LeaseUnavailable(reason, "l1")))
    service, payload = _service(device="cuda:0")

    async def boom(payload):
        raise AssertionError("forward pass must not run without a lease")
    service._run_forward = boom  # type: ignore[method-assign]

    result = asyncio.run(service.run_prediction_task(payload))

    assert (result.ok, result.error_code) == (False, "gpu_contended")
    assert result.error == f"gpu2 contended: {reason}"


@pytest.mark.parametrize("error", [PoolRpcTimeout(full=True), ConnectionError("redis down")])
def test_unreachable_pool_refuses_with_its_own_code_never_ungated(monkeypatch, error):
    monkeypatch.setattr(main, "gpu_lease", FakePool(error))
    service, payload = _service(device="cuda:0")

    async def boom(payload):
        raise AssertionError("an unreachable pool must not mean an ungated forward pass")
    service._run_forward = boom  # type: ignore[method-assign]

    result = asyncio.run(service.run_prediction_task(payload))

    assert (result.ok, result.error_code) == (False, "gpu_pool_unreachable")


def test_short_pool_rpc_timeout_is_contention_not_a_dead_pool(monkeypatch):
    """With a 2 s deadline the lease RPC timeout is always the shortened one (full=False), which
    says nothing about the pool's health (client.PoolRpcTimeout docstring): report contention."""
    monkeypatch.setattr(main, "gpu_lease", FakePool(PoolRpcTimeout(full=False)))
    service, payload = _service(device="cuda:0")
    _stub_forward(service)

    result = asyncio.run(service.run_prediction_task(payload))

    assert (result.ok, result.error_code) == (False, "gpu_contended")


def test_timed_out_forward_keeps_the_lease_while_it_still_runs(monkeypatch):
    """wait_for cannot kill the forward-pass thread. On the card, the lease must stay held while that
    work is still computing (up to one more WM_TIMEOUT_S), or diffusion could be granted on top."""
    pool = FakePool()
    monkeypatch.setattr(main, "gpu_lease", pool)
    monkeypatch.setattr(main.settings, "WM_TIMEOUT_S", 0.05)
    service, payload = _service(device="cuda:0")
    finished: list = []

    async def slow(payload):
        await asyncio.sleep(0.08)          # past WM_TIMEOUT_S, inside the second window
        finished.append(pool.held)
        return torch.zeros(1, 4), torch.zeros(1, 4)
    service._run_forward = slow  # type: ignore[method-assign]

    result = asyncio.run(service.run_prediction_task(payload))

    assert (result.ok, result.error_code) == (False, "timeout")
    assert finished == [True], "the lease was released while the forward pass was still running"
    assert pool.released[0].release_outcome == "upstream_error"


def test_no_bus_on_cuda_refuses_instead_of_running_ungated(monkeypatch):
    pool = FakePool()
    monkeypatch.setattr(main, "gpu_lease", pool)
    service, payload = _service(device="cuda:0", bus=None)
    _stub_forward(service)

    result = asyncio.run(service.run_prediction_task(payload))

    assert (result.ok, result.error_code) == (False, "gpu_pool_unreachable")
    assert pool.calls == []


def test_cpu_fallback_never_asks_the_pool(monkeypatch):
    pool = FakePool()
    monkeypatch.setattr(main, "gpu_lease", pool)
    service, payload = _service(device="cpu", bus=None)

    result = asyncio.run(service.run_prediction_task(payload))

    assert result.ok is True
    assert pool.calls == []


def test_forward_failure_releases_the_lease_as_upstream_error(monkeypatch):
    pool = FakePool()
    monkeypatch.setattr(main, "gpu_lease", pool)
    service, payload = _service(device="cuda:0")

    async def boom(payload):
        raise RuntimeError("CUDA error: CUDA-capable device(s) is/are busy or unavailable")
    service._run_forward = boom  # type: ignore[method-assign]

    result = asyncio.run(service.run_prediction_task(payload))

    assert (result.ok, result.error_code) == (False, "forward_failed")
    assert not pool.held, "a stuck lease would keep diffusion off gpu2 after this request gave up"
    assert pool.released[0].release_outcome == "upstream_error"


def test_no_permit_code_left():
    """Kill means kill: the durable-runs /capacity permit is gone from this service."""
    src = (main.__file__)
    text = open(src).read()
    assert "capacity_client" not in text and "GpuCapacityPermit" not in text
    assert not any(k.startswith("WM_GPU2_CAPACITY") for k in main.Settings.model_fields)
