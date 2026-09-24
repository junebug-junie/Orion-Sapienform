"""GPU2 capacity mutex on the forward-pass call site.

This service shares circe's GPU2 with orion-diffusion-host, with no
OS-level arbitration -- live-confirmed 2026-09-24 (two back-to-back "CUDA
error: CUDA-capable device(s) is/are busy or unavailable" failures on
diffusion-host's side). These tests assert the precedence model: a short
acquire budget must fail fast (never run the forward pass) rather than
block this service's own callers, and CPU fallback must never even try to
acquire -- there's no shared-hardware contention to arbitrate there.
"""
from __future__ import annotations

import asyncio

import pytest
import torch

from app import main
from app.main import WorldModelService
from app.model import FeatureGroupDims, WorldModel
from orion.durable_admission.capacity_client import CapacityRejected
from orion.schemas.world_model import WorldModelFeatureGroupV1, WorldModelTaskRequestPayload, WorldModelTrajectoryStepV1


def _dims() -> FeatureGroupDims:
    return FeatureGroupDims(
        biometrics=4, affect=2, execution_context=2, memory_pointers=4, temporal=1, vision_embedding=8
    )


def _step(ts: float, dims: FeatureGroupDims) -> WorldModelTrajectoryStepV1:
    def grp(dim: int) -> WorldModelFeatureGroupV1:
        return WorldModelFeatureGroupV1(dim=dim, vector=[0.1] * dim)

    return WorldModelTrajectoryStepV1(
        ts=ts,
        biometrics=grp(dims.biometrics),
        affect=grp(dims.affect),
        execution_context=grp(dims.execution_context),
        memory_pointers=grp(dims.memory_pointers),
        temporal=grp(dims.temporal),
        vision_embedding=grp(dims.vision_embedding),
    )


def _service(*, device: str) -> tuple[WorldModelService, WorldModelTaskRequestPayload]:
    service = WorldModelService()
    dims = _dims()
    service.dims = dims
    service.model = WorldModel(
        dims, fusion_dim=16, d_model=16, nhead=2, num_layers=1, dim_feedforward=32, max_window=8, state_dim=16
    )
    service.model.eval()
    service.device = device
    service._inflight_sem = asyncio.Semaphore(2)
    steps = [_step(float(i), dims) for i in range(3)]
    payload = WorldModelTaskRequestPayload(task_type="predict_next_state", trajectory=steps)
    return service, payload


class FakePermit:
    instances: list["FakePermit"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.acquired = False
        self.closed = False
        self.acquire_error: Exception | None = None
        FakePermit.instances.append(self)

    async def acquire(self) -> "FakePermit":
        if self.acquire_error is not None:
            raise self.acquire_error
        self.acquired = True
        return self

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    FakePermit.instances.clear()
    monkeypatch.setattr(main.settings, "WM_GPU2_CAPACITY_ENABLED", True)


def test_cpu_fallback_never_constructs_a_permit(monkeypatch):
    """No shared-hardware contention on CPU -- must not depend on
    durable-runs being reachable at all."""
    called = []
    monkeypatch.setattr(main, "GpuCapacityPermit", lambda **k: called.append(k) or FakePermit(**k))
    service, payload = _service(device="cpu")

    result = asyncio.run(service.run_prediction_task(payload))

    assert result.ok is True
    assert called == []


def test_disabled_flag_never_constructs_a_permit(monkeypatch):
    monkeypatch.setattr(main.settings, "WM_GPU2_CAPACITY_ENABLED", False)
    called = []
    monkeypatch.setattr(main, "GpuCapacityPermit", lambda **k: called.append(k) or FakePermit(**k))
    service, payload = _service(device="cuda:2")

    # This sandbox has no GPU -- stub the forward pass (same pattern as the
    # other cuda: tests below) so this test exercises the flag check only,
    # not real CUDA tensor placement.
    async def _stub_run_forward(payload):
        return torch.zeros(1, 4), torch.zeros(1, 4)
    service._run_forward = _stub_run_forward  # type: ignore[method-assign]

    result = asyncio.run(service.run_prediction_task(payload))

    assert result.ok is True
    assert called == []


def test_contended_gpu_fails_fast_without_running_the_forward_pass(monkeypatch):
    def _make(**kwargs):
        permit = FakePermit(**kwargs)
        permit.acquire_error = CapacityRejected("capacity_wait_budget_exhausted")
        return permit
    monkeypatch.setattr(main, "GpuCapacityPermit", _make)
    service, payload = _service(device="cuda:2")

    async def _boom(payload):
        raise AssertionError("forward pass must not run while GPU2 is contended")
    service._run_forward = _boom  # type: ignore[method-assign]

    result = asyncio.run(service.run_prediction_task(payload))

    assert result.ok is False
    assert result.error_code == "gpu_contended"
    assert FakePermit.instances and not FakePermit.instances[0].acquired


def test_permit_acquired_before_forward_pass_and_released_after(monkeypatch):
    # Stubs _run_forward directly (same pattern as this suite's own
    # test_run_prediction_task_honors_wm_timeout_s) rather than building
    # real cuda: tensors -- this sandbox has no GPU, and the property under
    # test is permit ordering, not real forward-pass math.
    monkeypatch.setattr(main, "GpuCapacityPermit", FakePermit)
    service, payload = _service(device="cuda:2")
    order: list[str] = []

    async def _stub_run_forward(payload):
        order.append("forward")
        assert FakePermit.instances[0].acquired
        assert not FakePermit.instances[0].closed
        return torch.zeros(1, 4), torch.zeros(1, 4)
    service._run_forward = _stub_run_forward  # type: ignore[method-assign]

    result = asyncio.run(service.run_prediction_task(payload))

    assert result.ok is True
    assert order == ["forward"]
    assert FakePermit.instances[0].acquired and FakePermit.instances[0].closed


def test_permit_released_even_when_forward_pass_raises(monkeypatch):
    monkeypatch.setattr(main, "GpuCapacityPermit", FakePermit)
    service, payload = _service(device="cuda:2")

    async def _boom(payload):
        raise RuntimeError("CUDA error: CUDA-capable device(s) is/are busy or unavailable")
    service._run_forward = _boom  # type: ignore[method-assign]

    result = asyncio.run(service.run_prediction_task(payload))

    assert result.ok is False
    assert result.error_code == "forward_failed"
    assert FakePermit.instances[0].acquired and FakePermit.instances[0].closed, (
        "a stuck, unreleased permit would starve the other side of the mutex "
        "(orion-diffusion-host) even after this request gives up"
    )
