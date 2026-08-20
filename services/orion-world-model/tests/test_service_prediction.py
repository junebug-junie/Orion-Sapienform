"""End-to-end (no bus, no HTTP) coverage of the CPU inference path: trajectory
payload -> tensors -> real forward pass -> WorldModelPredictionPayload.

This is the seam CLAUDE.md's "no empty-shell cognition" rule cares about --
proof the service does a real forward pass, not a stub, and that it says so
honestly (`model_untrained=True`) rather than dressing random-weight output
up as a working prediction.
"""

from __future__ import annotations

import asyncio

import pytest

from app.main import WorldModelService, trajectory_steps_to_tensors
from app.model import FeatureGroupDims
from app.settings import Settings
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


def test_trajectory_steps_to_tensors_shapes():
    dims = _dims()
    steps = [_step(float(i), dims) for i in range(3)]
    tensors = trajectory_steps_to_tensors(steps, dims, device="cpu")
    assert tensors["biometrics"].shape == (1, 3, 4)
    assert tensors["vision_embedding"].shape == (1, 3, 8)


def test_trajectory_steps_to_tensors_rejects_dim_mismatch():
    dims = _dims()
    steps = [_step(0.0, dims)]
    # Corrupt one group's declared dim so it disagrees with configured dims.
    steps[0].biometrics.dim = 99
    with pytest.raises(ValueError, match="does not match configured"):
        trajectory_steps_to_tensors(steps, dims, device="cpu")


def test_forward_no_grad_actually_disables_grad_across_to_thread():
    """Regression test: torch's grad-enabled flag is thread-local, so a bare
    `with torch.no_grad(): await asyncio.to_thread(model, x)` in the caller's
    coroutine does NOT disable grad inside the spawned worker thread
    (verified live while building app/main.py). `_forward_no_grad` must enter
    `torch.no_grad()` from *inside* the function that runs on the worker
    thread."""
    import torch

    from app.model import WorldModel

    async def body():
        service = WorldModelService()
        dims = _dims()
        service.dims = dims
        service.model = WorldModel(dims, fusion_dim=8, d_model=8, nhead=2, num_layers=1, dim_feedforward=16, max_window=4)
        service.model.eval()

        steps = [_step(0.0, dims)]
        tensors = trajectory_steps_to_tensors(steps, dims, device="cpu")

        mean, log_var = await asyncio.to_thread(service._forward_no_grad, tensors)
        assert mean.requires_grad is False
        assert log_var.requires_grad is False

    asyncio.run(body())


def test_run_prediction_task_real_forward_pass_cpu():
    """Constructs the service directly (no FastAPI lifespan, no bus) with a
    tiny model so this stays fast, then runs a real request through it."""

    async def body():
        service = WorldModelService()
        dims = _dims()
        service.dims = dims
        from app.model import WorldModel

        service.model = WorldModel(
            dims, fusion_dim=16, d_model=16, nhead=2, num_layers=1, dim_feedforward=32, max_window=8, state_dim=16
        )
        service.model.eval()
        service.device = "cpu"
        service._inflight_sem = asyncio.Semaphore(2)

        steps = [_step(float(i), dims) for i in range(3)]
        payload = WorldModelTaskRequestPayload(task_type="predict_next_state", trajectory=steps)

        result = await service.run_prediction_task(payload)
        assert result.ok is True
        assert result.model_untrained is True
        assert result.state_dim == 16
        assert result.window_len == 3
        assert len(result.predicted_mean) == 16
        assert len(result.predicted_log_var) == 16
        assert result.device == "cpu"

    asyncio.run(body())


def test_run_prediction_task_bad_trajectory_returns_error_not_exception():
    async def body():
        service = WorldModelService()
        dims = _dims()
        service.dims = dims
        from app.model import WorldModel

        service.model = WorldModel(dims, fusion_dim=8, d_model=8, nhead=2, num_layers=1, dim_feedforward=16, max_window=4)
        service.model.eval()
        service.device = "cpu"
        service._inflight_sem = asyncio.Semaphore(2)

        steps = [_step(0.0, dims)]
        steps[0].vision_embedding.dim = 3  # disagrees with configured dim=8
        payload = WorldModelTaskRequestPayload(task_type="predict_next_state", trajectory=steps)

        result = await service.run_prediction_task(payload)
        assert result.ok is False
        assert result.error_code == "bad_trajectory"

    asyncio.run(body())


def test_run_prediction_task_honors_wm_timeout_s():
    """WM_TIMEOUT_S was declared in Settings/.env_example but never actually
    read anywhere (code review finding) -- pins that app/main.py's
    asyncio.wait_for wrapping actually triggers on a slow forward pass and
    returns error_code=timeout rather than hanging or raising."""

    async def body():
        from app.model import WorldModel

        service = WorldModelService()
        dims = _dims()
        service.dims = dims
        # Not actually invoked (the test monkeypatches _run_forward below) --
        # only present so run_prediction_task's readiness check passes.
        service.model = WorldModel(dims, fusion_dim=4, d_model=4, nhead=1, num_layers=1, dim_feedforward=8, max_window=2)
        service.device = "cpu"
        service._inflight_sem = asyncio.Semaphore(2)

        async def _slow_run_forward(payload):
            await asyncio.sleep(0.5)
            raise AssertionError("should never reach this -- wait_for must time out first")

        service._run_forward = _slow_run_forward  # type: ignore[method-assign]

        import app.main as main_module

        original_timeout = main_module.settings.WM_TIMEOUT_S
        main_module.settings.WM_TIMEOUT_S = 0.05
        try:
            steps = [_step(0.0, dims)]
            payload = WorldModelTaskRequestPayload(task_type="predict_next_state", trajectory=steps)
            result = await service.run_prediction_task(payload)
        finally:
            main_module.settings.WM_TIMEOUT_S = original_timeout

        assert result.ok is False
        assert result.error_code == "timeout"

    asyncio.run(body())


def test_service_not_ready_returns_error_payload():
    async def body():
        service = WorldModelService()  # model/dims never set
        payload = WorldModelTaskRequestPayload(
            task_type="predict_next_state", trajectory=[_step(0.0, _dims())]
        )
        result = await service.run_prediction_task(payload)
        assert result.ok is False
        assert result.error_code == "service_not_ready"

    asyncio.run(body())


def test_settings_dims_match_service_default_construction():
    """Settings' default WM_DIM_* values must actually build a valid
    FeatureGroupDims (guards against a typo silently going unused)."""
    s = Settings()
    dims = FeatureGroupDims(
        biometrics=s.WM_DIM_BIOMETRICS,
        affect=s.WM_DIM_AFFECT,
        execution_context=s.WM_DIM_EXECUTION_CONTEXT,
        memory_pointers=s.WM_DIM_MEMORY_POINTERS,
        temporal=s.WM_DIM_TEMPORAL,
        vision_embedding=s.WM_DIM_VISION_EMBEDDING,
    )
    assert dims.total == 32 + 16 + 16 + 32 + 8 + 512
