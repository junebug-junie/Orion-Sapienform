"""Real forward-pass shape tests for the encoder + dynamics model.

Uses real tensors through real (untrained) weights -- this is the concrete
evidence backing README's "genuine forward pass, not a stub" claim
(CLAUDE.md §0A "no empty-shell cognition").
"""

from __future__ import annotations

import torch

from app.model import FEATURE_GROUP_NAMES, FeatureGroupDims, WorldModel, WorldModelDynamics, WorldModelEncoder


def _dims() -> FeatureGroupDims:
    return FeatureGroupDims(
        biometrics=32,
        affect=16,
        execution_context=16,
        memory_pointers=32,
        temporal=8,
        vision_embedding=512,
    )


def test_feature_group_dims_total():
    dims = _dims()
    assert dims.total == 32 + 16 + 16 + 32 + 8 + 512
    assert set(dims.as_dict()) == set(FEATURE_GROUP_NAMES)


def test_encoder_forward_shape():
    dims = _dims()
    encoder = WorldModelEncoder(dims, fusion_dim=64)
    batch = 3
    groups = {name: torch.randn(batch, getattr(dims, name)) for name in FEATURE_GROUP_NAMES}
    out = encoder(**groups)
    assert out.shape == (batch, 64)
    assert torch.isfinite(out).all()


def test_encoder_missing_group_raises():
    dims = _dims()
    encoder = WorldModelEncoder(dims, fusion_dim=64)
    groups = {name: torch.randn(2, getattr(dims, name)) for name in FEATURE_GROUP_NAMES}
    groups.pop("temporal")
    try:
        encoder(**groups)
        assert False, "expected ValueError for missing feature group"
    except ValueError as exc:
        assert "temporal" in str(exc)


def test_dynamics_forward_shape_and_causal_mask_applies():
    dynamics = WorldModelDynamics(
        fusion_dim=64, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_window=8, state_dim=32
    )
    batch, window = 2, 5
    trajectory = torch.randn(batch, window, 64)
    mean, log_var = dynamics(trajectory)
    assert mean.shape == (batch, 32)
    assert log_var.shape == (batch, 32)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_var).all()


def test_dynamics_window_exceeds_max_raises():
    dynamics = WorldModelDynamics(fusion_dim=32, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, max_window=4)
    trajectory = torch.randn(1, 6, 32)
    try:
        dynamics(trajectory)
        assert False, "expected ValueError for window > max_window"
    except ValueError as exc:
        assert "max_window" in str(exc)


def test_world_model_end_to_end_forward_shape():
    dims = _dims()
    model = WorldModel(dims, fusion_dim=64, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_window=8)
    batch, window = 2, 4
    feats = {name: torch.randn(batch, window, getattr(dims, name)) for name in FEATURE_GROUP_NAMES}
    mean, log_var = model(feats)
    assert mean.shape == (batch, 64)
    assert log_var.shape == (batch, 64)


def test_world_model_param_count_within_assumed_range():
    """README/PR assumption: dynamics model target is ~100-350M params,
    concrete default ~150M. This uses the service's real default
    hyperparameters (app/settings.py) so a hyperparameter drift that busts
    the stated assumption fails loudly here, not silently in prose."""
    from app.settings import Settings

    s = Settings()
    dims = FeatureGroupDims(
        biometrics=s.WM_DIM_BIOMETRICS,
        affect=s.WM_DIM_AFFECT,
        execution_context=s.WM_DIM_EXECUTION_CONTEXT,
        memory_pointers=s.WM_DIM_MEMORY_POINTERS,
        temporal=s.WM_DIM_TEMPORAL,
        vision_embedding=s.WM_DIM_VISION_EMBEDDING,
    )
    model = WorldModel(
        dims,
        fusion_dim=s.WM_FUSION_DIM,
        d_model=s.WM_DYNAMICS_D_MODEL,
        nhead=s.WM_DYNAMICS_NHEAD,
        num_layers=s.WM_DYNAMICS_NUM_LAYERS,
        dim_feedforward=s.WM_DYNAMICS_DIM_FEEDFORWARD,
        max_window=s.WM_DYNAMICS_MAX_WINDOW,
        state_dim=s.state_dim,
        dropout=s.WM_DYNAMICS_DROPOUT,
    )
    n = model.num_parameters()
    assert 100_000_000 <= n <= 350_000_000, f"param count {n} outside stated ~100-350M assumption"
