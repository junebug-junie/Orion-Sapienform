from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.profiles import VisionProfiles
from app.runner import VisionRunner


def _runner() -> VisionRunner:
    cfg_path = Path(__file__).resolve().parents[3] / "config" / "vision_profiles.yaml"
    profiles = VisionProfiles(str(cfg_path))
    profiles.load()
    tmp = tempfile.mkdtemp()
    return VisionRunner(profiles=profiles, enabled_names=["vlm_caption"], cache_dir=tmp)


def test_cast_is_a_noop_off_cuda():
    """Review finding, 2026-08-25: this helper now backs 3 call sites
    (embedding, detection, VLM) that previously each duplicated the same
    5-line block -- pins the shared behavior directly so a future edit to
    one caller's needs can't silently diverge from the others."""
    runner = _runner()
    model = MagicMock()
    inputs = {"pixel_values": torch.zeros(1, dtype=torch.float32)}

    out = runner._cast_inputs_to_model_dtype(inputs, model, "cpu")

    assert out is inputs
    model.parameters.assert_not_called()


def test_cast_moves_floating_tensors_to_model_dtype_on_cuda_string_device():
    """Doesn't require a real GPU -- ``.to(device="cuda:0", ...)`` on a CPU
    tensor with no CUDA available would itself raise, so this stubs
    ``.to`` on the tensor via a MagicMock-wrapped mock tensor instead of a
    real one, matching how the existing embedding/detection tests already
    have to work around the same constraint (no real cuda device in this
    environment's test run)."""
    runner = _runner()
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1, dtype=torch.float16)])

    floating = MagicMock()
    floating.dtype = torch.float32
    non_floating = MagicMock()
    non_floating.dtype = torch.int64
    inputs = {"pixel_values": floating, "input_ids": non_floating}

    import app.runner as runner_module
    real_is_floating_point = runner_module.torch.is_floating_point
    runner_module.torch.is_floating_point = lambda t: t is floating
    try:
        out = runner._cast_inputs_to_model_dtype(inputs, model, "cuda:0")
    finally:
        runner_module.torch.is_floating_point = real_is_floating_point

    floating.to.assert_called_once_with(device="cuda:0", dtype=torch.float16)
    non_floating.to.assert_called_once_with(device="cuda:0", dtype=torch.int64)
    assert set(out.keys()) == {"pixel_values", "input_ids"}
