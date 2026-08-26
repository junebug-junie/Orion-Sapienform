from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model_manager import ModelManager


def _patched_facenet(**overrides):
    fake_mtcnn = MagicMock()
    fake_resnet_instance = MagicMock()
    fake_resnet_instance.eval.return_value = fake_resnet_instance
    defaults = dict(
        mtcnn=patch("facenet_pytorch.MTCNN", return_value=fake_mtcnn),
        resnet=patch("facenet_pytorch.InceptionResnetV1", return_value=fake_resnet_instance),
    )
    defaults.update(overrides)
    return defaults, fake_mtcnn, fake_resnet_instance


def test_load_face_identity_models_returns_resnet_and_mtcnn():
    mgr = ModelManager()
    patches, fake_mtcnn, fake_resnet = _patched_facenet()
    with patches["mtcnn"] as mtcnn_ctor, patches["resnet"] as resnet_ctor:
        model, processor = mgr.load_face_identity_models(profile_name="identity_face", device="cpu")

    resnet_ctor.assert_called_once_with(pretrained="vggface2")
    mtcnn_ctor.assert_called_once()
    assert model is fake_resnet
    assert processor is fake_mtcnn


def test_load_face_identity_models_caches_by_profile_and_device():
    """Second call for the same (profile, device) must not reconstruct the
    models -- matches every other loader's caching contract in this file."""
    mgr = ModelManager()
    patches, _, _ = _patched_facenet()
    with patches["mtcnn"] as mtcnn_ctor, patches["resnet"] as resnet_ctor:
        mgr.load_face_identity_models(profile_name="identity_face", device="cpu")
        mgr.load_face_identity_models(profile_name="identity_face", device="cpu")

    resnet_ctor.assert_called_once()
    mtcnn_ctor.assert_called_once()


def test_load_face_identity_models_sets_torch_home_before_import(monkeypatch):
    """Review-relevant, not review-caught: facenet-pytorch's weights
    download via torch.hub, which defaults to ~/.cache/torch/hub and
    ignores this service's own MODEL_CACHE_DIR/HF_HOME conventions --
    without this, the container silently re-downloads ~107MB on every
    restart instead of using the persistent volume mount every other
    loader in this file already gets."""
    import os

    monkeypatch.delenv("TORCH_HOME", raising=False)
    mgr = ModelManager()
    patches, _, _ = _patched_facenet()
    with patches["mtcnn"], patches["resnet"]:
        mgr.load_face_identity_models(
            profile_name="identity_face_torchhome", device="cpu", torch_home="/mnt/telemetry/models/vision"
        )

    assert os.environ.get("TORCH_HOME") == "/mnt/telemetry/models/vision"


def test_load_face_identity_models_without_torch_home_does_not_set_env(monkeypatch):
    import os

    monkeypatch.delenv("TORCH_HOME", raising=False)
    mgr = ModelManager()
    patches, _, _ = _patched_facenet()
    with patches["mtcnn"], patches["resnet"]:
        mgr.load_face_identity_models(profile_name="identity_face_no_torchhome", device="cpu")

    assert "TORCH_HOME" not in os.environ


def test_load_face_identity_models_casts_embedder_to_dtype_on_cuda():
    """Review finding, 2026-08-26: this loader previously always ran the
    embedder in fp32 on CUDA, unlike every other CUDA-loaded profile in
    this file, which get a real fp16 cast via _torch_dtype's own "auto"
    default. Confirms the resnet's own .to() actually receives the
    resolved dtype, not just that dtype is accepted as a parameter."""
    import torch

    mgr = ModelManager()
    patches, _, fake_resnet = _patched_facenet()
    with patches["mtcnn"], patches["resnet"]:
        mgr.load_face_identity_models(profile_name="identity_face_cuda", device="cuda:0", dtype="auto")

    fake_resnet.to.assert_called_once_with(device="cuda:0", dtype=torch.float16)


def test_load_face_identity_models_cpu_never_calls_to():
    """CPU path must not call .to() at all -- matches every other loader's
    own `if device.startswith("cuda")` guard in this file."""
    mgr = ModelManager()
    patches, _, fake_resnet = _patched_facenet()
    with patches["mtcnn"], patches["resnet"]:
        mgr.load_face_identity_models(profile_name="identity_face_cpu_dtype", device="cpu", dtype="auto")

    fake_resnet.to.assert_not_called()
