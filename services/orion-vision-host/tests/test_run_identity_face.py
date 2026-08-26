from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.profiles import VisionProfiles
from app.runner import VisionRunner
import app.runner as runner_module


def _runner() -> VisionRunner:
    cfg_path = Path(__file__).resolve().parents[3] / "config" / "vision_profiles.yaml"
    profiles = VisionProfiles(str(cfg_path))
    profiles.load()
    tmp = tempfile.mkdtemp()
    return VisionRunner(profiles=profiles, enabled_names=["identity_face"], cache_dir=tmp)


def _image_path(tmp_path) -> str:
    p = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8)).save(p)
    return str(p)


def test_run_identity_face_no_face_detected_returns_empty_candidates(monkeypatch, tmp_path):
    runner = _runner()
    profile = runner.profiles.get_profile("identity_face")

    fake_model = MagicMock()
    fake_mtcnn = MagicMock(return_value=(None, [None]))
    monkeypatch.setattr(runner.models, "load_face_identity_models", lambda **kw: (fake_model, fake_mtcnn))
    monkeypatch.setattr(runner_module, "settings", MagicMock(
        MODEL_CACHE_DIR="/tmp", IDENTITY_ENROLLED_SUBJECT="juniper", IDENTITY_GALLERY_DIR=str(tmp_path)
    ))

    warnings = []
    result = runner._run_identity_face(profile, {"image_path": _image_path(tmp_path)}, "cpu", warnings)

    assert result["identities"]["candidates"] == []
    assert "no_face_detected" in warnings


def test_run_identity_face_gallery_not_enrolled_flags_warning_and_returns_unknown(monkeypatch, tmp_path):
    runner = _runner()
    profile = runner.profiles.get_profile("identity_face")

    fake_face = torch.zeros((1, 3, 4, 4))
    fake_model = MagicMock(return_value=torch.zeros((1, 512)))
    fake_mtcnn = MagicMock(return_value=(fake_face, [0.99]))
    monkeypatch.setattr(runner.models, "load_face_identity_models", lambda **kw: (fake_model, fake_mtcnn))
    monkeypatch.setattr(runner_module, "settings", MagicMock(
        MODEL_CACHE_DIR="/tmp", IDENTITY_ENROLLED_SUBJECT="juniper", IDENTITY_GALLERY_DIR=str(tmp_path)
    ))

    warnings = []
    result = runner._run_identity_face(profile, {"image_path": _image_path(tmp_path)}, "cpu", warnings)

    assert "identity_gallery_not_enrolled" in warnings
    assert result["identities"]["gallery_enrolled"] is False
    assert len(result["identities"]["candidates"]) == 1
    assert result["identities"]["candidates"][0]["subject"] == "unknown"
    assert result["identities"]["candidates"][0]["state"] == "unsure"
    assert result["identities"]["candidates"][0]["reason"] == "not_enrolled"


def test_run_identity_face_matches_enrolled_subject(monkeypatch, tmp_path):
    """A real gallery embedding, a query embedding identical to it -- must
    come back as a real "probable" match on the enrolled subject's name."""
    runner = _runner()
    profile = runner.profiles.get_profile("identity_face")

    import numpy as np
    from app.identity_gallery import save_gallery_embedding

    gallery_vec = np.array([1.0] + [0.0] * 511, dtype=np.float32)
    save_gallery_embedding(str(tmp_path), "juniper", gallery_vec, sample_count=1)

    query_embedding = torch.tensor(gallery_vec).unsqueeze(0)  # identical -> similarity 1.0
    fake_face = torch.zeros((1, 3, 4, 4))
    fake_model = MagicMock(return_value=query_embedding)
    fake_mtcnn = MagicMock(return_value=(fake_face, [0.99]))
    monkeypatch.setattr(runner.models, "load_face_identity_models", lambda **kw: (fake_model, fake_mtcnn))
    monkeypatch.setattr(runner_module, "settings", MagicMock(
        MODEL_CACHE_DIR="/tmp", IDENTITY_ENROLLED_SUBJECT="juniper", IDENTITY_GALLERY_DIR=str(tmp_path)
    ))

    result = runner._run_identity_face(profile, {"image_path": _image_path(tmp_path)}, "cpu", [])

    assert result["identities"]["gallery_enrolled"] is True
    candidate = result["identities"]["candidates"][0]
    assert candidate["subject"] == "juniper"
    assert candidate["state"] == "probable"
    assert candidate["similarity"] == 1.0
    assert candidate["detect_confidence"] == 0.99


def test_run_identity_face_caps_at_max_candidates(monkeypatch, tmp_path):
    runner = _runner()
    profile = runner.profiles.get_profile("identity_face")
    profile.params = dict(profile.params, max_candidates=2)

    fake_faces = torch.zeros((5, 3, 4, 4))
    fake_model = MagicMock(side_effect=lambda faces: torch.zeros((faces.shape[0], 512)))
    fake_mtcnn = MagicMock(return_value=(fake_faces, [0.9, 0.8, 0.7, 0.6, 0.5]))
    monkeypatch.setattr(runner.models, "load_face_identity_models", lambda **kw: (fake_model, fake_mtcnn))
    monkeypatch.setattr(runner_module, "settings", MagicMock(
        MODEL_CACHE_DIR="/tmp", IDENTITY_ENROLLED_SUBJECT="juniper", IDENTITY_GALLERY_DIR=str(tmp_path)
    ))

    result = runner._run_identity_face(profile, {"image_path": _image_path(tmp_path)}, "cpu", [])

    assert len(result["identities"]["candidates"]) == 2


def test_run_identity_face_keeps_highest_confidence_faces_not_first_n(monkeypatch, tmp_path):
    """Review finding, 2026-08-26: MTCNN's own detection order is not
    confidence-ordered. A plain faces[:max_faces] would keep whichever
    faces happen to be first, silently dropping the enrolled subject's own
    (possibly lower-listed) face in a crowded frame. This uses
    out-of-order probs -- the highest confidence face is NOT first -- and
    checks the surviving candidates by their real detect_confidence, not
    just by count."""
    runner = _runner()
    profile = runner.profiles.get_profile("identity_face")
    profile.params = dict(profile.params, max_candidates=2)

    fake_faces = torch.zeros((5, 3, 4, 4))
    fake_model = MagicMock(side_effect=lambda faces: torch.zeros((faces.shape[0], 512)))
    # Deliberately unsorted -- the two highest (0.95, 0.9) are NOT first.
    fake_mtcnn = MagicMock(return_value=(fake_faces, [0.5, 0.95, 0.6, 0.9, 0.4]))
    monkeypatch.setattr(runner.models, "load_face_identity_models", lambda **kw: (fake_model, fake_mtcnn))
    monkeypatch.setattr(runner_module, "settings", MagicMock(
        MODEL_CACHE_DIR="/tmp", IDENTITY_ENROLLED_SUBJECT="juniper", IDENTITY_GALLERY_DIR=str(tmp_path)
    ))

    result = runner._run_identity_face(profile, {"image_path": _image_path(tmp_path)}, "cpu", [])

    candidates = result["identities"]["candidates"]
    assert len(candidates) == 2
    kept_confidences = sorted(c["detect_confidence"] for c in candidates)
    assert kept_confidences == [0.9, 0.95]


def test_run_identity_face_never_returns_raw_embedding(monkeypatch, tmp_path):
    """Non-negotiable, checked at this method's actual output boundary, not
    just inside identity_gallery.match_embedding's own unit tests."""
    runner = _runner()
    profile = runner.profiles.get_profile("identity_face")

    fake_face = torch.zeros((1, 3, 4, 4))
    fake_model = MagicMock(return_value=torch.rand((1, 512)))
    fake_mtcnn = MagicMock(return_value=(fake_face, [0.99]))
    monkeypatch.setattr(runner.models, "load_face_identity_models", lambda **kw: (fake_model, fake_mtcnn))
    monkeypatch.setattr(runner_module, "settings", MagicMock(
        MODEL_CACHE_DIR="/tmp", IDENTITY_ENROLLED_SUBJECT="juniper", IDENTITY_GALLERY_DIR=str(tmp_path)
    ))

    result = runner._run_identity_face(profile, {"image_path": _image_path(tmp_path)}, "cpu", [])

    # Review finding, 2026-08-26: the original version of this assertion
    # (`"embedding" not in result_str.lower() or "embedding_ref" not in
    # result`) was vacuously always-true -- `result` never has a top-level
    # "embedding_ref" key under any code path, so the right side of the
    # `or` was unconditionally True regardless of the left side. This is
    # the real check.
    result_str = str(result)
    assert "embedding" not in result_str.lower()
    for candidate in result["identities"]["candidates"]:
        assert set(candidate.keys()) <= {"subject", "similarity", "state", "reason", "detect_confidence"}
