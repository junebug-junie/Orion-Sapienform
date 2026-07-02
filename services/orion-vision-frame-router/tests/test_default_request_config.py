from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.policy import FrameDispatchPolicy, load_policy_file
from app.settings import Settings


def _cfg_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "vision_frame_router.yaml"


def test_defaults_baseline_disables_caption_and_embeddings() -> None:
    raw = load_policy_file(_cfg_path())
    assert raw["defaults"]["baseline"]["request"]["want_caption"] is False
    assert raw["defaults"]["baseline"]["request"]["want_embeddings"] is False


def test_defaults_triggered_enables_caption_and_embeddings() -> None:
    raw = load_policy_file(_cfg_path())
    assert raw["defaults"]["triggered"]["request"]["want_caption"] is True
    assert raw["defaults"]["triggered"]["request"]["want_embeddings"] is True


def test_camera_without_override_inherits_baseline_request() -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(_cfg_path()))
    policy = FrameDispatchPolicy.load(settings)

    merged, _name = policy.resolve_camera_policy("mock-cam-01")
    assert merged["baseline"]["request"]["want_caption"] is False
    assert merged["baseline"]["request"]["want_embeddings"] is False

    porch_request = policy.resolve_camera_policy("porch_eye")[0]["request"]
    assert "want_caption" not in porch_request
