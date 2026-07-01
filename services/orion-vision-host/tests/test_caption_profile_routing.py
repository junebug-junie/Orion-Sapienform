from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.profiles import VisionProfiles


def _load_profiles() -> VisionProfiles:
    cfg_path = Path(__file__).resolve().parents[3] / "config" / "vision_profiles.yaml"
    p = VisionProfiles(str(cfg_path))
    p.load()
    return p


def test_caption_frame_routes_to_vlm_caption() -> None:
    p = _load_profiles()
    assert p.resolve_target("caption_frame") == "vlm_caption"


def test_vlm_caption_kind_is_caption_frame() -> None:
    p = _load_profiles()
    assert p.get_profile("vlm_caption").kind == "caption_frame"


def test_vlm_vqa_untouched() -> None:
    p = _load_profiles()
    vqa = p.get_profile("vlm_vqa")
    assert vqa.kind == "vlm"
    assert vqa.enabled is False
