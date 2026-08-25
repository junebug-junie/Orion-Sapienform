from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_caption_frame_routes_to_vlm_caption(vision_profiles) -> None:
    assert vision_profiles.resolve_target("caption_frame") == "vlm_caption"


def test_vlm_caption_kind_is_caption_frame(vision_profiles) -> None:
    assert vision_profiles.get_profile("vlm_caption").kind == "caption_frame"


def test_vqa_routes_to_vlm_vqa(vision_profiles) -> None:
    assert vision_profiles.resolve_target("vqa") == "vlm_vqa"


def test_vlm_vqa_enabled_2026_08_20(vision_profiles) -> None:
    """Was disabled + kind unimplemented (see test_run_vlm_vqa.py for the
    execution-side coverage) until this same-day patch. Live VRAM headroom
    was checked (not assumed) before flipping this -- see the profile's own
    comment in config/vision_profiles.yaml."""
    vqa = vision_profiles.get_profile("vlm_vqa")
    assert vqa.kind == "vlm"
    assert vqa.enabled is True
    assert vqa.warm_on_start is False  # lazy-load only, deliberately not eager
