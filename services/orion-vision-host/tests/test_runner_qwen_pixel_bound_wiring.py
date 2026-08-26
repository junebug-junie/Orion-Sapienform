from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.profiles import VisionProfiles
from app.runner import VisionRunner
import app.runner as runner_module


def _runner() -> VisionRunner:
    cfg_path = Path(__file__).resolve().parents[3] / "config" / "vision_profiles.yaml"
    profiles = VisionProfiles(str(cfg_path))
    profiles.load()
    tmp = tempfile.mkdtemp()
    return VisionRunner(profiles=profiles, enabled_names=["vlm_caption", "vlm_vqa"], cache_dir=tmp)


def test_run_caption_frame_passes_settings_qwen_pixel_bounds(monkeypatch, tmp_path):
    """Review finding, 2026-08-25: qwen_min_pixels/max_pixels must actually
    reach load_vlm_captioner from the real request path (not just be
    threaded through the function signature) -- this is the runtime call
    site, distinct from test_model_manager_vlm_family.py's direct-call
    coverage of load_vlm_captioner itself."""
    runner = _runner()
    profile = runner.profiles.get_profile("vlm_caption")

    monkeypatch.setattr(runner_module.settings, "VISION_VLM_QWEN_MIN_PIXELS", 111)
    monkeypatch.setattr(runner_module.settings, "VISION_VLM_QWEN_MAX_PIXELS", 222)

    fake_model = MagicMock()
    fake_model.parameters.return_value = iter([])
    fake_processor = MagicMock()
    fake_processor.side_effect = lambda **kw: {"input_ids": MagicMock()}
    fake_model.generate.return_value = MagicMock()
    fake_processor.batch_decode.return_value = ["a caption long enough to pass sanitize_caption checks"]

    load_calls = {}

    def fake_load_vlm_captioner(**kwargs):
        load_calls.update(kwargs)
        return fake_model, fake_processor

    monkeypatch.setattr(runner.models, "load_vlm_captioner", fake_load_vlm_captioner)
    monkeypatch.setattr(runner, "_generate_vlm_text", lambda *a, **kw: "a caption long enough to pass checks")

    jpg = tmp_path / "frame.jpg"
    jpg.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 32)
    from PIL import Image
    Image.new("RGB", (4, 4)).save(jpg)

    runner._run_caption_frame(profile, {"image_path": str(jpg)}, "cpu", [])

    assert load_calls.get("qwen_min_pixels") == 111
    assert load_calls.get("qwen_max_pixels") == 222


def test_run_vlm_vqa_passes_settings_qwen_pixel_bounds(monkeypatch, tmp_path):
    runner = _runner()
    profile = runner.profiles.get_profile("vlm_vqa")

    monkeypatch.setattr(runner_module.settings, "VISION_VLM_QWEN_MIN_PIXELS", 333)
    monkeypatch.setattr(runner_module.settings, "VISION_VLM_QWEN_MAX_PIXELS", 444)

    fake_model = MagicMock()
    fake_processor = MagicMock()
    load_calls = {}

    def fake_load_vlm_captioner(**kwargs):
        load_calls.update(kwargs)
        return fake_model, fake_processor

    monkeypatch.setattr(runner.models, "load_vlm_captioner", fake_load_vlm_captioner)
    monkeypatch.setattr(runner, "_generate_vlm_text", lambda *a, **kw: "a real answer")

    jpg = tmp_path / "frame.jpg"
    from PIL import Image
    Image.new("RGB", (4, 4)).save(jpg)

    runner._run_vlm_vqa(profile, {"image_path": str(jpg), "question": "what is this?"}, "cpu", [])

    assert load_calls.get("qwen_min_pixels") == 333
    assert load_calls.get("qwen_max_pixels") == 444
