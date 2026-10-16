"""_run_caption_frame with the walkway request (2026-10-15): the caller's
prompt reaches the model, the agreed null answer becomes no caption, and a
model that cannot take the prompt returns no caption instead of its own."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import runner as runner_mod
from app.profiles import VisionProfiles
from app.runner import VisionRunner

PROMPT = ("Are there any people, animals or vehicles in this image? If there are none, reply with only: "
          "nothing notable. If there are, describe them and what they are doing in one short sentence.")
REQ = {"image_path": "/frames/x.jpg", "caption_prompt": PROMPT, "caption_null_answers": ["nothing notable"]}
PROFILE = SimpleNamespace(name="vlm_caption", model_id="", dtype="", params={})


def _runner(monkeypatch, model_id: str, reply: str):
    cfg_path = Path(__file__).resolve().parents[3] / "config" / "vision_profiles.yaml"
    profiles = VisionProfiles(str(cfg_path))
    profiles.load()
    r = VisionRunner(profiles=profiles, enabled_names=["vlm_caption"], cache_dir=tempfile.mkdtemp())
    monkeypatch.setattr(runner_mod.settings, "VISION_VLM_MODEL_ID", model_id)
    monkeypatch.setattr(runner_mod, "_load_image_from_request", lambda req: Image.new("RGB", (4, 4)))
    r.models = MagicMock()
    r.models.load_vlm_captioner.return_value = (MagicMock(), MagicMock())
    seen = {}

    def fake_generate(model, processor, img, text_prompt, model_id, device, max_tokens, temperature):
        seen["prompt"] = text_prompt
        return reply

    monkeypatch.setattr(r, "_generate_vlm_text", fake_generate)
    return r, seen


def test_qwen_gets_the_walkway_prompt_and_real_caption_stands(monkeypatch):
    r, seen = _runner(monkeypatch, "Qwen/Qwen2-VL-2B-Instruct", "A man walks a dog along the sidewalk.")
    warnings: list = []
    out = r._run_caption_frame(PROFILE, dict(REQ), "cpu", warnings)
    assert seen["prompt"] == PROMPT
    assert out["caption"] == {"text": "A man walks a dog along the sidewalk.", "confidence": 1.0}
    assert warnings == []


@pytest.mark.parametrize("reply", ["nothing notable", "Nothing notable.", " NOTHING NOTABLE "])
def test_null_answer_is_no_caption_never_confidence_one(monkeypatch, reply):
    r, _ = _runner(monkeypatch, "Qwen/Qwen2-VL-2B-Instruct", reply)
    warnings: list = []
    out = r._run_caption_frame(PROFILE, dict(REQ), "cpu", warnings)
    assert out["caption"] == {"text": "", "confidence": 0.0, "rejected_reason": "null_answer"}
    assert warnings == ["caption_rejected:null_answer"]


def test_blip_on_the_lane_returns_no_caption_not_its_own(monkeypatch):
    r, seen = _runner(monkeypatch, "Salesforce/blip-image-captioning-base", "a camera is sitting on a bench")
    warnings: list = []
    out = r._run_caption_frame(PROFILE, dict(REQ), "cpu", warnings)
    assert out["caption"] == {"text": "", "confidence": 0.0, "rejected_reason": "caller_prompt_unsupported"}
    assert "prompt" not in seen  # never generated
    r.models.load_vlm_captioner.assert_not_called()


def test_cam0_blip_without_caller_prompt_unchanged(monkeypatch):
    r, seen = _runner(monkeypatch, "Salesforce/blip-image-captioning-base", "a woman sitting at a desk in an office")
    out = r._run_caption_frame(PROFILE, {"image_path": "/frames/x.jpg"}, "cpu", [])
    assert seen["prompt"] is None
    assert out["caption"]["text"] == "a woman sitting at a desk in an office"


def test_caption_crop_boxes_reach_the_model_as_a_crop(monkeypatch):
    """2026-10-16: the model must see the detector's region, not a ~1 MP
    downscale of the whole 2560x1920 frame (24/24 "nothing notable")."""
    r, seen = _runner(monkeypatch, "Qwen/Qwen2-VL-2B-Instruct", "A man walks a dog along the sidewalk.")
    monkeypatch.setattr(runner_mod, "_load_image_from_request", lambda req: Image.new("RGB", (2560, 1920)))
    sizes = {}

    def fake_generate(model, processor, img, text_prompt, model_id, device, max_tokens, temperature):
        sizes["img"] = img.size
        return "A man walks a dog along the sidewalk."

    monkeypatch.setattr(r, "_generate_vlm_text", fake_generate)
    req = dict(REQ, caption_crop_boxes=[[1280, 454, 1408, 518]], caption_crop_pad=0.5)
    out = r._run_caption_frame(PROFILE, req, "cpu", [])
    assert sizes["img"] != (2560, 1920)
    assert sizes["img"][0] < 1000 and sizes["img"][1] < 1000
    assert out["crop_box"] is not None
    assert out["caption"]["text"] == "A man walks a dog along the sidewalk."


def test_no_crop_boxes_captions_the_whole_frame(monkeypatch):
    r, _ = _runner(monkeypatch, "Qwen/Qwen2-VL-2B-Instruct", "A man walks a dog along the sidewalk.")
    out = r._run_caption_frame(PROFILE, dict(REQ), "cpu", [])
    assert out["crop_box"] is None
