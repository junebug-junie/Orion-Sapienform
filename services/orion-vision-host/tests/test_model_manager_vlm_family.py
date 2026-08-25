from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model_manager import ModelManager


def _patched_loaders(**overrides):
    """Every transformers loader load_vlm_captioner might reach for, all
    stubbed by default -- a test only needs to override the one branch it's
    exercising and assert the others were never touched, which is the real
    regression this file guards against (a new elif accidentally shadowing
    an existing branch, or falling through to the wrong one)."""
    defaults = dict(
        auto_processor=patch("transformers.AutoProcessor.from_pretrained", return_value=MagicMock()),
        blip2=patch("transformers.Blip2ForConditionalGeneration.from_pretrained", return_value=MagicMock()),
        blip=patch("transformers.BlipForConditionalGeneration.from_pretrained", return_value=MagicMock()),
        qwen2_vl=patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained", return_value=MagicMock()),
        qwen2_5_vl=patch("transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained", return_value=MagicMock()),
        generic=patch("transformers.AutoModelForVision2Seq.from_pretrained", return_value=MagicMock()),
    )
    defaults.update(overrides)
    return defaults


def test_load_vlm_captioner_selects_qwen2_vl_class_for_qwen2_vl_model_id():
    mgr = ModelManager()
    patches = _patched_loaders()
    with patches["auto_processor"], patches["blip2"], patches["blip"], \
         patches["qwen2_vl"] as qwen_load, patches["qwen2_5_vl"] as qwen25_load, \
         patches["generic"] as generic_load:
        model, processor = mgr.load_vlm_captioner(
            profile_name="vlm_caption", device="cpu", dtype="fp16",
            model_id="Qwen/Qwen2-VL-2B-Instruct",
        )

    qwen_load.assert_called_once()
    assert qwen_load.call_args.args[0] == "Qwen/Qwen2-VL-2B-Instruct"
    qwen25_load.assert_not_called()
    generic_load.assert_not_called()
    assert model is qwen_load.return_value


def test_load_vlm_captioner_selects_qwen2_5_vl_class_for_qwen2_5_vl_model_id():
    mgr = ModelManager()
    patches = _patched_loaders()
    with patches["auto_processor"], patches["blip2"], patches["blip"], \
         patches["qwen2_vl"] as qwen_load, patches["qwen2_5_vl"] as qwen25_load, \
         patches["generic"] as generic_load:
        model, processor = mgr.load_vlm_captioner(
            profile_name="vlm_caption", device="cpu", dtype="fp16",
            model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        )

    qwen25_load.assert_called_once()
    qwen_load.assert_not_called()
    generic_load.assert_not_called()
    assert model is qwen25_load.return_value


def test_load_vlm_captioner_still_selects_blip2_for_blip2_model_id():
    """Regression: the two new Qwen elif branches must not shadow the
    pre-existing blip2 branch -- "blip2" contains neither "qwen2-vl" nor
    "qwen2.5-vl" so this shouldn't be possible, but the whole point of this
    module is that string-based family routing is exactly the kind of
    thing that quietly breaks on refactor."""
    mgr = ModelManager()
    patches = _patched_loaders()
    with patches["auto_processor"], patches["blip2"] as blip2_load, patches["blip"], \
         patches["qwen2_vl"] as qwen_load, patches["qwen2_5_vl"] as qwen25_load, \
         patches["generic"] as generic_load:
        model, processor = mgr.load_vlm_captioner(
            profile_name="vlm_caption", device="cpu", dtype="fp16",
            model_id="Salesforce/blip2-opt-2.7b",
        )

    blip2_load.assert_called_once()
    qwen_load.assert_not_called()
    qwen25_load.assert_not_called()
    generic_load.assert_not_called()
    assert model is blip2_load.return_value


def test_load_vlm_captioner_passes_qwen_pixel_bounds_to_processor():
    """Review finding, 2026-08-25: Qwen2-VL/2.5-VL's "naive dynamic
    resolution" processor scales visual-token count (and VRAM) with input
    image resolution, uncapped, unless min_pixels/max_pixels are passed.
    Confirms load_vlm_captioner's caller-supplied bounds actually reach
    AutoProcessor.from_pretrained for both Qwen generations -- not just
    that the kwargs exist on the function signature."""
    mgr = ModelManager()
    patches = _patched_loaders()
    with patches["auto_processor"] as processor_load, patches["blip2"], patches["blip"], \
         patches["qwen2_vl"], patches["qwen2_5_vl"], patches["generic"]:
        mgr.load_vlm_captioner(
            profile_name="vlm_caption", device="cpu", dtype="fp16",
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            qwen_min_pixels=200704, qwen_max_pixels=1003520,
        )

    processor_load.assert_called_once()
    _, kwargs = processor_load.call_args
    assert kwargs.get("min_pixels") == 200704
    assert kwargs.get("max_pixels") == 1003520


def test_load_vlm_captioner_qwen_pixel_bounds_default_to_none():
    """Caller omitting the new kwargs (e.g. an older/simpler call site, or
    a test that doesn't care) must not accidentally pass a real numeric
    default into the processor -- None means "use the checkpoint's own
    default", not "0 pixels"."""
    mgr = ModelManager()
    patches = _patched_loaders()
    with patches["auto_processor"] as processor_load, patches["blip2"], patches["blip"], \
         patches["qwen2_vl"], patches["qwen2_5_vl"], patches["generic"]:
        mgr.load_vlm_captioner(
            profile_name="vlm_caption", device="cpu", dtype="fp16",
            model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        )

    _, kwargs = processor_load.call_args
    assert kwargs.get("min_pixels") is None
    assert kwargs.get("max_pixels") is None


def test_load_vlm_captioner_unrecognized_model_id_still_falls_back_to_generic():
    mgr = ModelManager()
    patches = _patched_loaders()
    with patches["auto_processor"], patches["blip2"], patches["blip"], \
         patches["qwen2_vl"] as qwen_load, patches["qwen2_5_vl"] as qwen25_load, \
         patches["generic"] as generic_load:
        model, processor = mgr.load_vlm_captioner(
            profile_name="vlm_caption", device="cpu", dtype="fp16",
            model_id="some-org/some-future-vlm",
        )

    generic_load.assert_called_once()
    qwen_load.assert_not_called()
    qwen25_load.assert_not_called()
    assert model is generic_load.return_value
