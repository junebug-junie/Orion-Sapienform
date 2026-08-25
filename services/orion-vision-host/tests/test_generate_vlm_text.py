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


def _runner() -> VisionRunner:
    cfg_path = Path(__file__).resolve().parents[3] / "config" / "vision_profiles.yaml"
    profiles = VisionProfiles(str(cfg_path))
    profiles.load()
    tmp = tempfile.mkdtemp()
    return VisionRunner(profiles=profiles, enabled_names=["vlm_caption", "vlm_vqa"], cache_dir=tmp)


def _fake_model() -> MagicMock:
    model = MagicMock()
    # next(model.parameters()).dtype is only read on the device.startswith("cuda")
    # branch -- these tests run on "cpu" so it's never touched, but a real
    # single-element iterator is cheap to provide anyway.
    model.parameters.return_value = iter([torch.zeros(1, dtype=torch.float32)])
    return model


def _img() -> Image.Image:
    return Image.new("RGB", (4, 4), color=(10, 20, 30))


def test_generate_vlm_text_blip_path_calls_processor_directly() -> None:
    """BLIP/BLIP2 path: processor(images=, text=) directly, full-sequence
    decode -- this must stay exactly the shape the live BLIP deployment on
    athena already depends on. No apply_chat_template, no token slicing."""
    runner = _runner()
    model = _fake_model()
    processor = MagicMock()
    processor.side_effect = lambda **kw: {"input_ids": torch.zeros((1, 3), dtype=torch.long)}
    model.generate.return_value = torch.zeros((1, 5), dtype=torch.long)
    processor.batch_decode.return_value = ["a real caption of the room"]

    text = runner._generate_vlm_text(
        model, processor, _img(), "Describe this image.",
        "Salesforce/blip-image-captioning-base", "cpu", 32, 0.2,
    )

    assert text == "a real caption of the room"
    _, kwargs = processor.call_args
    assert kwargs.get("text") == "Describe this image."
    processor.apply_chat_template.assert_not_called()
    # Full 5-token sequence handed to batch_decode, not sliced.
    decoded_arg = processor.batch_decode.call_args[0][0]
    assert decoded_arg.shape == (1, 5)


def test_generate_vlm_text_qwen2_vl_path_uses_chat_template() -> None:
    """Qwen2-VL path: builds a chat message via apply_chat_template with
    the real prompt text, then calls processor(text=, images=) with the
    templated string -- not the raw prompt BLIP would get."""
    runner = _runner()
    model = _fake_model()
    processor = MagicMock()
    processor.apply_chat_template.return_value = "<templated prompt>"
    processor.side_effect = lambda **kw: {"input_ids": torch.zeros((1, 7), dtype=torch.long)}
    model.generate.return_value = torch.arange(11, dtype=torch.long).unsqueeze(0)
    processor.batch_decode.return_value = ["the real answer"]

    text = runner._generate_vlm_text(
        model, processor, _img(), "what is in this image?",
        "Qwen/Qwen2-VL-2B-Instruct", "cpu", 32, 0.2,
    )

    assert text == "the real answer"
    processor.apply_chat_template.assert_called_once()
    messages = processor.apply_chat_template.call_args[0][0]
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0] == {"type": "image"}
    assert messages[0]["content"][1] == {"type": "text", "text": "what is in this image?"}

    _, kwargs = processor.call_args
    assert kwargs.get("text") == ["<templated prompt>"]


def test_generate_vlm_text_qwen2_vl_path_trims_echoed_prompt_by_input_length() -> None:
    """The bug this exists to avoid: decoding the FULL generated sequence
    (as the BLIP path correctly does) would hand the caller the entire
    chat-templated prompt echoed back plus the real answer glued on. Must
    slice by the real input token length instead of string-matching a
    prefix off the decoded text."""
    runner = _runner()
    model = _fake_model()
    processor = MagicMock()
    processor.apply_chat_template.return_value = "<templated prompt>"
    input_len = 7
    processor.side_effect = lambda **kw: {"input_ids": torch.zeros((1, input_len), dtype=torch.long)}
    # 7 echoed prompt tokens + 4 new tokens.
    model.generate.return_value = torch.arange(11, dtype=torch.long).unsqueeze(0)

    runner._generate_vlm_text(
        model, processor, _img(), "is the door open?",
        "Qwen/Qwen2.5-VL-3B-Instruct", "cpu", 32, 0.2,
    )

    decoded_arg = processor.batch_decode.call_args[0][0]
    assert len(decoded_arg) == 1
    assert decoded_arg[0].shape[0] == 11 - input_len


def test_generate_vlm_text_non_qwen_model_id_never_calls_apply_chat_template() -> None:
    """Any non-Qwen model_id (including ones this repo doesn't ship today)
    must fall through to the plain BLIP-style call -- apply_chat_template
    is gated strictly by vlm_family.is_chat_template_vlm, not by "anything
    unrecognized defaults to chat mode"."""
    runner = _runner()
    model = _fake_model()
    processor = MagicMock()
    processor.side_effect = lambda **kw: {"input_ids": torch.zeros((1, 3), dtype=torch.long)}
    model.generate.return_value = torch.zeros((1, 4), dtype=torch.long)
    processor.batch_decode.return_value = ["ok"]

    runner._generate_vlm_text(
        model, processor, _img(), "hi", "some-org/some-random-vlm", "cpu", 16, 0.0,
    )

    processor.apply_chat_template.assert_not_called()
