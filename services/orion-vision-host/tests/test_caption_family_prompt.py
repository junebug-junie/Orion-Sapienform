from __future__ import annotations

"""Blank-caption regression (docs/superpowers/specs/2026-10-08-vision-blank-
captions-and-on-demand-look-design.md). BLIP continues the text it is given
instead of following it, so handing it CAPTION_PROMPT made it echo the
prompt, the sanitizer rejected the echo, and every cam0 caption published as
{"text": "", "confidence": 1.0}."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.runner as runner_mod
from app.artifacts import build_artifact_payload
from app.caption_sanitize import CAPTION_PROMPT, sanitize_caption
from app.models import VisionResult
from app.vlm_family import caption_prompt_for

BLIP = "Salesforce/blip-image-captioning-base"
QWEN = "Qwen/Qwen2-VL-2B-Instruct"
SCENE = "a woman sitting at a desk in an office"


class _EchoingBlipProcessor:
    """Behaves like BLIP-base did live on cam0: given text, the decoded output
    is that text echoed back; given no text, a real scene caption."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._last_text: str | None = None

    def __call__(self, **kw):
        self.calls.append(kw)
        self._last_text = kw.get("text")
        return {"input_ids": torch.zeros((1, 3), dtype=torch.long)}

    def batch_decode(self, ids, skip_special_tokens=True):
        return [self._last_text.lower() if self._last_text else SCENE]


def _fake_model() -> MagicMock:
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1, dtype=torch.float32)])
    model.generate.return_value = torch.zeros((1, 5), dtype=torch.long)
    return model


def _run_caption(vlm_runner, monkeypatch, model_id: str, processor) -> tuple[dict, list[str]]:
    monkeypatch.setattr(runner_mod.settings, "VISION_VLM_MODEL_ID", model_id)
    monkeypatch.setattr(runner_mod, "_load_image_from_request", lambda req: Image.new("RGB", (4, 4)))
    monkeypatch.setattr(vlm_runner.models, "load_vlm_captioner", lambda **kw: (_fake_model(), processor))
    profile = vlm_runner.profiles.get_profile("vlm_caption")
    warnings: list[str] = []
    out = vlm_runner._run_caption_frame(profile, {"image_path": "x.jpg"}, "cpu", warnings)
    return out, warnings


def test_caption_prompt_for_family() -> None:
    assert caption_prompt_for(BLIP) is None
    assert caption_prompt_for("Salesforce/blip2-opt-2.7b") is None
    assert caption_prompt_for(QWEN) == CAPTION_PROMPT
    assert caption_prompt_for("Qwen/Qwen2.5-VL-3B-Instruct") == CAPTION_PROMPT


def test_echoing_blip_still_produces_a_caption(vlm_runner, monkeypatch) -> None:
    processor = _EchoingBlipProcessor()
    out, warnings = _run_caption(vlm_runner, monkeypatch, BLIP, processor)

    assert out["caption"] == {"text": SCENE, "confidence": 1.0}
    assert warnings == []
    # No text reached BLIP at all: unconditional captioning.
    assert "text" not in processor.calls[0]


def test_echo_filter_was_not_loosened() -> None:
    # The fix is "don't send BLIP the prompt", not "accept the echo".
    assert sanitize_caption(CAPTION_PROMPT)[1] is False
    assert sanitize_caption(CAPTION_PROMPT.lower())[2] == "prompt_echo"


def test_rejected_caption_is_published_as_rejected(vlm_runner, monkeypatch) -> None:
    processor = MagicMock()
    processor.side_effect = lambda **kw: {"input_ids": torch.zeros((1, 3), dtype=torch.long)}
    processor.batch_decode.return_value = ["a room"]  # under the 12-char floor

    out, warnings = _run_caption(vlm_runner, monkeypatch, BLIP, processor)

    assert out["caption"] == {"text": "", "confidence": 0.0, "rejected_reason": "too_short"}
    assert warnings == ["caption_rejected:too_short"]

    payload = build_artifact_payload(
        VisionResult(corr_id="c1", ok=True, task_type="retina_fast", device="cpu", artifacts=out)
    )
    wire = payload.model_dump()["outputs"]["caption"]
    assert wire == {"text": "", "confidence": 0.0, "rejected_reason": "too_short"}


def test_chat_template_vlm_still_gets_the_instruction(vlm_runner, monkeypatch) -> None:
    processor = MagicMock()
    processor.apply_chat_template.return_value = "<templated>"
    processor.side_effect = lambda **kw: {"input_ids": torch.zeros((1, 3), dtype=torch.long)}
    processor.batch_decode.return_value = ["one person seated at a desk, one chair"]

    out, _ = _run_caption(vlm_runner, monkeypatch, QWEN, processor)

    messages = processor.apply_chat_template.call_args[0][0]
    assert messages[0]["content"][1] == {"type": "text", "text": CAPTION_PROMPT}
    assert out["caption"]["text"] == "one person seated at a desk, one chair"
