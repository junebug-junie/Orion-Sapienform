from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.vlm_family import (
    is_chat_template_vlm,
    is_qwen2_5_vl_model,
    is_qwen2_vl_model,
)


def test_qwen2_vl_model_ids_detected():
    assert is_qwen2_vl_model("Qwen/Qwen2-VL-2B-Instruct") is True
    assert is_qwen2_vl_model("Qwen/Qwen2-VL-7B-Instruct") is True


def test_qwen2_5_vl_model_ids_detected():
    assert is_qwen2_5_vl_model("Qwen/Qwen2.5-VL-3B-Instruct") is True
    assert is_qwen2_5_vl_model("Qwen/Qwen2.5-VL-7B-Instruct") is True


def test_qwen2_vl_check_does_not_match_qwen2_5_vl_ids():
    """The plain "qwen2-vl" substring check must not accidentally also
    match "qwen2.5-vl" ids -- they route to different transformers classes
    in model_manager.py, so a false match would load the wrong model."""
    assert is_qwen2_vl_model("Qwen/Qwen2.5-VL-3B-Instruct") is False


def test_non_qwen_model_ids_not_detected():
    assert is_qwen2_vl_model("Salesforce/blip-image-captioning-base") is False
    assert is_qwen2_vl_model("Salesforce/blip2-opt-2.7b") is False
    assert is_qwen2_5_vl_model("Salesforce/blip-image-captioning-base") is False


def test_is_chat_template_vlm_true_for_both_qwen_generations():
    assert is_chat_template_vlm("Qwen/Qwen2-VL-2B-Instruct") is True
    assert is_chat_template_vlm("Qwen/Qwen2.5-VL-3B-Instruct") is True


def test_is_chat_template_vlm_false_for_blip_family():
    assert is_chat_template_vlm("Salesforce/blip-image-captioning-base") is False
    assert is_chat_template_vlm("Salesforce/blip2-opt-2.7b") is False


def test_detection_is_case_insensitive():
    assert is_qwen2_vl_model("QWEN/QWEN2-VL-2B-INSTRUCT") is True


def test_empty_or_none_model_id_does_not_raise():
    assert is_qwen2_vl_model("") is False
    assert is_qwen2_5_vl_model("") is False
    assert is_chat_template_vlm("") is False
