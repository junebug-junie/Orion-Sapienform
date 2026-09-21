#!/usr/bin/env python3
"""Lightweight check for the Circe DeepSeek-V4.1 Flash profile (no pytest)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "services" / "orion-llamacpp-host"))

profiles_mod = importlib.import_module("app.profiles")
raw = yaml.safe_load((REPO / "config" / "llm_profiles.yaml").read_text(encoding="utf-8"))
key = "deepseek-v41-flash-mxfp4-engram-4xv100-32gb-circe-test"
p = profiles_mod.LLMProfile(name=key, **raw["profiles"][key])
assert p.gpu.device_ids == [0, 1, 2, 3], p.gpu.device_ids
assert p.llamacpp.moe_stream is True
assert p.llamacpp.moe_stream_cache == 64
assert p.llamacpp.moe_stream_l2 == 96
assert p.llamacpp.moe_stream_io_threads == 4
assert p.llamacpp.repo_id == "JigSawPT/DeepSeek-V4.1-Flash-GGUF"
assert p.llamacpp.filename == "DeepSeek-V4.1-Flash-MXFP4-engram-00001-of-00011.gguf"
assert p.llamacpp.model_root.endswith("DeepSeek-V4.1-Flash-MXFP4-engram")
assert p.llamacpp.tensor_split == "1,1,1,1"
assert p.llamacpp.flash_attn == "off"
print("YAML+schema OK:", p.display_name)
