"""The shared GPU3 profile has ten 32k slots and selectable thinking."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import yaml


PROFILE = "qwen35-9b-udq5kxl-v100-32gb-circe-shared-32k"


def _value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_qwen35_shared_launch_has_ten_32k_slots_and_no_thinking_lockout(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "config/llm_profiles.yaml").read_text())
    profile_cfg = config["profiles"][PROFILE]
    assert profile_cfg["llamacpp"]["hf_repo_id"] == "unsloth/Qwen3.5-9B-GGUF"
    assert profile_cfg["llamacpp"]["hf_filename"] == "Qwen3.5-9B-UD-Q5_K_XL.gguf"
    assert profile_cfg["llamacpp"]["ctx_size"] == 327680
    assert profile_cfg["llamacpp"]["n_parallel"] == 10

    monkeypatch.setenv("LLM_PROFILE_NAME", PROFILE)
    monkeypatch.setenv("LLM_PROFILES_CONFIG_PATH", str(root / "config/llm_profiles.yaml"))
    main = importlib.import_module("app.main")
    settings = importlib.import_module("app.settings").settings
    model = importlib.import_module("app.profiles")
    profile = model.LLMProfile(name=PROFILE, **profile_cfg)
    monkeypatch.setattr(main, "_ensure_model_file", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "_get_supported_llama_server_flags", lambda _bin: {
        "--jinja", "--reasoning", "--reasoning-format", "--chat-template-kwargs",
        "--reasoning-budget", "--flash-attn", "--no-context-shift", "--n-predict",
        "--temp", "--top-k", "--top-p", "--min-p", "--presence-penalty",
    })
    monkeypatch.setattr(main, "_get_llama_server_build", lambda _bin: 10398)
    monkeypatch.setattr(settings, "llamacpp_model_path_override", None)
    cmd, _env = main.build_llama_server_cmd_and_env(profile)
    assert _value(cmd, "-m") == "/models/gguf/Qwen3.5-9B-UD-Q5_K_XL.gguf"
    assert _value(cmd, "--ctx-size") == "327680"
    assert _value(cmd, "--parallel") == "10"
    assert _value(cmd, "--n-predict") == "8192"
    assert _value(cmd, "--reasoning") == "auto"
    assert _value(cmd, "--reasoning-format") == "deepseek"
    assert json.loads(_value(cmd, "--chat-template-kwargs")) == {"enable_thinking": False}
    assert "--reasoning-budget" not in cmd
