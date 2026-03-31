from __future__ import annotations

import importlib
from pathlib import Path

import yaml


def _find_flag_value(cmd: list[str], flag: str) -> str:
    idx = cmd.index(flag)
    return cmd[idx + 1]


def test_qwen3_64k_profile_forwards_validated_flags(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "config" / "llm_profiles.yaml"

    monkeypatch.setenv("LLM_PROFILE_NAME", "qwen3-30b-a3b-q4km-atlas-agent-64k-think")
    monkeypatch.setenv("LLM_PROFILES_CONFIG_PATH", str(config_path))

    main = importlib.import_module("app.main")
    settings_mod = importlib.import_module("app.settings")
    profiles_mod = importlib.import_module("app.profiles")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile_cfg = raw["profiles"]["qwen3-30b-a3b-q4km-atlas-agent-64k-think"]
    profile = profiles_mod.LLMProfile(name="qwen3-30b-a3b-q4km-atlas-agent-64k-think", **profile_cfg)

    monkeypatch.setattr(main, "_ensure_model_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        settings_mod.settings,
        "llamacpp_model_path_override",
        "/models/gguf/Qwen3-30B-A3B-Q4_K_M.gguf",
    )

    cmd, _env = main.build_llama_server_cmd_and_env(profile)
    print("effective argv:", " ".join(cmd))

    assert "--reasoning" in cmd
    assert _find_flag_value(cmd, "--reasoning") == "on"
    assert _find_flag_value(cmd, "--reasoning-format") == "deepseek"
    assert _find_flag_value(cmd, "--chat-template-kwargs") == '{"enable_thinking":true}'
    assert _find_flag_value(cmd, "--flash-attn") == "on"
    assert _find_flag_value(cmd, "--rope-scaling") == "yarn"
    assert _find_flag_value(cmd, "--rope-scale") == "2.0"
    assert _find_flag_value(cmd, "--yarn-orig-ctx") == "40960"
    assert "--no-context-shift" in cmd
    assert _find_flag_value(cmd, "--split-mode") == "row"
    assert _find_flag_value(cmd, "--tensor-split") == "1,1"
    assert _find_flag_value(cmd, "--n-predict") == "16384"
    assert _find_flag_value(cmd, "--temp") == "0.6"
    assert _find_flag_value(cmd, "--top-k") == "20"
    assert _find_flag_value(cmd, "--top-p") == "0.95"
    assert _find_flag_value(cmd, "--min-p") == "0.0"
    assert _find_flag_value(cmd, "--presence-penalty") == "1.5"
