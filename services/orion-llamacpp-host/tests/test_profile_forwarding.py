from __future__ import annotations

import importlib
import json
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
        main,
        "_get_supported_llama_server_flags",
        lambda _server_bin: {
            "--jinja",
            "--reasoning",
            "--reasoning-format",
            "--chat-template-kwargs",
            "--flash-attn",
            "--rope-scaling",
            "--rope-scale",
            "--yarn-orig-ctx",
            "--no-context-shift",
            "--split-mode",
            "--tensor-split",
            "--n-predict",
            "--temp",
            "--top-k",
            "--top-p",
            "--min-p",
            "--presence-penalty",
        },
    )
    monkeypatch.setattr(main, "_get_llama_server_build", lambda _server_bin: 6000)
    monkeypatch.setattr(
        settings_mod.settings,
        "llamacpp_model_path_override",
        "/models/gguf/Qwen3-30B-A3B-Q4_K_M.gguf",
    )

    cmd, _env = main.build_llama_server_cmd_and_env(profile)
    print("effective argv:", " ".join(cmd))

    assert "--reasoning" in cmd
    assert _find_flag_value(cmd, "--reasoning") == "on"
    assert "--jinja" in cmd
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


def test_qwen3_64k_profile_skips_unsupported_reasoning_flag(monkeypatch):
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
        main,
        "_get_supported_llama_server_flags",
        lambda _server_bin: {
            "--jinja",
            "--reasoning-format",
            "--flash-attn",
            "--rope-scaling",
            "--rope-scale",
            "--yarn-orig-ctx",
            "--no-context-shift",
            "--split-mode",
            "--tensor-split",
            "--n-predict",
            "--temp",
            "--top-k",
            "--top-p",
            "--min-p",
            "--presence-penalty",
        },
    )
    monkeypatch.setattr(main, "_get_llama_server_build", lambda _server_bin: 5332)
    monkeypatch.setattr(
        settings_mod.settings,
        "llamacpp_model_path_override",
        "/models/gguf/Qwen3-30B-A3B-Q4_K_M.gguf",
    )

    cmd, _env = main.build_llama_server_cmd_and_env(profile)
    assert "--reasoning" not in cmd
    assert "--jinja" in cmd
    assert _find_flag_value(cmd, "--reasoning-format") == "deepseek"
    assert "--chat-template-kwargs" not in cmd
    flash_attn_idx = cmd.index("--flash-attn")
    assert flash_attn_idx + 1 >= len(cmd) or cmd[flash_attn_idx + 1].startswith("-")
    no_context_shift_idx = cmd.index("--no-context-shift")
    assert no_context_shift_idx + 1 >= len(cmd) or cmd[no_context_shift_idx + 1].startswith("-")
    print("effective argv (b5332):", " ".join(cmd))


def test_qwen3_8b_balanced_profile_forwards_enable_thinking_false(monkeypatch):
    """config/llm_profiles qwen3-8b-q4km-v100-16gb-balanced: non-thinking template kwargs."""
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "config" / "llm_profiles.yaml"

    monkeypatch.setenv("LLM_PROFILE_NAME", "qwen3-8b-q4km-v100-16gb-balanced")
    monkeypatch.setenv("LLM_PROFILES_CONFIG_PATH", str(config_path))

    main = importlib.import_module("app.main")
    settings_mod = importlib.import_module("app.settings")
    profiles_mod = importlib.import_module("app.profiles")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile_cfg = raw["profiles"]["qwen3-8b-q4km-v100-16gb-balanced"]
    profile = profiles_mod.LLMProfile(name="qwen3-8b-q4km-v100-16gb-balanced", **profile_cfg)

    monkeypatch.setattr(main, "_ensure_model_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main,
        "_get_supported_llama_server_flags",
        lambda _server_bin: {
            "--jinja",
            "--chat-template-kwargs",
            "--reasoning-budget",
            "--no-context-shift",
            "--n-predict",
            "--temp",
            "--top-k",
            "--top-p",
            "--min-p",
            "--presence-penalty",
        },
    )
    monkeypatch.setattr(main, "_get_llama_server_build", lambda _server_bin: 6000)
    monkeypatch.setattr(
        settings_mod.settings,
        "llamacpp_model_path_override",
        "/models/gguf/Qwen_Qwen3-8B-Q4_K_M.gguf",
    )

    cmd, _env = main.build_llama_server_cmd_and_env(profile)
    assert "--chat-template-kwargs" in cmd
    kwargs_val = _find_flag_value(cmd, "--chat-template-kwargs")
    assert json.loads(kwargs_val) == {"enable_thinking": False}
    assert "--reasoning-budget" in cmd
    assert _find_flag_value(cmd, "--reasoning-budget") == "0"
    assert "--jinja" in cmd
    assert cmd.index("--jinja") < cmd.index("--chat-template-kwargs")


def test_qwen3_8b_atlas_metacog_profile_q5km_single_lane_16k(monkeypatch):
    """Atlas metacog: Q5_K_M, n_parallel=1 -> full 16k ctx per slot, reasoning off."""
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "config" / "llm_profiles.yaml"

    main = importlib.import_module("app.main")
    settings_mod = importlib.import_module("app.settings")
    profiles_mod = importlib.import_module("app.profiles")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile_cfg = raw["profiles"]["qwen3-8b-q5km-v100-16gb-atlas-metacog-16k"]
    profile = profiles_mod.LLMProfile(name="qwen3-8b-q5km-v100-16gb-atlas-metacog-16k", **profile_cfg)

    monkeypatch.setattr(main, "_ensure_model_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main,
        "_get_supported_llama_server_flags",
        lambda _server_bin: {
            "--jinja",
            "--reasoning",
            "--reasoning-budget",
            "--ctx-size",
            "--parallel",
            "--no-context-shift",
            "--n-predict",
            "--temp",
            "--top-k",
            "--top-p",
            "--min-p",
            "--presence-penalty",
        },
    )
    monkeypatch.setattr(main, "_get_llama_server_build", lambda _server_bin: 6000)
    monkeypatch.setattr(
        settings_mod.settings,
        "llamacpp_model_path_override",
        "/models/gguf/Qwen_Qwen3-8B-Q5_K_M.gguf",
    )

    cmd, _env = main.build_llama_server_cmd_and_env(profile)
    assert _find_flag_value(cmd, "--reasoning") == "off"
    assert _find_flag_value(cmd, "--ctx-size") == "16384"
    assert _find_flag_value(cmd, "--parallel") == "1"
    assert _find_flag_value(cmd, "--reasoning-budget") == "0"
    assert _find_flag_value(cmd, "--n-predict") == "4096"
    assert _find_flag_value(cmd, "--temp") == "0.35"
    assert "--chat-template-kwargs" not in cmd


def test_qwen3_implicit_budget_off_emits_jinja_and_budget(monkeypatch):
    """Synthetic profile: only chat_template_kwargs={'enable_thinking': False} → policy emits --jinja and --reasoning-budget 0."""
    main = importlib.import_module("app.main")
    settings_mod = importlib.import_module("app.settings")
    profiles_mod = importlib.import_module("app.profiles")

    profile = profiles_mod.LLMProfile(
        name="synthetic-qwen3-implicit-off",
        backend="llamacpp",
        model_id="/models/gguf/Qwen_Qwen3-8B-Q4_K_M.gguf",
        gpu=profiles_mod.GPUConfig(num_gpus=1),
        llamacpp=profiles_mod.LlamaCppConfig(
            chat_template_kwargs={"enable_thinking": False},
        ),
    )

    monkeypatch.setattr(main, "_ensure_model_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main,
        "_get_supported_llama_server_flags",
        lambda _server_bin: {
            "--jinja",
            "--chat-template-kwargs",
            "--reasoning-budget",
        },
    )
    monkeypatch.setattr(main, "_get_llama_server_build", lambda _server_bin: 6000)
    monkeypatch.setattr(
        settings_mod.settings,
        "llamacpp_model_path_override",
        "/models/gguf/Qwen_Qwen3-8B-Q4_K_M.gguf",
    )

    cmd, _env = main.build_llama_server_cmd_and_env(profile)
    assert "--jinja" in cmd
    assert "--reasoning-budget" in cmd
    assert _find_flag_value(cmd, "--reasoning-budget") == "0"
    assert "--chat-template-kwargs" in cmd
    assert cmd.index("--jinja") < cmd.index("--chat-template-kwargs")


def test_shard_filenames_for_download_expands_multi_part_gguf(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE_NAME", "qwen3-coder-next-q4km-2xv100-32gb-agent-breadth")
    main = importlib.import_module("app.main")
    filename = "Qwen3-Coder-Next-Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00001-of-00004.gguf"
    shards = main._shard_filenames_for_download(filename)
    assert len(shards) == 4
    assert shards[0].endswith("00001-of-00004.gguf")
    assert shards[-1].endswith("00004-of-00004.gguf")


def test_qwen3_coder_next_profile_uses_hf_shard_paths():
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "config" / "llm_profiles.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for key in (
        "qwen3-coder-next-q4km-2xv100-32gb-agent-breadth",
        "qwen3-coder-next-q5km-2xv100-32gb-agent-depth",
    ):
        filename = raw["profiles"][key]["llamacpp"]["hf_filename"]
        assert "-Q4_K_M-" in filename or "-Q5_K_M-" in filename
        assert filename.endswith("-00001-of-00004.gguf")


def test_gemma4_31b_multimodal_profile_forwards_mmproj_and_image_flags(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "config" / "llm_profiles.yaml"

    monkeypatch.setenv("LLM_PROFILE_NAME", "gemma4-31b-it-q4km-2xv100-32gb-multimodal")
    monkeypatch.setenv("LLM_PROFILES_CONFIG_PATH", str(config_path))

    main = importlib.import_module("app.main")
    settings_mod = importlib.import_module("app.settings")
    profiles_mod = importlib.import_module("app.profiles")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile_cfg = raw["profiles"]["gemma4-31b-it-q4km-2xv100-32gb-multimodal"]
    profile = profiles_mod.LLMProfile(name="gemma4-31b-it-q4km-2xv100-32gb-multimodal", **profile_cfg)

    monkeypatch.setattr(main, "_ensure_model_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main,
        "_ensure_mmproj_file",
        lambda _cfg: "/models/gguf/mmproj-gemma-4-31B-it-bf16.gguf",
    )
    monkeypatch.setattr(
        main,
        "_get_supported_llama_server_flags",
        lambda _server_bin: {
            "--jinja",
            "--chat-template-kwargs",
            "--flash-attn",
            "--no-context-shift",
            "--split-mode",
            "--tensor-split",
            "--mmproj",
            "--ubatch-size",
            "--image-min-tokens",
            "--image-max-tokens",
            "--n-predict",
            "--temp",
            "--top-k",
            "--top-p",
            "--min-p",
            "--presence-penalty",
        },
    )
    monkeypatch.setattr(main, "_get_llama_server_build", lambda _server_bin: 8740)
    monkeypatch.setattr(
        settings_mod.settings,
        "llamacpp_model_path_override",
        "/models/gguf/gemma-4-31B-it-Q4_K_M.gguf",
    )

    cmd, _env = main.build_llama_server_cmd_and_env(profile)

    assert profile.supports_vision is True
    assert "--jinja" in cmd
    assert _find_flag_value(cmd, "--mmproj") == "/models/gguf/mmproj-gemma-4-31B-it-bf16.gguf"
    assert _find_flag_value(cmd, "--ubatch-size") == "2048"
    assert _find_flag_value(cmd, "--image-min-tokens") == "280"
    assert _find_flag_value(cmd, "--image-max-tokens") == "560"
    assert _find_flag_value(cmd, "--ctx-size") == "65536"
    assert _find_flag_value(cmd, "--temp") == "1.0"
    assert _find_flag_value(cmd, "--top-k") == "64"
    assert _find_flag_value(cmd, "--top-p") == "0.95"


_MUSE_GLIMMER_SUPPORTED_FLAGS_BASE = {
    "--jinja",
    "--no-context-shift",
    "--mmproj",
    "--ubatch-size",
    "--image-min-tokens",
    "--image-max-tokens",
    "--n-predict",
    "--temp",
    "--top-k",
    "--top-p",
    "--min-p",
    "--presence-penalty",
}


def _load_muse_glimmer_profile(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "config" / "llm_profiles.yaml"

    monkeypatch.setenv("LLM_PROFILE_NAME", "muse-glimmer-30b-udq4kxl-v100-32gb-agent-vision")
    monkeypatch.setenv("LLM_PROFILES_CONFIG_PATH", str(config_path))

    main = importlib.import_module("app.main")
    settings_mod = importlib.import_module("app.settings")
    profiles_mod = importlib.import_module("app.profiles")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile_cfg = raw["profiles"]["muse-glimmer-30b-udq4kxl-v100-32gb-agent-vision"]
    profile = profiles_mod.LLMProfile(name="muse-glimmer-30b-udq4kxl-v100-32gb-agent-vision", **profile_cfg)

    monkeypatch.setattr(main, "_ensure_model_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main,
        "_ensure_mmproj_file",
        lambda _cfg: "/models/gguf/mmproj-Muse-Glimmer-30B-Q8_0.gguf",
    )
    monkeypatch.setattr(
        main,
        "_ensure_draft_file",
        lambda _cfg: "/models/gguf/dflash-kquant.gguf",
    )
    monkeypatch.setattr(
        settings_mod.settings,
        "llamacpp_model_path_override",
        "/models/gguf/Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
    )
    return main, profile


def test_muse_glimmer_30b_agent_vision_profile_forwards_flags_on_supporting_build(monkeypatch):
    main, profile = _load_muse_glimmer_profile(monkeypatch)

    monkeypatch.setattr(
        main,
        "_get_supported_llama_server_flags",
        lambda _server_bin: _MUSE_GLIMMER_SUPPORTED_FLAGS_BASE
        | {"--model-draft", "--n-gpu-layers-draft", "--spec-type", "--spec-draft-n-max"},
    )
    # server-cuda-b10398: past both the DFlash merge (~b9766) and the Muse Glimmer
    # architecture merge (ggml-org/llama.cpp#26841, 2026-08-10) this profile needs.
    monkeypatch.setattr(main, "_get_llama_server_build", lambda _server_bin: 10398)

    cmd, _env = main.build_llama_server_cmd_and_env(profile)

    assert profile.supports_vision is True
    assert profile.supports_tools is True
    assert profile.llamacpp.spec_type == "draft-dflash"
    assert "--jinja" in cmd
    assert _find_flag_value(cmd, "--mmproj") == "/models/gguf/mmproj-Muse-Glimmer-30B-Q8_0.gguf"
    assert _find_flag_value(cmd, "--ubatch-size") == "1024"
    assert _find_flag_value(cmd, "--image-min-tokens") == "256"
    assert _find_flag_value(cmd, "--image-max-tokens") == "1024"
    assert _find_flag_value(cmd, "--ctx-size") == "32768"
    assert _find_flag_value(cmd, "--batch-size") == "1024"
    assert _find_flag_value(cmd, "--temp") == "1.0"
    assert _find_flag_value(cmd, "--top-k") == "64"
    assert _find_flag_value(cmd, "--top-p") == "0.95"
    assert _find_flag_value(cmd, "--n-predict") == "16384"
    # DFlash drafter: --model-draft still carries the path, --spec-type/--spec-draft-n-max
    # select the block-diffusion path, and the classic draft_min/draft_max tuning knobs
    # (not set on this profile) must not appear since they don't apply to this spec_type.
    assert _find_flag_value(cmd, "--model-draft") == "/models/gguf/dflash-kquant.gguf"
    assert _find_flag_value(cmd, "--spec-type") == "draft-dflash"
    assert _find_flag_value(cmd, "--spec-draft-n-max") == "16"
    assert "--draft-min" not in cmd
    assert "--draft-max" not in cmd


def test_muse_glimmer_30b_agent_vision_profile_skips_draft_entirely_on_old_build(monkeypatch, caplog):
    import logging

    main, profile = _load_muse_glimmer_profile(monkeypatch)

    # server-cuda-b8740 (this stack's current default pin, ~April 2026): advertises
    # --model-draft (classic speculative decoding existed long before DFlash) but not
    # --spec-type. A dflash-architecture draft GGUF is not safely interpretable via the
    # classic path, so the builder must skip the draft model entirely rather than
    # falling back to plain --model-draft.
    monkeypatch.setattr(
        main,
        "_get_supported_llama_server_flags",
        lambda _server_bin: _MUSE_GLIMMER_SUPPORTED_FLAGS_BASE
        | {"--model-draft", "--n-gpu-layers-draft", "--draft-min", "--draft-max"},
    )
    monkeypatch.setattr(main, "_get_llama_server_build", lambda _server_bin: 8740)

    with caplog.at_level(logging.ERROR):
        cmd, _env = main.build_llama_server_cmd_and_env(profile)

    assert "--model-draft" not in cmd
    assert "--spec-type" not in cmd
    assert "--spec-draft-n-max" not in cmd
    assert "--draft-min" not in cmd
    assert "--draft-max" not in cmd
    # The main model still launches -- this is a graceful degrade, not a crash.
    assert _find_flag_value(cmd, "--mmproj") == "/models/gguf/mmproj-Muse-Glimmer-30B-Q8_0.gguf"
    assert any("spec_type" in rec.message and "does not advertise --spec-type" in rec.message for rec in caplog.records)


def test_draft_fields_emit_speculative_decoding_flags(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("LLM_PROFILE_NAME", "unit-test")

    main = importlib.import_module("app.main")
    profiles_mod = importlib.import_module("app.profiles")
    settings_mod = importlib.import_module("app.settings")

    profile = profiles_mod.LLMProfile(
        name="unit-draft-enabled",
        backend="llamacpp",
        model_id="unit-target",
        gpu=profiles_mod.GPUConfig(num_gpus=1, tensor_parallel_size=1, device_ids=[0]),
        llamacpp=profiles_mod.LlamaCppConfig(
            model_root="/models/gguf",
            repo_id="example/target",
            filename="target.gguf",
            draft_filename="Qwen3-0.6B-Q4_K_M.gguf",
            draft_repo_id="unsloth/Qwen3-0.6B-GGUF",
            n_gpu_layers_draft=99,
            draft_min=4,
            draft_max=16,
            host="0.0.0.0",
            port=8080,
            ctx_size=8192,
            n_gpu_layers=99,
            threads=8,
            n_parallel=1,
            batch_size=512,
        ),
    )

    monkeypatch.setattr(main, "_ensure_model_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main,
        "_ensure_draft_file",
        lambda _cfg: "/models/gguf/Qwen3-0.6B-Q4_K_M.gguf",
    )
    monkeypatch.setattr(
        main,
        "_get_supported_llama_server_flags",
        lambda _server_bin: {
            "--model-draft",
            "--draft-min",
            "--draft-max",
            "--n-gpu-layers-draft",
            "--n-predict",
            "--temp",
        },
    )
    monkeypatch.setattr(main, "_get_llama_server_build", lambda _server_bin: 8740)
    monkeypatch.setattr(
        settings_mod.settings,
        "llamacpp_model_path_override",
        "/models/gguf/target.gguf",
    )

    with caplog.at_level(logging.ERROR):
        cmd, _env = main.build_llama_server_cmd_and_env(profile)

    assert _find_flag_value(cmd, "--model-draft") == "/models/gguf/Qwen3-0.6B-Q4_K_M.gguf"
    assert _find_flag_value(cmd, "--n-gpu-layers-draft") == "99"
    assert _find_flag_value(cmd, "--draft-min") == "4"
    assert _find_flag_value(cmd, "--draft-max") == "16"
    assert not any("draft" in rec.message.lower() and rec.levelno >= logging.ERROR for rec in caplog.records)


def test_draft_unset_emits_no_draft_flags(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE_NAME", "unit-test")

    main = importlib.import_module("app.main")
    profiles_mod = importlib.import_module("app.profiles")
    settings_mod = importlib.import_module("app.settings")

    profile = profiles_mod.LLMProfile(
        name="unit-draft-unset",
        backend="llamacpp",
        model_id="unit-target",
        gpu=profiles_mod.GPUConfig(num_gpus=1, tensor_parallel_size=1, device_ids=[0]),
        llamacpp=profiles_mod.LlamaCppConfig(
            model_root="/models/gguf",
            repo_id="example/target",
            filename="target.gguf",
            host="0.0.0.0",
            port=8080,
            ctx_size=8192,
            n_gpu_layers=99,
            threads=8,
            n_parallel=1,
            batch_size=512,
        ),
    )

    monkeypatch.setattr(main, "_ensure_model_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main,
        "_get_supported_llama_server_flags",
        lambda _server_bin: {
            "--model-draft",
            "--draft-min",
            "--draft-max",
            "--n-gpu-layers-draft",
        },
    )
    monkeypatch.setattr(main, "_get_llama_server_build", lambda _server_bin: 8740)
    monkeypatch.setattr(
        settings_mod.settings,
        "llamacpp_model_path_override",
        "/models/gguf/target.gguf",
    )

    ensure_calls: list[object] = []

    def _unexpected_ensure(cfg):
        ensure_calls.append(cfg)
        raise AssertionError("_ensure_draft_file must not run when draft_filename is unset")

    monkeypatch.setattr(main, "_ensure_draft_file", _unexpected_ensure)

    cmd, _env = main.build_llama_server_cmd_and_env(profile)

    assert "--model-draft" not in cmd
    assert "--draft-min" not in cmd
    assert "--draft-max" not in cmd
    assert "--n-gpu-layers-draft" not in cmd
    assert ensure_calls == []


def test_draft_requested_but_flags_unsupported_omits_draft_without_crash(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("LLM_PROFILE_NAME", "unit-test")

    main = importlib.import_module("app.main")
    profiles_mod = importlib.import_module("app.profiles")
    settings_mod = importlib.import_module("app.settings")

    profile = profiles_mod.LLMProfile(
        name="unit-draft-unsupported",
        backend="llamacpp",
        model_id="unit-target",
        gpu=profiles_mod.GPUConfig(num_gpus=1, tensor_parallel_size=1, device_ids=[0]),
        llamacpp=profiles_mod.LlamaCppConfig(
            model_root="/models/gguf",
            repo_id="example/target",
            filename="target.gguf",
            draft_filename="Qwen3-0.6B-Q4_K_M.gguf",
            draft_repo_id="unsloth/Qwen3-0.6B-GGUF",
            n_gpu_layers_draft=99,
            draft_min=4,
            draft_max=16,
            host="0.0.0.0",
            port=8080,
            ctx_size=8192,
            n_gpu_layers=99,
            threads=8,
            n_parallel=1,
            batch_size=512,
        ),
    )

    monkeypatch.setattr(main, "_ensure_model_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main,
        "_ensure_draft_file",
        lambda _cfg: "/models/gguf/Qwen3-0.6B-Q4_K_M.gguf",
    )
    # Probe succeeded but draft flags absent (old binary).
    monkeypatch.setattr(
        main,
        "_get_supported_llama_server_flags",
        lambda _server_bin: {"--temp", "--top-k"},
    )
    monkeypatch.setattr(main, "_get_llama_server_build", lambda _server_bin: 4719)
    monkeypatch.setattr(
        settings_mod.settings,
        "llamacpp_model_path_override",
        "/models/gguf/target.gguf",
    )

    with caplog.at_level(logging.ERROR):
        cmd, _env = main.build_llama_server_cmd_and_env(profile)

    assert cmd[0].endswith("llama-server") or "llama-server" in cmd[0]
    assert "-m" in cmd
    assert "--model-draft" not in cmd
    assert "--draft-min" not in cmd
    assert "--draft-max" not in cmd
    assert "--n-gpu-layers-draft" not in cmd
    assert any(
        "draft" in rec.message.lower() and "model-draft" in rec.message.lower()
        for rec in caplog.records
        if rec.levelno >= logging.ERROR
    )
