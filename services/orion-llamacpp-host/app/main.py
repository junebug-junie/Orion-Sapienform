# services/orion-llamacpp-host/app/main.py
from __future__ import annotations


import logging
import os
import asyncio
import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

if os.environ.get("CUDA_VISIBLE_DEVICES_OVERRIDE"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES_OVERRIDE"]

from huggingface_hub import hf_hub_download

from .settings import settings
from .profiles import LLMProfile, LlamaCppConfig
from .thinking_policy import resolve_thinking_launch_policy

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1

logger = logging.getLogger("llamacpp-host")


BOOT_ID = str(uuid.uuid4())
_LLAMA_FLAG_PATTERN = re.compile(r"--([a-z0-9][a-z0-9-]*)")
_LLAMA_BUILD_PATTERN = re.compile(r"version:\s*(\d+)")
_GGUF_SHARD_PATTERN = re.compile(r"^(.+/)(.+)-(\d{5})-of-(\d{5})\.gguf$")

# llama-server --spec-type values that load a draft GGUF with no classic-draft
# equivalent (docs/speculative.md): each needs --model-draft + --spec-type +
# --spec-draft-n-max together, and none of them are reachable via the pre-
# --spec-type flags (--draft-min/--draft-max), unlike "draft-simple"/unset.
_BLOCK_DRAFT_SPEC_TYPES = {"draft-dflash", "draft-dspark", "draft-mtp"}
# --spec-type values that need no draft GGUF file at all (in-context lookup).
_NGRAM_SPEC_TYPES = {
    "ngram-cache",
    "ngram-simple",
    "ngram-map-k",
    "ngram-map-k4v",
    "ngram-mod",
}


def _flag_confirmed_supported(supported_flags: Optional[Set[str]], flag: str) -> bool:
    """True only when a --help probe actually ran and listed `flag`.

    Fails CLOSED (unlike most gating in this file, which fails open when the
    probe itself failed/returned None) -- reserved for call sites where guessing
    wrong risks loading an incompatible GGUF architecture rather than just
    silently omitting an optional flag.
    """
    return supported_flags is not None and flag in supported_flags


def _shard_filenames_for_download(filename: str) -> list[str]:
    """Expand a multi-part GGUF first-shard path into all shard filenames."""
    match = _GGUF_SHARD_PATTERN.match(filename)
    if not match:
        return [filename]
    prefix_dir, stem, _first_idx, total = match.groups()
    total_n = int(total)
    if total_n <= 1:
        return [filename]
    return [f"{prefix_dir}{stem}-{idx:05d}-of-{total}.gguf" for idx in range(1, total_n + 1)]


def _ensure_hf_gguf_file(
    *,
    model_root: str,
    repo_id: str,
    filename: str,
    label: str,
) -> Path:
    """Download a GGUF (or shard set) from HuggingFace when missing locally."""
    target = Path(model_root) / filename
    if target.exists():
        return target

    Path(model_root).mkdir(parents=True, exist_ok=True)

    for shard_filename in _shard_filenames_for_download(filename):
        shard_path = Path(model_root) / shard_filename
        if shard_path.exists():
            continue
        logger.info("Downloading %s/%s -> %s (%s)", repo_id, shard_filename, model_root, label)
        hf_hub_download(
            repo_id=repo_id,
            filename=shard_filename,
            local_dir=model_root,
            local_dir_use_symlinks=False,
            token=settings.hf_token,
        )

    if not target.exists():
        raise FileNotFoundError(f"Download completed but {label} still missing at: {target}")
    return target


def _ensure_model_file(model_path: str, dl: Optional[LlamaCppConfig]) -> None:
    """
    Ensure the GGUF exists at model_path. If not, use dl.{repo_id,filename,model_root}
    to download it into model_root.
    """
    p = Path(model_path)
    if p.exists():
        return

    if dl is None or not dl.repo_id or not dl.filename:
        raise FileNotFoundError(
            f"Model not found and no download spec available: {model_path}"
        )

    _ensure_hf_gguf_file(
        model_root=dl.model_root,
        repo_id=dl.repo_id,
        filename=dl.filename,
        label="language model",
    )

    if not p.exists():
        raise FileNotFoundError(f"Download completed but model still missing at: {model_path}")


def _ensure_mmproj_file(cfg: LlamaCppConfig) -> Optional[str]:
    """Ensure multimodal projector GGUF exists; return concrete path for --mmproj."""
    if not cfg.mmproj_filename:
        return None

    repo_id = cfg.mmproj_repo_id or cfg.repo_id
    if not repo_id:
        raise FileNotFoundError(
            "Profile requests mmproj_filename but no mmproj_repo_id or repo_id is configured"
        )

    mmproj_path = _ensure_hf_gguf_file(
        model_root=cfg.model_root,
        repo_id=repo_id,
        filename=cfg.mmproj_filename,
        label="mmproj",
    )
    return str(mmproj_path)


def _ensure_draft_file(cfg: LlamaCppConfig) -> Optional[str]:
    """Ensure draft GGUF exists; return concrete path for --model-draft."""
    if not cfg.draft_filename:
        return None

    repo_id = cfg.draft_repo_id or cfg.repo_id
    if not repo_id:
        raise FileNotFoundError(
            "Profile requests draft_filename but no draft_repo_id or repo_id is configured"
        )

    draft_path = _ensure_hf_gguf_file(
        model_root=cfg.model_root,
        repo_id=repo_id,
        filename=cfg.draft_filename,
        label="draft model",
    )
    return str(draft_path)


def _ensure_ngram_file(cfg: LlamaCppConfig) -> Optional[str]:
    """Ensure the PLE/n-gram table GGUF exists; return concrete path for --model-ngram."""
    if not cfg.ngram_filename:
        return None

    repo_id = cfg.ngram_repo_id or cfg.repo_id
    if not repo_id:
        raise FileNotFoundError(
            "Profile requests ngram_filename but no ngram_repo_id or repo_id is configured"
        )

    ngram_path = _ensure_hf_gguf_file(
        model_root=cfg.model_root,
        repo_id=repo_id,
        filename=cfg.ngram_filename,
        label="ngram table",
    )
    return str(ngram_path)


def _resolve_runtime(profile: LLMProfile) -> Tuple[str, LlamaCppConfig, Dict[str, str]]:
    """
    Returns: (model_path, runtime_cfg, env)

    - model_path is a concrete /models/.../*.gguf inside container
    - runtime_cfg is profile.llamacpp with .env overrides applied
    - env includes CUDA_VISIBLE_DEVICES derived from profile.gpu.device_ids unless overridden
    """
    if profile.llamacpp is None:
        raise RuntimeError(
            f"Profile '{profile.name}' backend=llamacpp requires a 'llamacpp:' block in llm_profiles.yaml"
        )

    cfg = profile.llamacpp

    # Apply overrides (only when set)
    if settings.llamacpp_host_override is not None:
        cfg.host = settings.llamacpp_host_override
    if settings.llamacpp_port_override is not None:
        cfg.port = settings.llamacpp_port_override
    if settings.llamacpp_ctx_size_override is not None:
        cfg.ctx_size = settings.llamacpp_ctx_size_override
    if settings.llamacpp_n_gpu_layers_override is not None:
        cfg.n_gpu_layers = settings.llamacpp_n_gpu_layers_override
    if settings.llamacpp_threads_override is not None:
        cfg.threads = settings.llamacpp_threads_override
    if settings.llamacpp_n_parallel_override is not None:
        cfg.n_parallel = settings.llamacpp_n_parallel_override
    if settings.llamacpp_batch_size_override is not None:
        cfg.batch_size = settings.llamacpp_batch_size_override

    # Concrete model path resolution
    if settings.llamacpp_model_path_override:
        model_path = settings.llamacpp_model_path_override
    else:
        # Prefer llamacpp.filename + model_root
        if cfg.filename:
            model_path = str(Path(cfg.model_root) / cfg.filename)
        else:
            # Allow profile.model_id to be a direct absolute gguf path if desired
            if profile.model_id.endswith(".gguf") and profile.model_id.startswith("/"):
                model_path = profile.model_id
            else:
                raise RuntimeError(
                    f"Profile '{profile.name}' is missing llamacpp.filename and model_id is not a direct /.../*.gguf path"
                )

    # Environment
    env = os.environ.copy()
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    if settings.cuda_visible_devices_override:
        env["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices_override
    elif profile.gpu.device_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in profile.gpu.device_ids)

    return model_path, cfg, env


@lru_cache(maxsize=4)
def _get_supported_llama_server_flags(server_bin: str) -> Optional[Set[str]]:
    """
    Detect supported CLI flags from `llama-server --help`.
    Returns None when capability probing fails so caller can preserve legacy behavior.
    """
    try:
        result = subprocess.run(
            [server_bin, "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not inspect llama-server flags via --help: %s", exc)
        return None

    if result.returncode not in (0, 1):
        logger.warning("llama-server --help returned unexpected code=%s", result.returncode)
        return None

    help_text = f"{result.stdout}\n{result.stderr}"
    flags = {f"--{match.group(1)}" for match in _LLAMA_FLAG_PATTERN.finditer(help_text)}
    return flags or None


@lru_cache(maxsize=4)
def _get_llama_server_build(server_bin: str) -> Optional[int]:
    """
    Detect llama.cpp numeric build via `llama-server --version`.
    Returns None if probing fails.
    """
    try:
        result = subprocess.run(
            [server_bin, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not inspect llama-server build via --version: %s", exc)
        return None

    if result.returncode != 0:
        logger.warning("llama-server --version returned unexpected code=%s", result.returncode)
        return None

    version_text = f"{result.stdout}\n{result.stderr}"
    match = _LLAMA_BUILD_PATTERN.search(version_text)
    if match is None:
        return None
    return int(match.group(1))


def build_llama_server_cmd_and_env(profile: LLMProfile) -> Tuple[List[str], Dict[str, str]]:
    model_path, cfg, env = _resolve_runtime(profile)

    # Ensure GGUF exists (download if needed)
    _ensure_model_file(model_path, cfg)
    mmproj_path = _ensure_mmproj_file(cfg)

    # llama-server binary inside your built image
    server_bin = "/app/llama-server"
    if not Path(server_bin).exists():
        server_bin = "/app/llama.cpp/build/bin/llama-server"

    cmd: List[str] = [
        server_bin,
        "-m",
        model_path,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--ctx-size",
        str(cfg.ctx_size),
        "--n-gpu-layers",
        str(cfg.n_gpu_layers),
        "--threads",
        str(cfg.threads),
        "--parallel",
        str(cfg.n_parallel),
        "--batch-size",
        str(cfg.batch_size),
    ]

    supported_flags = _get_supported_llama_server_flags(server_bin)
    detected_build = _get_llama_server_build(server_bin)
    is_b5332_compatible = detected_build is not None and detected_build <= 5332

    def append_flag(flag: str, value: Optional[str] = None) -> None:
        if supported_flags is not None and flag not in supported_flags:
            logger.warning("Skipping unsupported llama-server option for this binary: %s", flag)
            return
        cmd.append(flag)
        if value is not None:
            cmd.append(value)

    policy = resolve_thinking_launch_policy(cfg, supported_flags)

    def ensure_jinja() -> None:
        if "--jinja" not in cmd:
            append_flag("--jinja")

    reasoning_format_emitted = False
    if cfg.reasoning is not None:
        if is_b5332_compatible:
            logger.info("Skipping --reasoning for llama.cpp build %s (not supported)", detected_build)
        else:
            append_flag("--reasoning", cfg.reasoning)
    if cfg.reasoning_format is not None:
        ensure_jinja()
        append_flag("--reasoning-format", cfg.reasoning_format)
        reasoning_format_emitted = "--reasoning-format" in cmd
    if cfg.chat_template_kwargs is not None:
        # Do not gate on legacy build numbers: if the binary advertises --chat-template-kwargs in
        # --help, append_flag emits it (per-profile enable_thinking / Qwen3 template kwargs). Older
        # builds without the flag log a skip from append_flag and keep default template behavior.
        if policy.require_jinja:
            ensure_jinja()
        append_flag("--chat-template-kwargs", json.dumps(cfg.chat_template_kwargs, separators=(",", ":")))

    if policy.effective_reasoning_budget is not None:
        if policy.require_jinja:
            ensure_jinja()
        append_flag("--reasoning-budget", str(int(policy.effective_reasoning_budget)))

    if reasoning_format_emitted and "--jinja" not in cmd:
        logger.warning("--reasoning-format requested but --jinja could not be emitted")

    if cfg.flash_attn is not None:
        if is_b5332_compatible:
            if cfg.flash_attn == "on":
                append_flag("--flash-attn")
            elif cfg.flash_attn != "off":
                logger.warning(
                    "Skipping --flash-attn value '%s' for llama.cpp build %s; build expects bare switch",
                    cfg.flash_attn,
                    detected_build,
                )
        else:
            append_flag("--flash-attn", cfg.flash_attn)
    if cfg.rope_scaling is not None:
        append_flag("--rope-scaling", cfg.rope_scaling)
    if cfg.rope_scale is not None:
        append_flag("--rope-scale", str(cfg.rope_scale))
    if cfg.yarn_orig_ctx is not None:
        append_flag("--yarn-orig-ctx", str(cfg.yarn_orig_ctx))
    if cfg.no_context_shift is True:
        append_flag("--no-context-shift")
    if cfg.split_mode is not None:
        append_flag("--split-mode", cfg.split_mode)
    if cfg.tensor_split is not None:
        append_flag("--tensor-split", cfg.tensor_split)

    if mmproj_path is not None:
        ensure_jinja()
        append_flag("--mmproj", mmproj_path)
    if cfg.ubatch_size is not None:
        append_flag("--ubatch-size", str(cfg.ubatch_size))
    if cfg.image_min_tokens is not None:
        append_flag("--image-min-tokens", str(cfg.image_min_tokens))
    if cfg.image_max_tokens is not None:
        append_flag("--image-max-tokens", str(cfg.image_max_tokens))

    if cfg.n_predict is not None:
        append_flag("--n-predict", str(cfg.n_predict))
    if cfg.temperature is not None:
        append_flag("--temp", str(cfg.temperature))
    if cfg.top_k is not None:
        append_flag("--top-k", str(cfg.top_k))
    if cfg.top_p is not None:
        append_flag("--top-p", str(cfg.top_p))
    if cfg.min_p is not None:
        append_flag("--min-p", str(cfg.min_p))
    if cfg.presence_penalty is not None:
        append_flag("--presence-penalty", str(cfg.presence_penalty))

    # Speculative decoding. Three distinct llama-server mechanisms live behind
    # llamacpp.spec_type (docs/speculative.md):
    #  1. Classic small-LM draft: spec_type unset, or explicitly "draft-simple"/"none".
    #     --model-draft + --draft-min/--draft-max. Predates --spec-type entirely, so an
    #     older binary lacking --spec-type can still run this via the bare flags --
    #     --spec-type is emitted too when explicitly requested and the binary happens
    #     to support it, but its absence is NOT a reason to skip the draft model.
    #  2. Block-drafting types (draft-dflash/draft-dspark/draft-mtp): need a GGUF draft
    #     file (--model-draft) PLUS --spec-type + --spec-draft-n-max, and have no
    #     legacy-flag equivalent -- an older binary without --spec-type cannot run
    #     these at all (loading the draft GGUF via bare --model-draft would either fail
    #     to load or misinterpret its architecture), so --spec-type support is a hard,
    #     fail-closed prerequisite for these types specifically, not a best-effort extra.
    #  3. N-gram types (ngram-*): no draft GGUF file -- pure in-context lookup, selected
    #     via --spec-type alone, so this branch runs independently of draft_filename.
    if cfg.spec_type is not None and cfg.spec_type in _NGRAM_SPEC_TYPES:
        if not _flag_confirmed_supported(supported_flags, "--spec-type"):
            logger.error(
                "Profile requested spec_type=%s (n-gram drafting, no draft GGUF needed) but "
                "this llama-server does not advertise --spec-type in --help; omitting "
                "speculative decoding. Upgrade LLAMACPP_IMAGE_TAG or unset spec_type.",
                cfg.spec_type,
            )
        else:
            append_flag("--spec-type", cfg.spec_type)
            if cfg.spec_draft_n_max is not None:
                append_flag("--spec-draft-n-max", str(int(cfg.spec_draft_n_max)))

    if cfg.draft_filename:
        draft_supported = (
            supported_flags is None or "--model-draft" in supported_flags
        )
        block_drafting = cfg.spec_type in _BLOCK_DRAFT_SPEC_TYPES
        if not draft_supported:
            logger.error(
                "Profile requested draft_filename=%s but this llama-server does not advertise "
                "--model-draft in --help; omitting draft speculative decoding and launching "
                "the main model only. Upgrade LLAMACPP_IMAGE_TAG or unset draft_filename.",
                cfg.draft_filename,
            )
        elif block_drafting and not _flag_confirmed_supported(supported_flags, "--spec-type"):
            logger.error(
                "Profile requested draft_filename=%s with spec_type=%s but this llama-server "
                "does not advertise --spec-type in --help (or its --help could not be probed); "
                "omitting draft speculative decoding entirely (not falling back to plain "
                "--model-draft -- a %s-architecture draft GGUF is not safely loadable via the "
                "classic draft path). Upgrade LLAMACPP_IMAGE_TAG past the spec_type's upstream "
                "merge.",
                cfg.draft_filename,
                cfg.spec_type,
                cfg.spec_type,
            )
        else:
            draft_path = _ensure_draft_file(cfg)
            if draft_path is None:
                logger.error(
                    "Profile requested draft_filename=%s but draft path resolved to None; "
                    "omitting draft speculative decoding.",
                    cfg.draft_filename,
                )
            else:
                append_flag("--model-draft", draft_path)
                if cfg.n_gpu_layers_draft is not None:
                    append_flag("--n-gpu-layers-draft", str(int(cfg.n_gpu_layers_draft)))
                if block_drafting:
                    append_flag("--spec-type", cfg.spec_type)
                    if cfg.spec_draft_n_max is not None:
                        append_flag("--spec-draft-n-max", str(int(cfg.spec_draft_n_max)))
                else:
                    # Classic path (spec_type unset, "draft-simple", or "none"). Emit
                    # --spec-type too when explicitly requested and the binary happens to
                    # support it (harmless/more precise), but its absence never blocks the
                    # draft model here -- unlike block-drafting, this path has a working
                    # fallback (the bare --model-draft/--draft-min/--draft-max flags).
                    if cfg.spec_type is not None and _flag_confirmed_supported(
                        supported_flags, "--spec-type"
                    ):
                        append_flag("--spec-type", cfg.spec_type)
                        if cfg.spec_draft_n_max is not None:
                            append_flag("--spec-draft-n-max", str(int(cfg.spec_draft_n_max)))
                    if cfg.draft_min is not None:
                        append_flag("--draft-min", str(int(cfg.draft_min)))
                    if cfg.draft_max is not None:
                        append_flag("--draft-max", str(int(cfg.draft_max)))
                if "--model-draft" not in cmd:
                    logger.error(
                        "Profile requested draft_filename=%s but --model-draft was not emitted; "
                        "omitting draft speculative decoding.",
                        cfg.draft_filename,
                    )

    # Qwen3.8-Flash-Next / "qwen4exp" PLE/n-gram table (--model-ngram, ggml-org/llama.cpp#27742).
    # Independent of the draft-model block above -- this is a second required file for one
    # architecture, not a drafter, and the flag is not gated by any spec_type value.
    if cfg.ngram_filename:
        if not _flag_confirmed_supported(supported_flags, "--model-ngram"):
            logger.error(
                "Profile requested ngram_filename=%s but this llama-server does not advertise "
                "--model-ngram in --help; omitting it and launching the main model only -- this "
                "architecture's PLE/n-gram table will NOT be loaded. Upgrade LLAMACPP_IMAGE_TAG "
                "past the qwen4exp merge (ggml-org/llama.cpp#27742, ~b10666) or unset ngram_filename.",
                cfg.ngram_filename,
            )
        else:
            ngram_path = _ensure_ngram_file(cfg)
            if ngram_path is None:
                logger.error(
                    "Profile requested ngram_filename=%s but ngram path resolved to None; "
                    "omitting --model-ngram.",
                    cfg.ngram_filename,
                )
            else:
                append_flag("--model-ngram", ngram_path)
                if cfg.ngram_load_mode is not None:
                    append_flag("--ngram-load-mode", cfg.ngram_load_mode)

    if cfg.chat_template_kwargs is not None and "--chat-template-kwargs" not in cmd:
        logger.error(
            "Profile requested chat_template_kwargs but --chat-template-kwargs was not emitted "
            "(unsupported by this llama-server or missing from --help). Qwen3 may stay in default thinking; "
            "upgrade llama-server or set reasoning_budget: 0 if supported."
        )
    if policy.effective_reasoning_budget is not None and "--reasoning-budget" not in cmd:
        logger.error(
            "Profile requested reasoning_budget but --reasoning-budget was not emitted; "
            "upgrade llama-server to a build that supports this flag."
        )

    if (
        cfg.chat_template_kwargs is not None
        and cfg.chat_template_kwargs.get("enable_thinking") is False
        and policy.effective_reasoning_budget is None
        and supported_flags is not None
        and "--reasoning-budget" not in supported_flags
    ):
        logger.warning(
            "Profile requests enable_thinking=false via chat_template_kwargs but this llama-server "
            "binary does not advertise --reasoning-budget; thinking may remain on until llama-server "
            "is upgraded to a build that supports --reasoning-budget."
        )

    logger.info(
        "thinking_launch_policy intent=%s effective_reasoning_budget=%s jinja_in_cmd=%s",
        policy.intent_label,
        policy.effective_reasoning_budget,
        "--jinja" in cmd,
    )
    logger.info("Effective llama-server argv: %s", " ".join(cmd))

    return cmd, env

# Heartbeat Coroutine
async def heartbeat_loop(settings):
    # Initialize a local bus just for this script
    bus = OrionBusAsync(url=settings.orion_bus_url, enabled=True)
    await bus.connect()

    logger.info("Heartbeat loop started.")
    try:
        while True:
            try:
                payload = SystemHealthV1(
                    service=settings.service_name,
                    version=settings.service_version,
                    boot_id=BOOT_ID,
                    last_seen_ts=datetime.now(timezone.utc),
                    node="llamacpp-node",
                    status="ok",
                    # heartbeat_interval_sec must match this loop's real period. Left at the
                    # schema default of 10.0, orion-equilibrium-service computes
                    # grace = interval * EQUILIBRIUM_GRACE_MULTIPLIER (3.0) = 30.0s and marks the
                    # service "down" once delta > grace (service.py's status check). Publishing
                    # every 30s leaves ZERO margin, so any event-loop delay or bus latency flips
                    # it to down, emits a spurious transition and pushes distress_score.
                    heartbeat_interval_sec=30.0,
                ).model_dump(mode="json")

                await bus.publish("orion:system:health", BaseEnvelope(
                    kind="system.health.v1",
                    source=ServiceRef(name=settings.service_name, version=settings.service_version),
                    payload=payload
                ))
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

            await asyncio.sleep(30)
    except asyncio.CancelledError:
        logger.info("Heartbeat loop stopping...")
    finally:
        await bus.close()

#  Main Entrypoint
async def _main_async():
    logging.basicConfig(
        level=logging.INFO,
        format="[LLAMACPP] %(levelname)s - %(name)s - %(message)s",
    )
    if settings.cuda_visible_devices_override:
        os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices_override

    profile = settings.resolve_profile()
    logger.info(
        "Starting %s v%s profile=%s",
        settings.service_name,
        settings.service_version,
        profile.name,
    )

    cmd, env = build_llama_server_cmd_and_env(profile)
    logger.info("Launching llama-server: %s", " ".join(cmd))

    # Start the heartbeat in background
    hb_task = asyncio.create_task(heartbeat_loop(settings))

    # Create subprocess
    process = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        stdout=None, # Inherit
        stderr=None
    )

    try:
        # Wait for the server process to exit
        await process.wait()
    finally:
        # Clean up heartbeat
        hb_task.cancel()
        await hb_task

def main():
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
