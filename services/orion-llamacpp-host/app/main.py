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

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1

logger = logging.getLogger("llamacpp-host")


BOOT_ID = str(uuid.uuid4())
_LLAMA_FLAG_PATTERN = re.compile(r"--([a-z0-9][a-z0-9-]*)")
_LLAMA_BUILD_PATTERN = re.compile(r"version:\s*(\d+)")


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

    Path(dl.model_root).mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s/%s -> %s", dl.repo_id, dl.filename, dl.model_root)
    hf_hub_download(
        repo_id=dl.repo_id,
        filename=dl.filename,
        local_dir=dl.model_root,
        local_dir_use_symlinks=False,
        token=settings.hf_token,
    )

    if not p.exists():
        raise FileNotFoundError(f"Download completed but model still missing at: {model_path}")


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

    reasoning_format_emitted = False
    if cfg.reasoning is not None:
        if is_b5332_compatible:
            logger.info("Skipping --reasoning for llama.cpp build %s (not supported)", detected_build)
        else:
            append_flag("--reasoning", cfg.reasoning)
    if cfg.reasoning_format is not None:
        append_flag("--jinja")
        append_flag("--reasoning-format", cfg.reasoning_format)
        reasoning_format_emitted = "--reasoning-format" in cmd
    if cfg.chat_template_kwargs is not None and not is_b5332_compatible:
        append_flag("--chat-template-kwargs", json.dumps(cfg.chat_template_kwargs, separators=(",", ":")))
    elif cfg.chat_template_kwargs is not None and is_b5332_compatible:
        logger.info(
            "Skipping --chat-template-kwargs for llama.cpp build %s (not documented/supported)",
            detected_build,
        )

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
                    status="ok"
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
