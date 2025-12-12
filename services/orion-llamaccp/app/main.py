# services/orion-llamaccp/app/main.py

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, List, Tuple

from .settings import settings

logger = logging.getLogger("orion-llamacpp")


def build_llamacpp_command_and_env() -> Tuple[List[str], Dict[str, str]]:
    """
    Build the llama.cpp `llama-server` command + environment based on:

      - LLM profiles (config/llm_profiles.yaml)
      - LLM_PROFILE_NAME / LLM_MODEL_ID / LLAMACPP_MODEL
      - LLAMACPP_* env overrides

    We assume llama.cpp was built into /app/llama.cpp with `llama-server` at:
      /app/llama.cpp/build/bin/llama-server
    """
    model_id, gpu_cfg = settings.resolve_model_and_gpu()
    gpu_cfg = gpu_cfg or {}

    logger.info(
        "Resolved llama.cpp config: model_id=%s gpu_cfg=%s",
        model_id,
        gpu_cfg,
    )

    # Interpret model_id as a concrete GGUF path inside the container.
    model_path = model_id

    # Context length: prefer profile's max_model_len, fall back to settings.ctx_size
    ctx_size = int(gpu_cfg.get("max_model_len", settings.ctx_size))

    # Batch size: prefer profile's max_batch_tokens, else settings.batch_size
    batch_size = int(gpu_cfg.get("max_batch_tokens") or settings.batch_size)

    # Parallelism: profile.max_concurrent_requests is more of a logical hint;
    # we still derive n_parallel from settings by default.
    n_parallel = int(
        gpu_cfg.get("max_concurrent_requests") or settings.n_parallel
    )

    # N GPU layers (rough tuning knob, only from settings right now)
    n_gpu_layers = int(settings.n_gpu_layers)

    # Base llama-server path (compiled in Dockerfile)
    server_bin = "/app/llama.cpp/build/bin/llama-server"

    cmd: List[str] = [
        server_bin,
        "-m",
        model_path,
        "--host",
        settings.host,
        "--port",
        str(settings.port),
        "--ctx-size",
        str(ctx_size),
        "--batch-size",
        str(batch_size),
        "--threads",
        str(settings.threads),
        "--n-parallel",
        str(n_parallel),
        "--n-gpu-layers",
        str(n_gpu_layers),
    ]

    # Env: start from container env
    env = os.environ.copy()

    # Respect profile.device_ids if present (and override env CUDA_VISIBLE_DEVICES)
    device_ids = gpu_cfg.get("device_ids")
    if device_ids:
        cuda_visible = ",".join(str(i) for i in device_ids)
        logger.info("Using CUDA_VISIBLE_DEVICES=%s from profile.device_ids", cuda_visible)
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    elif env.get("CUDA_VISIBLE_DEVICES"):
        logger.info(
            "Using existing CUDA_VISIBLE_DEVICES=%s from env",
            env["CUDA_VISIBLE_DEVICES"],
        )
    else:
        logger.info("No explicit CUDA_VISIBLE_DEVICES provided; using all GPUs visible in container")

    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    return cmd, env


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[LLAMACPP] %(levelname)s - %(name)s - %(message)s",
    )

    logger.info(
        "Starting %s v%s (host=%s port=%s profile=%s)",
        settings.service_name,
        settings.service_version,
        settings.host,
        settings.port,
        settings.llm_profile_name,
    )

    try:
        cmd, env = build_llamacpp_command_and_env()
    except Exception as e:
        logger.error("Failed to resolve llama.cpp configuration: %s", e, exc_info=True)
        raise

    logger.info("Launching llama-server with command: %s", " ".join(cmd))

    # Run llama-server in the foreground; if it exits, the container restarts.
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
