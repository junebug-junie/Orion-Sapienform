# services/orion-vllm/app/main.py
from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import List, Dict, Any

from .settings import settings


logger = logging.getLogger("orion-vllm")


def build_vllm_command_and_env() -> tuple[List[str], Dict[str, str]]:
    """
    Build the vLLM OpenAI server command + environment based on Settings + profiles.

    - Model + GPU profile are resolved via settings.resolve_model_and_gpu().
    - GPU profile values (e.g., tensor_parallel_size, max_model_len,
      gpu_memory_fraction, cuda_visible_devices) override Settings defaults when present.
    """
    model_id, gpu_cfg = settings.resolve_model_and_gpu()
    gpu_cfg = gpu_cfg or {}

    logger.info("Resolved vLLM config: model_id=%s gpu_cfg=%s", model_id, gpu_cfg)

    # gpu_memory_fraction: prefer profile value, fall back to Settings default
    gpu_memory_fraction = float(
        gpu_cfg.get("gpu_memory_fraction", settings.gpu_memory_fraction)
    )

    cmd: List[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        settings.host,
        "--port",
        str(settings.port),
        "--model",
        model_id,
        "--gpu-memory-utilization",
        str(gpu_memory_fraction),
    ]

    # tensor_parallel_size (optional, from profile)
    tp = gpu_cfg.get("tensor_parallel_size")
    if tp is not None:
        cmd += ["--tensor-parallel-size", str(tp)]

    # max_model_len (optional, from profile)
    max_len = gpu_cfg.get("max_model_len")
    if max_len is not None:
        cmd += ["--max-model-len", str(max_len)]

    # Optional future: derive max_num_seqs from batch tokens / concurrency here
    # using gpu_cfg["max_batch_tokens"] or max_concurrent_requests.

    if settings.download_dir:
        cmd += ["--download-dir", str(settings.download_dir)]

    if settings.enforce_eager:
        cmd += ["--enforce-eager"]

    # Env: start from current process env
    env = os.environ.copy()

    # 🔒 Bridge hf_token → HF_TOKEN / HUGGING_FACE_HUB_TOKEN for vLLM/HF hub
    # Prefer settings.hf_token, but fall back to a raw env var named "hf_token" if present.
    hf_token = getattr(settings, "hf_token", None) or os.environ.get("hf_token")
    if hf_token:
        # Don't clobber if user *explicitly* set these already
        if not env.get("HF_TOKEN"):
            env["HF_TOKEN"] = hf_token
        if not env.get("HUGGING_FACE_HUB_TOKEN"):
            env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        logger.info("[VLLM] Hugging Face token wired into env for model downloads")

    # CUDA: let Settings/profile drive CUDA_VISIBLE_DEVICES
    cuda_visible = gpu_cfg.get("cuda_visible_devices")
    if cuda_visible:
        logger.info("Using CUDA_VISIBLE_DEVICES=%s (from settings/profile)", cuda_visible)
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)

    # Stable mapping index -> physical GPU (important with mixed cards)
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    return cmd, env


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[VLLM] %(levelname)s - %(name)s - %(message)s",
    )

    logger.info(
        "Starting %s v%s (host=%s port=%s)",
        settings.service_name,
        settings.service_version,
        settings.host,
        settings.port,
    )

    try:
        cmd, env = build_vllm_command_and_env()
    except Exception as e:
        logger.error("Failed to resolve vLLM configuration: %s", e, exc_info=True)
        raise

    logger.info("vLLM command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
