# services/orion-vllm/app/main.py
from __future__ import annotations

import logging
import subprocess
import sys

from .settings import settings


logger = logging.getLogger("orion-vllm")


def build_vllm_command() -> list[str]:
    """
    Build the vLLM OpenAI server command based on Settings + profiles.
    """
    model_id, gpu_cfg = settings.resolve_model_and_gpu()

    logger.info(
        "Resolved vLLM config: model_id=%s gpu_cfg=%s",
        model_id,
        gpu_cfg,
    )

    cmd: list[str] = [
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
        str(settings.gpu_memory_fraction),
    ]

    tp = gpu_cfg.get("tensor_parallel_size")
    if tp:
        cmd += ["--tensor-parallel-size", str(tp)]

    max_len = gpu_cfg.get("max_model_len")
    if max_len:
        cmd += ["--max-model-len", str(max_len)]

    # Optional: if you later want to derive max_num_seqs from batch tokens / concurrency,
    # you can do it here based on gpu_cfg["max_batch_tokens"] or max_concurrent_requests.

    if settings.download_dir:
        cmd += ["--download-dir", str(settings.download_dir)]

    if settings.enforce_eager:
        cmd += ["--enforce-eager"]

    return cmd


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
        cmd = build_vllm_command()
    except Exception as e:
        logger.error("Failed to resolve vLLM configuration: %s", e, exc_info=True)
        raise

    logger.info("vLLM command: %s", " ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
