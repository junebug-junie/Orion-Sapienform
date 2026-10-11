# services/orion-vllm-host/app/main.py
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import sys
from typing import List, Dict, Any, Optional

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from .settings import settings


logger = logging.getLogger("orion-vllm-host")


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

    # Limit total tokens in a batch (controls KV cache footprint)
    max_batch_tokens = gpu_cfg.get("max_batch_tokens")
    if max_batch_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(max_batch_tokens)]

    # Limit concurrent requests (max active sequences)
    max_concurrent = gpu_cfg.get("max_concurrent_requests")
    if max_concurrent is not None:
        cmd += ["--max-num-seqs", str(max_concurrent)]

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

   # Help PyTorch deal with fragmentation on these tight V100s
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    serving = settings.resolve_serving()
    if serving.get("pool_discovery") is True:
        cmd += ["--middleware", "app.discovery.pool_server_info"]
    for key in ("served_model_name", "dtype", "kv_cache_dtype", "reasoning_parser", "tool_call_parser"):
        if serving.get(key) is not None:
            cmd += ["--" + key.replace("_", "-"), str(serving[key])]
    for key in ("trust_remote_code", "enable_auto_tool_choice"):
        if serving.get(key) is True:
            cmd.append("--" + key.replace("_", "-"))
    if serving.get("speculative_config"):
        cmd += ["--speculative-config", json.dumps(serving["speculative_config"])]
    for key, value in (serving.get("env") or {}).items():
        if key not in {"VLLM_USE_V2_MODEL_RUNNER", "VLLM_WORKER_MULTIPROC_METHOD",
                       "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"}:
            raise ValueError(f"Unsupported vLLM profile environment key: {key}")
        env[key] = str(value)
    return cmd, env


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Independent of the vLLM server subprocess this service
    launches -- see docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.service_name,
            service_version=settings.service_version,
            node_name=settings.node_name,
            bus_url=settings.orion_bus_url,
            bus_enabled=settings.orion_bus_enabled,
            heartbeat_interval_sec=settings.heartbeat_interval_sec,
        )
    )


def run_vllm_server_blocking() -> None:
    """The service's original (pre-heartbeat) synchronous launch flow, unchanged. Run inside
    asyncio.to_thread() by _main_async() so the heartbeat chassis's own event loop stays
    responsive while this blocks on the vLLM subprocess."""
    try:
        cmd, env = build_vllm_command_and_env()
    except Exception as e:
        logger.error("Failed to resolve vLLM configuration: %s", e, exc_info=True)
        raise

    logger.info("vLLM command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


async def announce_loop() -> None:
    """A fresh declaration is necessary but never sufficient for a pool grant."""
    from orion.core.bus.async_service import OrionBusAsync
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.schemas.gpu_pool import (
        LLM_WORKER_ANNOUNCE_CHANNEL, LLM_WORKER_ANNOUNCE_KIND, LlmWorkerAnnounceV1,
    )
    if not settings.orion_bus_enabled or not settings.llm_role or not settings.llm_announce_port:
        return
    if not settings.profile_name:
        raise ValueError("Pool announcement requires explicit VLLM_PROFILE_NAME")
    _, gpu = settings.resolve_model_and_gpu()
    bus = OrionBusAsync(settings.orion_bus_url, enabled=True)
    try:
        while True:
            try:
                await bus.connect()
                payload = LlmWorkerAnnounceV1(
                    host=settings.llm_announce_host, role=settings.llm_role,
                    profile_name=settings.profile_name, port=settings.llm_announce_port,
                    cuda_visible_devices=gpu.get("cuda_visible_devices"), service_name=settings.service_name,
                )
                await bus.publish(LLM_WORKER_ANNOUNCE_CHANNEL, BaseEnvelope(
                    kind=LLM_WORKER_ANNOUNCE_KIND,
                    source=ServiceRef(name=settings.service_name, version=settings.service_version,
                                      node=settings.node_name), payload=payload.model_dump(mode="json")))
            except Exception:
                logger.warning("vllm_worker_announce_failed", exc_info=True)
            await asyncio.sleep(30)
    finally:
        await bus.close()


async def _main_async() -> None:
    settings.validate_announcement()
    logger.info(
        "Starting %s v%s (host=%s port=%s)",
        settings.service_name,
        settings.service_version,
        settings.host,
        settings.port,
    )

    # Awaited (not fired concurrently) before the launch flow starts, matching PR #1350's
    # pilot-5 shape -- an unreachable bus can add up to connect_timeout_sec (default 10s) to
    # startup, bounded and non-fatal (caught below), not unbounded blocking.
    heartbeat_chassis: Optional[HeartbeatOnly] = None
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            settings.service_name,
            settings.heartbeat_interval_sec,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        heartbeat_chassis = None

    # Known limitation (found in review, not fixed here -- bounded to interactive dev use, same
    # tradeoff as orion-ollama-host's identical asyncio.to_thread() conversion): a worker thread
    # running a blocking subprocess.run() cannot be cancelled once started, so an interactive
    # Ctrl-C (SIGINT) can leave the interpreter hanging until the vLLM subprocess exits on its
    # own, whereas before this patch the same blocking call ran on the main thread and Ctrl-C
    # interrupted it immediately. `docker stop` (SIGTERM) is unaffected either way.
    announce_task = asyncio.create_task(announce_loop())
    try:
        await asyncio.to_thread(run_vllm_server_blocking)
    finally:
        announce_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await announce_task
        if heartbeat_chassis is not None:
            try:
                await heartbeat_chassis.stop()
            except Exception as exc:
                logger.warning("system_health_heartbeat_stop_error error=%s", exc)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[VLLM] %(levelname)s - %(name)s - %(message)s",
    )
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
