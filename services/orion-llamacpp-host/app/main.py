# services/orion-llamacpp-host/app/main.py
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from huggingface_hub import hf_hub_download

from .settings import settings
from .profiles import LLMProfile, LlamaCppConfig

logger = logging.getLogger("orion-llamacpp-host")


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

    return cmd, env


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[LLAMACPP] %(levelname)s - %(name)s - %(message)s",
    )

    profile = settings.resolve_profile()
    logger.info(
        "Starting %s v%s profile=%s",
        settings.service_name,
        settings.service_version,
        profile.name,
    )

    cmd, env = build_llama_server_cmd_and_env(profile)
    logger.info("Launching llama-server: %s", " ".join(cmd))

    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
