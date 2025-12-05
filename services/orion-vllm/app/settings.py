# services/orion-vllm/app/settings.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from pydantic_settings import BaseSettings
from pydantic import Field
import yaml


class Settings(BaseSettings):
    # Service identity
    service_name: str = Field("orion-vllm", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")

    # HTTP bind
    host: str = Field("0.0.0.0", alias="VLLM_HOST")
    port: int = Field(8000, alias="VLLM_PORT")

    # Profiles config (same YAML concept as llm-gateway)
    llm_profiles_config_path: Path = Field(
        default=Path("/app/config/llm_profiles.yaml"),
        alias="LLM_PROFILES_CONFIG_PATH",
        description="Path to YAML defining LLM profiles",
    )

    # Which profile to use from llm_profiles.yaml on THIS node
    profile_name: Optional[str] = Field(
        default=None,
        alias="VLLM_PROFILE_NAME",
        description="Name of LLM profile to use for this vLLM node",
    )

    # Direct model override (takes precedence over profile.model_id)
    model_id: Optional[str] = Field(
        default=None,
        alias="VLLM_MODEL_ID",
        description="Model identifier passed to vLLM (overrides profile.model_id if set)",
    )

    # Runtime-only knobs (env defaults that profiles can override)
    gpu_memory_fraction: float = Field(
        default=0.9,
        alias="VLLM_GPU_MEMORY_FRACTION",
        description="Fraction of GPU memory vLLM is allowed to use (--gpu-memory-utilization)",
    )

    download_dir: Optional[Path] = Field(
        default=Path("/models"),
        alias="VLLM_DOWNLOAD_DIR",
        description="Download/cache directory for vLLM models",
    )

    enforce_eager: bool = Field(
        default=False,
        alias="VLLM_ENFORCE_EAGER",
        description="Whether to pass --enforce-eager to vLLM",
    )

    # Optional explicit override for CUDA_VISIBLE_DEVICES
    cuda_visible_devices: Optional[str] = Field(
        default=None,
        alias="VLLM_CUDA_VISIBLE_DEVICES",
        description=(
            "Explicit CUDA_VISIBLE_DEVICES override. "
            "If unset, derived from profile.gpu.device_ids when present."
        ),
    )

    class Config:
        env_file = ".env"
        extra = "ignore"  # ignore any extra env vars (like old VLLM_MAX_MODEL_LEN, etc.)

    # ─────────────────────────────────────────────
    # Profile loading
    # ─────────────────────────────────────────────
    def _load_profiles(self) -> Dict[str, Any]:
        """
        Load raw profile dicts from YAML (same file used by llm-gateway).
        """
        path = self.llm_profiles_config_path
        try:
            if not path.exists():
                return {}
        except Exception:
            return {}

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return raw.get("profiles", {}) or {}

    def resolve_model_and_gpu(self) -> Tuple[str, Dict[str, Any]]:
        """
        Resolve (model_id, gpu_cfg) using:
          1) Explicit env: VLLM_MODEL_ID (optional)
          2) LLM profile referenced by VLLM_PROFILE_NAME
          3) If no profile_name set, first profile in the YAML (if any)

        gpu_cfg is a normalized dict with keys like:
          - gpu_memory_fraction
          - tensor_parallel_size
          - max_model_len
          - max_batch_tokens
          - max_concurrent_requests
          - device_ids
          - cuda_visible_devices
        """
        profiles = self._load_profiles()

        # Choose profile
        profile_name = self.profile_name
        if not profile_name and profiles:
            # Fallback: first defined profile for this node
            profile_name = next(iter(profiles.keys()))

        profile = profiles.get(profile_name) if profile_name else None

        # 1) Model ID: env wins over profile
        model_id = self.model_id
        if not model_id and profile is not None:
            model_id = profile.get("model_id")

        if not model_id:
            raise ValueError(
                "No model configured for vLLM. "
                "Set VLLM_MODEL_ID or VLLM_PROFILE_NAME (with model_id in profiles YAML)."
            )

        gpu_cfg: Dict[str, Any] = {}

        # 2) GPU settings: purely from profile.gpu
        device_ids = None
        if profile is not None:
            gpu = profile.get("gpu") or {}

            if gpu.get("gpu_memory_fraction") is not None:
                gpu_cfg["gpu_memory_fraction"] = gpu.get("gpu_memory_fraction")

            if gpu.get("tensor_parallel_size") is not None:
                gpu_cfg["tensor_parallel_size"] = gpu.get("tensor_parallel_size")

            if gpu.get("max_model_len") is not None:
                gpu_cfg["max_model_len"] = gpu.get("max_model_len")

            if gpu.get("max_batch_tokens") is not None:
                gpu_cfg["max_batch_tokens"] = gpu.get("max_batch_tokens")

            if gpu.get("max_concurrent_requests") is not None:
                gpu_cfg["max_concurrent_requests"] = gpu.get("max_concurrent_requests")

            # Canonical GPU mapping: device_ids
            if gpu.get("device_ids") is not None:
                device_ids = gpu.get("device_ids")
                gpu_cfg["device_ids"] = device_ids

        # 3) CUDA_VISIBLE_DEVICES: env override wins, otherwise derive from device_ids.
        cuda_visible = self.cuda_visible_devices
        if cuda_visible is None and device_ids is not None:
            # Normalize list/tuple/int → "0,1,2"
            if isinstance(device_ids, (list, tuple)):
                cuda_visible = ",".join(str(i) for i in device_ids)
            else:
                cuda_visible = str(device_ids)

        if cuda_visible:
            gpu_cfg["cuda_visible_devices"] = cuda_visible

        return model_id, gpu_cfg


settings = Settings()
