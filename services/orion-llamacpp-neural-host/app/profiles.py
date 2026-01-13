from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, AliasChoices

BackendType = Literal["ollama", "vllm", "brain", "llamacpp"]


class GPUConfig(BaseModel):
    num_gpus: int = Field(..., description="How many GPUs this profile expects")
    tensor_parallel_size: int = Field(1, description="Doc-only for llamacpp")
    device_ids: Optional[List[int]] = Field(default=None)

    max_model_len: Optional[int] = None
    max_batch_tokens: Optional[int] = None
    max_concurrent_requests: Optional[int] = None
    gpu_memory_fraction: Optional[float] = None


class LlamaCppConfig(BaseModel):
    # avoid protected namespace warnings on fields like model_root
    model_config = ConfigDict(protected_namespaces=())

    model_root: str = Field("/models/gguf")

    # Canonical names (what code should use)
    repo_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("repo_id", "hf_repo_id"),
        description="HuggingFace repo id",
    )
    filename: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("filename", "hf_filename"),
        description="GGUF filename in repo",
    )

    host: str = "0.0.0.0"
    port: int = 8080
    ctx_size: int = 8192
    n_gpu_layers: int = 80
    threads: int = 16
    n_parallel: int = 2
    batch_size: int = 512


class LLMProfile(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    display_name: Optional[str] = None
    task_type: Literal["chat", "analysis", "tools", "embeddings", "multimodal"] = "chat"

    backend: BackendType
    model_id: str

    supports_tools: bool = False
    supports_embeddings: bool = False
    supports_vision: bool = False

    gpu: GPUConfig
    llamacpp: Optional[LlamaCppConfig] = None

    preferred_verbs: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class LLMProfileRegistry(BaseModel):
    profiles: Dict[str, LLMProfile] = {}

    def get(self, name: str) -> LLMProfile:
        try:
            return self.profiles[name]
        except KeyError:
            raise KeyError(f"LLM profile '{name}' not found in registry")
