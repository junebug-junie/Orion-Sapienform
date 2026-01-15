from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, AliasChoices

BackendType = Literal["ollama", "vllm", "brain", "llama-cola"]


class GPUConfig(BaseModel):
    num_gpus: int = Field(..., description="How many GPUs this profile expects")
    tensor_parallel_size: int = Field(1, description="Doc-only for llama-cola")
    device_ids: Optional[List[int]] = Field(default=None)

    max_model_len: Optional[int] = None
    max_batch_tokens: Optional[int] = None
    max_concurrent_requests: Optional[int] = None
    gpu_memory_fraction: Optional[float] = None


class LlamaColaConfig(BaseModel):
    # avoid protected namespace warnings on fields like model_root
    model_config = ConfigDict(protected_namespaces=())

    model_root: str = Field("/models/cola")

    # Canonical names (what code should use)
    repo_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("repo_id", "hf_repo_id"),
        description="HuggingFace repo id",
    )
    revision: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("revision", "hf_revision"),
        description="Optional HuggingFace revision/tag",
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Local path to a pretrained CoLA model directory",
    )

    host: str = "0.0.0.0"
    port: int = 8080
    max_new_tokens: int = 128


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
    llama_cola: Optional[LlamaColaConfig] = None

    preferred_verbs: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class LLMProfileRegistry(BaseModel):
    profiles: Dict[str, LLMProfile] = {}

    def get(self, name: str) -> LLMProfile:
        try:
            return self.profiles[name]
        except KeyError:
            raise KeyError(f"LLM profile '{name}' not found in registry")
