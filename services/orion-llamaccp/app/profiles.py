# services/orion-llamacpp/app/profiles.py

from __future__ import annotations
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict

BackendType = Literal["ollama", "vllm", "brain", "llamacpp"]


class GPUConfig(BaseModel):
    num_gpus: int = Field(..., description="How many GPUs this profile expects")
    tensor_parallel_size: int = Field(
        1, description="Tensor parallel world size (doc only for llamacpp)"
    )
    device_ids: Optional[List[int]] = Field(
        default=None,
        description="Optional explicit CUDA device IDs for this profile, e.g. [0,1,2]",
    )

    max_model_len: Optional[int] = Field(
        None, description="Optional override for max model length / context"
    )
    max_batch_tokens: Optional[int] = None
    max_concurrent_requests: Optional[int] = None
    gpu_memory_fraction: Optional[float] = Field(
        None, description="0-1 fraction hint for VRAM usage (doc only for llamacpp)"
    )


class LLMProfile(BaseModel):
    # Pydantic v2 config
    model_config = ConfigDict(
        protected_namespaces=(),  # disables the 'model_' reserved namespace
    )

    name: str
    display_name: Optional[str] = None

    task_type: Literal["chat", "analysis", "tools", "embeddings", "multimodal"] = "chat"

    backend: BackendType
    model_id: str = Field(..., description="Model identifier or GGUF path")

    supports_tools: bool = False
    supports_embeddings: bool = False
    supports_vision: bool = False

    gpu: GPUConfig

    preferred_verbs: List[str] = Field(
        default_factory=list,
        description="Semantic verb names ideally using this profile",
    )
    notes: Optional[str] = None


class LLMProfileRegistry(BaseModel):
    profiles: Dict[str, LLMProfile] = {}

    def get(self, name: str) -> LLMProfile:
        try:
            return self.profiles[name]
        except KeyError:
            raise KeyError(f"LLM profile '{name}' not found in registry")

    def for_verb(self, verb: str, default: Optional[str] = None) -> Optional[LLMProfile]:
        for p in self.profiles.values():
            if verb in p.preferred_verbs:
                return p
        if default:
            return self.profiles.get(default)
        return None
