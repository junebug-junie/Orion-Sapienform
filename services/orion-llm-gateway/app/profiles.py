from __future__ import annotations
from typing import Dict, Literal, Optional, List

from pydantic import BaseModel, Field, ConfigDict

BackendType = Literal["ollama", "vllm", "brain", "llamacpp"]


class GPUConfig(BaseModel):
    num_gpus: int = Field(..., description="How many GPUs this profile expects")
    tensor_parallel_size: int = Field(1, description="Tensor parallel world size")
    max_model_len: Optional[int] = Field(
        None, description="Optional override for max model length / context"
    )
    max_batch_tokens: Optional[int] = None
    max_concurrent_requests: Optional[int] = None
    gpu_memory_fraction: Optional[float] = Field(
        None, description="0-1 fraction hint for vLLM / other backends"
    )


class LLMProfile(BaseModel):
    # Pydantic v2 config to silence the model_id warning
    model_config = ConfigDict(
        protected_namespaces=(),  # disables the 'model_' reserved namespace
    )

    name: str
    display_name: Optional[str] = None

    # Logical usage
    task_type: Literal["chat", "analysis", "tools", "embeddings", "multimodal"] = "chat"

    # Backend + model wiring
    backend: BackendType
    model_id: str = Field(..., description="Model identifier understood by backend")

    # Capabilities
    supports_tools: bool = False
    supports_embeddings: bool = False
    supports_vision: bool = False

    # Performance & GPU layout (mostly documentation for now)
    gpu: GPUConfig

    # Semantic hinting
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
