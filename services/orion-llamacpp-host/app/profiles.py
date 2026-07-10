from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

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
    mmproj_filename: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("mmproj_filename", "hf_mmproj_filename"),
        description="Multimodal projector GGUF filename (llama-server --mmproj)",
    )
    mmproj_repo_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("mmproj_repo_id", "hf_mmproj_repo_id"),
        description="HF repo for mmproj; defaults to repo_id when omitted",
    )
    draft_filename: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("draft_filename", "hf_draft_filename"),
        description="Draft GGUF filename for llama-server speculative decoding (--model-draft)",
    )
    draft_repo_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("draft_repo_id", "hf_draft_repo_id"),
        description="HF repo for draft GGUF; defaults to repo_id when omitted",
    )
    n_gpu_layers_draft: Optional[int] = Field(
        default=None,
        description="Draft model GPU offload layers (--n-gpu-layers-draft)",
    )
    draft_min: Optional[int] = Field(
        default=None,
        description="Minimum draft tokens before verify (--draft-min)",
    )
    draft_max: Optional[int] = Field(
        default=None,
        description="Maximum draft tokens per speculation step (--draft-max)",
    )

    host: str = "0.0.0.0"
    port: int = 8080
    ctx_size: int = 8192
    n_gpu_layers: int = 80
    threads: int = 16
    n_parallel: int = 2
    batch_size: int = 512
    ubatch_size: Optional[int] = None
    image_min_tokens: Optional[int] = None
    image_max_tokens: Optional[int] = None

    reasoning: Optional[Literal["on", "off", "auto"]] = None
    reasoning_format: Optional[Literal["none", "deepseek", "deepseek-legacy", "auto"]] = None
    # llama-server --reasoning-budget (use 0 to disable Qwen3-style thinking; complements chat_template_kwargs)
    reasoning_budget: Optional[int] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None

    flash_attn: Optional[Literal["on", "off", "auto"]] = None
    rope_scaling: Optional[Literal["none", "linear", "yarn"]] = None
    rope_scale: Optional[float] = None
    yarn_orig_ctx: Optional[int] = None
    no_context_shift: Optional[bool] = None
    split_mode: Optional[Literal["none", "layer", "row"]] = None
    tensor_split: Optional[str] = None

    n_predict: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None


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
