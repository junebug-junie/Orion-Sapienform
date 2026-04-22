from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SubstrateSourceConfig(BaseModel):
    postgres_uri: str | None = None
    examples_jsonl: str | None = None
    import_run_ids: list[str] = Field(default_factory=list)
    limit: int | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> "SubstrateSourceConfig":
        if not self.postgres_uri and not self.examples_jsonl:
            raise ValueError("Set postgres_uri or examples_jsonl for substrate source")
        return self


class DatasetBuildConfig(BaseModel):
    source: SubstrateSourceConfig
    output_dir: str = "artifacts/chatgpt_qlora"
    foundry_build_dir: str | None = None
    foundry_partition: str = "sft_direct_orion"
    split_seed: str = "chatgpt-qlora-sft-v1"
    val_ratio: float = 0.1
    min_prompt_chars: int = 3
    min_response_chars: int = 3
    allowed_roles: list[str] = Field(default_factory=lambda: ["user", "assistant"])
    template_style: Literal["orion_chatml_v1"] = "orion_chatml_v1"


class QLoRAHyperParams(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


class TrainingConfig(BaseModel):
    base_model: str
    tokenizer_name: str | None = None
    output_dir: str = "artifacts/chatgpt_qlora"
    run_name: str = "chatgpt-sft"
    run_id: str | None = None
    max_seq_length: int = 2048
    epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 2
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = False
    qlora: QLoRAHyperParams = Field(default_factory=QLoRAHyperParams)


class EvalConfig(BaseModel):
    output_dir: str = "artifacts/chatgpt_qlora"
    max_new_tokens: int = 256
    sample_count: int = 5
    generation_temperature: float = 0.2
    generation_top_p: float = 0.9


class FoundryConfig(BaseModel):
    output_dir: str = "artifacts/chatgpt_qlora"
    build_name: str = "chatgpt-semantic-foundry-v1"
    preserve_unknown_concepts: bool = True
    include_oracle_rewrite_candidates: bool = True


class PipelineConfig(BaseModel):
    foundry: FoundryConfig = Field(default_factory=FoundryConfig)
    dataset: DatasetBuildConfig
    training: TrainingConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)


class CanonicalSftExample(BaseModel):
    example_id: str
    split: Literal["train", "val"]
    prompt: str
    response: str
    text: str
    import_run_id: str
    conversation_id: str
    turn_id: str | None = None
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    schema_version: Literal["chatgpt_qlora_dataset_manifest.v1"] = "chatgpt_qlora_dataset_manifest.v1"
    created_at: str
    source: SubstrateSourceConfig
    split_seed: str
    val_ratio: float
    template_style: str
    included_count: int
    excluded_count: int
    train_count: int
    val_count: int
    foundry_build_dir: str | None = None
    foundry_partition: str | None = None
    import_run_ids: list[str] = Field(default_factory=list)
    conversation_ids: list[str] = Field(default_factory=list)
    files: dict[str, str]

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()


class TrainingRunManifest(BaseModel):
    schema_version: Literal["chatgpt_qlora_training_manifest.v1"] = "chatgpt_qlora_training_manifest.v1"
    created_at: str
    run_name: str
    run_id: str
    base_model: str
    tokenizer_name: str
    chat_template: str
    training_config: dict[str, Any] = Field(default_factory=dict)
    dataset_manifest_path: str
    dataset_manifest_sha256: str
    foundry_build_dir: str | None = None
    foundry_partition: str | None = None
    adapter_output_dir: str
    training_metrics: dict[str, Any] = Field(default_factory=dict)
    library_versions: dict[str, str] = Field(default_factory=dict)
    status: Literal["complete", "failed", "simulated"]
    import_run_ids: list[str] = Field(default_factory=list)


class EvalManifest(BaseModel):
    schema_version: Literal["chatgpt_qlora_eval_manifest.v1"] = "chatgpt_qlora_eval_manifest.v1"
    created_at: str
    run_name: str
    format_validity: bool
    metric_summary: dict[str, Any] = Field(default_factory=dict)
    provenance_integrity: bool
    samples: list[dict[str, Any]] = Field(default_factory=list)


class AdapterArtifactManifest(BaseModel):
    schema_version: Literal["chatgpt_qlora_adapter_manifest.v1"] = "chatgpt_qlora_adapter_manifest.v1"
    created_at: str
    adapter_id: str
    run_id: str
    adapter_dir: str
    base_model: str
    tokenizer_name: str
    load_hints: dict[str, Any] = Field(default_factory=dict)
    dataset_manifest_path: str
    training_manifest_path: str
    eval_manifest_path: str
    import_run_ids: list[str] = Field(default_factory=list)


class FoundryBuildManifest(BaseModel):
    schema_version: Literal["chatgpt_semantic_foundry_manifest.v1"] = "chatgpt_semantic_foundry_manifest.v1"
    created_at: str
    build_name: str
    source: SubstrateSourceConfig
    files: dict[str, str]
    relationship_mode_distribution: dict[str, int] = Field(default_factory=dict)
    oracle_vs_orion_distribution: dict[str, int] = Field(default_factory=dict)
    developmental_fit_distribution: dict[str, int] = Field(default_factory=dict)
    partition_counts: dict[str, int] = Field(default_factory=dict)
    concept_domain_frequencies: dict[str, int] = Field(default_factory=dict)
    subdomain_frequencies: dict[str, int] = Field(default_factory=dict)
    discovered_new_concept_count: int = 0
    failure_mode_frequencies: dict[str, int] = Field(default_factory=dict)
    rewrite_candidate_count: int = 0
    teacher_exemplar_count: int = 0
    direct_sft_allowed_count: int = 0
    excluded_count: int = 0
    architecture_heavy_count: int = 0
    developmental_policy_count: int = 0
    ontology_density_distribution: dict[str, int] = Field(default_factory=dict)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
