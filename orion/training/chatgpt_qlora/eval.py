from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runtime import ensure_runtime_packages
from .schemas import AdapterArtifactManifest, DatasetManifest, EvalConfig, EvalManifest, TrainingRunManifest, ensure_dir


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_training_run(
    cfg: EvalConfig,
    *,
    training_manifest_path: str,
    simulate: bool = False,
) -> tuple[EvalManifest, AdapterArtifactManifest]:
    train_path = Path(training_manifest_path)
    t_manifest = TrainingRunManifest.model_validate_json(train_path.read_text(encoding="utf-8"))
    d_manifest = DatasetManifest.model_validate_json(Path(t_manifest.dataset_manifest_path).read_text(encoding="utf-8"))

    val_rows = _load_jsonl(Path(d_manifest.files["val"]))
    if not simulate and t_manifest.status == "simulated":
        raise RuntimeError("Cannot run real eval for a simulated training run")

    tokenizer = None
    base_model = None
    adapter_model = None
    torch = None
    device = None
    if not (simulate or t_manifest.status == "simulated"):
        ensure_runtime_packages()
        import torch as torch_mod
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = torch_mod
        tokenizer = AutoTokenizer.from_pretrained(t_manifest.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = AutoModelForCausalLM.from_pretrained(t_manifest.base_model).to(device)
        adapter_model = PeftModel.from_pretrained(base_model, t_manifest.adapter_output_dir)

    def _run_generations(prompt: str) -> tuple[str, str]:
        if simulate or t_manifest.status == "simulated":
            return "[simulated_generation_base]", "[simulated_generation_adapter]"

        assert tokenizer is not None and base_model is not None and adapter_model is not None and device is not None
        prompt_text = f"<|user|>\\n{prompt}\\n<|assistant|>\\n"
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        base_out = base_model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.generation_temperature,
            top_p=cfg.generation_top_p,
        )
        adapter_out = adapter_model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.generation_temperature,
            top_p=cfg.generation_top_p,
        )
        return tokenizer.decode(base_out[0], skip_special_tokens=True), tokenizer.decode(adapter_out[0], skip_special_tokens=True)

    samples: list[dict[str, Any]] = []
    for row in val_rows[: cfg.sample_count]:
        base_response, adapter_response = _run_generations(str(row.get("prompt") or ""))
        item = {
            "example_id": row.get("example_id"),
            "prompt": row.get("prompt"),
            "expected_response": row.get("response"),
            "import_run_id": row.get("import_run_id"),
            "conversation_id": row.get("conversation_id"),
            "base_response": base_response,
            "adapter_response": adapter_response,
        }
        samples.append(item)

    provenance_integrity = all(
        bool(s.get("import_run_id")) and bool(s.get("conversation_id")) for s in samples
    )

    eval_manifest = EvalManifest(
        created_at=_now(),
        run_name=t_manifest.run_name,
        format_validity=bool(d_manifest.files.get("train")) and bool(d_manifest.files.get("val")),
        metric_summary=t_manifest.training_metrics,
        provenance_integrity=provenance_integrity,
        samples=samples,
    )

    eval_dir = ensure_dir(cfg.output_dir) / "eval" / t_manifest.run_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / "eval_manifest.json"
    eval_path.write_text(eval_manifest.model_dump_json(indent=2), encoding="utf-8")

    adapter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{t_manifest.run_id}:{t_manifest.run_name}:{t_manifest.dataset_manifest_sha256}"))
    adapter_manifest = AdapterArtifactManifest(
        created_at=_now(),
        adapter_id=adapter_id,
        run_id=t_manifest.run_id,
        adapter_dir=t_manifest.adapter_output_dir,
        base_model=t_manifest.base_model,
        tokenizer_name=t_manifest.tokenizer_name,
        load_hints={
            "library": "peft",
            "base_model": t_manifest.base_model,
            "adapter_path": t_manifest.adapter_output_dir,
            "chat_template": t_manifest.chat_template,
            "run_id": t_manifest.run_id,
            "foundry_build_dir": t_manifest.foundry_build_dir,
            "foundry_partition": t_manifest.foundry_partition,
        },
        dataset_manifest_path=t_manifest.dataset_manifest_path,
        training_manifest_path=str(train_path),
        eval_manifest_path=str(eval_path),
        import_run_ids=t_manifest.import_run_ids,
    )
    adapter_path = eval_dir / "adapter_artifact_manifest.json"
    adapter_path.write_text(adapter_manifest.model_dump_json(indent=2), encoding="utf-8")

    return eval_manifest, adapter_manifest
