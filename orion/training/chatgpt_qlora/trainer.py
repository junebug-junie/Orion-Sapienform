from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

from .runtime import ensure_runtime_packages
from .schemas import DatasetManifest, TrainingConfig, TrainingRunManifest, ensure_dir


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_qlora_training(
    cfg: TrainingConfig,
    *,
    dataset_manifest_path: str,
    simulate: bool = False,
) -> TrainingRunManifest:
    dataset_path = Path(dataset_manifest_path)
    manifest = DatasetManifest.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    run_id = cfg.run_id or str(uuid.uuid4())

    run_dir = ensure_dir(cfg.output_dir) / "training" / cfg.run_name
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_name = cfg.tokenizer_name or cfg.base_model

    if simulate:
        metrics = {
            "train_loss": 0.0,
            "eval_loss": 0.0,
            "note": "simulated_run",
            "train_examples": manifest.train_count,
            "val_examples": manifest.val_count,
        }
        lib_versions = {}
        (adapter_dir / "SIMULATED_ADAPTER.txt").write_text(
            "Simulated adapter artifact for pipeline validation.\n", encoding="utf-8"
        )
        status = "simulated"
    else:
        ensure_runtime_packages()

        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer

        if manifest.train_count <= 0:
            raise RuntimeError("Dataset manifest has zero train examples; cannot run real training")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=(torch.bfloat16 if cfg.compute_dtype == "bfloat16" else torch.float16),
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=bnb_config,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_file = manifest.files["train"]
        val_file = manifest.files["val"]
        ds = load_dataset("json", data_files={"train": train_file, "validation": val_file})

        lora_cfg = LoraConfig(
            r=cfg.qlora.r,
            lora_alpha=cfg.qlora.alpha,
            lora_dropout=cfg.qlora.dropout,
            target_modules=cfg.qlora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        args = TrainingArguments(
            output_dir=str(run_dir / "hf_checkpoints"),
            run_name=cfg.run_name,
            num_train_epochs=cfg.epochs,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            per_device_train_batch_size=cfg.train_batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            evaluation_strategy=("steps" if manifest.val_count > 0 else "no"),
            eval_steps=cfg.eval_steps,
            bf16=(cfg.compute_dtype == "bfloat16"),
            fp16=(cfg.compute_dtype == "float16"),
            report_to=[],
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            args=args,
            peft_config=lora_cfg,
            processing_class=tokenizer,
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
        )
        trainer.train()
        eval_metrics = trainer.evaluate() if manifest.val_count > 0 else {}
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        train_loss = None
        for item in trainer.state.log_history:
            if "loss" in item:
                train_loss = item["loss"]
        metrics = {
            "train_loss": train_loss,
            "eval_loss": eval_metrics.get("eval_loss"),
            "train_runtime": eval_metrics.get("train_runtime"),
            "train_examples": manifest.train_count,
            "val_examples": manifest.val_count,
            "max_steps": cfg.max_steps,
        }
        lib_versions = {}
        for lib in ("torch", "transformers", "datasets", "bitsandbytes", "peft", "trl"):
            try:
                module = import_module(lib)
                lib_versions[lib] = getattr(module, "__version__", "unknown")
            except Exception:
                lib_versions[lib] = "unknown"
        status = "complete"

    train_manifest = TrainingRunManifest(
        created_at=_now(),
        run_name=cfg.run_name,
        run_id=run_id,
        base_model=cfg.base_model,
        tokenizer_name=tokenizer_name,
        chat_template="orion_chatml_v1",
        training_config=cfg.model_dump(mode="json"),
        dataset_manifest_path=str(dataset_path),
        dataset_manifest_sha256=_sha256_file(dataset_path),
        foundry_build_dir=manifest.foundry_build_dir,
        foundry_partition=manifest.foundry_partition,
        adapter_output_dir=str(adapter_dir),
        training_metrics=metrics,
        library_versions=lib_versions,
        status=status,
        import_run_ids=manifest.import_run_ids,
    )
    train_manifest_path = run_dir / "training_manifest.json"
    train_manifest_path.write_text(train_manifest.model_dump_json(indent=2), encoding="utf-8")

    (run_dir / "adapter_registry_stub.json").write_text(
        json.dumps(
            {
                "adapter_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{run_id}:{cfg.run_name}:{train_manifest.dataset_manifest_sha256}")),
                "run_id": run_id,
                "run_name": cfg.run_name,
                "adapter_dir": str(adapter_dir),
                "base_model": cfg.base_model,
                "tokenizer_name": tokenizer_name,
                "chat_template": "orion_chatml_v1",
                "dataset_manifest_path": str(dataset_path),
                "dataset_manifest_sha256": train_manifest.dataset_manifest_sha256,
                "import_run_ids": manifest.import_run_ids,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return train_manifest
