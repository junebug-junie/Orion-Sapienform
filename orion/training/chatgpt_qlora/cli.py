from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import build_sft_dataset
from .eval import evaluate_training_run
from .foundry import build_semantic_foundry
from .schemas import PipelineConfig
from .trainer import run_qlora_training


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orion-native ChatGPT QLoRA SFT pipeline")
    p.add_argument("--config", required=True, help="Path to JSON pipeline config")
    p.add_argument(
        "--phase",
        choices=["foundry", "dataset", "train", "eval", "run-all"],
        default="run-all",
    )
    p.add_argument("--simulate", action="store_true", help="Run training/eval without HF runtime")
    p.add_argument("--dataset-manifest", default=None)
    p.add_argument("--training-manifest", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = PipelineConfig.model_validate_json(Path(args.config).read_text(encoding="utf-8"))

    dataset_manifest_path = args.dataset_manifest
    training_manifest_path = args.training_manifest
    foundry_manifest_path = Path(cfg.foundry.output_dir) / "foundry" / cfg.foundry.build_name / "foundry_manifest.json"

    if args.phase in {"foundry", "run-all"}:
        foundry_manifest = build_semantic_foundry(cfg.foundry, cfg.dataset.source)
        cfg.dataset.foundry_build_dir = str(foundry_manifest_path.parent)
        print(json.dumps({"phase": "foundry", "manifest": foundry_manifest.model_dump(mode="json")}, indent=2))

    if args.phase in {"dataset", "run-all"}:
        dataset_manifest = build_sft_dataset(cfg.dataset)
        dataset_manifest_path = str(Path(cfg.dataset.output_dir) / "dataset" / "dataset_manifest.json")
        print(json.dumps({"phase": "dataset", "manifest": dataset_manifest.model_dump(mode="json")}, indent=2))

    if args.phase in {"train", "run-all"}:
        if not dataset_manifest_path:
            raise ValueError("dataset manifest required for training phase")
        training_manifest = run_qlora_training(
            cfg.training,
            dataset_manifest_path=dataset_manifest_path,
            simulate=args.simulate,
        )
        training_manifest_path = str(Path(cfg.training.output_dir) / "training" / cfg.training.run_name / "training_manifest.json")
        print(json.dumps({"phase": "train", "manifest": training_manifest.model_dump(mode="json")}, indent=2))

    if args.phase in {"eval", "run-all"}:
        if not training_manifest_path:
            raise ValueError("training manifest required for eval phase")
        eval_manifest, adapter_manifest = evaluate_training_run(
            cfg.eval,
            training_manifest_path=training_manifest_path,
            simulate=args.simulate,
        )
        print(
            json.dumps(
                {
                    "phase": "eval",
                    "eval_manifest": eval_manifest.model_dump(mode="json"),
                    "adapter_manifest": adapter_manifest.model_dump(mode="json"),
                },
                indent=2,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
