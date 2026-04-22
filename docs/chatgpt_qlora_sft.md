# ChatGPT QLoRA SFT (Orion-native)

This module now includes a semantic foundry that builds multi-layer corpus outputs (annotations, entities, relations, routing partitions) before SFT dataset creation.

## Inputs
- Preferred substrate input: `chat_gpt_derived_example` rows (via `postgres_uri` + optional `import_run_ids`).
- Local deterministic path: JSONL exported in the same shape as `chat_gpt_derived_example`.

## Canonical example contract
Each materialized row contains:
- `example_id`, `split`, `prompt`, `response`, `text`
- lineage fields: `import_run_id`, `conversation_id`, `turn_id`, `user_message_id`, `assistant_message_id`
- `tags`, `metadata`

## Artifacts
Default root: `artifacts/chatgpt_qlora/`
- `foundry/<build_name>/foundry_manifest.json`
- `foundry/<build_name>/annotations.example.jsonl`
- `foundry/<build_name>/entities.example.jsonl`
- `foundry/<build_name>/relations.example.jsonl`
- `foundry/<build_name>/conversation.summary.jsonl`
- `foundry/<build_name>/partition.sft_direct_orion.jsonl`
- `foundry/<build_name>/partition.sft_rewritten_oracle.jsonl`
- `foundry/<build_name>/partition.rubric_negative_examples.jsonl`
- `foundry/<build_name>/partition.ontology_graph.jsonl`
- `dataset/dataset_manifest.json`
- `dataset/examples.all.jsonl`
- `dataset/examples.train.jsonl`
- `dataset/examples.val.jsonl`
- `training/<run_name>/training_manifest.json`
- `training/<run_name>/adapter/` (saved adapter/tokenizer or simulated placeholder)
- `eval/<run_name>/eval_manifest.json`
- `eval/<run_name>/adapter_artifact_manifest.json`

## Run
```bash
python scripts/run_chatgpt_qlora_sft.py \
  --config orion/training/chatgpt_qlora/examples/pipeline_config.example.json \
  --phase run-all \
  --simulate
```

## Real-run smoke command (Orion hardware)
```bash
python scripts/run_chatgpt_qlora_sft.py \
  --config /path/to/chatgpt_qlora_real_smoke.json \
  --phase run-all
```

Recommended minimal real-smoke config settings:
- dataset source uses `postgres_uri` with `import_run_ids` and small `limit`
- `max_steps` set to `10`
- `save_steps` / `eval_steps` set small (for example `5`)
- `train_batch_size=1`, `gradient_accumulation_steps=1`

## Notes
- `--simulate` validates dataset + lineage + manifest wiring without requiring heavy HF runtime deps.
- Real training mode requires: `torch`, `transformers`, `datasets`, `bitsandbytes`, `peft`, `trl`.
- Real eval mode produces side-by-side `base_response` and `adapter_response` on held-out prompts in `eval_manifest.json`.
- Default routing policy excludes `frontier_oracle` from direct SFT and routes it to rewrite-first partitions.
