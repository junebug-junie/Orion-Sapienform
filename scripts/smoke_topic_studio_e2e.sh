#!/usr/bin/env bash
set -euo pipefail

HUB_BASE_URL=${HUB_BASE_URL:-http://localhost:8080}

if ! html=$(curl -fsS "${HUB_BASE_URL}/"); then
  echo "FAIL: Hub not reachable at ${HUB_BASE_URL}." >&2
  exit 1
fi
if ! printf '%s' "$html" | rg -q "data-split-pane"; then
  echo "FAIL: Topic Studio split pane not found in hub HTML." >&2
  exit 1
fi

datasets_json=$(curl -fsS "${HUB_BASE_URL}/api/topic-foundry/datasets")
dataset_id=$(jq -r '.datasets[0].dataset_id // empty' <<<"$datasets_json")
if [[ -z "$dataset_id" ]]; then
  echo "FAIL: No datasets returned from /datasets." >&2
  exit 1
fi

dataset_spec=$(jq -c '.datasets[0]' <<<"$datasets_json")
source_table=$(jq -r '.source_table' <<<"$dataset_spec")
schema_name="public"
schemas_json=$(curl -fsS "${HUB_BASE_URL}/api/topic-foundry/introspect/schemas" || echo "")
if [[ -n "$schemas_json" ]]; then
  schema_name=$(jq -r '.schemas[0] // "public"' <<<"$schemas_json")
fi
if [[ "$source_table" != *.* ]]; then
  source_table="${schema_name}.${source_table}"
fi
qualified_dataset_spec=$(jq -c --arg source_table "$source_table" '.source_table=$source_table' <<<"$dataset_spec")

preview_payload=$(jq -c --argjson dataset "$qualified_dataset_spec" '{
  dataset: $dataset,
  windowing: {
    block_mode: "rows",
    segmentation_mode: "time_gap",
    time_gap_seconds: 900,
    max_window_seconds: 7200,
    min_blocks_per_segment: 1,
    max_chars: 1200
  },
  limit: 50
}' <<<"{}")

preview_status=$(curl -s -o /tmp/topic_preview.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$preview_payload" \
  "${HUB_BASE_URL}/api/topic-foundry/datasets/preview")
if [[ "$preview_status" != 2* ]]; then
  echo "FAIL: Preview request failed (status ${preview_status})." >&2
  cat /tmp/topic_preview.json >&2 || true
  exit 1
fi

model_payload=$(jq -c --arg dataset_id "$dataset_id" '{
  name: "smoke-e2e",
  version: "v1",
  stage: "development",
  dataset_id: $dataset_id,
  model_spec: {
    algorithm: "hdbscan",
    metric: "cosine",
    min_cluster_size: 5,
    params: {}
  },
  windowing_spec: {
    block_mode: "rows",
    segmentation_mode: "time_gap",
    time_gap_seconds: 900,
    max_window_seconds: 7200,
    min_blocks_per_segment: 1,
    max_chars: 1200
  },
  metadata: {}
}' <<<"{}")

model_status=$(curl -s -o /tmp/topic_model.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$model_payload" \
  "${HUB_BASE_URL}/api/topic-foundry/models")
if [[ "$model_status" != 2* ]]; then
  echo "FAIL: Model create failed (status ${model_status})." >&2
  cat /tmp/topic_model.json >&2 || true
  exit 1
fi
model_id=$(jq -r '.model_id // empty' /tmp/topic_model.json)
if [[ -z "$model_id" ]]; then
  echo "FAIL: Model response missing model_id." >&2
  cat /tmp/topic_model.json >&2 || true
  exit 1
fi

train_payload=$(jq -c --arg model_id "$model_id" --arg dataset_id "$dataset_id" '{
  model_id: $model_id,
  dataset_id: $dataset_id
}' <<<"{}")
train_status=$(curl -s -o /tmp/topic_train.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$train_payload" \
  "${HUB_BASE_URL}/api/topic-foundry/runs/train")
if [[ "$train_status" != 2* ]]; then
  echo "FAIL: Train request failed (status ${train_status})." >&2
  cat /tmp/topic_train.json >&2 || true
  exit 1
fi
run_id=$(jq -r '.run_id // empty' /tmp/topic_train.json)
if [[ -z "$run_id" ]]; then
  echo "FAIL: Train response missing run_id." >&2
  cat /tmp/topic_train.json >&2 || true
  exit 1
fi

run_status="queued"
for _ in {1..20}; do
  run_json=$(curl -fsS "${HUB_BASE_URL}/api/topic-foundry/runs/${run_id}")
  run_status=$(jq -r '.status // empty' <<<"$run_json")
  if [[ "$run_status" == "failed" ]]; then
    echo "FAIL: Run failed." >&2
    jq . <<<"$run_json" >&2 || true
    exit 1
  fi
  if [[ "$run_status" == "complete" ]]; then
    break
  fi
  sleep 3
done

segments_status=$(curl -s -o /tmp/topic_segments.json -w "%{http_code}" \
  "${HUB_BASE_URL}/api/topic-foundry/segments?run_id=${run_id}&limit=1&offset=0&format=wrapped")
if [[ "$segments_status" != 2* ]]; then
  echo "FAIL: Segments request failed (status ${segments_status})." >&2
  cat /tmp/topic_segments.json >&2 || true
  exit 1
fi
segment_id=$(jq -r '.items[0].segment_id // empty' /tmp/topic_segments.json)
if [[ -z "$segment_id" ]]; then
  echo "FAIL: No segments returned for run ${run_id}." >&2
  cat /tmp/topic_segments.json >&2 || true
  exit 1
fi

full_text_status=$(curl -s -o /tmp/topic_full_text.json -w "%{http_code}" \
  "${HUB_BASE_URL}/api/topic-foundry/segments/${segment_id}/full_text")
if [[ "$full_text_status" != 2* ]]; then
  echo "FAIL: Full text request failed (status ${full_text_status})." >&2
  cat /tmp/topic_full_text.json >&2 || true
  exit 1
fi

echo "Topic Studio E2E smoke checks passed."
