#!/usr/bin/env bash
set -euo pipefail

HUB_BASE_URL=${HUB_BASE_URL:-http://localhost:8080}
RUN_ID=${RUN_ID:-}
GROUP_BY_COLUMN=${GROUP_BY_COLUMN:-}
DATASET_ID=${DATASET_ID:-}

if ! html=$(curl -fsS "${HUB_BASE_URL}/"); then
  echo "SKIP: Hub not reachable at ${HUB_BASE_URL}."
  exit 0
fi
if ! printf '%s' "$html" | rg -q "hubTabButton"; then
  echo "Header nav not found in hub HTML." >&2
  exit 1
fi
if ! printf '%s' "$html" | rg -q "data-split-pane"; then
  echo "Topic Studio split pane not found in hub HTML." >&2
  exit 1
fi

if [[ -z "$RUN_ID" ]]; then
  echo "SKIP: RUN_ID not set for segments/full_text check."
  exit 0
fi

segments_url="${HUB_BASE_URL}/api/topic-foundry/segments?run_id=${RUN_ID}&include_snippet=true&include_bounds=true&limit=1&offset=0&format=wrapped&sort_by=created_at&sort_dir=desc"
seg_status=$(curl -s -o /tmp/hub_segments.json -w "%{http_code}" "$segments_url")
if [[ "$seg_status" != 2* ]]; then
  echo "Unexpected status from segments: ${seg_status}" >&2
  cat /tmp/hub_segments.json >&2 || true
  exit 1
fi
segment_id=$(jq -r '.items[0].segment_id // empty' /tmp/hub_segments.json)
if [[ -z "$segment_id" ]]; then
  echo "No segment_id returned from segments endpoint." >&2
  cat /tmp/hub_segments.json >&2 || true
  exit 1
fi

full_text_status=$(curl -s -o /tmp/hub_segment_full_text.json -w "%{http_code}" \
  "${HUB_BASE_URL}/api/topic-foundry/segments/${segment_id}/full_text")
if [[ "$full_text_status" != 2* ]]; then
  echo "Unexpected status from full_text: ${full_text_status}" >&2
  cat /tmp/hub_segment_full_text.json >&2 || true
  exit 1
fi

if [[ -z "$DATASET_ID" || -z "$GROUP_BY_COLUMN" ]]; then
  echo "SKIP: DATASET_ID or GROUP_BY_COLUMN not set for group_by windowing check."
  exit 0
fi

model_payload=$(cat <<JSON
{
  "name": "smoke-group-by",
  "version": "v1",
  "stage": "development",
  "dataset_id": "${DATASET_ID}",
  "model_spec": {
    "algorithm": "hdbscan",
    "metric": "cosine",
    "min_cluster_size": 5,
    "params": {}
  },
  "windowing_spec": {
    "block_mode": "group_by_column",
    "group_by": "${GROUP_BY_COLUMN}",
    "segmentation_mode": "time_gap",
    "time_gap_seconds": 900,
    "max_window_seconds": 7200,
    "min_blocks_per_segment": 1,
    "max_chars": 6000
  },
  "metadata": {}
}
JSON
)

model_status=$(curl -s -o /tmp/hub_group_by_model.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$model_payload" \
  "${HUB_BASE_URL}/api/topic-foundry/models")
if [[ "$model_status" != 2* ]]; then
  echo "Unexpected status from /models (group_by): ${model_status}" >&2
  cat /tmp/hub_group_by_model.json >&2 || true
  exit 1
fi
model_id=$(jq -r '.model_id // empty' /tmp/hub_group_by_model.json)
if [[ -z "$model_id" ]]; then
  echo "No model_id returned from group_by model create." >&2
  cat /tmp/hub_group_by_model.json >&2 || true
  exit 1
fi

train_payload=$(cat <<JSON
{
  "model_id": "${model_id}",
  "dataset_id": "${DATASET_ID}"
}
JSON
)

train_status=$(curl -s -o /tmp/hub_group_by_train.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$train_payload" \
  "${HUB_BASE_URL}/api/topic-foundry/runs/train")
if [[ "$train_status" != 2* ]]; then
  echo "Unexpected status from /runs/train (group_by): ${train_status}" >&2
  cat /tmp/hub_group_by_train.json >&2 || true
  exit 1
fi

echo "Topic Studio split pane + full_text + group_by windowing checks passed."
