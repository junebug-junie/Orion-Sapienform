#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://localhost:8030}
DATASET_ID=${DATASET_ID:-}

if [[ -z "$DATASET_ID" ]]; then
  echo "DATASET_ID is required." >&2
  exit 1
fi

model_payload=$(cat <<JSON
{
  "name": "smoke-cosine",
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
    "block_mode": "turn_pairs",
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

model_status=$(curl -s -o /tmp/topic_foundry_cosine_model.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$model_payload" \
  "${BASE_URL}/models")

if [[ "$model_status" != 2* ]]; then
  echo "Unexpected status from /models: ${model_status}" >&2
  cat /tmp/topic_foundry_cosine_model.json >&2 || true
  exit 1
fi

model_id=$(jq -r '.model_id // empty' /tmp/topic_foundry_cosine_model.json)
if [[ -z "$model_id" ]]; then
  echo "Model creation did not return model_id." >&2
  cat /tmp/topic_foundry_cosine_model.json >&2 || true
  exit 1
fi

train_payload=$(cat <<JSON
{
  "model_id": "${model_id}",
  "dataset_id": "${DATASET_ID}"
}
JSON
)

train_status=$(curl -s -o /tmp/topic_foundry_cosine_train.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$train_payload" \
  "${BASE_URL}/runs/train")

if [[ "$train_status" != 2* ]]; then
  echo "Unexpected status from /runs/train: ${train_status}" >&2
  cat /tmp/topic_foundry_cosine_train.json >&2 || true
  exit 1
fi

run_id=$(jq -r '.run_id // empty' /tmp/topic_foundry_cosine_train.json)
if [[ -z "$run_id" ]]; then
  echo "Train response did not return run_id." >&2
  cat /tmp/topic_foundry_cosine_train.json >&2 || true
  exit 1
fi

timeout_secs=${TIMEOUT_SECS:-300}
sleep_secs=${SLEEP_SECS:-5}
deadline=$((SECONDS + timeout_secs))

while (( SECONDS < deadline )); do
  status=$(curl -s "${BASE_URL}/runs/${run_id}" | jq -r '.status // empty')
  error=$(curl -s "${BASE_URL}/runs/${run_id}" | jq -r '.error // empty')
  if [[ "$status" == "complete" ]]; then
    echo "Training completed."
    exit 0
  fi
  if [[ "$status" == "failed" ]]; then
    if [[ "$error" == *"Unrecognized metric"* ]]; then
      echo "Training failed due to unrecognized cosine metric." >&2
      exit 1
    fi
    echo "Training failed: ${error}" >&2
    exit 1
  fi
  sleep "$sleep_secs"
done

echo "Training did not complete within timeout." >&2
exit 1
