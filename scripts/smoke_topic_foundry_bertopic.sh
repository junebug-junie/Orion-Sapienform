#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${TOPIC_FOUNDRY_BASE_URL:-http://localhost:8615}"
DATASET_ID="${1:-${TOPIC_FOUNDRY_DATASET_ID:-}}"
if [[ -z "$DATASET_ID" ]]; then
  echo "usage: $0 <dataset_id>"
  exit 1
fi

MODEL_NAME="smoke-bertopic-$(date +%s)"
MODEL_VERSION="v1"

create_model_payload=$(cat <<JSON
{
  "name": "$MODEL_NAME",
  "version": "$MODEL_VERSION",
  "stage": "candidate",
  "dataset_id": "$DATASET_ID",
  "model_spec": {
    "algorithm": "hdbscan",
    "metric": "cosine",
    "model_meta": {
      "engine": "bertopic",
      "embedding_backend": "vector_host",
      "reducer": "umap",
      "clusterer": "hdbscan",
      "representation": "ctfidf"
    }
  },
  "windowing_spec": {"windowing_mode": "time_gap"},
  "metadata": {}
}
JSON
)

model_json=$(curl -fsS -X POST "$BASE_URL/models" -H 'content-type: application/json' -d "$create_model_payload")
model_id=$(echo "$model_json" | jq -r '.model_id')

echo "created model_id=$model_id"

run_train() {
  local mode="$1"
  local params="$2"
  curl -fsS -X POST "$BASE_URL/runs/train" -H 'content-type: application/json' -d "{\"model_id\":\"$model_id\",\"dataset_id\":\"$DATASET_ID\",\"topic_mode\":\"$mode\",\"topic_mode_params\":$params}" | jq -r '.run_id'
}

wait_run() {
  local run_id="$1"
  for _ in $(seq 1 120); do
    status=$(curl -fsS "$BASE_URL/runs/$run_id" | jq -r '.status')
    [[ "$status" == "complete" ]] && return 0
    [[ "$status" == "failed" ]] && return 1
    sleep 2
  done
  return 1
}

standard_run=$(run_train standard '{}')
wait_run "$standard_run"
curl -fsS "$BASE_URL/runs/$standard_run" | jq '{run_id,doc_count,cluster_count,outlier_rate,artifacts}'

guided_run=$(run_train guided '{"seed_topic_list":[["gpu","cuda","v100"],["yard","xeriscape","pergola"]]}')
wait_run "$guided_run"

zeroshot_run=$(run_train zeroshot '{"zeroshot_topic_list":["Orion architecture","home lab hardware","family logistics"]}')
wait_run "$zeroshot_run"

echo "smoke bertopic complete"
