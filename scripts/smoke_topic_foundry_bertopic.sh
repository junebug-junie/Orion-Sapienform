#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${TOPIC_FOUNDRY_BASE_URL:-http://localhost:8615}"
NOW="$(date -u +%Y%m%d%H%M%S)"

require_jq() { command -v jq >/dev/null 2>&1 || { echo "jq required"; exit 1; }; }
require_jq

echo "[smoke] capabilities"
cap="$(curl -fsS "$BASE_URL/capabilities")"
echo "$cap" | jq -e '.capabilities.topic_modeling and .backends.embedding_backends and .backends.reducers and .backends.clusterers and .backends.representations' >/dev/null

dataset_id="${TOPIC_FOUNDRY_DATASET_ID:-}"
if [[ -z "$dataset_id" ]]; then
  echo "Set TOPIC_FOUNDRY_DATASET_ID to run full smoke." >&2
  exit 2
fi

model_payload="$(jq -n --arg name "smoke-bertopic-$NOW" --arg version "$NOW" --arg ds "$dataset_id" '{name:$name,version:$version,stage:"development",dataset_id:$ds,model_spec:{algorithm:"hdbscan",min_cluster_size:5,metric:"cosine",params:{}},windowing_spec:{windowing_mode:"document",time_gap_minutes:15,max_chars:6000},model_meta:{topic_engine:"bertopic",embedding_backend:"vector_host",embedding_model:"sentence-transformers/all-MiniLM-L6-v2",reducer:"umap",clusterer:"hdbscan",representation:"ctfidf"},metadata:{}}')"
model_res="$(curl -fsS -X POST "$BASE_URL/models" -H 'Content-Type: application/json' -d "$model_payload")"
model_id="$(echo "$model_res" | jq -r '.model_id')"

run_train () {
  local mode="$1"; shift
  local mode_params="$1"; shift
  local payload
  payload="$(jq -n --arg mid "$model_id" --arg ds "$dataset_id" --arg mode "$mode" --argjson mode_params "$mode_params" '{model_id:$mid,dataset_id:$ds,topic_mode:$mode,topic_mode_params:$mode_params}')"
  local run_res run_id final
  run_res="$(curl -fsS -X POST "$BASE_URL/runs/train" -H 'Content-Type: application/json' -d "$payload")"
  run_id="$(echo "$run_res" | jq -r '.run_id')"
  for _ in {1..60}; do
    final="$(curl -fsS "$BASE_URL/runs/$run_id")"
    status="$(echo "$final" | jq -r '.status')"
    [[ "$status" == "complete" || "$status" == "failed" ]] && break
    sleep 2
  done
  echo "$final" | jq -e '.status == "complete" and .stats.doc_count > 0 and .stats.cluster_count >= 0 and .stats.outlier_rate >= 0 and .stats.outlier_rate <= 1 and .stats.topic_mode != null and (.artifact_paths|type=="object")' >/dev/null
  echo "[smoke] mode=$mode assertions passed run_id=$run_id"
}

run_train standard '{"nonce":"first"}'
run_train standard '{"nonce":"second"}'
run_train guided '{"seed_topic_list":["support","billing"]}'
run_train zeroshot '{"zeroshot_topic_list":["support","billing"]}'

echo "[smoke] all assertions passed"
