#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8615}
TOPIC_FOUNDRY_BASE_URL=${TOPIC_FOUNDRY_BASE_URL:-"http://localhost:${PORT}"}
START_AT=${START_AT:-"$(date -u -d '30 days ago' +'%Y-%m-%dT%H:%M:%SZ')"}
END_AT=${END_AT:-"$(date -u +'%Y-%m-%dT%H:%M:%SZ')"}
TIMEOUT_SECS=${TIMEOUT_SECS:-600}
SLEEP_SECS=${SLEEP_SECS:-2}
MAX_CHARS=${MAX_CHARS:-6000}

SOURCE_TABLE=${SOURCE_TABLE:-chat_history_log}
ID_COLUMN=${ID_COLUMN:-chat_id}
TIME_COLUMN=${TIME_COLUMN:-created_at}
TEXT_COLUMNS=${TEXT_COLUMNS:-prompt,response}
EMBEDDING_URL=${EMBEDDING_URL:-"http://orion-vector-host:8320/embedding"}
MODEL_NAME=${MODEL_NAME:-topic-foundry-llm}
MODEL_VERSION=${MODEL_VERSION:-v1}

DATASET_ID=${DATASET_ID:-""}
MODEL_ID=${MODEL_ID:-""}

if [[ "${TOPIC_FOUNDRY_LLM_ENABLE:-false}" != "true" ]]; then
  echo "LLM boundary judge disabled; skipping." >&2
  exit 0
fi

if [[ "${TOPIC_FOUNDRY_LLM_USE_BUS:-true}" == "true" ]]; then
  echo "LLM boundary judge running via bus; ensure ORION_BUS_URL is reachable." >&2
fi

if [[ -z "$DATASET_ID" ]]; then
  dataset_payload=$(jq -n \
    --arg name "${MODEL_NAME}-dataset" \
    --arg source_table "$SOURCE_TABLE" \
    --arg id_column "$ID_COLUMN" \
    --arg time_column "$TIME_COLUMN" \
    --arg text_columns "$TEXT_COLUMNS" \
    '{
      name: $name,
      source_table: $source_table,
      id_column: $id_column,
      time_column: $time_column,
      text_columns: ($text_columns | split(",")),
      where_sql: null,
      where_params: null,
      timezone: "UTC"
    }')

  dataset_resp=$(curl -fsS -X POST "${TOPIC_FOUNDRY_BASE_URL}/datasets" \
    -H 'Content-Type: application/json' \
    -d "$dataset_payload")
  DATASET_ID=$(echo "$dataset_resp" | jq -r '.dataset_id')
fi

if [[ -z "$MODEL_ID" ]]; then
  model_payload=$(jq -n \
    --arg name "$MODEL_NAME" \
    --arg version "$MODEL_VERSION" \
    --arg dataset_id "$DATASET_ID" \
    --arg embedding_url "$EMBEDDING_URL" \
    --argjson max_chars "$MAX_CHARS" \
    '{
      name: $name,
      version: $version,
      stage: "development",
      dataset_id: $dataset_id,
      model_spec: {
        algorithm: "hdbscan",
        embedding_source_url: $embedding_url,
        min_cluster_size: 15,
        metric: "cosine",
        params: {}
      },
      windowing_spec: {
        block_mode: "turn_pairs",
        include_roles: ["user", "assistant"],
        segmentation_mode: "hybrid_llm",
        semantic_split_threshold: 0.75,
        confirm_edges_k: 2,
        smoothing_window: 3,
        llm_boundary_context_blocks: 3,
        llm_boundary_max_chars: 4000,
        llm_candidate_top_k: 200,
        llm_candidate_strategy: "semantic_low_sim",
        llm_candidate_threshold: 0.75,
        time_gap_seconds: 900,
        max_window_seconds: 7200,
        min_blocks_per_segment: 1,
        max_chars: $max_chars
      },
      metadata: {}
    }')

  model_resp=$(curl -fsS -X POST "${TOPIC_FOUNDRY_BASE_URL}/models" \
    -H 'Content-Type: application/json' \
    -d "$model_payload")
  MODEL_ID=$(echo "$model_resp" | jq -r '.model_id')
fi

run_payload=$(jq -n \
  --arg model_id "$MODEL_ID" \
  --arg dataset_id "$DATASET_ID" \
  --arg start_at "$START_AT" \
  --arg end_at "$END_AT" \
  '{
    model_id: $model_id,
    dataset_id: $dataset_id,
    start_at: $start_at,
    end_at: $end_at
  }')

run_resp=$(curl -fsS -X POST "${TOPIC_FOUNDRY_BASE_URL}/runs/train" \
  -H 'Content-Type: application/json' \
  -d "$run_payload")

RUN_ID=$(echo "$run_resp" | jq -r '.run_id')

start_ts=$(date +%s)
while true; do
  run_status_resp=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/runs/${RUN_ID}")
  status=$(echo "$run_status_resp" | jq -r '.status')
  if [[ "$status" == "complete" ]]; then
    break
  fi
  if [[ "$status" == "failed" ]]; then
    echo "Run failed: ${run_status_resp}" >&2
    exit 1
  fi
  now_ts=$(date +%s)
  elapsed=$((now_ts - start_ts))
  if [[ "$elapsed" -ge "$TIMEOUT_SECS" ]]; then
    echo "Run timed out after ${TIMEOUT_SECS}s" >&2
    exit 1
  fi
  sleep "$SLEEP_SECS"
done

run_dir=$(echo "$run_status_resp" | jq -r '.artifact_paths.run_dir')
if [[ -z "$run_dir" || "$run_dir" == "null" ]]; then
  echo "run_dir not available in artifact_paths" >&2
  exit 1
fi

if [[ ! -f "${run_dir}/boundary_judgements.jsonl" ]]; then
  echo "boundary_judgements.jsonl not found at ${run_dir}" >&2
  exit 1
fi

echo "Boundary judgements found: ${run_dir}/boundary_judgements.jsonl"
