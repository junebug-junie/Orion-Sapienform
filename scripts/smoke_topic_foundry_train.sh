#!/usr/bin/env bash
set -euo pipefail

resolve_base_url() {
  local cli_base_url="${1:-}"
  if [[ -n "$cli_base_url" ]]; then
    echo "$cli_base_url"
    return
  fi
  if [[ -n "${TOPIC_FOUNDRY_BASE_URL:-}" ]]; then
    echo "$TOPIC_FOUNDRY_BASE_URL"
    return
  fi
  if [[ -n "${HUB_BASE_URL:-}" ]]; then
    echo "${HUB_BASE_URL%/}/api/topic-foundry"
    return
  fi
  echo "http://127.0.0.1:8080/api/topic-foundry"
}

BASE_URL="$(resolve_base_url "${1:-}")"
BASE_URL="${BASE_URL%/}"

datasets=$(curl -fsS "${BASE_URL}/datasets")
dataset=$(jq -c '.datasets[0] // empty' <<<"$datasets")
if [[ -z "$dataset" || "$dataset" == "null" ]]; then
  echo "FAIL: No datasets returned from /datasets" >&2
  exit 1
fi

source_table=$(jq -r '.source_table' <<<"$dataset")
schema="public"
table="$source_table"
if [[ "$source_table" == *.* ]]; then
  schema="${source_table%%.*}"
  table="${source_table##*.}"
fi

columns=$(curl -fsS "${BASE_URL}/introspect/columns?schema=${schema}&table=${table}")
boundary_column=$(jq -r '.columns[] | select((.data_type | test("char|text|int")) or (.udt_name=="uuid")) | .column_name' <<<"$columns" | head -n 1)
if [[ -z "$boundary_column" ]]; then
  echo "FAIL: No boundary column candidates found for ${schema}.${table}" >&2
  exit 1
fi

micro_dataset_payload=$(jq -c --arg boundary_column "$boundary_column" '{
  name: ("smoke-boundary-" + (now|tostring)),
  source_table: .source_table,
  id_column: .id_column,
  time_column: .time_column,
  text_columns: .text_columns,
  boundary_column: $boundary_column,
  boundary_strategy: "column",
  timezone: .timezone
}' <<<"$dataset")

micro_dataset_status=$(curl -s -o /tmp/topic_foundry_dataset.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$micro_dataset_payload" \
  "${BASE_URL}/datasets")
if [[ "$micro_dataset_status" != 2* ]]; then
  echo "FAIL: Boundary dataset create failed (status ${micro_dataset_status})" >&2
  cat /tmp/topic_foundry_dataset.json >&2 || true
  exit 1
fi
micro_dataset_id=$(jq -r '.dataset_id // empty' /tmp/topic_foundry_dataset.json)
if [[ -z "$micro_dataset_id" ]]; then
  echo "FAIL: Boundary dataset response missing dataset_id" >&2
  cat /tmp/topic_foundry_dataset.json >&2 || true
  exit 1
fi

macro_model_payload=$(jq -c --arg dataset_id "$(jq -r '.dataset_id' <<<"$dataset")" '{
  name: ("smoke-macro-" + (now|tostring)),
  version: "v1",
  stage: "development",
  dataset_id: $dataset_id,
  model_spec: { algorithm: "hdbscan", metric: "cosine", min_cluster_size: 5, params: {} },
  windowing_spec: { windowing_mode: "time_gap", segmentation_mode: "time_gap", time_gap_seconds: 900, max_window_seconds: 7200, min_blocks_per_segment: 1, max_chars: 1200 },
  metadata: {}
}' <<<"{}")

macro_model_status=$(curl -s -o /tmp/topic_foundry_macro_model.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$macro_model_payload" \
  "${BASE_URL}/models")
if [[ "$macro_model_status" != 2* ]]; then
  echo "FAIL: Macro model create failed (status ${macro_model_status})" >&2
  cat /tmp/topic_foundry_macro_model.json >&2 || true
  exit 1
fi
macro_model_id=$(jq -r '.model_id // empty' /tmp/topic_foundry_macro_model.json)

micro_model_payload=$(jq -c --arg dataset_id "$micro_dataset_id" '{
  name: ("smoke-micro-" + (now|tostring)),
  version: "v1",
  stage: "development",
  dataset_id: $dataset_id,
  model_spec: { algorithm: "hdbscan", metric: "cosine", min_cluster_size: 5, params: {} },
  windowing_spec: { windowing_mode: "conversation_bound_then_time_gap", segmentation_mode: "time_gap", time_gap_seconds: 900, max_window_seconds: 7200, min_blocks_per_segment: 1, max_chars: 1200 },
  metadata: {}
}' <<<"{}")

micro_model_status=$(curl -s -o /tmp/topic_foundry_micro_model.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$micro_model_payload" \
  "${BASE_URL}/models")
if [[ "$micro_model_status" != 2* ]]; then
  echo "FAIL: Micro model create failed (status ${micro_model_status})" >&2
  cat /tmp/topic_foundry_micro_model.json >&2 || true
  exit 1
fi
micro_model_id=$(jq -r '.model_id // empty' /tmp/topic_foundry_micro_model.json)

macro_train_status=$(curl -s -o /tmp/topic_foundry_macro_train.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "{\"model_id\":\"${macro_model_id}\",\"dataset_id\":\"$(jq -r '.dataset_id' <<<"$dataset")\",\"run_scope\":\"macro\"}" \
  "${BASE_URL}/runs/train")
if [[ "$macro_train_status" != 2* ]]; then
  echo "FAIL: Macro train failed (status ${macro_train_status})" >&2
  cat /tmp/topic_foundry_macro_train.json >&2 || true
  exit 1
fi

micro_train_status=$(curl -s -o /tmp/topic_foundry_micro_train.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "{\"model_id\":\"${micro_model_id}\",\"dataset_id\":\"${micro_dataset_id}\",\"run_scope\":\"micro\"}" \
  "${BASE_URL}/runs/train")
if [[ "$micro_train_status" != 2* ]]; then
  echo "FAIL: Micro train failed (status ${micro_train_status})" >&2
  cat /tmp/topic_foundry_micro_train.json >&2 || true
  exit 1
fi

echo "Topic Foundry macro + micro train smoke checks passed."
