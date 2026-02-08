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

schemas=$(curl -fsS "${BASE_URL}/introspect/schemas")
schema=$(jq -r '.schemas[0] // empty' <<<"$schemas")
if [[ -z "$schema" ]]; then
  echo "FAIL: No schemas returned from /introspect/schemas" >&2
  exit 1
fi

tables=$(curl -fsS "${BASE_URL}/introspect/tables?schema=${schema}")
table=$(jq -r '.tables[0].table_name // empty' <<<"$tables")
if [[ -z "$table" ]]; then
  echo "FAIL: No tables returned for schema ${schema}" >&2
  exit 1
fi

columns=$(curl -fsS "${BASE_URL}/introspect/columns?schema=${schema}&table=${table}")
id_column=$(jq -r '.columns[] | select(.udt_name=="uuid" or (.data_type|test("int"))) | .column_name' <<<"$columns" | head -n 1)
time_column=$(jq -r '.columns[] | select(.data_type|test("timestamp|date|time")) | .column_name' <<<"$columns" | head -n 1)
text_columns=$(jq -r '[.columns[] | select(.data_type|test("char|text|varchar")) | .column_name] | .[:2]' <<<"$columns")
boundary_column=$(jq -r '.columns[0].column_name // empty' <<<"$columns")

if [[ -z "$id_column" || -z "$time_column" || "$text_columns" == "[]" || -z "$boundary_column" ]]; then
  echo "FAIL: Unable to derive id/time/text/boundary columns from ${schema}.${table}" >&2
  exit 1
fi

create_payload=$(jq -c --arg name "smoke-dataset-$(date +%s)" \
  --arg source_table "${schema}.${table}" \
  --arg id_column "$id_column" \
  --arg time_column "$time_column" \
  --argjson text_columns "$text_columns" \
  '{
    name: $name,
    source_table: $source_table,
    id_column: $id_column,
    time_column: $time_column,
    text_columns: $text_columns,
    timezone: "UTC"
  }' <<<"{}")

create_status=$(curl -s -o /tmp/topic_foundry_dataset_create.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$create_payload" \
  "${BASE_URL}/datasets")
if [[ "$create_status" != 2* ]]; then
  echo "FAIL: Dataset create failed (status ${create_status})" >&2
  cat /tmp/topic_foundry_dataset_create.json >&2 || true
  exit 1
fi

dataset_id=$(jq -r '.dataset_id // empty' /tmp/topic_foundry_dataset_create.json)
if [[ -z "$dataset_id" ]]; then
  echo "FAIL: Dataset create response missing dataset_id" >&2
  exit 1
fi

patch_payload=$(jq -c --arg boundary_column "$boundary_column" '{"boundary_column": $boundary_column, "boundary_strategy": "column"}' <<<"{}")
patch_status=$(curl -s -o /tmp/topic_foundry_dataset_patch.json -w "%{http_code}" \
  -X PATCH \
  -H "Content-Type: application/json" \
  -d "$patch_payload" \
  "${BASE_URL}/datasets/${dataset_id}")
if [[ "$patch_status" != 2* ]]; then
  echo "FAIL: Dataset patch failed (status ${patch_status})" >&2
  cat /tmp/topic_foundry_dataset_patch.json >&2 || true
  exit 1
fi

updated_boundary=$(jq -r '.boundary_column // empty' /tmp/topic_foundry_dataset_patch.json)
if [[ -z "$updated_boundary" ]]; then
  echo "FAIL: Dataset patch response missing boundary_column" >&2
  cat /tmp/topic_foundry_dataset_patch.json >&2 || true
  exit 1
fi

echo "Topic Foundry dataset create/update smoke checks passed."
