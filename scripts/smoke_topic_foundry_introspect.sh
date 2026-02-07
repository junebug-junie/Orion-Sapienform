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
column=$(jq -r '.columns[0].column_name // empty' <<<"$columns")
if [[ -z "$column" ]]; then
  echo "FAIL: No columns returned for ${schema}.${table}" >&2
  exit 1
fi

fingerprint=$(curl -fsS "${BASE_URL}/introspect/table_fingerprint?schema=${schema}&table=${table}")
exists=$(jq -r '.exists' <<<"$fingerprint")
if [[ "$exists" != "true" ]]; then
  echo "FAIL: Table fingerprint indicates ${schema}.${table} missing" >&2
  exit 1
fi

echo "Topic Foundry introspection smoke checks passed."
