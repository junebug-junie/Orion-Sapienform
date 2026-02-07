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

echo "Using BASE_URL=${BASE_URL}"

scripts=(
  "scripts/smoke_topic_foundry_introspect.sh"
  "scripts/smoke_topic_foundry_preview.sh"
  "scripts/smoke_topic_foundry_train.sh"
  "scripts/smoke_topic_foundry_facets.sh"
  "scripts/smoke_topic_foundry_enrich.sh"
)

for script in "${scripts[@]}"; do
  echo "Running ${script}..."
  bash "${script}" "${BASE_URL}"
done

echo "Topic Foundry full smoke chain passed."
