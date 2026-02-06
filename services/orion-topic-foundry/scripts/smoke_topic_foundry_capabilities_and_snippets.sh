#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8615}
TOPIC_FOUNDRY_BASE_URL=${TOPIC_FOUNDRY_BASE_URL:-"http://localhost:${PORT}"}
RUN_ID=${RUN_ID:-""}

capabilities=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/capabilities")
echo "$capabilities" | jq

echo "$capabilities" | jq -e '
  .service
  and .version
  and .node
  and (.llm_enabled == true or .llm_enabled == false)
  and (.segmentation_modes_supported | type == "array" and length > 0)
  and (.enricher_modes_supported | type == "array" and length > 0)
  and (.defaults | type == "object")
' >/dev/null

if [[ -z "$RUN_ID" ]]; then
  RUN_ID=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/runs?limit=1" | jq -r '.runs[0].run_id // empty')
fi

if [[ -z "$RUN_ID" ]]; then
  echo "No runs found; skipping snippet assertion."
  exit 0
fi

segments=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/segments?run_id=${RUN_ID}&include_snippet=true")
echo "$segments" | jq

segment_count=$(echo "$segments" | jq '.segments | length')
if [[ "$segment_count" -eq 0 ]]; then
  echo "No segments found for run ${RUN_ID}; skipping snippet assertion."
  exit 0
fi

echo "$segments" | jq -e 'any(.segments[]; (.snippet // "") | length > 0)' >/dev/null
