#!/usr/bin/env bash
set -euo pipefail

HUB_BASE="${HUB_BASE:-http://localhost:8080}"
AUTH_TOKEN="${SMOKE_TOKEN:-${TOPIC_FOUNDRY_TOKEN:-}}"
EXPECT_LLM="${SMOKE_EXPECT_LLM:-0}"

READY_URL="${HUB_BASE%/}/api/topic-foundry/ready"
CAP_URL="${HUB_BASE%/}/api/topic-foundry/capabilities"

hdrs=(-H "accept: application/json")
if [[ -n "$AUTH_TOKEN" ]]; then
  hdrs+=( -H "authorization: Bearer ${AUTH_TOKEN}" )
fi

echo "[contract] hub=${HUB_BASE}"
echo "[contract] GET ${READY_URL}"
ready_json="$(curl -fsS "${hdrs[@]}" "$READY_URL")"

echo "[contract] GET ${CAP_URL}"
cap_json="$(curl -fsS "${hdrs[@]}" "$CAP_URL")"

jq -e '.ok | type == "boolean"' <<<"$ready_json" >/dev/null
jq -e '.checks | type == "object"' <<<"$ready_json" >/dev/null

jq -e '.service == "topic-foundry" or .service == "orion-topic-foundry"' <<<"$cap_json" >/dev/null
jq -e '.segmentation_modes_supported | type == "array"' <<<"$cap_json" >/dev/null
jq -e '.supported_metrics | type == "array"' <<<"$cap_json" >/dev/null
jq -e '.llm_enabled | type == "boolean"' <<<"$cap_json" >/dev/null

if [[ "$EXPECT_LLM" == "1" ]]; then
  jq -e '.llm_enabled == true' <<<"$cap_json" >/dev/null
  jq -e '.llm_transport == "bus"' <<<"$cap_json" >/dev/null
  jq -e '(.llm_bus_route // "") | length > 0' <<<"$cap_json" >/dev/null
fi

jq -r '
  {
    service,
    llm_enabled,
    llm_transport,
    llm_bus_route,
    segmentation_modes: (.segmentation_modes_supported | length),
    metrics: (.supported_metrics | length)
  }
' <<<"$cap_json"

echo "[contract] PASS"
