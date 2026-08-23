#!/usr/bin/env bash
set -euo pipefail
HUB_PORT="${HUB_PORT:-8080}"
BASE="http://localhost:${HUB_PORT}"

echo "1. Biometrics substrate chain (prerequisite):"
curl -s "${BASE}/api/substrate/biometrics-node/atlas/latest" | jq '.active_node_pressure_projection'

echo "2. Latest field state:"
curl -s "${BASE}/api/substrate/field/latest" | jq .

echo "3. Atlas node field:"
curl -s "${BASE}/api/substrate/field/node/atlas" | jq .

echo "4. LLM inference capability field:"
curl -s "${BASE}/api/substrate/field/capability/llm_inference" | jq .

echo "5. Athena cabinet sensor channels (expect keys present; values should move with Nano, not ratchet up):"
curl -s "${BASE}/api/substrate/field/node/athena" | jq '{
  cabinet_climate_activity,
  cabinet_particulate_activity,
  cabinet_em_activity,
  cabinet_uv_activity,
  cabinet_vibration_activity,
  cabinet_proximity_activity,
  cabinet_sensor_staleness
}'
