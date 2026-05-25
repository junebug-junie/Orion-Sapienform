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
