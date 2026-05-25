#!/usr/bin/env bash
set -euo pipefail

HUB_PORT="${HUB_PORT:-8080}"
BASE="http://localhost:${HUB_PORT}"

echo "1. Check biometrics grammar events (manual):"
echo "   redis-cli --raw SUBSCRIBE orion:grammar:event"
echo ""

echo "2. Query latest node biometrics projection chain..."
curl -s "${BASE}/api/substrate/biometrics-node/atlas/latest" | jq .

echo ""
echo "3. Query active node pressure projection..."
curl -s "${BASE}/api/substrate/node-pressure/latest" | jq .

echo ""
echo "4. Verify latest chain contains emission + receipt labels..."
curl -s "${BASE}/api/substrate/biometrics-node/atlas/latest" | jq '.event_chain'
