#!/usr/bin/env bash
set -euo pipefail

HUB_URL="${HUB_URL:-http://localhost:8080}"
TIMEOUT="${TIMEOUT:-25}"
FAIL=0

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAIL=1; }

corr_id="smoke-agent-chain-$(date +%s)"
payload=$(cat <<JSON
{
  "messages": [{"role":"user","content":"Please delegate this to agent chain and then summarize."}],
  "mode": "agent",
  "use_recall": false,
  "request_id": "${corr_id}",
  "options": {"allowed_verbs": ["agent_chain"]}
}
JSON
)

echo "[1/2] Trigger supervised request corr_id=${corr_id}"
resp="$(curl -fsS --max-time "$TIMEOUT" -H 'content-type: application/json' -d "$payload" "$HUB_URL/api/chat")" || { fail "Hub /api/chat request failed"; resp=''; }
if [[ -n "$resp" ]]; then
  pass "Hub request completed"
fi

echo "[2/2] Verify AgentChainService RPC emit in docker logs"
if docker logs "${PROJECT:-orion}-cortex-exec" 2>&1 | rg -q "RPC emit -> .*AgentChainService.*${corr_id}"; then
  pass "Detected AgentChainService RPC emit for corr_id=${corr_id}"
else
  fail "No AgentChainService RPC emit found for corr_id=${corr_id}"
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo "SMOKE RESULT: FAIL"
  exit 1
fi

echo "SMOKE RESULT: PASS"
