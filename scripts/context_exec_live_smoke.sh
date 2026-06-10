#!/usr/bin/env bash
# Live-stack smoke for orion-context-exec (#661).
# Requires: context-exec on :8096, optional cortex-exec with CONTEXT_EXEC_ENABLED=true.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CTX_PORT="${CONTEXT_EXEC_PORT:-8096}"
CTX_BASE="http://127.0.0.1:${CTX_PORT}"

echo "== context-exec health =="
health="$(curl -sf "${CTX_BASE}/health")"
echo "$health" | python3 -m json.tool
echo "$health" | python3 -c "
import json,sys
d=json.load(sys.stdin)
assert d.get('ok') is True, d
assert d.get('service')=='orion-context-exec', d
assert d.get('write_enabled') is False, d
assert d.get('max_depth')==1, d
print('health ok')
"

echo "== direct belief_provenance run =="
curl -sf "${CTX_BASE}/context-exec/run" \
  -H 'content-type: application/json' \
  -d '{
    "text": "Where did Orion get the claim that I am from Denver?",
    "mode": "belief_provenance",
    "expected_artifact_type": "BeliefProvenanceReportV1"
  }' | python3 -m json.tool | head -40

echo "== HTTP agent-chain compat (no bus alias) =="
curl -sf "${CTX_BASE}/agent/chain/run" \
  -H 'content-type: application/json' \
  -d '{"text":"Why did corr 99 fail open?","mode":"agent"}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); assert d.get('text'); assert 'context_exec' in (d.get('structured') or {}); print('compat ok')"

echo "== compat alias check (must be false in .env) =="
grep -E '^CONTEXT_EXEC_COMPAT_AGENT_CHAIN_ENABLED=' services/orion-context-exec/.env \
  | grep -E '=false$' >/dev/null \
  || { echo "FAIL: set CONTEXT_EXEC_COMPAT_AGENT_CHAIN_ENABLED=false in services/orion-context-exec/.env"; exit 1; }
echo "compat alias disabled ok"

echo "SMOKE PASS"
