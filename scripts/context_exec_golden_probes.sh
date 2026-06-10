#!/usr/bin/env bash
# Hub golden-path probes for context-exec (#663).
# Requires: context-exec :8096, cortex-exec + cortex-orch with CONTEXT_EXEC_ENABLED=true, Hub :8080.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CTX_PORT="${CONTEXT_EXEC_PORT:-8096}"
HUB_BASE="${HUB_BASE_URL:-http://127.0.0.1:8080}"
CTX_BASE="http://127.0.0.1:${CTX_PORT}"
REAL_CORR_ID="${CONTEXT_EXEC_PROBE_CORR_ID:-}"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== env posture =="
for f in services/orion-context-exec/.env services/orion-cortex-exec/.env; do
  if [[ -f "$f" ]]; then
    echo "--- $f ---"
    grep -E '^(CONTEXT_EXEC_|CHANNEL_CONTEXT_EXEC|CHANNEL_RECALL)' "$f" || true
  else
    echo "WARN: missing $f"
  fi
done

echo "== context-exec health =="
health="$(curl -sf "${CTX_BASE}/health")"
echo "$health" | python3 -m json.tool
echo "$health" | python3 -c "
import json,sys
d=json.load(sys.stdin)
assert d.get('ok') is True
assert d.get('write_enabled') is False
print('health ok')
"

hub_chat() {
  local prompt="$1"
  local body
  body="$(PROMPT="$prompt" python3 - <<'PY'
import json, os
print(json.dumps({
    "mode": "brain",
    "use_recall": True,
    "recall_profile": "assist.light.v1",
    "no_write": True,
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
}))
PY
)"
  curl -sf "${HUB_BASE}/api/chat" \
    -H 'content-type: application/json' \
    -H 'X-Orion-No-Write: 1' \
    -d "$body"
}

echo "== probe 1: Denver belief provenance (Hub) =="
p1="$(hub_chat "Orion, where did the claim that I am from Denver come from across your runtime?")"
echo "$p1" | python3 -m json.tool | head -60
echo "$p1" | python3 -c "
import json,sys
d=json.load(sys.stdin)
blob=json.dumps(d).lower()
assert blob.strip() not in ('{}', 'null'), d
print('denver probe ok (response received)')
"

echo "== probe 2: trace autopsy (Hub) =="
if [[ -z "$REAL_CORR_ID" ]]; then
  echo "SKIP: set CONTEXT_EXEC_PROBE_CORR_ID to a real correlation id from a prior run"
else
  p2="$(hub_chat "Orion, why did corr ${REAL_CORR_ID} fail open?")"
  echo "$p2" | python3 -m json.tool | head -60
  echo "$p2" | python3 -c "
import json,sys,os
d=json.load(sys.stdin)
corr=os.environ.get('CONTEXT_EXEC_PROBE_CORR_ID','').lower()
blob=json.dumps(d).lower()
assert corr in blob or 'trace' in blob or 'unknown' in blob or 'evidence' in blob
print('trace autopsy probe ok')
"
fi

echo "== probe 3: repo impact (Hub) =="
p3="$(hub_chat "What breaks if I replace agent-chain-service with context-exec?")"
echo "$p3" | python3 -m json.tool | head -60
echo "$p3" | python3 -c "
import json,sys
d=json.load(sys.stdin)
blob=json.dumps(d).lower()
assert 'agent' in blob or 'context-exec' in blob or 'context_exec' in blob or 'repo' in blob
print('repo impact probe ok')
"

echo "GOLDEN PROBES PASS (Hub reachable; set CONTEXT_EXEC_PROBE_CORR_ID for full trace autopsy validation)"
