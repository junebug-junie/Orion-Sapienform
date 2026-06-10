#!/usr/bin/env bash
# Hub golden-path probes for context-exec — must prove routing, not just HTTP 200 (#664).
# Requires: context-exec :8096, cortex-exec + cortex-orch with CONTEXT_EXEC_ENABLED=true, Hub :8080.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CTX_PORT="${CONTEXT_EXEC_PORT:-8096}"
HUB_BASE="${HUB_BASE_URL:-http://127.0.0.1:8080}"
CTX_BASE="http://127.0.0.1:${CTX_PORT}"
REAL_CORR_ID="${CONTEXT_EXEC_PROBE_CORR_ID:-}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== env posture =="
for f in services/orion-context-exec/.env services/orion-cortex-exec/.env services/orion-cortex-orch/.env; do
  if [[ -f "$f" ]]; then
    echo "--- $f ---"
    grep -E '^(CONTEXT_EXEC_|CHANNEL_CONTEXT_EXEC|CHANNEL_RECALL)' "$f" || true
  else
    echo "WARN: missing $f"
  fi
done

echo "== context-exec direct organ check =="
direct="$(curl -sf "${CTX_BASE}/context-exec/run" \
  -H 'content-type: application/json' \
  -d '{
    "text": "Where did Orion get the claim that I am from Denver?",
    "mode": "belief_provenance",
    "expected_artifact_type": "BeliefProvenanceReportV1"
  }')"
echo "$direct" | python3 -m json.tool 2>/dev/null | head -40 || true
echo "$direct" | python3 -c "
import json,sys
d=json.load(sys.stdin)
assert d.get('status') == 'ok', d
assert d.get('artifact_type') == 'BeliefProvenanceReportV1', d
print('direct context-exec organ ok')
"

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

hub_chat_json() {
  python3 - "$1" <<'PY'
import json, os, sys
spec = json.loads(sys.argv[1])
print(json.dumps(spec))
PY
}

hub_post() {
  local body="$1"
  curl -sf --max-time "${HUB_CHAT_TIMEOUT_SEC:-180}" "${HUB_BASE}/api/chat" \
    -H 'content-type: application/json' \
    -H 'X-Orion-No-Write: 1' \
    -d "$body"
}

echo "== probe 1: Denver belief provenance (Hub, mode=auto) =="
p1_body="$(hub_chat_json '{
  "mode": "auto",
  "use_recall": true,
  "no_write": true,
  "messages": [{
    "role": "user",
    "content": "Orion, where did the claim that I am from Denver come from across your runtime?"
  }],
  "options": {
    "route_intent": "auto",
    "answer_contract": {
      "request_kind": "runtime_debug",
      "requires_runtime_grounding": true,
      "allow_unverified_specifics": false,
      "preferred_render_style": "answer"
    }
  }
}')"
p1="$(hub_post "$p1_body")"
echo "$p1" | python3 -m json.tool 2>/dev/null | head -80 || true
echo "$p1" | python3 -c "
import json,sys
from scripts.context_exec_probe_lib import assert_hub_context_exec_routing
d=json.load(sys.stdin)
assert_hub_context_exec_routing(d, probe_name='denver_provenance', expected_mode='belief_provenance')
"

echo "== probe 2: trace autopsy (Hub, mode=auto) =="
if [[ -z "$REAL_CORR_ID" ]]; then
  echo "SKIP: set CONTEXT_EXEC_PROBE_CORR_ID to a real correlation id from a prior run"
else
  p2_body="$(REAL_CORR_ID="$REAL_CORR_ID" python3 - <<'PY'
import json, os
corr = os.environ["REAL_CORR_ID"]
print(json.dumps({
    "mode": "auto",
    "use_recall": True,
    "no_write": True,
    "messages": [{"role": "user", "content": f"Orion, why did corr {corr} fail open?"}],
    "options": {
        "route_intent": "auto",
        "answer_contract": {
            "request_kind": "runtime_debug",
            "requires_runtime_grounding": True,
            "allow_unverified_specifics": False,
            "preferred_render_style": "answer",
        },
    },
}))
PY
)"
  p2="$(hub_post "$p2_body")"
  echo "$p2" | python3 -m json.tool 2>/dev/null | head -80 || true
  echo "$p2" | python3 -c "
import json,sys
from scripts.context_exec_probe_lib import assert_hub_context_exec_routing
d=json.load(sys.stdin)
assert_hub_context_exec_routing(d, probe_name='trace_autopsy', expected_mode='trace_autopsy')
"
fi

echo "== probe 3: repo impact (Hub, mode=auto) =="
p3_body="$(hub_chat_json '{
  "mode": "auto",
  "use_recall": true,
  "no_write": true,
  "messages": [{
    "role": "user",
    "content": "What breaks if I replace agent-chain-service with context-exec? Ground this in my repo."
  }],
  "options": {
    "route_intent": "auto",
    "answer_contract": {
      "request_kind": "repo_technical",
      "requires_repo_grounding": true,
      "allow_unverified_specifics": false,
      "preferred_render_style": "answer"
    }
  }
}')"
p3="$(hub_post "$p3_body")"
echo "$p3" | python3 -m json.tool 2>/dev/null | head -80 || true
echo "$p3" | python3 -c "
import json,sys
from scripts.context_exec_probe_lib import assert_hub_context_exec_routing
d=json.load(sys.stdin)
assert_hub_context_exec_routing(d, probe_name='repo_impact', expected_mode='repo_impact_analysis')
"

echo "GOLDEN PROBES PASS — context-exec routing verified through Hub"
