#!/usr/bin/env bash
set -euo pipefail

REQUEST_TEXT='Dry-run cleanup of stopped containers.'
SERVICES=(
  orion-athena-cortex-orch
  orion-athena-cortex-exec
  orion-athena-agent-chain
  orion-athena-planner-react
)

TMP_DIR="${TMP_DIR:-$(mktemp -d /tmp/orion-live-verify.XXXXXX)}"
START_TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

fail() {
  echo "FAIL: $*"
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

need_cmd docker
need_cmd curl
need_cmd python3

for svc in "${SERVICES[@]}"; do
  docker inspect "$svc" >/dev/null 2>&1 || fail "required container not found: $svc"
done

HUB_CONTAINER="${HUB_CONTAINER:-orion-athena-hub}"
docker inspect "$HUB_CONTAINER" >/dev/null 2>&1 || fail "hub container not found: $HUB_CONTAINER"

HUB_PORT="$({ docker inspect "$HUB_CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' || true; } | awk -F= '/^HUB_PORT=/{print $2; exit}')"
HUB_PORT="${HUB_PORT:-8000}"
HUB_URL="${HUB_URL:-http://127.0.0.1:${HUB_PORT}/api/chat}"

REQ_JSON="${TMP_DIR}/request.json"
cat >"$REQ_JSON" <<JSON
{
  "mode": "agent",
  "messages": [
    {"role": "user", "content": "${REQUEST_TEXT}"}
  ],
  "use_recall": false,
  "no_write": true
}
JSON

echo "Triggering live ingress via ${HUB_URL}"
RESP_JSON="${TMP_DIR}/response.json"
HTTP_CODE="$(curl -sS -o "$RESP_JSON" -w '%{http_code}' -H 'Content-Type: application/json' -X POST "$HUB_URL" --data-binary "@$REQ_JSON")"
[[ "$HTTP_CODE" == "200" ]] || fail "hub request failed, HTTP ${HTTP_CODE}, body=$(cat "$RESP_JSON")"
CORR_ID="$(python3 - <<'PY' "$RESP_JSON"
import json,sys
p=json.load(open(sys.argv[1]))
print(p.get('correlation_id') or '')
PY
)"
[[ -n "$CORR_ID" ]] || fail "missing correlation_id in hub response: $(cat "$RESP_JSON")"

echo "correlation_id=${CORR_ID}"

# Give services a small window to flush async logs.
sleep 4

for svc in "${SERVICES[@]}"; do
  docker logs --since "$START_TS" --timestamps "$svc" >"${TMP_DIR}/${svc}.log" 2>&1 || true
  grep -F "$CORR_ID" "${TMP_DIR}/${svc}.log" >"${TMP_DIR}/${svc}.corr.log" || true
  if [[ ! -s "${TMP_DIR}/${svc}.corr.log" ]]; then
    cp "${TMP_DIR}/${svc}.log" "${TMP_DIR}/${svc}.corr.log"
  fi

done

find_line() {
  local file="$1"; shift
  local pattern="$1"
  grep -E "$pattern" "$file" | head -n 1 || true
}

E1="$(find_line "${TMP_DIR}/orion-athena-cortex-exec.corr.log" 'bound_capability_request_received.*selected_verb=housekeep_runtime')"
[[ -n "$E1" ]] || fail "missing evidence #1 (supervisor selected housekeep_runtime)"

E2="$(find_line "${TMP_DIR}/orion-athena-agent-chain.corr.log" 'bound_capability_direct_execute=1.*selected_verb=housekeep_runtime.*selected_verb_preserved=1')"
[[ -n "$E2" ]] || fail "missing evidence #2 (selected_verb preserved into agent-chain)"

E3="$(find_line "${TMP_DIR}/orion-athena-cortex-orch.corr.log" 'orch_publish_verb_runtime.*skills\.runtime\.')"
[[ -n "$E3" ]] || fail "missing evidence #3 (downstream concrete skills.runtime.* invocation)"

E4="$(find_line "${TMP_DIR}/orion-athena-cortex-exec.corr.log" 'final_text_assembly .*verb=skills\.runtime\..*final_len=[1-9]')"
[[ -n "$E4" ]] || fail "missing evidence #4 (non-empty terminal output for concrete runtime skill)"

if grep -E 'bound_capability_execution_timeout' "${TMP_DIR}/orion-athena-agent-chain.corr.log" >/dev/null 2>&1; then
  fail "evidence #5 failed (bound_capability_execution_timeout present)"
fi

TIMINGS_JSON="${TMP_DIR}/timings.json"
python3 - <<'PY' "${TMP_DIR}/orion-athena-cortex-exec.corr.log" "$TIMINGS_JSON"
import json,re,sys
from datetime import datetime

log = open(sys.argv[1], encoding='utf-8', errors='ignore').read().splitlines()
out = {}

def first_ts(pattern):
    rx = re.compile(pattern)
    for line in log:
        if rx.search(line):
            ts = line.split(' ', 1)[0]
            try:
                return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp(), line
            except Exception:
                continue
    return None, None

pairs = {
  'planner_react': (r'rpc -> PlannerReactService', r'ok <- PlannerReactService'),
  'agent_chain': (r'rpc -> AgentChainService', r'ok <- AgentChainService'),
  'concrete_skill': (r'verb_runtime_intake .*trigger=legacy\.plan', r'final_text_assembly .*verb=skills\.runtime\.'),
}
for k,(s,e) in pairs.items():
    s_ts,s_line = first_ts(s)
    e_ts,e_line = first_ts(e)
    if s_ts is None or e_ts is None or e_ts < s_ts:
        print(f"MISSING:{k}")
        sys.exit(2)
    out[k] = {
        'duration_sec': round(e_ts - s_ts, 3),
        'start_line': s_line,
        'end_line': e_line,
    }
json.dump(out, open(sys.argv[2], 'w'), indent=2)
PY

echo "PASS: live bound capability verification"
echo "Evidence #1: $E1"
echo "Evidence #2: $E2"
echo "Evidence #3: $E3"
echo "Evidence #4: $E4"
echo "Evidence #5: no bound_capability_execution_timeout in agent-chain logs"
echo "Evidence #6 timings: $(cat "$TIMINGS_JSON")"
