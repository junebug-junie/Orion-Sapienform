#!/usr/bin/env bash
set -euo pipefail

HUB_URL="${HUB_URL:-http://localhost:8080}"
TIMEOUT="${TIMEOUT:-20}"
FAIL=0

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAIL=1; }

json_get() {
  local expr="$1"
  python - "$expr" <<'PY'
import json,sys
expr=sys.argv[1]
obj=json.load(sys.stdin)
val=obj
for part in expr.split('.'):
    if not part:
        continue
    if isinstance(val, dict):
        val=val.get(part)
    else:
        val=None
        break
print("" if val is None else val)
PY
}

echo "[1/4] Checking /api/verbs active flags"
verbs_json="$(curl -fsS --max-time "$TIMEOUT" "$HUB_URL/api/verbs?include_inactive=1")" || { fail "Unable to query /api/verbs"; verbs_json=''; }
if [[ -n "$verbs_json" ]]; then
  has_active="$(printf '%s' "$verbs_json" | python - <<'PY'
import json,sys
obj=json.load(sys.stdin)
verbs=obj.get('verbs') or []
ok=bool(verbs) and all('active' in v for v in verbs)
print('1' if ok else '0')
PY
)"
  if [[ "$has_active" == "1" ]]; then pass "/api/verbs returns active flags"; else fail "/api/verbs missing active flags"; fi
fi

echo "[2/4] Brain mode default verb should be chat_general"
brain_payload='{"messages":[{"role":"user","content":"smoke brain default"}],"mode":"brain","use_recall":false}'
brain_json="$(curl -fsS --max-time "$TIMEOUT" -H 'content-type: application/json' -d "$brain_payload" "$HUB_URL/api/chat")" || { fail "Brain mode request failed"; brain_json=''; }
if [[ -n "$brain_json" ]]; then
  brain_verb="$(printf '%s' "$brain_json" | python - <<'PY'
import json,sys
obj=json.load(sys.stdin)
print(((obj.get('raw') or {}).get('verb')) or '')
PY
)"
  if [[ "$brain_verb" == "chat_general" ]]; then pass "Brain mode defaulted to chat_general"; else fail "Expected brain raw.verb=chat_general got '$brain_verb'"; fi
fi

echo "[3/4] Agent mode with no verb should not be overwritten to chat_general"
agent_payload='{"messages":[{"role":"user","content":"smoke agent default"}],"mode":"agent","use_recall":false}'
agent_json="$(curl -fsS --max-time "$TIMEOUT" -H 'content-type: application/json' -d "$agent_payload" "$HUB_URL/api/chat")" || { fail "Agent mode request failed"; agent_json=''; }
if [[ -n "$agent_json" ]]; then
  agent_verb="$(printf '%s' "$agent_json" | python - <<'PY'
import json,sys
obj=json.load(sys.stdin)
print(((obj.get('raw') or {}).get('verb')) or '')
PY
)"
  if [[ "$agent_verb" != "chat_general" ]]; then pass "Agent mode did not default to chat_general (verb='$agent_verb')"; else fail "Agent mode was overwritten to chat_general"; fi
fi

echo "[4/4] Explicit inactive verb override should fail clearly"
disabled_payload='{"messages":[{"role":"user","content":"smoke inactive"}],"mode":"brain","verbs":["dream_simple"],"use_recall":false}'
disabled_json="$(curl -fsS --max-time "$TIMEOUT" -H 'content-type: application/json' -d "$disabled_payload" "$HUB_URL/api/chat")" || { fail "Inactive verb request transport failed"; disabled_json=''; }
if [[ -n "$disabled_json" ]]; then
  inactive_error="$(printf '%s' "$disabled_json" | python - <<'PY'
import json,sys
obj=json.load(sys.stdin)
msg=(obj.get('error') or '') + ' ' + (obj.get('message') or '')
print('1' if 'inactive_verb' in msg or 'inactive on node' in msg else '0')
PY
)"
  if [[ "$inactive_error" == "1" ]]; then pass "Inactive verb override returns clear error"; else fail "Inactive verb override did not return expected error"; fi
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo "SMOKE RESULT: FAIL"
  exit 1
fi

echo "SMOKE RESULT: PASS"
