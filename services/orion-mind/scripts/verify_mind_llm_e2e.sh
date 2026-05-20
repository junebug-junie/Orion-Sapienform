#!/usr/bin/env bash
# End-to-end Mind+LLM verification: preflight → Hub brain chat → mind_runs artifact → phase gates.
# Usage: ./services/orion-mind/scripts/verify_mind_llm_e2e.sh [HUB_BASE_URL]
set -euo pipefail

HUB_BASE="${1:-http://127.0.0.1:8080}"
SESSION_ID="${ORION_SESSION_ID:-$(uuidgen 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4())')}"
CHAT_TIMEOUT="${CHAT_TIMEOUT_SEC:-180}"
PROMPT="${VERIFY_PROMPT:-mind llm e2e verify $(date -u +%H:%M:%S) — one short reply.}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}PASS${NC} $*"; }
fail() { echo -e "${RED}FAIL${NC} $*"; FAILED=1; }
warn() { echo -e "${YELLOW}WARN${NC} $*"; }

FAILED=0

echo "=== Mind LLM E2E verification ==="
echo "hub=$HUB_BASE session=$SESSION_ID"

echo ""
echo "--- 1. Infrastructure preflight ---"

check_url() {
  local name="$1" url="$2"
  if curl -sf -m 8 "$url" >/dev/null 2>&1; then
    pass "$name reachable ($url)"
  else
    fail "$name unreachable ($url)"
  fi
}

check_url "hub" "$HUB_BASE/health"
check_url "mind-direct" "http://127.0.0.1:6611/health"

if docker exec orion-mind printenv MIND_LLM_SYNTHESIS_ENABLED 2>/dev/null | grep -q true; then
  pass "orion-mind MIND_LLM_SYNTHESIS_ENABLED=true"
else
  fail "orion-mind MIND_LLM_SYNTHESIS_ENABLED not true"
fi

if docker exec orion-mind printenv ORION_BUS_URL 2>/dev/null | grep -q redis; then
  pass "orion-mind ORION_BUS_URL set ($(docker exec orion-mind printenv ORION_BUS_URL 2>/dev/null))"
else
  fail "orion-mind ORION_BUS_URL missing"
fi

if docker exec orion-athena-cortex-orch curl -sf http://orion-mind:6611/health >/dev/null 2>&1; then
  pass "orch → mind HTTP (/health)"
else
  fail "orch cannot reach orion-mind:6611"
fi

GW_ROUTES=$(docker exec orion-llm-gateway printenv LLM_GATEWAY_ROUTE_TABLE_JSON 2>/dev/null || true)
for route in quick metacog chat; do
  if echo "$GW_ROUTES" | grep -q "\"$route\""; then
    pass "gateway route table includes $route"
  else
    fail "gateway route table missing $route"
  fi
done

echo ""
echo "--- 2. Hub brain chat (mind_enabled) ---"
CHAT_PAYLOAD=$(cat <<EOF
{
  "mode": "brain",
  "verbs": ["chat_general"],
  "messages": [{"role": "user", "content": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$PROMPT")}],
  "use_recall": false,
  "context": {"metadata": {"mind_enabled": true}},
  "surface_context": {"surface": "verify_script", "hub_chat_lane": "brain"}
}
EOF
)

CHAT_OUT=$(mktemp)
HTTP_CODE=$(curl -sS -m "$CHAT_TIMEOUT" -o "$CHAT_OUT" -w '%{http_code}' \
  -X POST "$HUB_BASE/api/chat" \
  -H "Content-Type: application/json" \
  -H "X-Orion-Session-Id: $SESSION_ID" \
  -d "$CHAT_PAYLOAD" || echo "000")

if [[ "$HTTP_CODE" != "200" ]]; then
  fail "POST /api/chat HTTP $HTTP_CODE"
  head -c 2000 "$CHAT_OUT" 2>/dev/null || true
  echo ""
else
  pass "POST /api/chat HTTP 200"
fi

CORR=$(python3 - <<'PY' "$CHAT_OUT"
import json, sys
path = sys.argv[1]
try:
    data = json.load(open(path))
except Exception:
    print("")
    sys.exit(0)
print(data.get("correlation_id") or "")
PY
)

if [[ -z "$CORR" ]]; then
  fail "chat response missing correlation_id"
else
  pass "correlation_id=$CORR"
fi

REPLY_LEN=$(python3 - <<'PY' "$CHAT_OUT"
import json, sys
data = json.load(open(sys.argv[1]))
text = data.get("text") or data.get("response") or ""
print(len(str(text)))
PY
)
if [[ "${REPLY_LEN:-0}" -gt 0 ]]; then
  pass "assistant reply received (${REPLY_LEN} chars)"
else
  warn "no assistant text in chat response (Mind may still have run)"
fi

echo ""
echo "--- 3. mind_runs artifact (Hub DB via API) ---"
sleep 2
RUNS_OUT=$(mktemp)
RUNS_CODE=$(curl -sS -m 30 -o "$RUNS_OUT" -w '%{http_code}' \
  "$HUB_BASE/api/mind/runs?correlation_id=$CORR" \
  -H "X-Orion-Session-Id: $SESSION_ID" || echo "000")

if [[ "$RUNS_CODE" != "200" ]]; then
  fail "GET /api/mind/runs HTTP $RUNS_CODE"
else
  pass "GET /api/mind/runs HTTP 200"
fi

python3 - <<'PY' "$RUNS_OUT" "$CORR"
import json, sys

path, corr = sys.argv[1], sys.argv[2]
data = json.load(open(path))
items = data if isinstance(data, list) else data.get("items") or data.get("runs") or []
if not items:
    print("FAIL no mind_runs row for correlation (Orch→Mind write path broken or mind skipped)")
    sys.exit(2)

row = items[0]
print(f"PASS mind_run_id={row.get('mind_run_id')} ok={row.get('ok')}")

result = row.get("result_jsonb") or row.get("result") or {}
if isinstance(result, str):
    try:
        result = json.loads(result)
    except Exception:
        result = {}

quality = result.get("mind_quality") or (result.get("brief") or {}).get("mind_quality")
machine = (result.get("brief") or {}).get("machine_contract") or {}
fail_open = machine.get("mind.llm_fail_open_to_deterministic")
error_code = machine.get("mind.llm_synthesis_error_code")
stance_skip = (result.get("brief") or {}).get("mind_authorized_for_stance_skip")

print(f"  mind_quality={quality}")
print(f"  fail_open={fail_open}")
print(f"  error_code={error_code}")
print(f"  stance_skip={stance_skip}")

phases = machine.get("mind.phase_telemetry") or []
for phase in ("semantic_synthesis", "active_frontier_judge", "stance_handoff"):
    rec = next((t for t in phases if t.get("phase_name") == phase), None)
    if not rec:
        print(f"  FAIL phase missing: {phase}")
        continue
    status = rec.get("status")
    vok = rec.get("validation_ok")
    err = rec.get("error")
    raw_n = rec.get("raw_claim_count")
    ret_n = rec.get("retained_claim_count")
    print(f"  phase {phase}: status={status} validation_ok={vok} raw={raw_n} retained={ret_n} error={err}")

gates_ok = True
if quality != "meaningful_synthesis":
    print(f"FAIL expected mind_quality=meaningful_synthesis got {quality}")
    gates_ok = False
if fail_open:
    print(f"FAIL mind.llm_fail_open_to_deterministic={fail_open}")
    gates_ok = False
sem = next((t for t in phases if t.get("phase_name") == "semantic_synthesis"), {})
if sem and sem.get("retained_claim_count", 0) < 1:
    print("FAIL semantic retained_claim_count < 1")
    gates_ok = False
if not stance_skip:
    print("WARN mind_authorized_for_stance_skip is false (may still be valid if legacy stance used)")

if gates_ok:
    print("PASS all Mind LLM quality gates")
    sys.exit(0)
sys.exit(3)
PY
GATE_RC=$?
[[ $GATE_RC -eq 0 ]] || FAILED=1

echo ""
echo "--- 4. Orch / Mind log hints (same correlation) ---"
if [[ -n "$CORR" ]]; then
  docker logs orion-athena-cortex-orch 2>&1 | rg "$CORR" | rg -i 'mind_run_artifact|mind_http|mind_skipped' | tail -5 || warn "no orch mind lines for correlation"
  docker logs orion-mind 2>&1 | rg "$CORR" | tail -8 || warn "no mind log lines for correlation"
fi

echo ""
if [[ "$FAILED" -eq 0 && "$GATE_RC" -eq 0 ]]; then
  echo -e "${GREEN}E2E VERIFIED${NC} — Hub brain path produced meaningful_synthesis Mind run."
  exit 0
fi
echo -e "${RED}E2E FAILED${NC} — see gates above; fix infra before chasing more shape normalizers."
exit 1
