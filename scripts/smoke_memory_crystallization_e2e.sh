#!/usr/bin/env bash
# Live-stack smoke: memory crystallization propose → validate → approve → active-packet
set -euo pipefail

: "${ORION_HUB_URL:?set ORION_HUB_URL (e.g. http://127.0.0.1:8080)}"
: "${ORION_HUB_SESSION_ID:?set ORION_HUB_SESSION_ID}"
: "${RECALL_PG_DSN:?set RECALL_PG_DSN (same contract as Hub RECALL_PG_DSN)}"

if ! command -v curl >/dev/null 2>&1; then echo "FAIL: curl missing"; exit 1; fi
if ! command -v jq >/dev/null 2>&1; then echo "FAIL: jq missing"; exit 1; fi

BASE="${ORION_HUB_URL%/}"
HDR=(-H "Content-Type: application/json" -H "X-Orion-Session-Id: ${ORION_HUB_SESSION_ID}")
STAMP="$(date -u +%Y%m%d%H%M%S)"

echo "== health: projection =="
curl -sS "${HDR[@]}" "${BASE}/api/memory/crystallizations/projection/health" | jq -c .

echo "== propose =="
PROPOSE=$(curl -sS -w "\n%{http_code}" -X POST "${BASE}/api/memory/crystallizations/propose" "${HDR[@]}" -d "$(jq -n --arg s "Smoke ${STAMP}" '{
  kind: "stance",
  subject: "Smoke crystallization",
  summary: $s,
  scope: ["project:orion"],
  planning_effects: ["require governor approval"],
  retrieval_affordances: ["retrieve_when:smoke"],
  evidence: [{source_kind: "operator_note", source_id: "smoke-1", excerpt: "smoke"}],
  proposed_by: "smoke"
}')")
BODY="$(echo "$PROPOSE" | head -n -1)"
CODE="$(echo "$PROPOSE" | tail -n 1)"
[[ "$CODE" == "200" ]] || { echo "FAIL propose HTTP $CODE body=$BODY"; exit 1; }
CID="$(echo "$BODY" | jq -r '.crystallization_id // empty')"
[[ -n "$CID" ]] || { echo "FAIL no crystallization_id"; exit 1; }

echo "== validate =="
VAL=$(curl -sS -w "\n%{http_code}" -X POST "${BASE}/api/memory/crystallizations/proposals/${CID}/validate" "${HDR[@]}")
VAL_BODY="$(echo "$VAL" | head -n -1)"
VAL_CODE="$(echo "$VAL" | tail -n 1)"
[[ "$VAL_CODE" == "200" ]] || { echo "FAIL validate HTTP $VAL_CODE"; exit 1; }
echo "$VAL_BODY" | jq -c '{valid, errors, detection}'

echo "== approve =="
APP=$(curl -sS -w "\n%{http_code}" -X POST "${BASE}/api/memory/crystallizations/proposals/${CID}/approve" "${HDR[@]}" -d '{}')
APP_BODY="$(echo "$APP" | head -n -1)"
APP_CODE="$(echo "$APP" | tail -n 1)"
[[ "$APP_CODE" == "200" ]] || { echo "FAIL approve HTTP $APP_CODE body=$APP_BODY"; exit 1; }
STATUS="$(echo "$APP_BODY" | jq -r '.status')"
[[ "$STATUS" == "active" ]] || { echo "FAIL status=$STATUS"; exit 1; }

echo "== active-packet =="
PKT=$(curl -sS -w "\n%{http_code}" -X POST "${BASE}/api/memory/active-packet" "${HDR[@]}" -d "{\"query\":\"memory smoke ${STAMP}\",\"task_type\":\"architecture\",\"seed_crystallization_id\":\"${CID}\"}")
PKT_BODY="$(echo "$PKT" | head -n -1)"
PKT_CODE="$(echo "$PKT" | tail -n 1)"
[[ "$PKT_CODE" == "200" ]] || { echo "FAIL active-packet HTTP $PKT_CODE"; exit 1; }
echo "$PKT_BODY" | jq -c '{crystallization_refs, card_refs, chroma_refs, retrieval_trace}'

echo "PASS smoke_memory_crystallization_e2e crystallization_id=${CID}"
