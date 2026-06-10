#!/usr/bin/env bash
# Smoke: memory crystallization propose → validate → approve (Hub API)
set -euo pipefail

HUB_BASE_URL="${ORION_HUB_URL:-http://localhost:8080}"
SESSION_ID="${ORION_HUB_SESSION_ID:?Set ORION_HUB_SESSION_ID}"

HDR=(-H "Content-Type: application/json" -H "X-Orion-Session-Id: ${SESSION_ID}")

echo "== propose =="
PROPOSE=$(curl -sS -X POST "${HUB_BASE_URL}/api/memory/crystallizations/propose" "${HDR[@]}" -d '{
  "kind": "stance",
  "subject": "Smoke crystallization",
  "summary": "Local-first memory governance smoke test",
  "scope": ["project:orion"],
  "planning_effects": ["require governor approval"],
  "retrieval_affordances": ["retrieve_when:smoke"],
  "evidence": [{"source_kind": "operator_note", "source_id": "smoke-1", "excerpt": "smoke"}],
  "proposed_by": "smoke"
}')
echo "${PROPOSE}" | head -c 400
echo ""

CID=$(echo "${PROPOSE}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('crystallization_id',''))")
if [[ -z "${CID}" ]]; then
  echo "FAIL: no crystallization_id in propose response"
  exit 1
fi

echo "== validate =="
curl -sS -X POST "${HUB_BASE_URL}/api/memory/crystallizations/proposals/${CID}/validate" "${HDR[@]}" | head -c 300
echo ""

echo "== approve =="
APPROVE=$(curl -sS -X POST "${HUB_BASE_URL}/api/memory/crystallizations/proposals/${CID}/approve" "${HDR[@]}" -d '{}')
echo "${APPROVE}" | head -c 500
echo ""

STATUS=$(echo "${APPROVE}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")
if [[ "${STATUS}" != "active" ]]; then
  echo "FAIL: expected status=active got ${STATUS}"
  exit 1
fi

echo "== active-packet =="
curl -sS -X POST "${HUB_BASE_URL}/api/memory/active-packet" "${HDR[@]}" -d '{"query":"memory governance smoke","task_type":"architecture"}' | head -c 400
echo ""

echo "PASS smoke_memory_crystallization_e2e crystallization_id=${CID}"
